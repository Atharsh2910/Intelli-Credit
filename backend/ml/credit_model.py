"""
Intelli-Credit: ML Credit Scoring Pipeline
- Ensemble learning (XGBoost + LightGBM + RandomForest) for PD prediction
- Loan limit regression model
- Interest rate pricing model
- SHAP explainability
- Prophet time-series forecasting
"""

import os
import json
import warnings
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Feature Definitions (must match GAN generator)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "dscr", "interest_coverage_ratio", "ebitda_margin_pct",
    "revenue_growth_3yr_cagr", "pat_margin_pct",
    "debt_equity_ratio", "tangible_net_worth_cr", "promoter_contribution_pct",
    "cash_accrual_to_debt",
    "current_ratio", "quick_ratio",
    "facr", "security_coverage",
    "gst_compliance_score", "gstr_2a_3b_discrepancy_pct",
    "bank_credit_debit_ratio", "emi_bounce_count", "avg_bank_utilisation_pct",
    "cibil_score", "litigation_count", "years_in_business",
    "promoter_experience_yrs", "secondary_research_risk",
    "sector_outlook_score", "macro_risk_score",
]


# ---------------------------------------------------------------------------
# Ensemble Credit Risk Model
# ---------------------------------------------------------------------------

class CreditRiskModel:
    """Ensemble ML model for Probability of Default (PD) prediction."""

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(self.model_dir, exist_ok=True)

        self.xgb_model = None
        self.lgb_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.shap_explainer = None
        self.is_trained = False
        self.training_metrics = {}

        # Try to load pre-trained model
        self._try_load()

    def train(self, data: pd.DataFrame, target_col: str = "default") -> Dict[str, Any]:
        """Train ensemble models on data."""
        # Prepare features
        feature_cols = [c for c in FEATURE_NAMES if c in data.columns]
        X = data[feature_cols].copy()
        y = data[target_col].copy()

        # Handle missing values
        X = X.fillna(X.median())

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )

        # Apply SMOTE for class imbalance
        if HAS_SMOTE and y_train.mean() < 0.3:
            smote = SMOTE(random_state=42, sampling_strategy=0.4)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # 1. XGBoost
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # 2. LightGBM
        if HAS_LGBM:
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )
            self.lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        # 3. Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.rf_model.fit(X_train, y_train)

        # Evaluate ensemble
        y_pred_proba = self._ensemble_predict_proba(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "auc_roc": round(float(roc_auc_score(y_test, y_pred_proba)), 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "default_rate_train": round(float(y_train.mean()), 4),
            "default_rate_test": round(float(y_test.mean()), 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(feature_cols),
            "features_used": feature_cols,
        }

        self.training_metrics = metrics
        self.is_trained = True

        # Initialize SHAP
        if HAS_SHAP:
            self.shap_explainer = shap.TreeExplainer(self.xgb_model)

        # Save models
        self._save()

        return metrics

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict default probability for a single company."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Run train() or load a pre-trained model.")

        # Build feature vector
        X = pd.DataFrame([features])
        for col in FEATURE_NAMES:
            if col not in X.columns:
                X[col] = 0.0

        X = X[FEATURE_NAMES].fillna(0)
        X_scaled = self.scaler.transform(X)

        # Ensemble prediction
        pd_prob = float(self._ensemble_predict_proba(X_scaled)[0])

        # SHAP explanation
        shap_values = self._get_shap_values(X_scaled)

        return {
            "default_probability": round(pd_prob, 4),
            "credit_score": self._prob_to_score(pd_prob),
            "shap_explanation": shap_values,
        }

    def _ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction."""
        probas = []
        weights = []

        if self.xgb_model:
            probas.append(self.xgb_model.predict_proba(X)[:, 1])
            weights.append(0.45)

        if self.lgb_model:
            probas.append(self.lgb_model.predict_proba(X)[:, 1])
            weights.append(0.30)

        if self.rf_model:
            probas.append(self.rf_model.predict_proba(X)[:, 1])
            weights.append(0.25 if self.lgb_model else 0.55)

        if not probas:
            return np.array([0.5])

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        return sum(p * w for p, w in zip(probas, weights))

    def _get_shap_values(self, X_scaled: np.ndarray) -> List[Dict]:
        """Get SHAP values for explanation."""
        if not HAS_SHAP or self.shap_explainer is None:
            return []

        try:
            sv = self.shap_explainer.shap_values(X_scaled)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]

            shap_dict = {}
            for i, name in enumerate(FEATURE_NAMES):
                if i < sv.shape[1]:
                    shap_dict[name] = float(sv[0][i])

            # Sort by absolute impact
            sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

            return [
                {
                    "feature": name,
                    "impact": round(val, 4),
                    "direction": "risk_increasing" if val > 0 else "risk_decreasing",
                    "description": self._feature_description(name, val),
                }
                for name, val in sorted_features[:10]  # Top 10
            ]
        except Exception:
            return []

    def _feature_description(self, feature: str, shap_value: float) -> str:
        """Human-readable description of SHAP impact."""
        direction = "increases" if shap_value > 0 else "decreases"
        magnitude = "strongly" if abs(shap_value) > 0.15 else "moderately" if abs(shap_value) > 0.05 else "slightly"

        descriptions = {
            "dscr": f"Debt Service Coverage Ratio {magnitude} {direction} default risk",
            "debt_equity_ratio": f"Debt-to-Equity ratio {magnitude} {direction} default risk",
            "cibil_score": f"CIBIL score {magnitude} {direction} default risk",
            "emi_bounce_count": f"EMI bounce history {magnitude} {direction} default risk",
            "litigation_count": f"Active litigations {magnitude} {direction} default risk",
            "gst_compliance_score": f"GST compliance {magnitude} {direction} default risk",
            "gstr_2a_3b_discrepancy_pct": f"GST return discrepancy {magnitude} {direction} default risk",
            "current_ratio": f"Current ratio {magnitude} {direction} default risk",
            "ebitda_margin_pct": f"EBITDA margin {magnitude} {direction} default risk",
            "revenue_growth_3yr_cagr": f"Revenue growth {magnitude} {direction} default risk",
            "security_coverage": f"Security coverage {magnitude} {direction} default risk",
            "years_in_business": f"Business vintage {magnitude} {direction} default risk",
            "promoter_experience_yrs": f"Promoter experience {magnitude} {direction} default risk",
            "avg_bank_utilisation_pct": f"Bank utilisation {magnitude} {direction} default risk",
            "sector_outlook_score": f"Sector outlook {magnitude} {direction} default risk",
        }
        return descriptions.get(feature, f"{feature} {magnitude} {direction} default risk")

    def _prob_to_score(self, prob: float) -> int:
        """Convert default probability to credit score (300-900 scale)."""
        # Higher prob → lower score
        score = 900 - int(prob * 600)
        return max(300, min(900, score))

    def _save(self) -> None:
        """Save all model artifacts."""
        if self.xgb_model:
            joblib.dump(self.xgb_model, os.path.join(self.model_dir, "xgb_model.pkl"))
        if self.lgb_model:
            joblib.dump(self.lgb_model, os.path.join(self.model_dir, "lgb_model.pkl"))
        if self.rf_model:
            joblib.dump(self.rf_model, os.path.join(self.model_dir, "rf_model.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))
        with open(os.path.join(self.model_dir, "metrics.json"), "w") as f:
            json.dump(self.training_metrics, f, indent=2)

    def _try_load(self) -> None:
        """Try to load pre-trained models."""
        try:
            xgb_path = os.path.join(self.model_dir, "xgb_model.pkl")
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if os.path.exists(xgb_path) and os.path.exists(scaler_path):
                self.xgb_model = joblib.load(xgb_path)
                self.scaler = joblib.load(scaler_path)

                lgb_path = os.path.join(self.model_dir, "lgb_model.pkl")
                if os.path.exists(lgb_path):
                    self.lgb_model = joblib.load(lgb_path)

                rf_path = os.path.join(self.model_dir, "rf_model.pkl")
                if os.path.exists(rf_path):
                    self.rf_model = joblib.load(rf_path)

                metrics_path = os.path.join(self.model_dir, "metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        self.training_metrics = json.load(f)

                if HAS_SHAP:
                    self.shap_explainer = shap.TreeExplainer(self.xgb_model)

                self.is_trained = True
        except Exception:
            self.is_trained = False


# ---------------------------------------------------------------------------
# Loan Limit Regression Model
# ---------------------------------------------------------------------------

class LoanLimitModel:
    """Predicts safe credit exposure (loan amount)."""

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), "models")
        self.gb_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self._try_load()

    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train loan limit models."""
        feature_cols = [c for c in FEATURE_NAMES if c in data.columns]
        # Additional features for loan sizing
        extra_cols = ["revenue_cr", "tangible_net_worth_cr"]
        all_cols = [c for c in feature_cols + extra_cols if c in data.columns]

        X = data[all_cols].fillna(0)
        y = data["loan_amount_cr"]

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Gradient Boosting
        self.gb_model = GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42,
        )
        self.gb_model.fit(X_train, y_train)

        # Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=150, max_depth=6, random_state=42, n_jobs=-1,
        )
        self.rf_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self._ensemble_predict(X_test)
        metrics = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "r2": round(float(r2_score(y_test, y_pred)), 4),
        }

        self.is_trained = True
        self._save()
        return metrics

    def predict(self, features: Dict[str, float]) -> float:
        """Predict loan amount in crores."""
        if not self.is_trained:
            return self._heuristic_predict(features)

        X = pd.DataFrame([features])
        all_cols = FEATURE_NAMES + ["revenue_cr"]
        for col in all_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[[c for c in all_cols if c in X.columns]].fillna(0)
        X_scaled = self.scaler.transform(X)

        return float(max(0.1, self._ensemble_predict(X_scaled)[0]))

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        if self.gb_model:
            preds.append(self.gb_model.predict(X) * 0.55)
        if self.rf_model:
            preds.append(self.rf_model.predict(X) * 0.45)
        return sum(preds) if preds else np.array([1.0])

    def _heuristic_predict(self, features: Dict) -> float:
        """Fallback heuristic when model not trained."""
        tnw = features.get("tangible_net_worth_cr", 2.0)
        dscr = features.get("dscr", 1.3)
        security = features.get("security_coverage", 1.2)
        revenue = features.get("revenue_cr", 5.0)

        tnw_based = tnw * 1.5
        dscr_factor = min(dscr / 1.25, 1.5)
        security_factor = min(security / 1.33, 1.5)
        revenue_based = revenue * 0.3

        return round(min(tnw_based * dscr_factor, revenue_based * security_factor), 2)

    def _save(self):
        if self.gb_model:
            joblib.dump(self.gb_model, os.path.join(self.model_dir, "loan_gb_model.pkl"))
        if self.rf_model:
            joblib.dump(self.rf_model, os.path.join(self.model_dir, "loan_rf_model.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "loan_scaler.pkl"))

    def _try_load(self):
        try:
            gb_path = os.path.join(self.model_dir, "loan_gb_model.pkl")
            scaler_path = os.path.join(self.model_dir, "loan_scaler.pkl")
            if os.path.exists(gb_path) and os.path.exists(scaler_path):
                self.gb_model = joblib.load(gb_path)
                self.scaler = joblib.load(scaler_path)
                rf_path = os.path.join(self.model_dir, "loan_rf_model.pkl")
                if os.path.exists(rf_path):
                    self.rf_model = joblib.load(rf_path)
                self.is_trained = True
        except Exception:
            self.is_trained = False


# ---------------------------------------------------------------------------
# Interest Rate Pricing Model
# ---------------------------------------------------------------------------

class InterestRateModel:
    """Predicts risk-based pricing (interest rate)."""

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), "models")
        self.ridge_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self._try_load()

    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train interest rate models."""
        feature_cols = [c for c in FEATURE_NAMES if c in data.columns]
        X = data[feature_cols].fillna(0)
        y = data["interest_rate_pct"]

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Ridge Regression
        self.ridge_model = Ridge(alpha=1.0)
        self.ridge_model.fit(X_train, y_train)

        # Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1,
        )
        self.rf_model.fit(X_train, y_train)

        y_pred = self._ensemble_predict(X_test)
        metrics = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "r2": round(float(r2_score(y_test, y_pred)), 4),
        }

        self.is_trained = True
        self._save()
        return metrics

    def predict(self, features: Dict[str, float]) -> float:
        """Predict interest rate."""
        if not self.is_trained:
            return self._heuristic_predict(features)

        X = pd.DataFrame([features])
        for col in FEATURE_NAMES:
            if col not in X.columns:
                X[col] = 0.0
        X = X[FEATURE_NAMES].fillna(0)
        X_scaled = self.scaler.transform(X)

        rate = float(self._ensemble_predict(X_scaled)[0])
        return round(max(7.0, min(22.0, rate)), 2)

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        if self.ridge_model:
            preds.append(self.ridge_model.predict(X) * 0.4)
        if self.rf_model:
            preds.append(self.rf_model.predict(X) * 0.6)
        return sum(preds) if preds else np.array([12.0])

    def _heuristic_predict(self, features: Dict) -> float:
        """Fallback heuristic interest rate."""
        base_rate = 10.5
        cibil = features.get("cibil_score", 700)
        dscr = features.get("dscr", 1.3)

        # CIBIL adjustment
        if cibil >= 750:
            base_rate -= 0.5
        elif cibil < 650:
            base_rate += 1.0

        # DSCR adjustment
        if dscr >= 2.0:
            base_rate -= 0.5
        elif dscr < 1.25:
            base_rate += 1.0

        return round(max(7.0, min(22.0, base_rate)), 2)

    def _save(self):
        if self.ridge_model:
            joblib.dump(self.ridge_model, os.path.join(self.model_dir, "rate_ridge_model.pkl"))
        if self.rf_model:
            joblib.dump(self.rf_model, os.path.join(self.model_dir, "rate_rf_model.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "rate_scaler.pkl"))

    def _try_load(self):
        try:
            ridge_path = os.path.join(self.model_dir, "rate_ridge_model.pkl")
            scaler_path = os.path.join(self.model_dir, "rate_scaler.pkl")
            if os.path.exists(ridge_path) and os.path.exists(scaler_path):
                self.ridge_model = joblib.load(ridge_path)
                self.scaler = joblib.load(scaler_path)
                rf_path = os.path.join(self.model_dir, "rate_rf_model.pkl")
                if os.path.exists(rf_path):
                    self.rf_model = joblib.load(rf_path)
                self.is_trained = True
        except Exception:
            self.is_trained = False


# ---------------------------------------------------------------------------
# Time Series Forecasting
# ---------------------------------------------------------------------------

class CashflowForecaster:
    """Prophet-based cashflow and revenue forecasting."""

    def __init__(self):
        self.model = None

    def forecast(self, historical_data: List[Dict], periods: int = 8) -> Dict[str, Any]:
        """
        Forecast future values.

        historical_data: list of {"date": "YYYY-MM-DD", "value": float}
        periods: number of future periods (quarters)
        """
        try:
            from prophet import Prophet

            df = pd.DataFrame(historical_data)
            df.columns = ["ds", "y"]
            df["ds"] = pd.to_datetime(df["ds"])

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
            )
            model.fit(df)

            future = model.make_future_dataframe(periods=periods, freq="QE")
            forecast = model.predict(future)

            # Extract forecasted values
            future_rows = forecast[forecast["ds"] > df["ds"].max()]

            predictions = []
            for _, row in future_rows.iterrows():
                predictions.append({
                    "date": row["ds"].strftime("%Y-%m-%d"),
                    "predicted_value": round(float(row["yhat"]), 2),
                    "lower_bound": round(float(row["yhat_lower"]), 2),
                    "upper_bound": round(float(row["yhat_upper"]), 2),
                })

            # Detect liquidity stress
            last_actual = float(df["y"].iloc[-1])
            avg_predicted = np.mean([p["predicted_value"] for p in predictions[:4]])
            stress_flag = avg_predicted < last_actual * 0.8  # >20% decline

            return {
                "predictions": predictions,
                "trend": "declining" if avg_predicted < last_actual else "growing",
                "liquidity_stress": stress_flag,
                "stress_details": f"Predicted {abs(1 - avg_predicted / last_actual) * 100:.1f}% {'decline' if stress_flag else 'growth'} in next 4 quarters" if last_actual > 0 else "Insufficient data",
            }

        except Exception as e:
            # Fallback: simple linear extrapolation
            return self._simple_forecast(historical_data, periods)

    def _simple_forecast(self, data: List[Dict], periods: int) -> Dict[str, Any]:
        """Simple linear extrapolation fallback."""
        if not data or len(data) < 2:
            return {"predictions": [], "trend": "unknown", "liquidity_stress": False}

        values = [d["value"] for d in data]
        n = len(values)
        x = np.arange(n)
        slope = np.polyfit(x, values, 1)[0]

        predictions = []
        last_val = values[-1]
        for i in range(periods):
            pred_val = last_val + slope * (i + 1)
            predictions.append({
                "date": f"Q{i + 1}_forecast",
                "predicted_value": round(float(pred_val), 2),
                "lower_bound": round(float(pred_val * 0.9), 2),
                "upper_bound": round(float(pred_val * 1.1), 2),
            })

        avg_pred = np.mean([p["predicted_value"] for p in predictions[:4]])
        stress_flag = avg_pred < last_val * 0.8

        return {
            "predictions": predictions,
            "trend": "declining" if slope < 0 else "growing",
            "liquidity_stress": stress_flag,
            "stress_details": f"Linear trend: slope={slope:.2f} per period",
        }
