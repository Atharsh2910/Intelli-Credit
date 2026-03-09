"""
Intelli-Credit: GAN Synthetic Data Generator
Uses CTGAN to generate realistic synthetic SME financial datasets.
- Seed dataset creation with Indian SME distributions
- CTGAN training and generation
- Distribution validation (KS test, correlation)
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# CTGAN import with fallback
try:
    from ctgan import CTGAN
    HAS_CTGAN = True
except ImportError:
    HAS_CTGAN = False


# ---------------------------------------------------------------------------
# Feature Definitions
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # Capacity (5)
    "dscr", "interest_coverage_ratio", "ebitda_margin_pct",
    "revenue_growth_3yr_cagr", "pat_margin_pct",
    # Capital (4)
    "debt_equity_ratio", "tangible_net_worth_cr", "promoter_contribution_pct",
    "cash_accrual_to_debt",
    # Liquidity (2)
    "current_ratio", "quick_ratio",
    # Collateral (2)
    "facr", "security_coverage",
    # GST & Banking (5)
    "gst_compliance_score", "gstr_2a_3b_discrepancy_pct",
    "bank_credit_debit_ratio", "emi_bounce_count", "avg_bank_utilisation_pct",
    # Character (5)
    "cibil_score", "litigation_count", "years_in_business",
    "promoter_experience_yrs", "secondary_research_risk",
    # Conditions (2)
    "sector_outlook_score", "macro_risk_score",
]

# Indian SME distribution parameters (based on RBI/MSME data patterns)
FEATURE_DISTRIBUTIONS = {
    "dscr":                       {"dist": "lognormal", "mean": 1.4, "sigma": 0.4, "min": 0.3, "max": 5.0},
    "interest_coverage_ratio":    {"dist": "lognormal", "mean": 2.5, "sigma": 0.6, "min": 0.5, "max": 10.0},
    "ebitda_margin_pct":          {"dist": "normal",    "mean": 12, "std": 6, "min": -10, "max": 40},
    "revenue_growth_3yr_cagr":    {"dist": "normal",    "mean": 10, "std": 12, "min": -30, "max": 60},
    "pat_margin_pct":             {"dist": "normal",    "mean": 5, "std": 4, "min": -15, "max": 25},
    "debt_equity_ratio":          {"dist": "lognormal", "mean": 1.2, "sigma": 0.5, "min": 0.1, "max": 6.0},
    "tangible_net_worth_cr":      {"dist": "lognormal", "mean": 3, "sigma": 1, "min": 0.1, "max": 100},
    "promoter_contribution_pct":  {"dist": "normal",    "mean": 55, "std": 15, "min": 20, "max": 100},
    "cash_accrual_to_debt":       {"dist": "normal",    "mean": 0.15, "std": 0.08, "min": 0, "max": 0.5},
    "current_ratio":              {"dist": "lognormal", "mean": 1.3, "sigma": 0.3, "min": 0.5, "max": 4.0},
    "quick_ratio":                {"dist": "lognormal", "mean": 0.9, "sigma": 0.3, "min": 0.2, "max": 3.0},
    "facr":                       {"dist": "lognormal", "mean": 1.5, "sigma": 0.4, "min": 0.5, "max": 5.0},
    "security_coverage":          {"dist": "lognormal", "mean": 1.4, "sigma": 0.3, "min": 0.5, "max": 4.0},
    "gst_compliance_score":       {"dist": "normal",    "mean": 7, "std": 2, "min": 0, "max": 10},
    "gstr_2a_3b_discrepancy_pct": {"dist": "exponential", "scale": 8, "min": 0, "max": 60},
    "bank_credit_debit_ratio":    {"dist": "normal",    "mean": 1.05, "std": 0.15, "min": 0.5, "max": 2.0},
    "emi_bounce_count":           {"dist": "poisson",   "lam": 1.5, "min": 0, "max": 20},
    "avg_bank_utilisation_pct":   {"dist": "normal",    "mean": 65, "std": 18, "min": 10, "max": 100},
    "cibil_score":                {"dist": "normal",    "mean": 700, "std": 60, "min": 300, "max": 900},
    "litigation_count":           {"dist": "poisson",   "lam": 2, "min": 0, "max": 20},
    "years_in_business":          {"dist": "lognormal", "mean": 2.5, "sigma": 0.7, "min": 0.5, "max": 50},
    "promoter_experience_yrs":    {"dist": "lognormal", "mean": 2.8, "sigma": 0.5, "min": 1, "max": 40},
    "secondary_research_risk":    {"dist": "categorical", "probs": {0: 0.5, 1: 0.35, 2: 0.15}},
    "sector_outlook_score":       {"dist": "normal",    "mean": 6, "std": 1.5, "min": 1, "max": 10},
    "macro_risk_score":           {"dist": "normal",    "mean": 5, "std": 1.5, "min": 1, "max": 10},
}

# Additional columns for diversity
ADDITIONAL_COLUMNS = {
    "revenue_cr": {"dist": "lognormal", "mean": 2.5, "sigma": 1.2, "min": 0.5, "max": 500},
    "loan_amount_cr": {"dist": "lognormal", "mean": 1.5, "sigma": 0.8, "min": 0.1, "max": 100},
    "interest_rate_pct": {"dist": "normal", "mean": 12, "std": 2.5, "min": 7, "max": 22},
}


class SyntheticDataGenerator:
    """CTGAN-based synthetic SME financial data generator."""

    def __init__(self, epochs: int = 300, batch_size: int = 500):
        self.epochs = epochs
        self.batch_size = batch_size
        self.ctgan = None
        self._seed_data = None

    def generate_seed_dataset(self, n_samples: int = 2000) -> pd.DataFrame:
        """Generate realistic seed dataset based on Indian SME distributions."""
        np.random.seed(42)
        data = {}

        # Generate features
        all_defs = {**FEATURE_DISTRIBUTIONS, **ADDITIONAL_COLUMNS}
        for col_name, params in all_defs.items():
            data[col_name] = self._sample_distribution(params, n_samples)

        df = pd.DataFrame(data)

        # Generate correlated default labels using ML-inspired logic
        df["default"] = self._generate_default_labels(df)

        # Introduce realistic correlations
        df = self._add_correlations(df)

        self._seed_data = df
        return df

    def _sample_distribution(self, params: Dict, n: int) -> np.ndarray:
        """Sample from specified distribution."""
        dist = params["dist"]
        lo = params.get("min", -np.inf)
        hi = params.get("max", np.inf)

        if dist == "normal":
            values = np.random.normal(params["mean"], params["std"], n)
        elif dist == "lognormal":
            values = np.random.lognormal(params["mean"], params["sigma"], n)
            # Rescale lognormal to sensible range
            values = values / np.median(values) * np.exp(params["mean"])
        elif dist == "exponential":
            values = np.random.exponential(params["scale"], n)
        elif dist == "poisson":
            values = np.random.poisson(params["lam"], n).astype(float)
        elif dist == "categorical":
            choices = list(params["probs"].keys())
            probs = list(params["probs"].values())
            values = np.random.choice(choices, size=n, p=probs).astype(float)
        else:
            values = np.random.normal(0, 1, n)

        return np.clip(values, lo, hi)

    def _generate_default_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate realistic default labels using feature-based scoring."""
        n = len(df)
        risk_score = np.zeros(n)

        # Higher DSCR → lower risk
        if "dscr" in df.columns:
            risk_score -= np.clip((df["dscr"] - 1.25) * 0.5, -1, 1)

        # Higher debt_equity → higher risk
        if "debt_equity_ratio" in df.columns:
            risk_score += np.clip((df["debt_equity_ratio"] - 2.0) * 0.3, -0.5, 1)

        # Lower CIBIL → higher risk
        if "cibil_score" in df.columns:
            risk_score -= np.clip((df["cibil_score"] - 650) / 100 * 0.4, -0.5, 1)

        # More bounces → higher risk
        if "emi_bounce_count" in df.columns:
            risk_score += np.clip(df["emi_bounce_count"] * 0.1, 0, 0.5)

        # More litigation → higher risk
        if "litigation_count" in df.columns:
            risk_score += np.clip(df["litigation_count"] * 0.05, 0, 0.3)

        # GST discrepancy → higher risk
        if "gstr_2a_3b_discrepancy_pct" in df.columns:
            risk_score += np.clip((df["gstr_2a_3b_discrepancy_pct"] - 10) * 0.02, 0, 0.3)

        # Low current ratio → higher risk
        if "current_ratio" in df.columns:
            risk_score -= np.clip((df["current_ratio"] - 1.1) * 0.3, -0.5, 0.5)

        # High utilisation → higher risk
        if "avg_bank_utilisation_pct" in df.columns:
            risk_score += np.clip((df["avg_bank_utilisation_pct"] - 80) * 0.015, -0.3, 0.3)

        # Add noise
        risk_score += np.random.normal(0, 0.3, n)

        # Convert to probability and sample
        prob_default = 1 / (1 + np.exp(-risk_score))  # Sigmoid
        labels = (np.random.random(n) < prob_default).astype(int)

        # Target ~20% default rate (realistic for SMEs)
        target_rate = 0.20
        current_rate = labels.mean()
        if current_rate > target_rate + 0.05:
            # Randomly flip some 1s to 0s
            ones_idx = np.where(labels == 1)[0]
            n_flip = int((current_rate - target_rate) * n)
            flip_idx = np.random.choice(ones_idx, size=min(n_flip, len(ones_idx)), replace=False)
            labels[flip_idx] = 0
        elif current_rate < target_rate - 0.05:
            zeros_idx = np.where(labels == 0)[0]
            n_flip = int((target_rate - current_rate) * n)
            flip_idx = np.random.choice(zeros_idx, size=min(n_flip, len(zeros_idx)), replace=False)
            labels[flip_idx] = 1

        return labels

    def _add_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic correlations between features."""
        # Revenue growth ↔ EBITDA margin (modest positive)
        if "revenue_growth_3yr_cagr" in df.columns and "ebitda_margin_pct" in df.columns:
            noise = np.random.normal(0, 2, len(df))
            df["ebitda_margin_pct"] = df["ebitda_margin_pct"] + df["revenue_growth_3yr_cagr"] * 0.1 + noise
            df["ebitda_margin_pct"] = df["ebitda_margin_pct"].clip(-10, 40)

        # PAT margin correlated with EBITDA margin
        if "pat_margin_pct" in df.columns and "ebitda_margin_pct" in df.columns:
            df["pat_margin_pct"] = df["ebitda_margin_pct"] * 0.5 + np.random.normal(0, 2, len(df))
            df["pat_margin_pct"] = df["pat_margin_pct"].clip(-15, 25)

        # Quick ratio < current ratio
        if "quick_ratio" in df.columns and "current_ratio" in df.columns:
            df["quick_ratio"] = df["current_ratio"] * np.random.uniform(0.5, 0.9, len(df))
            df["quick_ratio"] = df["quick_ratio"].clip(0.2, 3.0)

        # Loan amount proportional to revenue
        if "loan_amount_cr" in df.columns and "revenue_cr" in df.columns:
            df["loan_amount_cr"] = df["revenue_cr"] * np.random.uniform(0.2, 0.8, len(df))
            df["loan_amount_cr"] = df["loan_amount_cr"].clip(0.1, 100)

        # Interest rate inversely correlated with CIBIL
        if "interest_rate_pct" in df.columns and "cibil_score" in df.columns:
            cibil_adj = (750 - df["cibil_score"]) / 100 * 1.5
            df["interest_rate_pct"] = 11 + cibil_adj + np.random.normal(0, 1, len(df))
            df["interest_rate_pct"] = df["interest_rate_pct"].clip(7, 22)

        return df

    def train_ctgan(self, data: Optional[pd.DataFrame] = None) -> None:
        """Train CTGAN on seed or provided data."""
        if not HAS_CTGAN:
            print("CTGAN not available. Using seed data generation only.")
            return

        if data is None:
            if self._seed_data is None:
                data = self.generate_seed_dataset()
            else:
                data = self._seed_data

        discrete_columns = ["default", "emi_bounce_count", "litigation_count", "secondary_research_risk"]
        discrete_columns = [c for c in discrete_columns if c in data.columns]

        self.ctgan = CTGAN(
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=True,
        )
        self.ctgan.fit(data, discrete_columns=discrete_columns)

    def generate(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate synthetic data using trained CTGAN or seed distributions."""
        if self.ctgan is not None and HAS_CTGAN:
            synthetic = self.ctgan.sample(n_samples)
            # Clip to realistic ranges
            for col, params in {**FEATURE_DISTRIBUTIONS, **ADDITIONAL_COLUMNS}.items():
                if col in synthetic.columns:
                    synthetic[col] = synthetic[col].clip(params.get("min", -np.inf), params.get("max", np.inf))
            return synthetic
        else:
            # Fallback to distribution-based generation
            return self.generate_seed_dataset(n_samples)

    def validate_distribution(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
        """Validate synthetic data against real data using KS test."""
        results = {}
        common_cols = [c for c in real.columns if c in synthetic.columns and real[c].dtype in [np.float64, np.int64, float, int]]

        for col in common_cols:
            ks_stat, p_value = stats.ks_2samp(real[col].dropna(), synthetic[col].dropna())
            results[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 4),
                "pass": p_value > 0.05,  # Fail to reject null hypothesis → similar distributions
                "real_mean": round(float(real[col].mean()), 4),
                "synthetic_mean": round(float(synthetic[col].mean()), 4),
                "real_std": round(float(real[col].std()), 4),
                "synthetic_std": round(float(synthetic[col].std()), 4),
            }

        # Correlation comparison
        real_corr = real[common_cols].corr()
        synthetic_corr = synthetic[common_cols].corr()
        corr_diff = (real_corr - synthetic_corr).abs().mean().mean()

        total_pass = sum(1 for v in results.values() if v["pass"])
        total = len(results)

        return {
            "feature_tests": results,
            "pass_rate": f"{total_pass}/{total}",
            "avg_correlation_difference": round(float(corr_diff), 4),
            "overall_quality": "GOOD" if total_pass / max(total, 1) > 0.7 else "FAIR" if total_pass / max(total, 1) > 0.5 else "POOR",
        }

    def save_model(self, path: str) -> None:
        """Save trained CTGAN model."""
        if self.ctgan and HAS_CTGAN:
            self.ctgan.save(path)

    def load_model(self, path: str) -> None:
        """Load trained CTGAN model."""
        if HAS_CTGAN and os.path.exists(path):
            self.ctgan = CTGAN.load(path)

    def get_feature_names(self) -> List[str]:
        """Return the list of ML feature names."""
        return FEATURE_NAMES.copy()
