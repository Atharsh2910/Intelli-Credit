"""
Intelli-Credit: Model Training Script
One-time script to generate synthetic data and train all ML models.
    python scripts/train_model.py
"""

import os
import sys
import json

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.gan.synthetic_generator import SyntheticDataGenerator
from backend.ml.credit_model import CreditRiskModel, LoanLimitModel, InterestRateModel

MODEL_DIR = os.path.join(project_root, "backend", "ml", "models")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 60)
    print("  INTELLI-CREDIT: ML Model Training Pipeline")
    print("=" * 60)

    # Step 1: Generate Synthetic Data
    print("\n[1/5] Generating synthetic SME financial data...")
    gen = SyntheticDataGenerator()
    seed_data = gen.generate_seed_dataset(n_samples=2000)
    print(f"  ✓ Seed dataset: {len(seed_data)} rows, {len(seed_data.columns)} columns")
    print(f"  ✓ Default rate: {seed_data['default'].mean():.1%}")

    # Step 2: Try CTGAN training
    print("\n[2/5] Training CTGAN for synthetic data augmentation...")
    try:
        gen.train_ctgan(seed_data)
        synthetic = gen.generate(n_samples=5000)
        print(f"  ✓ CTGAN generated {len(synthetic)} synthetic rows")

        # Validate
        validation = gen.validate_distribution(seed_data, synthetic)
        print(f"  ✓ Distribution quality: {validation['overall_quality']}")
        print(f"  ✓ Feature test pass rate: {validation['pass_rate']}")

        training_data = synthetic
    except Exception as e:
        print(f"  ⚠ CTGAN unavailable ({e}), using seed data")
        training_data = gen.generate_seed_dataset(n_samples=5000)

    # Step 3: Train Credit Risk Model
    print("\n[3/5] Training Credit Risk Ensemble (XGBoost + LightGBM + RF)...")
    risk_model = CreditRiskModel(MODEL_DIR)
    risk_metrics = risk_model.train(training_data)
    print(f"  ✓ AUC-ROC: {risk_metrics['auc_roc']}")
    print(f"  ✓ Accuracy: {risk_metrics['accuracy']}")
    print(f"  ✓ Features: {risk_metrics['n_features']}")

    # Step 4: Train Loan Limit Model
    print("\n[4/5] Training Loan Limit Regression (GBoost + RF)...")
    loan_model = LoanLimitModel(MODEL_DIR)
    loan_metrics = loan_model.train(training_data)
    print(f"  ✓ RMSE: {loan_metrics['rmse']}")
    print(f"  ✓ MAE: {loan_metrics['mae']}")
    print(f"  ✓ R²: {loan_metrics['r2']}")

    # Step 5: Train Interest Rate Model
    print("\n[5/5] Training Interest Rate Model (Ridge + RF)...")
    rate_model = InterestRateModel(MODEL_DIR)
    rate_metrics = rate_model.train(training_data)
    print(f"  ✓ RMSE: {rate_metrics['rmse']}")
    print(f"  ✓ MAE: {rate_metrics['mae']}")
    print(f"  ✓ R²: {rate_metrics['r2']}")

    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  Models saved to: {MODEL_DIR}")

    all_metrics = {
        "credit_risk": risk_metrics,
        "loan_limit": loan_metrics,
        "interest_rate": rate_metrics,
    }
    metrics_path = os.path.join(MODEL_DIR, "all_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")

    # Quick test
    print("\n  Quick prediction test:")
    test_features = {
        "dscr": 1.5, "interest_coverage_ratio": 3.0, "ebitda_margin_pct": 14,
        "revenue_growth_3yr_cagr": 12, "pat_margin_pct": 6,
        "debt_equity_ratio": 1.1, "tangible_net_worth_cr": 5, "promoter_contribution_pct": 60,
        "cash_accrual_to_debt": 0.18, "current_ratio": 1.4, "quick_ratio": 1.0,
        "facr": 1.6, "security_coverage": 1.5, "gst_compliance_score": 8,
        "gstr_2a_3b_discrepancy_pct": 4, "bank_credit_debit_ratio": 1.1,
        "emi_bounce_count": 0, "avg_bank_utilisation_pct": 55, "cibil_score": 740,
        "litigation_count": 1, "years_in_business": 12, "promoter_experience_yrs": 15,
        "secondary_research_risk": 0, "sector_outlook_score": 7, "macro_risk_score": 4,
    }
    pred = risk_model.predict(test_features)
    print(f"  Default probability: {pred['default_probability']:.2%}")
    print(f"  Credit score: {pred['credit_score']}/900")
    print(f"  Top risk driver: {pred['shap_explanation'][0]['feature'] if pred['shap_explanation'] else 'N/A'}")
    print(f"\n  All models ready for deployment!")


if __name__ == "__main__":
    main()
