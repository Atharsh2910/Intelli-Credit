"""
Intelli-Credit: Decision Engine
ML-based credit grading, loan sizing, and interest rate determination.
"""

import json
from typing import Dict, List, Any

from backend.ml.credit_model import CreditRiskModel, LoanLimitModel, InterestRateModel


class CreditDecisionEngine:
    """Produces ML-driven credit decisions."""

    def __init__(self, model_dir: str = None):
        self.risk_model = CreditRiskModel(model_dir)
        self.loan_model = LoanLimitModel(model_dir)
        self.rate_model = InterestRateModel(model_dir)

    def decide(self, features: Dict[str, float], agent_analysis: Dict = None) -> Dict[str, Any]:
        """Generate complete credit decision."""
        # ML predictions
        risk_result = self.risk_model.predict(features)
        pd_prob = risk_result["default_probability"]
        credit_score = risk_result["credit_score"]
        shap_explanation = risk_result["shap_explanation"]

        # Loan amount
        loan_amount = self.loan_model.predict(features)

        # Interest rate
        interest_rate = self.rate_model.predict(features)

        # Credit grade using ML probability + agent risk score
        ai_risk_score = 0
        if agent_analysis:
            ai_risk_score = agent_analysis.get("risk_assessment", {}).get("risk_score", 0)
        combined = 0.60 * pd_prob + 0.40 * (ai_risk_score / 100)

        grade, decision = self._grade_decision(combined)

        # Apply security haircut to loan amount
        security_coverage = features.get("security_coverage", 1.2)
        haircut = self._compute_haircut(security_coverage, grade)
        final_loan = round(loan_amount * (1 - haircut), 2)

        # Covenants
        covenants = self._suggest_covenants(grade, features, agent_analysis)

        return {
            "decision": decision,
            "credit_grade": grade,
            "credit_score": credit_score,
            "default_probability": round(pd_prob * 100, 2),
            "combined_score": round(combined, 4),
            "loan_amount_cr": round(loan_amount, 2),
            "haircut_pct": round(haircut * 100, 1),
            "final_loan_amount_cr": final_loan,
            "interest_rate_pct": interest_rate,
            "shap_explanation": shap_explanation,
            "covenants": covenants,
            "pricing_breakdown": self._pricing_breakdown(grade, features, interest_rate),
        }

    def _grade_decision(self, score: float):
        if score < 0.15:
            return "A", "APPROVE"
        elif score < 0.30:
            return "B", "APPROVE"
        elif score < 0.50:
            return "C", "APPROVE_WITH_CONDITIONS"
        elif score < 0.70:
            return "D", "DECLINE_WITH_REVIEW"
        else:
            return "E", "DECLINE"

    def _compute_haircut(self, security_coverage: float, grade: str) -> float:
        base = {"A": 0.05, "B": 0.10, "C": 0.15, "D": 0.25, "E": 0.35}
        h = base.get(grade, 0.20)
        if security_coverage < 1.0:
            h += 0.10
        return min(0.50, h)

    def _pricing_breakdown(self, grade, features, final_rate) -> Dict:
        bands = {"A": (8.5, 10.0), "B": (10.0, 12.0), "C": (12.0, 14.5), "D": (14.5, 18.0), "E": (18.0, 22.0)}
        lo, hi = bands.get(grade, (12, 15))
        base = (lo + hi) / 2

        adjustments = []
        dscr = features.get("dscr", 1.3)
        if dscr >= 2.0:
            adjustments.append({"factor": "Strong DSCR (≥2.0x)", "adjustment": -0.5})
        elif dscr < 1.25:
            adjustments.append({"factor": "Weak DSCR (<1.25x)", "adjustment": 0.5})

        cibil = features.get("cibil_score", 700)
        if cibil >= 750:
            adjustments.append({"factor": "Strong CIBIL (≥750)", "adjustment": -0.25})
        elif cibil < 650:
            adjustments.append({"factor": "Low CIBIL (<650)", "adjustment": 0.25})

        vintage = features.get("years_in_business", 5)
        if vintage >= 10:
            adjustments.append({"factor": "Long vintage (≥10yr)", "adjustment": -0.25})
        elif vintage < 3:
            adjustments.append({"factor": "Short vintage (<3yr)", "adjustment": 0.25})

        return {
            "grade_band": f"{lo}%–{hi}%",
            "base_rate": round(base, 2),
            "adjustments": adjustments,
            "final_rate": final_rate,
        }

    def _suggest_covenants(self, grade, features, agent_analysis) -> List[str]:
        covenants = []
        if grade in ("C", "D"):
            covenants.append("Maintain minimum DSCR of 1.25x (tested quarterly)")
            covenants.append("Submit quarterly financial statements within 45 days")
            covenants.append("Maintain Debt Service Reserve Account (DSRA) for 2 quarters")

        if features.get("emi_bounce_count", 0) > 0:
            covenants.append("No EMI/ECS bounces — any bounce triggers review")

        if features.get("debt_equity_ratio", 0) > 2.5:
            covenants.append(f"Reduce D/E ratio to below 2.5x within 12 months")

        if features.get("gstr_2a_3b_discrepancy_pct", 0) > 10:
            covenants.append("Resolve GST ITC discrepancies before first disbursement")

        if agent_analysis:
            fraud_level = agent_analysis.get("fraud_analysis", {}).get("risk_level", "LOW")
            if fraud_level in ("HIGH", "CRITICAL"):
                covenants.append("Enhanced due diligence before disbursement")
                covenants.append("Promoter personal guarantee required")

        if not covenants:
            covenants.append("Standard monitoring — annual review")

        return covenants
