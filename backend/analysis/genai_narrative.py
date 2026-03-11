"""
Intelli-Credit: GenAI Narrative Generator
Generates prose-form executive summary narratives using OpenAI.
Falls back to template-based generation when LLM is unavailable.
"""

import os
import json
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class GenAINarrativeGenerator:
    """Generates executive summary narratives with GenAI."""

    def __init__(self):
        self.client = None
        if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, agent_analysis: Dict, decision: Dict = None,
                 company_info: Dict = None) -> Dict[str, Any]:
        """Generate an executive narrative from analysis results."""
        if self.client:
            try:
                narrative = self._llm_generate(agent_analysis, decision, company_info)
                return {
                    "narrative": narrative,
                    "generated_by": "GenAI (GPT-4o)",
                    "generated_at": datetime.now().isoformat(),
                }
            except Exception:
                pass

        narrative = self._template_generate(agent_analysis, decision, company_info)
        return {
            "narrative": narrative,
            "generated_by": "Template Engine",
            "generated_at": datetime.now().isoformat(),
        }

    def _llm_generate(self, analysis: Dict, decision: Dict, company_info: Dict) -> str:
        """Generate narrative using LLM."""
        info = company_info or analysis.get("company_info", {})
        prompt = f"""You are a senior credit analyst at an Indian bank. Write a comprehensive executive summary narrative for a credit appraisal.

Company: {json.dumps(info, indent=2)}

Financial Analysis: {json.dumps(analysis.get('financial_analysis', {}).get('ratios', {}), indent=2)}

Risk Assessment: {json.dumps(analysis.get('risk_assessment', {}), indent=2)}

Committee Synthesis: {json.dumps(analysis.get('committee_synthesis', {}).get('synthesis', {}), indent=2, default=str)}

Decision: {json.dumps(decision or {}, indent=2)}

Write a professional, flowing narrative (300-500 words) covering:
1. Company Overview & Background
2. Key Financial Highlights (cite specific ratios)  
3. Risk Assessment Summary
4. Credit Quality Assessment
5. Recommendation Rationale

Use formal financial language. Use ₹ for currency. Write in third person.
Do NOT use markdown headers or bullet points — write in paragraph form only."""

        response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200,
        )
        return response.choices[0].message.content.strip()

    def _template_generate(self, analysis: Dict, decision: Dict = None,
                           company_info: Dict = None) -> str:
        """Generate narrative using templates."""
        info = company_info or analysis.get("company_info", {})
        ratios = analysis.get("financial_analysis", {}).get("ratios", {})
        risk = analysis.get("risk_assessment", {})
        committee = analysis.get("committee_synthesis", {})
        five_cs = committee.get("five_cs", {})
        synthesis = committee.get("synthesis", {})

        name = info.get("company_name", "The Company")
        industry = info.get("industry", "its sector")
        vintage = info.get("years_in_business", "N/A")

        dscr = ratios.get("dscr", "N/A")
        de = ratios.get("debt_equity_ratio", "N/A")
        ebitda = ratios.get("ebitda_margin_pct", "N/A")
        icr = ratios.get("interest_coverage_ratio", "N/A")
        cagr = ratios.get("revenue_growth_3yr_cagr", "N/A")
        cr = ratios.get("current_ratio", "N/A")

        risk_level = risk.get("risk_level", "N/A")
        risk_score = risk.get("risk_score", "N/A")
        risk_count = len(risk.get("risk_factors", []))

        total_five_cs = five_cs.get("total", {}).get("score", "N/A")
        decision_val = "N/A"
        grade = "N/A"
        loan = "N/A"
        rate = "N/A"

        if decision:
            decision_val = decision.get("decision", "N/A")
            grade = decision.get("credit_grade", "N/A")
            loan = decision.get("final_loan_amount_cr", "N/A")
            rate = decision.get("interest_rate_pct", "N/A")

        narrative = f"""{name} is a {industry} company with a business vintage of {vintage} years. The entity has approached the bank for a credit facility, and this narrative summarizes the key findings from the automated credit assessment conducted by the Intelli-Credit AI engine.

From a financial standpoint, {name} reports an EBITDA margin of {ebitda}% and a debt-to-equity ratio of {de}x. The Debt Service Coverage Ratio (DSCR) stands at {dscr}x, while the Interest Coverage Ratio is {icr}x. Revenue has grown at a 3-year CAGR of {cagr}%, and the current ratio is {cr}x. These metrics have been evaluated against RBI prudential norms and industry benchmarks.

The comprehensive risk assessment has identified {risk_count} risk factor(s), resulting in an overall risk classification of {risk_level} with a risk score of {risk_score}/100. The risk evaluation encompasses financial leverage, regulatory compliance, litigation history, GST cross-validation, banking behavior, and fraud network analysis.

Under the Five Cs of Credit framework, {name} has achieved a composite score of {total_five_cs}/100, reflecting the aggregate assessment across Character, Capacity, Capital, Collateral, and Conditions. Each dimension has been independently scored to provide a balanced view of creditworthiness.

Based on the ML ensemble scoring (XGBoost, LightGBM, Random Forest) combined with the multi-agent AI analysis, the credit recommendation is {decision_val} with a credit grade of {grade}. The recommended facility size is ₹{loan} Crores at an interest rate of {rate}% per annum. This assessment has been generated using explainable AI with SHAP feature attribution to ensure full transparency in the decision-making process.

This narrative was auto-generated by the Intelli-Credit AI Engine on {datetime.now().strftime('%d %B %Y')}."""

        return narrative.strip()
