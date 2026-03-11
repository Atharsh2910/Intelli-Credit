"""
Intelli-Credit: SWOT Analysis Generator
Generates Strengths, Weaknesses, Opportunities, and Threats
from the multi-agent credit analysis results.
"""

import os
import json
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class SWOTAnalyzer:
    """Generates comprehensive SWOT analysis from credit analysis data."""

    def __init__(self):
        self.client = None
        if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, agent_analysis: Dict, decision: Dict = None,
                 company_info: Dict = None) -> Dict[str, Any]:
        """Generate SWOT analysis from analysis results."""
        if self.client:
            try:
                return self._llm_generate(agent_analysis, decision, company_info)
            except Exception:
                pass
        return self._rule_generate(agent_analysis, decision, company_info)

    def _llm_generate(self, analysis: Dict, decision: Dict,
                      company_info: Dict) -> Dict[str, Any]:
        """Generate SWOT using LLM."""
        info = company_info or analysis.get("company_info", {})
        prompt = f"""Generate a SWOT analysis for a corporate credit assessment.

Company: {json.dumps(info, indent=2)}

Financial Analysis: {json.dumps(analysis.get('financial_analysis', {}), indent=2)}

Research Findings: {json.dumps(analysis.get('research', {}), indent=2, default=str)}

Risk Assessment: {json.dumps(analysis.get('risk_assessment', {}), indent=2)}

Decision: {json.dumps(decision or {}, indent=2)}

Return ONLY valid JSON with this structure:
{{
  "strengths": [
    {{"title": "...", "detail": "..."}},
    ...
  ],
  "weaknesses": [
    {{"title": "...", "detail": "..."}},
    ...
  ],
  "opportunities": [
    {{"title": "...", "detail": "..."}},
    ...
  ],
  "threats": [
    {{"title": "...", "detail": "..."}},
    ...
  ],
  "summary": "2-3 sentence overall SWOT summary"
}}

Provide 3-5 items per quadrant. Be specific using data from the analysis.
Focus on credit-relevant factors for an Indian corporate borrower."""

        response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)

    def _rule_generate(self, analysis: Dict, decision: Dict = None,
                       company_info: Dict = None) -> Dict[str, Any]:
        """Generate SWOT using rule-based logic."""
        info = company_info or analysis.get("company_info", {})
        ratios = analysis.get("financial_analysis", {}).get("ratios", {})
        research = analysis.get("research", {})
        risk = analysis.get("risk_assessment", {})
        fraud = analysis.get("fraud_analysis", {})
        liquidity = analysis.get("liquidity_analysis", {})
        five_cs = analysis.get("committee_synthesis", {}).get("five_cs", {})

        strengths = []
        weaknesses = []
        opportunities = []
        threats = []

        # ── Strengths ──
        dscr = ratios.get("dscr", 0)
        if dscr >= 1.5:
            strengths.append({
                "title": "Strong Debt Service Coverage",
                "detail": f"DSCR of {dscr}x demonstrates robust ability to service debt obligations."
            })

        icr = ratios.get("interest_coverage_ratio", 0)
        if icr >= 2.0:
            strengths.append({
                "title": "Healthy Interest Coverage",
                "detail": f"Interest coverage ratio of {icr}x provides comfortable margin for interest payments."
            })

        ebitda_margin = ratios.get("ebitda_margin_pct", 0)
        if ebitda_margin >= 12:
            strengths.append({
                "title": "Strong Profitability",
                "detail": f"EBITDA margin of {ebitda_margin}% indicates efficient operations."
            })

        cagr = ratios.get("revenue_growth_3yr_cagr", 0)
        if cagr >= 10:
            strengths.append({
                "title": "Consistent Revenue Growth",
                "detail": f"3-year revenue CAGR of {cagr}% shows strong top-line momentum."
            })

        de_ratio = ratios.get("debt_equity_ratio", 99)
        if de_ratio <= 1.5:
            strengths.append({
                "title": "Conservative Capital Structure",
                "detail": f"Debt-to-equity ratio of {de_ratio}x reflects prudent leverage."
            })

        vintage = info.get("years_in_business", 0)
        if vintage >= 10:
            strengths.append({
                "title": "Established Business Vintage",
                "detail": f"{vintage} years in business provides track record and credibility."
            })

        cibil = info.get("cibil_score", 0)
        if cibil >= 750:
            strengths.append({
                "title": "Excellent Credit History",
                "detail": f"CIBIL score of {cibil} reflects strong creditworthiness."
            })

        if not strengths:
            strengths.append({
                "title": "Operating Business",
                "detail": "Company has an established operating history."
            })

        # ── Weaknesses ──
        if dscr < 1.25 and dscr > 0:
            weaknesses.append({
                "title": "Weak Debt Service Coverage",
                "detail": f"DSCR of {dscr}x is below the RBI minimum of 1.25x."
            })

        if de_ratio > 2.5:
            weaknesses.append({
                "title": "High Leverage",
                "detail": f"Debt-to-equity ratio of {de_ratio}x indicates elevated financial risk."
            })

        if ebitda_margin < 8 and ebitda_margin > 0:
            weaknesses.append({
                "title": "Thin Margins",
                "detail": f"EBITDA margin of {ebitda_margin}% limits buffer against downturns."
            })

        cr = ratios.get("current_ratio", 0)
        if cr < 1.1 and cr > 0:
            weaknesses.append({
                "title": "Tight Liquidity",
                "detail": f"Current ratio of {cr}x below RBI's 1.10x threshold."
            })

        if cibil > 0 and cibil < 650:
            weaknesses.append({
                "title": "Poor Credit Score",
                "detail": f"CIBIL score of {cibil} indicates past repayment issues."
            })

        if liquidity.get("liquidity_grade") in ("STRESSED", "CRITICAL"):
            weaknesses.append({
                "title": "Liquidity Stress",
                "detail": f"Liquidity grade: {liquidity['liquidity_grade']}. Working capital may be constrained."
            })

        if not weaknesses:
            weaknesses.append({
                "title": "Limited Diversification",
                "detail": "Concentration risk may exist in revenue sources or geography."
            })

        # ── Opportunities ──
        sector = research.get("sector_outlook", [])
        if sector:
            opportunities.append({
                "title": "Favorable Sector Outlook",
                "detail": "Industry research indicates positive growth trends in the sector."
            })

        if cagr > 0:
            opportunities.append({
                "title": "Revenue Expansion Potential",
                "detail": f"Historical {cagr}% growth rate suggests scope for scale-up with adequate financing."
            })

        sc = info.get("security_coverage", 0)
        if sc >= 1.5:
            opportunities.append({
                "title": "Strong Collateral Position",
                "detail": f"Security coverage of {sc}x allows headroom for additional credit facilities."
            })

        opportunities.append({
            "title": "Credit Facility Utilization",
            "detail": "Structured credit facility can support working capital and capex needs."
        })

        if not opportunities:
            opportunities.append({
                "title": "Market Growth",
                "detail": "Indian economy growth provides baseline expansion opportunity."
            })

        # ── Threats ──
        risk_factors = risk.get("risk_factors", [])
        for rf in risk_factors[:3]:
            threats.append({
                "title": rf.get("factor", "Risk Factor"),
                "detail": rf.get("detail", "Identified during risk assessment.")
            })

        if research.get("regulatory_flags"):
            threats.append({
                "title": "Regulatory Concerns",
                "detail": "Regulatory flags detected during background research."
            })

        fraud_level = fraud.get("risk_level", "LOW")
        if fraud_level in ("HIGH", "CRITICAL"):
            threats.append({
                "title": "Fraud Network Risk",
                "detail": f"Fraud graph analysis indicates {fraud_level} risk level."
            })

        if not threats:
            threats.append({
                "title": "Macroeconomic Uncertainty",
                "detail": "External economic factors could impact business performance."
            })

        total_score = five_cs.get("total", {}).get("score", 0)
        summary = (
            f"SWOT analysis based on Five Cs score of {total_score}/100. "
            f"Identified {len(strengths)} strengths, {len(weaknesses)} weaknesses, "
            f"{len(opportunities)} opportunities, and {len(threats)} threats. "
            f"Overall risk level: {risk.get('risk_level', 'N/A')}."
        )

        return {
            "strengths": strengths[:5],
            "weaknesses": weaknesses[:5],
            "opportunities": opportunities[:5],
            "threats": threats[:5],
            "summary": summary,
        }
