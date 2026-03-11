"""
Intelli-Credit: Credit Appraisal Memo Generator
Generates formal CAM structured by Five Cs of Credit.
Supports both text and PDF output.
"""

import os
import io
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


class CAMGenerator:
    """Generates Credit Appraisal Memo (CAM)."""

    def __init__(self):
        self.client = None
        if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, decision: Dict, agent_analysis: Dict, company_info: Dict = None) -> Dict[str, Any]:
        """Generate a complete CAM."""
        info = company_info or agent_analysis.get("company_info", {})

        if self.client:
            memo_text = self._llm_generate(decision, agent_analysis, info)
        else:
            memo_text = self._template_generate(decision, agent_analysis, info)

        return {
            "memo_text": memo_text,
            "generated_at": datetime.now().isoformat(),
            "company_name": info.get("company_name", "Unknown"),
            "decision": decision.get("decision", "REVIEW"),
            "credit_grade": decision.get("credit_grade", "N/A"),
            "loan_amount_cr": decision.get("final_loan_amount_cr", 0),
            "interest_rate_pct": decision.get("interest_rate_pct", 0),
        }

    def _llm_generate(self, decision, analysis, info) -> str:
        prompt = f"""Generate a formal Credit Appraisal Memo (CAM) for an Indian bank credit committee.

Company Information:
{json.dumps(info, indent=2)}

Credit Decision:
{json.dumps(decision, indent=2)}

Agent Analysis Summary:
- Financial: {json.dumps(analysis.get('financial_analysis', {}).get('ratios', {}), indent=2)}
- Risk Level: {analysis.get('risk_assessment', {}).get('risk_level', 'N/A')}
- Risk Factors: {json.dumps(analysis.get('risk_assessment', {}).get('risk_factors', []), indent=2)}
- Five Cs: {json.dumps(analysis.get('committee_synthesis', {}).get('five_cs', {}), indent=2)}
- Fraud Risk: {analysis.get('fraud_analysis', {}).get('risk_level', 'N/A')}

Structure the memo using the Five Cs of Credit:
1. CHARACTER - Promoter background, business vintage, integrity
2. CAPACITY - Revenue, profitability, DSCR, repayment ability
3. CAPITAL - Net worth, capital structure, promoter contribution
4. COLLATERAL - Security coverage, asset quality
5. CONDITIONS - Industry outlook, macro environment, liquidity

End with:
- RECOMMENDATION: Decision, loan amount, interest rate
- KEY RISKS: Top 3 risks
- COVENANTS: Suggested covenants
- SHAP EXPLANATION: Top factors that influenced the ML model

Make it professional, detailed (500+ words), and suitable for a credit committee presentation.
Use ₹ for currency. Include specific numbers from the analysis."""

        try:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return self._template_generate(decision, analysis, info)

    def _template_generate(self, decision, analysis, info) -> str:
        """Template-based CAM generation (no LLM needed)."""
        name = info.get("company_name", "Unknown Company")
        industry = info.get("industry", "N/A")
        vintage = info.get("years_in_business", "N/A")
        ratios = analysis.get("financial_analysis", {}).get("ratios", {})
        risk = analysis.get("risk_assessment", {})
        five_cs = analysis.get("committee_synthesis", {}).get("five_cs", {})
        fraud = analysis.get("fraud_analysis", {})

        shap = decision.get("shap_explanation", [])
        shap_text = "\n".join([
            f"  • {s['feature']}: {'+' if s['impact']>0 else ''}{s['impact']:.3f} ({s['direction']})"
            for s in shap[:5]
        ]) if shap else "  • Model explanation not available"

        covenants = decision.get("covenants", [])
        cov_text = "\n".join([f"  {i+1}. {c}" for i, c in enumerate(covenants)]) if covenants else "  Standard monitoring"

        risk_factors = risk.get("risk_factors", [])
        risk_text = "\n".join([
            f"  • [{r['severity']}] {r['factor']}: {r.get('detail','')}"
            for r in risk_factors[:5]
        ]) if risk_factors else "  No significant risk factors identified"

        memo = f"""
═══════════════════════════════════════════════════════════
              CREDIT APPRAISAL MEMO (CAM)
═══════════════════════════════════════════════════════════

Date: {datetime.now().strftime('%d %B %Y')}
Company: {name}
Industry: {industry}
Vintage: {vintage} years
CIN: {info.get('cin', 'N/A')}

───────────────────────────────────────────────────────────
1. CHARACTER (Score: {five_cs.get('character', {}).get('score', 'N/A')}/20)
───────────────────────────────────────────────────────────
  Promoter(s): {', '.join(info.get('promoter_names', ['N/A']))}
  Business Vintage: {vintage} years
  CIBIL Score: {info.get('cibil_score', ratios.get('cibil_score', 'N/A'))}
  Litigation Status: {len(analysis.get('research', {}).get('litigation_history', []))} active cases
  Fraud Network Risk: {fraud.get('risk_level', 'N/A')}

───────────────────────────────────────────────────────────
2. CAPACITY (Score: {five_cs.get('capacity', {}).get('score', 'N/A')}/20)
───────────────────────────────────────────────────────────
  Revenue Growth (3Y CAGR): {ratios.get('revenue_growth_3yr_cagr', 'N/A')}%
  EBITDA Margin: {ratios.get('ebitda_margin_pct', 'N/A')}%
  PAT Margin: {ratios.get('pat_margin_pct', 'N/A')}%
  DSCR: {ratios.get('dscr', 'N/A')}x
  Interest Coverage: {ratios.get('interest_coverage_ratio', 'N/A')}x

───────────────────────────────────────────────────────────
3. CAPITAL (Score: {five_cs.get('capital', {}).get('score', 'N/A')}/20)
───────────────────────────────────────────────────────────
  Tangible Net Worth: ₹{ratios.get('tangible_net_worth_cr', 'N/A')} Cr
  Debt-Equity Ratio: {ratios.get('debt_equity_ratio', 'N/A')}x
  Cash Accrual to Debt: {ratios.get('cash_accrual_to_debt', 'N/A')}

───────────────────────────────────────────────────────────
4. COLLATERAL (Score: {five_cs.get('collateral', {}).get('score', 'N/A')}/20)
───────────────────────────────────────────────────────────
  Security Coverage: {info.get('security_coverage', ratios.get('security_coverage', 'N/A'))}x
  FACR: {ratios.get('facr', 'N/A')}x
  Collateral Type: {info.get('collateral_type', 'N/A')}

───────────────────────────────────────────────────────────
5. CONDITIONS (Score: {five_cs.get('conditions', {}).get('score', 'N/A')}/20)
───────────────────────────────────────────────────────────
  Sector Outlook: {info.get('sector_outlook', 'N/A')}
  Liquidity Grade: {analysis.get('liquidity_analysis', {}).get('liquidity_grade', 'N/A')}
  Macro Risk Score: {info.get('macro_risk_score', 'N/A')}/10

═══════════════════════════════════════════════════════════
              RECOMMENDATION
═══════════════════════════════════════════════════════════

  Decision:        {decision.get('decision', 'REVIEW')}
  Credit Grade:    {decision.get('credit_grade', 'N/A')}
  Credit Score:    {decision.get('credit_score', 'N/A')}/900
  Default Prob:    {decision.get('default_probability', 'N/A')}%
  Loan Amount:     ₹{decision.get('final_loan_amount_cr', 'N/A')} Cr
  Interest Rate:   {decision.get('interest_rate_pct', 'N/A')}%
  Five Cs Total:   {five_cs.get('total', {}).get('score', 'N/A')}/100

───────────────────────────────────────────────────────────
  KEY RISKS
───────────────────────────────────────────────────────────
{risk_text}

───────────────────────────────────────────────────────────
  ML MODEL EXPLANATION (SHAP)
───────────────────────────────────────────────────────────
{shap_text}

───────────────────────────────────────────────────────────
  COVENANTS
───────────────────────────────────────────────────────────
{cov_text}

═══════════════════════════════════════════════════════════
  This memo was generated by Intelli-Credit AI Engine.
  All recommendations are ML-driven with explainable AI.
═══════════════════════════════════════════════════════════
"""
        return memo.strip()

    def generate_pdf(self, decision: Dict, agent_analysis: Dict,
                     company_info: Dict = None) -> bytes:
        """Generate a professional PDF version of the CAM.
        Returns raw PDF bytes suitable for streaming to the client.
        """
        from fpdf import FPDF

        cam_data = self.generate(decision, agent_analysis, company_info)
        info = company_info or agent_analysis.get("company_info", {})
        ratios = agent_analysis.get("financial_analysis", {}).get("ratios", {})
        risk = agent_analysis.get("risk_assessment", {})
        five_cs = agent_analysis.get("committee_synthesis", {}).get("five_cs", {})

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        # ── Title ──
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(11, 31, 58)  # Navy
        pdf.cell(0, 12, "CREDIT APPRAISAL MEMO (CAM)", ln=True, align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, f"Generated by Intelli-Credit AI Engine | {datetime.now().strftime('%d %B %Y')}", ln=True, align="C")
        pdf.ln(6)

        # ── Company Info ──
        pdf.set_draw_color(11, 31, 58)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(11, 31, 58)
        pdf.cell(0, 8, "ENTITY INFORMATION", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)

        info_rows = [
            ("Company", info.get("company_name", "N/A")),
            ("Industry", info.get("industry", "N/A")),
            ("Vintage", f"{info.get('years_in_business', 'N/A')} years"),
            ("CIN", info.get("cin", "N/A")),
            ("CIBIL Score", str(info.get("cibil_score", "N/A"))),
        ]
        for label, value in info_rows:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(45, 6, f"{label}:", align="L")
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 6, str(value), ln=True)
        pdf.ln(4)

        # ── Five Cs Scorecard ──
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(11, 31, 58)
        pdf.cell(0, 8, "FIVE Cs OF CREDIT SCORECARD", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)

        for c_name in ["character", "capacity", "capital", "collateral", "conditions"]:
            data = five_cs.get(c_name, {})
            score = data.get("score", 0)
            max_score = data.get("max", 20)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(45, 6, f"{c_name.upper()}:")
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 6, f"{score} / {max_score}", ln=True)

        total = five_cs.get("total", {})
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(45, 8, "TOTAL:")
        pdf.cell(0, 8, f"{total.get('score', 0)} / {total.get('max', 100)}", ln=True)
        pdf.ln(4)

        # ── Key Financial Ratios ──
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(11, 31, 58)
        pdf.cell(0, 8, "KEY FINANCIAL RATIOS", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)

        ratio_rows = [
            ("DSCR", f"{ratios.get('dscr', 'N/A')}x"),
            ("Interest Coverage", f"{ratios.get('interest_coverage_ratio', 'N/A')}x"),
            ("Debt-Equity Ratio", f"{ratios.get('debt_equity_ratio', 'N/A')}x"),
            ("EBITDA Margin", f"{ratios.get('ebitda_margin_pct', 'N/A')}%"),
            ("PAT Margin", f"{ratios.get('pat_margin_pct', 'N/A')}%"),
            ("Current Ratio", f"{ratios.get('current_ratio', 'N/A')}x"),
            ("Revenue Growth (3Y CAGR)", f"{ratios.get('revenue_growth_3yr_cagr', 'N/A')}%"),
        ]
        for label, value in ratio_rows:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(55, 6, f"{label}:")
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 6, str(value), ln=True)
        pdf.ln(4)

        # ── Risk Assessment ──
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(11, 31, 58)
        pdf.cell(0, 8, "RISK ASSESSMENT", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 6, f"Risk Level: {risk.get('risk_level', 'N/A')}  |  Risk Score: {risk.get('risk_score', 'N/A')}/100", ln=True)

        for rf in risk.get("risk_factors", [])[:5]:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(8, 6, chr(8226))  # Bullet
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 6, f"[{rf.get('severity', '')}] {rf.get('factor', '')}: {rf.get('detail', '')}", ln=True)
        pdf.ln(4)

        # ── Recommendation ──
        pdf.set_draw_color(11, 31, 58)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(11, 31, 58)
        pdf.cell(0, 8, "RECOMMENDATION", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)

        rec_rows = [
            ("Decision", decision.get("decision", "N/A")),
            ("Credit Grade", decision.get("credit_grade", "N/A")),
            ("Credit Score", f"{decision.get('credit_score', 'N/A')} / 900"),
            ("Default Probability", f"{decision.get('default_probability', 'N/A')}%"),
            ("Loan Amount", f"Rs. {decision.get('final_loan_amount_cr', 'N/A')} Cr"),
            ("Interest Rate", f"{decision.get('interest_rate_pct', 'N/A')}%"),
        ]
        for label, value in rec_rows:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(50, 7, f"{label}:")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 7, str(value), ln=True)
        pdf.ln(4)

        # ── Covenants ──
        covenants = decision.get("covenants", [])
        if covenants:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(11, 31, 58)
            pdf.cell(0, 8, "SUGGESTED COVENANTS", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            for i, cov in enumerate(covenants, 1):
                pdf.cell(0, 6, f"  {i}. {cov}", ln=True)
            pdf.ln(4)

        # ── SHAP Explanation ──
        shap = decision.get("shap_explanation", [])
        if shap:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(11, 31, 58)
            pdf.cell(0, 8, "ML MODEL EXPLANATION (SHAP)", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            for s in shap[:8]:
                direction = "+" if s.get("impact", 0) > 0 else ""
                pdf.cell(0, 6, f"  {s.get('feature', '')}: {direction}{s.get('impact', 0):.4f} ({s.get('direction', '')})", ln=True)
            pdf.ln(4)

        # ── Footer ──
        pdf.ln(6)
        pdf.set_draw_color(11, 31, 58)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 5, "This report was generated by Intelli-Credit AI Engine. All recommendations are ML-driven with explainable AI.", ln=True, align="C")
        pdf.cell(0, 5, "For internal use only. Not to be treated as investment advice.", ln=True, align="C")

        # Output as bytes
        return bytes(pdf.output())

