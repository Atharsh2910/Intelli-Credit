"""
Intelli-Credit: Multi-Agent Intelligence System
LangChain-based agent orchestration for credit analysis.
- Research Agent: web search, MCA, litigation, news
- Financial Agent: ratio computation, GST analysis
- Risk Agent: combines signals into risk factors
- Liquidity Agent: cashflow forecasting
- Fraud Agent: graph analytics integration
- Committee Moderator: synthesizes all → recommendation
"""

import os
import json
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain.tools import Tool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

try:
    from tavily import TavilyClient
    HAS_TAVILY = True
except ImportError:
    HAS_TAVILY = False

from backend.rag.document_intelligence import DocumentIntelligence
from backend.fraud_graph.graph_analytics import FraudGraphAnalyzer
from backend.ml.credit_model import CashflowForecaster


class ResearchAgent:
    """Crawls news, checks promoter backgrounds, sector outlook, litigation."""

    def __init__(self):
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if HAS_TAVILY and os.getenv("TAVILY_API_KEY") else None

    def research(self, company_name: str, promoter_names: List[str] = None, industry: str = "") -> Dict[str, Any]:
        results = {
            "company_news": [],
            "promoter_background": [],
            "sector_outlook": [],
            "litigation_history": [],
            "regulatory_flags": [],
            "overall_risk": "LOW",
        }

        if not self.tavily:
            results["note"] = "Web search unavailable — provide TAVILY_API_KEY"
            return results

        # Company news
        try:
            news = self.tavily.search(f"{company_name} India corporate news financial", max_results=5)
            results["company_news"] = [
                {"title": r.get("title",""), "url": r.get("url",""), "snippet": r.get("content","")[:200]}
                for r in news.get("results", [])
            ]
        except Exception:
            pass

        # Promoter background
        for name in (promoter_names or []):
            try:
                bg = self.tavily.search(f"{name} promoter director India fraud litigation defaulter", max_results=3)
                for r in bg.get("results", []):
                    results["promoter_background"].append({
                        "promoter": name, "title": r.get("title",""),
                        "url": r.get("url",""), "snippet": r.get("content","")[:200]
                    })
            except Exception:
                pass

        # Sector outlook
        if industry:
            try:
                sector = self.tavily.search(f"{industry} India sector outlook 2024 2025 growth", max_results=3)
                results["sector_outlook"] = [
                    {"title": r.get("title",""), "snippet": r.get("content","")[:200]}
                    for r in sector.get("results", [])
                ]
            except Exception:
                pass

        # Litigation / NCLT / DRT
        try:
            lit = self.tavily.search(f"{company_name} NCLT DRT litigation court India", max_results=3,
                                     include_domains=["nclt.gov.in", "drt.gov.in", "indiankanoon.org"])
            results["litigation_history"] = [
                {"title": r.get("title",""), "url": r.get("url",""), "snippet": r.get("content","")[:200]}
                for r in lit.get("results", [])
            ]
        except Exception:
            pass

        # RBI wilful defaulter check
        try:
            rbi = self.tavily.search(f"{company_name} RBI wilful defaulter", max_results=2)
            for r in rbi.get("results", []):
                if "wilful" in r.get("content", "").lower() or "defaulter" in r.get("content", "").lower():
                    results["regulatory_flags"].append({
                        "flag": "WILFUL_DEFAULTER_MENTION", "source": r.get("url",""),
                        "snippet": r.get("content","")[:200]
                    })
        except Exception:
            pass

        # Determine risk
        risk_signals = len(results["litigation_history"]) + len(results["regulatory_flags"])
        results["overall_risk"] = "HIGH" if risk_signals >= 3 else "MEDIUM" if risk_signals >= 1 else "LOW"

        return results


class FinancialAgent:
    """Computes financial ratios, analyzes GST, compares flows."""

    def analyze(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        revenue = financial_data.get("revenue_cr", 0)
        ebitda = financial_data.get("ebitda_cr", 0)
        pat = financial_data.get("pat_cr", 0)
        total_debt = financial_data.get("total_debt_cr", 0)
        net_worth = financial_data.get("net_worth_cr", 0)
        interest = financial_data.get("interest_expense_cr", 0)
        depreciation = financial_data.get("depreciation_cr", 0)
        current_assets = financial_data.get("current_assets_cr", 0)
        current_liabilities = financial_data.get("current_liabilities_cr", 0)
        inventory = financial_data.get("inventory_cr", 0)
        fixed_assets = financial_data.get("fixed_assets_cr", 0)
        cash_accrual = pat + depreciation
        annual_debt_service = financial_data.get("annual_debt_service_cr", interest + total_debt * 0.15)

        ratios = {
            "dscr": round(cash_accrual / max(annual_debt_service, 0.01), 2),
            "interest_coverage_ratio": round(ebitda / max(interest, 0.01), 2),
            "debt_equity_ratio": round(total_debt / max(net_worth, 0.01), 2),
            "current_ratio": round(current_assets / max(current_liabilities, 0.01), 2),
            "quick_ratio": round((current_assets - inventory) / max(current_liabilities, 0.01), 2),
            "ebitda_margin_pct": round(ebitda / max(revenue, 0.01) * 100, 2),
            "pat_margin_pct": round(pat / max(revenue, 0.01) * 100, 2),
            "facr": round(fixed_assets / max(total_debt, 0.01), 2),
            "cash_accrual_to_debt": round(cash_accrual / max(total_debt, 0.01), 2),
            "tangible_net_worth_cr": round(net_worth, 2),
        }

        # Revenue growth
        rev_history = financial_data.get("revenue_history", [])
        if len(rev_history) >= 2:
            cagr = ((rev_history[-1] / max(rev_history[0], 0.01)) ** (1 / len(rev_history)) - 1) * 100
            ratios["revenue_growth_3yr_cagr"] = round(cagr, 2)
        else:
            ratios["revenue_growth_3yr_cagr"] = 0

        # RBI benchmarks
        alerts = []
        if ratios["dscr"] < 1.25:
            alerts.append(f"DSCR {ratios['dscr']} below RBI minimum of 1.25x")
        if ratios["current_ratio"] < 1.10:
            alerts.append(f"Current ratio {ratios['current_ratio']} below RBI minimum 1.10")
        if ratios["debt_equity_ratio"] > 3.0:
            alerts.append(f"Debt-equity {ratios['debt_equity_ratio']} exceeds 3.0x threshold")
        if ratios["interest_coverage_ratio"] < 1.5:
            alerts.append(f"Interest coverage {ratios['interest_coverage_ratio']} dangerously low")

        return {"ratios": ratios, "alerts": alerts, "cash_accrual_cr": round(cash_accrual, 2)}


class RiskAgent:
    """Combines financial + research signals into risk assessment."""

    def assess(self, financial_analysis: Dict, research_data: Dict,
               gst_validation: Dict = None, bank_analysis: Dict = None,
               fraud_report: Dict = None) -> Dict[str, Any]:

        risk_factors = []
        risk_score = 0  # 0-100 scale

        # Financial risks
        ratios = financial_analysis.get("ratios", {})
        if ratios.get("dscr", 2) < 1.25:
            risk_factors.append({"category": "Financial", "factor": "Low DSCR", "severity": "HIGH",
                                  "detail": f"DSCR of {ratios['dscr']} below RBI minimum"})
            risk_score += 15
        if ratios.get("debt_equity_ratio", 0) > 3.0:
            risk_factors.append({"category": "Financial", "factor": "High leverage", "severity": "HIGH",
                                  "detail": f"D/E ratio of {ratios['debt_equity_ratio']}"})
            risk_score += 12
        if ratios.get("ebitda_margin_pct", 0) < 5:
            risk_factors.append({"category": "Financial", "factor": "Low profitability", "severity": "MEDIUM",
                                  "detail": f"EBITDA margin {ratios['ebitda_margin_pct']}%"})
            risk_score += 8

        # Research risks
        if research_data.get("regulatory_flags"):
            risk_factors.append({"category": "Regulatory", "factor": "Regulatory flags found",
                                  "severity": "CRITICAL", "detail": "Potential wilful defaulter or regulatory violation"})
            risk_score += 25
        if len(research_data.get("litigation_history", [])) >= 2:
            risk_factors.append({"category": "Legal", "factor": "Multiple litigations",
                                  "severity": "HIGH", "detail": f"{len(research_data['litigation_history'])} active cases"})
            risk_score += 12

        # GST risks
        if gst_validation:
            if gst_validation.get("discrepancy_flag"):
                risk_factors.append({"category": "GST", "factor": "ITC discrepancy",
                                      "severity": "HIGH", "detail": f"{gst_validation['discrepancy_pct']}% gap"})
                risk_score += 10
            if gst_validation.get("gst_bank_flag"):
                risk_factors.append({"category": "GST", "factor": "GST-Bank gap",
                                      "severity": "MEDIUM", "detail": f"{gst_validation['gst_bank_gap_pct']}% gap"})
                risk_score += 8

        # Bank risks
        if bank_analysis:
            bounces = bank_analysis.get("emi_bounce_count", 0)
            if bounces >= 3:
                risk_factors.append({"category": "Banking", "factor": "EMI bounces",
                                      "severity": "HIGH", "detail": f"{bounces} bounces detected"})
                risk_score += 15
            if bank_analysis.get("avg_utilisation_pct", 0) > 90:
                risk_factors.append({"category": "Banking", "factor": "Over-utilisation",
                                      "severity": "MEDIUM", "detail": f"{bank_analysis['avg_utilisation_pct']}% utilisation"})
                risk_score += 8

        # Fraud risks
        if fraud_report:
            fraud_level = fraud_report.get("risk_level", "LOW")
            if fraud_level in ("HIGH", "CRITICAL"):
                risk_factors.append({"category": "Fraud", "factor": f"Fraud risk: {fraud_level}",
                                      "severity": fraud_level, "detail": f"Score: {fraud_report['overall_risk_score']}/10"})
                risk_score += 20

        risk_score = min(100, risk_score)
        overall = "CRITICAL" if risk_score >= 70 else "HIGH" if risk_score >= 50 else "MEDIUM" if risk_score >= 25 else "LOW"

        return {
            "risk_factors": risk_factors,
            "risk_score": risk_score,
            "risk_level": overall,
            "risk_summary": f"{len(risk_factors)} risk factors identified. Overall: {overall} ({risk_score}/100)"
        }


class LiquidityAgent:
    """Forecasts cashflows and detects liquidity stress."""

    def __init__(self):
        self.forecaster = CashflowForecaster()

    def analyze(self, cashflow_data: List[Dict] = None, financial_data: Dict = None) -> Dict[str, Any]:
        result = {"forecast": None, "stress_indicators": [], "liquidity_grade": "ADEQUATE"}

        if cashflow_data and len(cashflow_data) >= 4:
            result["forecast"] = self.forecaster.forecast(cashflow_data)
            if result["forecast"].get("liquidity_stress"):
                result["stress_indicators"].append("Declining cashflow trend detected")
                result["liquidity_grade"] = "STRESSED"

        if financial_data:
            cr = financial_data.get("ratios", {}).get("current_ratio", 1.3)
            qr = financial_data.get("ratios", {}).get("quick_ratio", 0.9)
            if cr < 1.0:
                result["stress_indicators"].append(f"Current ratio {cr} below 1.0")
                result["liquidity_grade"] = "CRITICAL"
            elif cr < 1.1:
                result["stress_indicators"].append(f"Current ratio {cr} below RBI minimum 1.10")
                if result["liquidity_grade"] != "CRITICAL":
                    result["liquidity_grade"] = "STRESSED"
            if qr < 0.5:
                result["stress_indicators"].append(f"Quick ratio {qr} very low")

        return result


class FraudAgentWrapper:
    """Wrapper around graph analytics for agent integration."""

    def __init__(self):
        self.analyzer = FraudGraphAnalyzer()

    def analyze(self, company_data: Dict = None) -> Dict[str, Any]:
        if company_data and company_data.get("graph_companies") and company_data.get("graph_relationships"):
            self.analyzer.build_graph(company_data["graph_companies"], company_data["graph_relationships"])
            report = self.analyzer.analyze_company(company_data.get("company_id", "target"))
        else:
            report = self.analyzer.build_sample_graph(company_data.get("company_name", "Target Company") if company_data else "Target Company")

        from dataclasses import asdict
        return asdict(report)


class CommitteeModerator:
    """Synthesizes all agent outputs into final recommendation."""

    def __init__(self):
        self.llm = None
        if HAS_LANGCHAIN and os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0.1)

    def synthesize(self, financial: Dict, research: Dict, risk: Dict,
                   liquidity: Dict, fraud: Dict, company_info: Dict = None) -> Dict[str, Any]:
        """Produce committee-style synthesis."""

        # Build Five Cs assessment
        five_cs = self._assess_five_cs(financial, research, risk, liquidity, fraud, company_info)

        # If LLM available, generate detailed synthesis
        if self.llm:
            synthesis = self._llm_synthesis(financial, research, risk, liquidity, fraud, five_cs, company_info)
        else:
            synthesis = self._rule_synthesis(financial, research, risk, liquidity, fraud, five_cs)

        return {
            "five_cs": five_cs,
            "synthesis": synthesis,
            "committee_decision": synthesis.get("decision", "REVIEW"),
            "confidence": synthesis.get("confidence", 0.5),
        }

    def _assess_five_cs(self, financial, research, risk, liquidity, fraud, company_info) -> Dict:
        ratios = financial.get("ratios", {})
        risk_level = risk.get("risk_level", "MEDIUM")
        info = company_info or {}

        # Character (0-20)
        character = 15
        if research.get("regulatory_flags"):
            character -= 8
        if len(research.get("litigation_history", [])) >= 2:
            character -= 4
        if info.get("years_in_business", 0) >= 10:
            character += 3
        character = max(0, min(20, character))

        # Capacity (0-20)
        capacity = 10
        dscr = ratios.get("dscr", 1.3)
        if dscr >= 2.0:
            capacity += 5
        elif dscr >= 1.5:
            capacity += 3
        elif dscr < 1.25:
            capacity -= 5
        growth = ratios.get("revenue_growth_3yr_cagr", 0)
        if growth > 15:
            capacity += 3
        elif growth < 0:
            capacity -= 3
        capacity = max(0, min(20, capacity))

        # Capital (0-20)
        capital = 10
        de = ratios.get("debt_equity_ratio", 1.5)
        if de <= 1.0:
            capital += 5
        elif de <= 2.0:
            capital += 2
        elif de > 3.0:
            capital -= 5
        capital = max(0, min(20, capital))

        # Collateral (0-20)
        collateral = 10
        sc = ratios.get("security_coverage", 1.0)
        if sc is None:
            sc = info.get("security_coverage", 1.0)
        if sc >= 2.0:
            collateral += 5
        elif sc >= 1.5:
            collateral += 3
        elif sc < 1.0:
            collateral -= 5
        collateral = max(0, min(20, collateral))

        # Conditions (0-20)
        conditions = 12
        if liquidity.get("liquidity_grade") == "CRITICAL":
            conditions -= 6
        elif liquidity.get("liquidity_grade") == "STRESSED":
            conditions -= 3
        fraud_level = fraud.get("risk_level", "LOW")
        if fraud_level in ("HIGH", "CRITICAL"):
            conditions -= 5
        conditions = max(0, min(20, conditions))

        total = character + capacity + capital + collateral + conditions

        return {
            "character": {"score": character, "max": 20},
            "capacity": {"score": capacity, "max": 20},
            "capital": {"score": capital, "max": 20},
            "collateral": {"score": collateral, "max": 20},
            "conditions": {"score": conditions, "max": 20},
            "total": {"score": total, "max": 100},
        }

    def _llm_synthesis(self, financial, research, risk, liquidity, fraud, five_cs, company_info) -> Dict:
        prompt = f"""You are a senior credit committee member at an Indian bank. Synthesize the following analysis and provide your recommendation.

Company: {json.dumps(company_info or {}, indent=2)}

Financial Analysis: {json.dumps(financial, indent=2)}

Research Findings: {json.dumps(research, indent=2)}

Risk Assessment: {json.dumps(risk, indent=2)}

Liquidity Analysis: {json.dumps(liquidity, indent=2)}

Fraud Analysis: {json.dumps(fraud, indent=2)}

Five Cs Scorecard: {json.dumps(five_cs, indent=2)}

Provide your response as JSON with these keys:
- decision: APPROVE / APPROVE_WITH_CONDITIONS / DECLINE
- confidence: 0.0-1.0
- rationale: 2-3 sentence summary
- key_strengths: list of strings
- key_risks: list of strings
- conditions: list of suggested conditions/covenants (if applicable)
- dissenting_view: any counter-arguments to consider

Return ONLY valid JSON."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except Exception:
            return self._rule_synthesis(financial, research, risk, liquidity, fraud, five_cs)

    def _rule_synthesis(self, financial, research, risk, liquidity, fraud, five_cs) -> Dict:
        total = five_cs["total"]["score"]
        risk_level = risk.get("risk_level", "MEDIUM")

        if total >= 70 and risk_level in ("LOW", "MEDIUM"):
            decision = "APPROVE"
        elif total >= 50 and risk_level != "CRITICAL":
            decision = "APPROVE_WITH_CONDITIONS"
        else:
            decision = "DECLINE"

        return {
            "decision": decision,
            "confidence": min(0.95, total / 100),
            "rationale": f"Five Cs score: {total}/100. Risk level: {risk_level}.",
            "key_strengths": [a for a in financial.get("alerts", []) if "above" in a.lower()] or ["Adequate financial position"],
            "key_risks": [f["factor"] for f in risk.get("risk_factors", [])[:3]],
            "conditions": ["Quarterly financial monitoring", "DSRA maintenance"] if decision == "APPROVE_WITH_CONDITIONS" else [],
        }


class CreditAgentOrchestrator:
    """Orchestrates all agents for end-to-end credit analysis."""

    def __init__(self):
        self.research_agent = ResearchAgent()
        self.financial_agent = FinancialAgent()
        self.risk_agent = RiskAgent()
        self.liquidity_agent = LiquidityAgent()
        self.fraud_agent = FraudAgentWrapper()
        self.moderator = CommitteeModerator()
        self.doc_intelligence = DocumentIntelligence()

    def run_full_analysis(self, company_info: Dict, financial_data: Dict,
                          gst_validation: Dict = None, bank_analysis: Dict = None,
                          cashflow_data: List[Dict] = None, graph_data: Dict = None,
                          field_insights: Dict = None) -> Dict[str, Any]:
        """Run the complete multi-agent analysis pipeline."""

        # Phase 1: Financial Analysis
        financial_result = self.financial_agent.analyze(financial_data)

        # Phase 2: Research
        research_result = self.research_agent.research(
            company_name=company_info.get("company_name", ""),
            promoter_names=company_info.get("promoter_names", []),
            industry=company_info.get("industry", ""),
        )

        # Phase 3: Fraud Analysis
        fraud_data = graph_data or {"company_name": company_info.get("company_name", "")}
        fraud_result = self.fraud_agent.analyze(fraud_data)

        # Phase 4: Liquidity Analysis
        liquidity_result = self.liquidity_agent.analyze(cashflow_data, financial_result)

        # Phase 5: Risk Assessment (combines all)
        risk_result = self.risk_agent.assess(
            financial_result, research_result, gst_validation, bank_analysis, fraud_result
        )

        # Phase 6: Committee Synthesis
        committee_result = self.moderator.synthesize(
            financial_result, research_result, risk_result,
            liquidity_result, fraud_result, company_info
        )

        return {
            "company_info": company_info,
            "financial_analysis": financial_result,
            "research": research_result,
            "fraud_analysis": fraud_result,
            "liquidity_analysis": liquidity_result,
            "risk_assessment": risk_result,
            "committee_synthesis": committee_result,
            "field_insights": field_insights,
        }
