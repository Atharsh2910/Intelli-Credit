"""
Intelli-Credit: LangChain Multi-Agent Orchestrator
Executes the full credit analysis pipeline using LangChain agents.
Falls back to rule-based pipeline when LLM is not available.
"""

import json
import traceback
from typing import Dict, List, Optional, Any

from backend.agents.credit_agent import (
    ResearchAgent,
    FinancialAgent,
    RiskAgent,
    LiquidityAgent,
    FraudAgentWrapper,
    CommitteeModerator,
)

try:
    from backend.agents.langchain_agents import LangChainAgentFactory
    HAS_AGENT_FACTORY = True
except ImportError:
    HAS_AGENT_FACTORY = False

from backend.rag.document_intelligence import DocumentIntelligence


class LangChainCreditOrchestrator:
    """
    Multi-agent orchestrator for end-to-end credit analysis.

    When LangChain + OpenAI API key are available:
        Uses LangChain AgentExecutors with specialized prompts and tools.
        Each agent reasons about its domain and calls tools autonomously.

    When not available:
        Falls back to the existing rule-based pipeline (direct function calls).
    """

    def __init__(self):
        # Rule-based agents (always available as fallback)
        self.research_agent = ResearchAgent()
        self.financial_agent = FinancialAgent()
        self.risk_agent = RiskAgent()
        self.liquidity_agent = LiquidityAgent()
        self.fraud_agent = FraudAgentWrapper()
        self.moderator = CommitteeModerator()
        self.doc_intelligence = DocumentIntelligence()

        # LangChain agents (may be None if LLM not configured)
        self.agent_factory = None
        if HAS_AGENT_FACTORY:
            try:
                self.agent_factory = LangChainAgentFactory()
                if self.agent_factory.is_available:
                    print("🤖 LangChain agents initialized — using LLM-powered orchestration")
                else:
                    print("⚠️  LangChain available but no API key — using rule-based fallback")
                    self.agent_factory = None
            except Exception as e:
                print(f"⚠️  LangChain agent init failed: {e} — using rule-based fallback")
                self.agent_factory = None

    @property
    def use_langchain(self) -> bool:
        """Whether LangChain orchestration is active."""
        return self.agent_factory is not None and self.agent_factory.is_available

    def run_full_analysis(
        self,
        company_info: Dict,
        financial_data: Dict,
        gst_validation: Dict = None,
        bank_analysis: Dict = None,
        cashflow_data: List[Dict] = None,
        graph_data: Dict = None,
        field_insights: Dict = None,
    ) -> Dict[str, Any]:
        """
        Run the complete multi-agent analysis pipeline.
        Attempts LangChain orchestration first, falls back to rule-based if it fails.
        """
        if self.use_langchain:
            try:
                return self._langchain_pipeline(
                    company_info, financial_data, gst_validation,
                    bank_analysis, cashflow_data, graph_data, field_insights,
                )
            except Exception as e:
                print(f"⚠️  LangChain pipeline error: {e}")
                traceback.print_exc()
                print("⚠️  Falling back to rule-based pipeline")

        return self._rule_based_pipeline(
            company_info, financial_data, gst_validation,
            bank_analysis, cashflow_data, graph_data, field_insights,
        )

    # -----------------------------------------------------------------------
    # LangChain Pipeline
    # -----------------------------------------------------------------------

    def _langchain_pipeline(
        self,
        company_info: Dict,
        financial_data: Dict,
        gst_validation: Dict = None,
        bank_analysis: Dict = None,
        cashflow_data: List[Dict] = None,
        graph_data: Dict = None,
        field_insights: Dict = None,
    ) -> Dict[str, Any]:
        """Execute the full pipeline using LangChain agents."""

        company_name = company_info.get("company_name", "Unknown")

        # ── Phase 1: Financial Analysis ──────────────────────────────────
        financial_input = json.dumps(financial_data)
        if gst_validation:
            financial_prompt = (
                f"Analyze the financial data for {company_name}.\n"
                f"Financial Data: {financial_input}\n"
                f"GST Data is also available: {json.dumps(gst_validation)}\n"
                f"Run both financial_analysis and gst_validation tools, then synthesize."
            )
        else:
            financial_prompt = (
                f"Analyze the financial data for {company_name}.\n"
                f"Financial Data: {financial_input}\n"
                f"Run the financial_analysis tool and provide your assessment."
            )

        print(f"  📊 Phase 1: Financial Agent analyzing {company_name}...")
        financial_result = self.agent_factory.run_agent("financial", financial_prompt)

        # Ensure we have core ratios (merge tool output with agent synthesis)
        core_financial = self.financial_agent.analyze(financial_data)
        financial_result = self._merge_financial(financial_result, core_financial)

        # ── Phase 2: Research ────────────────────────────────────────────
        research_prompt = (
            f"Research company '{company_name}' in the '{company_info.get('industry', '')}' industry.\n"
            f"Promoters: {json.dumps(company_info.get('promoter_names', []))}\n"
            f"Search for news, promoter backgrounds, litigation, regulatory flags, and sector outlook."
        )

        print(f"  🔍 Phase 2: Research Agent investigating {company_name}...")
        research_result = self.agent_factory.run_agent("research", research_prompt)

        # Ensure we have core research data
        core_research = self.research_agent.research(
            company_name,
            company_info.get("promoter_names", []),
            company_info.get("industry", ""),
        )
        research_result = self._merge_research(research_result, core_research)

        # ── Phase 3: Fraud Analysis ──────────────────────────────────────
        fraud_data = graph_data or {"company_name": company_name}
        fraud_prompt = (
            f"Analyze the corporate network for fraud patterns related to {company_name}.\n"
            f"Company data: {json.dumps(fraud_data)}"
        )

        print(f"  🕵️ Phase 3: Fraud Agent analyzing {company_name} network...")
        fraud_result = self.agent_factory.run_agent("fraud", fraud_prompt)

        # Ensure core fraud data
        core_fraud = self.fraud_agent.analyze(fraud_data)
        fraud_result = self._merge_dicts(fraud_result, core_fraud)

        # ── Phase 4: Liquidity Analysis ──────────────────────────────────
        liquidity_input = {
            "cashflow_data": cashflow_data,
            "financial_data": core_financial,
        }
        liquidity_prompt = (
            f"Analyze the liquidity position of {company_name}.\n"
            f"Data: {json.dumps(liquidity_input, default=str)}"
        )

        print(f"  💧 Phase 4: Liquidity Agent analyzing {company_name}...")
        liquidity_result = self.agent_factory.run_agent("liquidity", liquidity_prompt)

        # Ensure core liquidity data
        core_liquidity = self.liquidity_agent.analyze(cashflow_data, core_financial)
        liquidity_result = self._merge_dicts(liquidity_result, core_liquidity)

        # ── Phase 5: Risk Assessment ─────────────────────────────────────
        risk_input = {
            "financial_analysis": core_financial,
            "research_data": core_research,
            "gst_validation": gst_validation,
            "bank_analysis": bank_analysis,
            "fraud_report": core_fraud,
        }
        risk_prompt = (
            f"Assess the overall credit risk for {company_name}.\n"
            f"Combined signals: {json.dumps(risk_input, default=str)}"
        )

        print(f"  ⚠️ Phase 5: Risk Agent assessing {company_name}...")
        risk_result = self.agent_factory.run_agent("risk", risk_prompt)

        # Ensure core risk data
        core_risk = self.risk_agent.assess(
            core_financial, core_research, gst_validation, bank_analysis, core_fraud
        )
        risk_result = self._merge_dicts(risk_result, core_risk)

        # ── Phase 6: Committee Synthesis ─────────────────────────────────
        five_cs = self.moderator._assess_five_cs(
            core_financial, core_research, core_risk,
            core_liquidity, core_fraud, company_info,
        )

        print(f"  🏛️ Phase 6: Committee Moderator synthesizing for {company_name}...")
        try:
            synthesis = self.agent_factory.run_committee_synthesis(
                financial_analysis=core_financial,
                research=research_result,
                fraud_analysis=fraud_result,
                liquidity_analysis=liquidity_result,
                risk_assessment=risk_result,
                company_info=company_info,
                five_cs=five_cs,
            )
        except Exception:
            synthesis = self.moderator._rule_synthesis(
                core_financial, core_research, core_risk,
                core_liquidity, core_fraud, five_cs,
            )

        committee_result = {
            "five_cs": five_cs,
            "synthesis": synthesis,
            "committee_decision": synthesis.get("decision", "REVIEW"),
            "confidence": synthesis.get("confidence", 0.5),
        }

        print(f"  ✅ LangChain pipeline complete for {company_name}")

        return {
            "company_info": company_info,
            "financial_analysis": core_financial,
            "research": research_result,
            "fraud_analysis": fraud_result,
            "liquidity_analysis": liquidity_result,
            "risk_assessment": risk_result,
            "committee_synthesis": committee_result,
            "field_insights": field_insights,
            "orchestration_mode": "langchain",
        }

    # -----------------------------------------------------------------------
    # Rule-Based Pipeline (Fallback)
    # -----------------------------------------------------------------------

    def _rule_based_pipeline(
        self,
        company_info: Dict,
        financial_data: Dict,
        gst_validation: Dict = None,
        bank_analysis: Dict = None,
        cashflow_data: List[Dict] = None,
        graph_data: Dict = None,
        field_insights: Dict = None,
    ) -> Dict[str, Any]:
        """Execute the pipeline using direct function calls (no LLM needed)."""

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

        # Phase 5: Risk Assessment
        risk_result = self.risk_agent.assess(
            financial_result, research_result, gst_validation, bank_analysis, fraud_result
        )

        # Phase 6: Committee Synthesis
        committee_result = self.moderator.synthesize(
            financial_result, research_result, risk_result,
            liquidity_result, fraud_result, company_info,
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
            "orchestration_mode": "rule_based",
        }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _merge_financial(agent_output: Dict, core_output: Dict) -> Dict:
        """Merge LangChain agent financial output with core computed ratios."""
        merged = dict(core_output)
        # Overlay agent insights (summaries, health grades) on top of core ratios
        for key in ("financial_health", "financial_summary", "gst_validation"):
            if key in agent_output:
                merged[key] = agent_output[key]
        return merged

    @staticmethod
    def _merge_research(agent_output: Dict, core_output: Dict) -> Dict:
        """Merge LangChain agent research output with core research data."""
        merged = dict(core_output)
        # The agent might produce a more detailed summary
        if "research_summary" in agent_output:
            merged["research_summary"] = agent_output["research_summary"]
        # Extend lists if the agent found additional items
        for list_key in ("company_news", "promoter_background", "sector_outlook",
                         "litigation_history", "regulatory_flags"):
            if list_key in agent_output and isinstance(agent_output[list_key], list):
                existing_set = {json.dumps(item, sort_keys=True) for item in merged.get(list_key, [])}
                for item in agent_output[list_key]:
                    if json.dumps(item, sort_keys=True) not in existing_set:
                        merged.setdefault(list_key, []).append(item)
        # Prefer the more severe risk level
        risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        agent_risk = risk_order.get(agent_output.get("overall_risk", "LOW"), 0)
        core_risk = risk_order.get(merged.get("overall_risk", "LOW"), 0)
        if agent_risk > core_risk:
            merged["overall_risk"] = agent_output["overall_risk"]
        return merged

    @staticmethod
    def _merge_dicts(agent_output: Dict, core_output: Dict) -> Dict:
        """Generic merge: core output as base, agent insights layered on top."""
        merged = dict(core_output)
        for key, val in agent_output.items():
            if key == "parse_error":
                continue
            if key == "raw_output":
                merged["agent_commentary"] = val
                continue
            # Only overwrite if the agent has a meaningful value
            if val is not None and val != "" and val != []:
                if key not in merged:
                    merged[key] = val
                elif isinstance(val, str) and key.endswith("_summary"):
                    merged[key] = val  # Prefer agent summaries
        return merged
