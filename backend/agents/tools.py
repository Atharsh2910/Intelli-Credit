"""
Intelli-Credit: LangChain Tool Definitions
Wraps all analysis functions as LangChain Tools for agent orchestration.
Each tool accepts/returns JSON strings per LangChain convention.
"""

import json
from typing import Optional, Dict

from langchain_core.tools import Tool


# ---------------------------------------------------------------------------
# Lazy-initialized shared instances (avoids import-time failures
# when heavy libs like fitz/xgboost aren't available)
# ---------------------------------------------------------------------------

_instances: Dict[str, object] = {}


def _get_instance(name: str):
    """Lazy-init shared agent/service instances."""
    if name not in _instances:
        if name == "research":
            from backend.agents.credit_agent import ResearchAgent
            _instances[name] = ResearchAgent()
        elif name == "financial":
            from backend.agents.credit_agent import FinancialAgent
            _instances[name] = FinancialAgent()
        elif name == "risk":
            from backend.agents.credit_agent import RiskAgent
            _instances[name] = RiskAgent()
        elif name == "liquidity":
            from backend.agents.credit_agent import LiquidityAgent
            _instances[name] = LiquidityAgent()
        elif name == "fraud":
            from backend.agents.credit_agent import FraudAgentWrapper
            _instances[name] = FraudAgentWrapper()
        elif name == "doc_intelligence":
            from backend.rag.document_intelligence import DocumentIntelligence
            _instances[name] = DocumentIntelligence()
        elif name == "ingestion":
            from backend.ingestion.data_ingestion import IngestionPipeline
            _instances[name] = IngestionPipeline()
    return _instances[name]


# ---------------------------------------------------------------------------
# Tool wrapper functions
# ---------------------------------------------------------------------------

def _financial_analysis_func(input_json: str) -> str:
    """Run financial ratio analysis on company financial data."""
    try:
        data = json.loads(input_json)
        result = _get_instance("financial").analyze(data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _web_research_func(input_json: str) -> str:
    """Conduct web research on a company: news, promoter background, sector outlook, litigation, regulatory flags."""
    try:
        data = json.loads(input_json)
        company_name = data.get("company_name", "")
        promoter_names = data.get("promoter_names", [])
        industry = data.get("industry", "")
        result = _get_instance("research").research(company_name, promoter_names, industry)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _fraud_graph_analysis_func(input_json: str) -> str:
    """Run fraud graph analytics using NetworkX to detect shell companies, circular trading, and promoter network risks."""
    try:
        data = json.loads(input_json)
        result = _get_instance("fraud").analyze(data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _liquidity_analysis_func(input_json: str) -> str:
    """Analyze liquidity and cashflow stress indicators."""
    try:
        data = json.loads(input_json)
        cashflow_data = data.get("cashflow_data")
        financial_data = data.get("financial_data")
        result = _get_instance("liquidity").analyze(cashflow_data, financial_data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _risk_assessment_func(input_json: str) -> str:
    """Combine all signals (financial, research, GST, bank, fraud) into a comprehensive risk assessment."""
    try:
        data = json.loads(input_json)
        result = _get_instance("risk").assess(
            financial_analysis=data.get("financial_analysis", {}),
            research_data=data.get("research_data", {}),
            gst_validation=data.get("gst_validation"),
            bank_analysis=data.get("bank_analysis"),
            fraud_report=data.get("fraud_report"),
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _document_search_func(input_json: str) -> str:
    """Search uploaded documents for relevant context using RAG (vector similarity search)."""
    try:
        data = json.loads(input_json)
        query = data.get("query", "")
        company_name = data.get("company_name", "")
        top_k = data.get("top_k", 5)
        results = _get_instance("doc_intelligence").search(query, company_name, top_k)
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _gst_validation_func(input_json: str) -> str:
    """Cross-validate GSTR-2A vs GSTR-3B and compare GST turnover against bank credits."""
    try:
        data = json.loads(input_json)
        gst_data = data.get("gst_data", data)
        bank_data = data.get("bank_data")
        result = _get_instance("ingestion").validate_gst(gst_data, bank_data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def create_all_tools():
    """Create and return all LangChain tools."""
    return {
        "financial_analysis": Tool(
            name="financial_analysis",
            func=_financial_analysis_func,
            description=(
                "Compute financial ratios and RBI benchmark alerts for a company. "
                "Input: JSON with keys like revenue_cr, ebitda_cr, pat_cr, total_debt_cr, "
                "net_worth_cr, interest_expense_cr, depreciation_cr, current_assets_cr, "
                "current_liabilities_cr, inventory_cr, fixed_assets_cr, annual_debt_service_cr, "
                "revenue_history (list of floats). "
                "Output: JSON with 'ratios' (DSCR, D/E, current ratio, etc.), 'alerts', 'cash_accrual_cr'."
            ),
        ),
        "web_research": Tool(
            name="web_research",
            func=_web_research_func,
            description=(
                "Conduct web research on a company using Tavily search. "
                "Input: JSON with 'company_name' (str), 'promoter_names' (list of str), 'industry' (str). "
                "Output: JSON with 'company_news', 'promoter_background', 'sector_outlook', "
                "'litigation_history', 'regulatory_flags', 'overall_risk' (LOW/MEDIUM/HIGH)."
            ),
        ),
        "fraud_graph_analysis": Tool(
            name="fraud_graph_analysis",
            func=_fraud_graph_analysis_func,
            description=(
                "Run fraud graph analytics using NetworkX. Detects shell companies, circular trading, "
                "shared director clusters, and promoter network risks. "
                "Input: JSON with optional 'company_name', 'graph_companies' (list), 'graph_relationships' (list). "
                "Output: JSON with 'overall_risk_score' (0-10), 'risk_level', 'shell_company_flags', "
                "'circular_trading_flags', 'shared_director_clusters', 'promoter_network_risks', "
                "'centrality_scores', 'recommendations'."
            ),
        ),
        "liquidity_analysis": Tool(
            name="liquidity_analysis",
            func=_liquidity_analysis_func,
            description=(
                "Analyze cashflow trends and detect liquidity stress. "
                "Input: JSON with 'cashflow_data' (list of {date, value}), 'financial_data' (dict with ratios). "
                "Output: JSON with 'forecast', 'stress_indicators', 'liquidity_grade' (ADEQUATE/STRESSED/CRITICAL)."
            ),
        ),
        "risk_assessment": Tool(
            name="risk_assessment",
            func=_risk_assessment_func,
            description=(
                "Combine all signals into a comprehensive risk assessment. "
                "Input: JSON with 'financial_analysis', 'research_data', 'gst_validation', "
                "'bank_analysis', 'fraud_report' — each being the output from the corresponding tool. "
                "Output: JSON with 'risk_factors' (list), 'risk_score' (0-100), "
                "'risk_level' (LOW/MEDIUM/HIGH/CRITICAL), 'risk_summary'."
            ),
        ),
        "document_search": Tool(
            name="document_search",
            func=_document_search_func,
            description=(
                "Search uploaded documents for relevant context using RAG vector similarity. "
                "Input: JSON with 'query' (str), 'company_name' (str), 'top_k' (int, default 5). "
                "Output: JSON list of matching document chunks with 'text', 'score', 'doc_type'."
            ),
        ),
        "gst_validation": Tool(
            name="gst_validation",
            func=_gst_validation_func,
            description=(
                "Cross-validate GSTR-2A vs GSTR-3B ITC and compare GST turnover against bank credits. "
                "Input: JSON with 'gst_data' containing 'company_name', 'period', 'gstr_2a_itc', "
                "'gstr_3b_itc', 'gst_turnover'; and optional 'bank_data' with 'total_credits'. "
                "Output: JSON with discrepancy_pct, discrepancy_flag, gst_bank_gap_pct, risk_signals."
            ),
        ),
    }
