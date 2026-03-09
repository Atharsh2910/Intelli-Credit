"""
Intelli-Credit: LangChain Agent Definitions
Specialized LangChain agents with focused prompts and curated tool subsets.
Each agent uses create_openai_tools_agent + AgentExecutor for structured tool calling.
"""

import os
import json
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_openai import ChatOpenAI
    from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    HAS_LANGCHAIN = True
except ImportError:
    try:
        # Fallback for older langchain versions
        from langchain_openai import ChatOpenAI
        from langchain.agents import AgentExecutor, create_openai_tools_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        HAS_LANGCHAIN = True
    except ImportError:
        HAS_LANGCHAIN = False

from backend.agents.tools import create_all_tools


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

RESEARCH_AGENT_PROMPT = """You are a senior credit research analyst at an Indian bank. Your role is to:

1. Search for company news, promoter backgrounds, sector outlook, and litigation history
2. Search uploaded documents for relevant context about the company
3. Identify regulatory red flags (RBI wilful defaulter, NCLT/DRT cases)
4. Assess the overall research risk level

You MUST use the available tools to gather information. Always use the web_research tool first 
with the company details, then supplement with document_search if a company name is provided.

After gathering information, synthesize your findings into a JSON response with these keys:
- company_news: list of relevant news items
- promoter_background: list of findings about promoters
- sector_outlook: list of sector insights
- litigation_history: list of litigation/court cases found
- regulatory_flags: list of regulatory red flags
- overall_risk: "LOW", "MEDIUM", or "HIGH"
- research_summary: 2-3 sentence summary of key findings

Return ONLY valid JSON."""

FINANCIAL_AGENT_PROMPT = """You are a financial analyst at an Indian bank specializing in corporate credit assessment. Your role is to:

1. Analyze financial statements and compute key ratios (DSCR, D/E, current ratio, etc.)
2. Check compliance with RBI regulatory benchmarks
3. Validate GST returns if GST data is provided
4. Identify financial red flags and strengths

You MUST use the financial_analysis tool with the provided financial data. 
If GST data is available, also use the gst_validation tool.

After running the tools, synthesize your findings into a JSON response with:
- ratios: dict of all computed financial ratios
- alerts: list of RBI benchmark violations
- cash_accrual_cr: cash accrual amount
- gst_validation: GST validation results (if applicable)
- financial_health: "STRONG", "ADEQUATE", "WEAK", or "CRITICAL"
- financial_summary: 2-3 sentence assessment

Return ONLY valid JSON."""

FRAUD_AGENT_PROMPT = """You are a fraud investigation specialist at an Indian bank. Your role is to:

1. Analyze corporate network graphs to detect fraud patterns
2. Identify shell companies, circular trading, and suspicious director networks
3. Assess promoter network risks and hidden connections
4. Provide actionable recommendations

You MUST use the fraud_graph_analysis tool with the provided company data.

After analysis, synthesize findings into a JSON response with:
- overall_risk_score: float 0-10
- risk_level: "LOW", "MEDIUM", "HIGH", or "CRITICAL"
- shell_company_flags: list of suspected shell entities
- circular_trading_flags: list of circular trading patterns
- shared_director_clusters: list of director clusters
- promoter_network_risks: list of promoter risks
- centrality_scores: dict of graph centrality metrics
- recommendations: list of actionable recommendations
- fraud_summary: 2-3 sentence summary

Return ONLY valid JSON."""

LIQUIDITY_AGENT_PROMPT = """You are a liquidity and cashflow analyst at an Indian bank. Your role is to:

1. Forecast future cashflows based on historical data
2. Detect liquidity stress indicators
3. Assess the company's ability to meet short-term obligations
4. Grade the overall liquidity position

You MUST use the liquidity_analysis tool with the provided data.

After analysis, provide a JSON response with:
- forecast: cashflow forecast data (if available)
- stress_indicators: list of detected stress signals
- liquidity_grade: "ADEQUATE", "STRESSED", or "CRITICAL"
- liquidity_summary: 2-3 sentence assessment

Return ONLY valid JSON."""

RISK_AGENT_PROMPT = """You are a senior risk officer at an Indian bank. Your role is to:

1. Combine signals from financial analysis, research, GST, banking, and fraud assessments
2. Identify and categorize all risk factors by severity
3. Compute an overall risk score (0-100 scale)
4. Determine the risk level and provide a risk summary

You MUST use the risk_assessment tool with data from all previous analyses.

After assessment, provide a JSON response with:
- risk_factors: list of {category, factor, severity, detail}
- risk_score: int 0-100
- risk_level: "LOW", "MEDIUM", "HIGH", or "CRITICAL"
- risk_summary: summary of findings
- top_risks: list of top 3 most critical risks

Return ONLY valid JSON."""

COMMITTEE_MODERATOR_PROMPT = """You are the chairperson of a credit committee at a major Indian bank. 
You are reviewing a credit proposal and must synthesize all agent analyses into a final recommendation.

You have received the following analyses:
- Financial Analysis: {financial_analysis}
- Research Findings: {research}
- Fraud Analysis: {fraud_analysis}
- Liquidity Analysis: {liquidity_analysis}
- Risk Assessment: {risk_assessment}
- Company Information: {company_info}
- Five Cs Scorecard: {five_cs}

Based on ALL the above, provide your committee decision as JSON:
- decision: "APPROVE", "APPROVE_WITH_CONDITIONS", or "DECLINE"
- confidence: float 0.0-1.0
- rationale: 2-3 sentence summary of your reasoning
- key_strengths: list of strengths supporting approval
- key_risks: list of key risks
- conditions: list of suggested covenants/conditions (if applicable)
- dissenting_view: counter-arguments to consider

Return ONLY valid JSON."""


# ---------------------------------------------------------------------------
# Agent Factory
# ---------------------------------------------------------------------------

def _create_llm(temperature: float = 0.1):
    """Create the ChatOpenAI LLM instance."""
    if not HAS_LANGCHAIN:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=temperature,
        api_key=api_key,
    )


def _create_agent_executor(
    system_prompt: str,
    tools: list,
    llm=None,
    max_iterations: int = 5,
) -> Optional["AgentExecutor"]:
    """Create a LangChain AgentExecutor with the given prompt and tools."""
    if not HAS_LANGCHAIN or llm is None:
        return None

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=os.getenv("APP_ENV") == "development",
        max_iterations=max_iterations,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )


class LangChainAgentFactory:
    """Creates and manages all LangChain agents."""

    def __init__(self):
        self.llm = _create_llm()
        self.all_tools = create_all_tools()
        self._agents = {}
        self._build_agents()

    @property
    def is_available(self) -> bool:
        """Check if LangChain agents are available (LLM configured)."""
        return self.llm is not None and HAS_LANGCHAIN

    def _build_agents(self):
        """Build all specialized agents."""
        if not self.is_available:
            return

        tools = self.all_tools

        # Research Agent
        self._agents["research"] = _create_agent_executor(
            system_prompt=RESEARCH_AGENT_PROMPT,
            tools=[tools["web_research"], tools["document_search"]],
            llm=self.llm,
            max_iterations=5,
        )

        # Financial Agent
        self._agents["financial"] = _create_agent_executor(
            system_prompt=FINANCIAL_AGENT_PROMPT,
            tools=[tools["financial_analysis"], tools["gst_validation"]],
            llm=self.llm,
            max_iterations=4,
        )

        # Fraud Agent
        self._agents["fraud"] = _create_agent_executor(
            system_prompt=FRAUD_AGENT_PROMPT,
            tools=[tools["fraud_graph_analysis"]],
            llm=self.llm,
            max_iterations=3,
        )

        # Liquidity Agent
        self._agents["liquidity"] = _create_agent_executor(
            system_prompt=LIQUIDITY_AGENT_PROMPT,
            tools=[tools["liquidity_analysis"]],
            llm=self.llm,
            max_iterations=3,
        )

        # Risk Agent
        self._agents["risk"] = _create_agent_executor(
            system_prompt=RISK_AGENT_PROMPT,
            tools=[tools["risk_assessment"]],
            llm=self.llm,
            max_iterations=3,
        )

    def get_agent(self, name: str) -> Optional["AgentExecutor"]:
        """Get an agent by name."""
        return self._agents.get(name)

    def run_agent(self, name: str, input_text: str) -> Dict[str, Any]:
        """Run a named agent and parse its JSON output."""
        agent = self._agents.get(name)
        if agent is None:
            raise RuntimeError(f"Agent '{name}' not available. Check LLM configuration.")

        result = agent.invoke({"input": input_text})
        output = result.get("output", "{}")

        # Parse JSON from output (handle markdown code fences)
        return self._parse_json_output(output)

    def run_committee_synthesis(
        self,
        financial_analysis: Dict,
        research: Dict,
        fraud_analysis: Dict,
        liquidity_analysis: Dict,
        risk_assessment: Dict,
        company_info: Dict,
        five_cs: Dict,
    ) -> Dict[str, Any]:
        """Run the committee moderator as a direct LLM call (no tools needed)."""
        if not self.is_available:
            raise RuntimeError("LLM not available for committee synthesis.")

        prompt = COMMITTEE_MODERATOR_PROMPT.format(
            financial_analysis=json.dumps(financial_analysis, indent=2),
            research=json.dumps(research, indent=2),
            fraud_analysis=json.dumps(fraud_analysis, indent=2),
            liquidity_analysis=json.dumps(liquidity_analysis, indent=2),
            risk_assessment=json.dumps(risk_assessment, indent=2),
            company_info=json.dumps(company_info, indent=2),
            five_cs=json.dumps(five_cs, indent=2),
        )

        response = self.llm.invoke(prompt)
        return self._parse_json_output(response.content)

    @staticmethod
    def _parse_json_output(text: str) -> Dict[str, Any]:
        """Parse JSON from LLM output, handling markdown code fences."""
        text = text.strip()

        # Remove markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return {"raw_output": text, "parse_error": True}
