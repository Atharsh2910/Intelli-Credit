"""
Intelli-Credit: FastAPI Backend
9 endpoints for the credit decisioning engine.
"""

import os
import sys
import json
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.ingestion.data_ingestion import IngestionPipeline
from backend.agents.credit_agent import CreditAgentOrchestrator
from backend.decision_engine.engine import CreditDecisionEngine
from backend.cam_generator.memo_generator import CAMGenerator
from backend.ml.credit_model import CreditRiskModel, LoanLimitModel, InterestRateModel
from backend.fraud_graph.graph_analytics import FraudGraphAnalyzer
from backend.rag.document_intelligence import DocumentIntelligence

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Intelli-Credit API",
    description="AI-Powered Corporate Credit Decisioning Engine",
    version="1.0.0",
)

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Service Instances
# ---------------------------------------------------------------------------

ingestion = IngestionPipeline()
orchestrator = CreditAgentOrchestrator()
model_dir = os.path.join(project_root, "backend", "ml", "models")
decision_engine = CreditDecisionEngine(model_dir)
cam_generator = CAMGenerator()
doc_intelligence = DocumentIntelligence()
fraud_analyzer = FraudGraphAnalyzer()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: bool
    version: str = "1.0.0"


class CompanyInfo(BaseModel):
    company_name: str
    industry: str = ""
    cin: str = ""
    years_in_business: float = 5
    promoter_names: List[str] = []
    promoter_experience_yrs: float = 10
    cibil_score: float = 700
    collateral_type: str = "immovable"
    security_coverage: float = 1.3
    sector_outlook: str = "stable"
    macro_risk_score: float = 5.0


class FinancialData(BaseModel):
    revenue_cr: float = 10
    ebitda_cr: float = 1.5
    pat_cr: float = 0.8
    total_debt_cr: float = 5
    net_worth_cr: float = 4
    interest_expense_cr: float = 0.5
    depreciation_cr: float = 0.3
    current_assets_cr: float = 6
    current_liabilities_cr: float = 4
    inventory_cr: float = 2
    fixed_assets_cr: float = 8
    annual_debt_service_cr: float = 1.2
    revenue_history: List[float] = [7, 8, 9, 10]
    promoter_contribution_pct: float = 55


class GSTData(BaseModel):
    company_name: str
    period: str = "FY2023-24"
    gstr_2a_itc: float = 0
    gstr_3b_itc: float = 0
    gst_turnover: float = 0


class BankData(BaseModel):
    total_credits: float = 0
    avg_monthly_balance: float = 0
    emi_bounce_count: int = 0
    avg_utilisation_pct: float = 65
    credit_debit_ratio: float = 1.05


class StructuredIngestionRequest(BaseModel):
    company_name: str
    gst_data: Optional[GSTData] = None
    bank_data: Optional[BankData] = None


class FieldInsights(BaseModel):
    factory_condition: str = "good"
    management_quality: str = "good"
    inventory_observation: str = ""
    workforce_observation: str = ""
    additional_notes: str = ""


class FullDecisionRequest(BaseModel):
    company_info: CompanyInfo
    financial_data: FinancialData
    gst_data: Optional[GSTData] = None
    bank_data: Optional[BankData] = None
    field_insights: Optional[FieldInsights] = None
    cashflow_history: Optional[List[Dict[str, Any]]] = None


class MLFeatures(BaseModel):
    dscr: float = 1.3
    interest_coverage_ratio: float = 2.5
    ebitda_margin_pct: float = 12
    revenue_growth_3yr_cagr: float = 10
    pat_margin_pct: float = 5
    debt_equity_ratio: float = 1.2
    tangible_net_worth_cr: float = 3
    promoter_contribution_pct: float = 55
    cash_accrual_to_debt: float = 0.15
    current_ratio: float = 1.3
    quick_ratio: float = 0.9
    facr: float = 1.5
    security_coverage: float = 1.3
    gst_compliance_score: float = 7
    gstr_2a_3b_discrepancy_pct: float = 5
    bank_credit_debit_ratio: float = 1.05
    emi_bounce_count: int = 0
    avg_bank_utilisation_pct: float = 65
    cibil_score: float = 700
    litigation_count: int = 0
    years_in_business: float = 5
    promoter_experience_yrs: float = 10
    secondary_research_risk: int = 0
    sector_outlook_score: float = 6
    macro_risk_score: float = 5


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=decision_engine.risk_model.is_trained,
    )


@app.post("/api/ingest/document")
async def ingest_document(
    file: UploadFile = File(...),
    company_name: str = Form("Unknown"),
    doc_type: str = Form("other"),
):
    """Upload and parse a PDF document."""
    try:
        file_bytes = await file.read()
        result = ingestion.ingest_document(
            file_bytes=file_bytes,
            filename=file.filename,
            company_name=company_name,
            doc_type=doc_type,
        )

        # Store in vector DB
        if result["chunks"]:
            store_result = doc_intelligence.upsert_chunks(result["chunks"], company_name)
            result["vector_store"] = store_result

        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/structured")
async def ingest_structured(request: StructuredIngestionRequest):
    """Process structured GST and bank data."""
    try:
        result = {}

        if request.gst_data:
            bank_dict = None
            if request.bank_data:
                bank_dict = {"total_credits": request.bank_data.total_credits}
            gst_result = ingestion.validate_gst(request.gst_data.model_dump(), bank_dict)
            result["gst_validation"] = gst_result

        if request.bank_data:
            result["bank_analysis"] = {
                "total_credits": request.bank_data.total_credits,
                "avg_monthly_balance": request.bank_data.avg_monthly_balance,
                "emi_bounce_count": request.bank_data.emi_bounce_count,
                "avg_utilisation_pct": request.bank_data.avg_utilisation_pct,
                "credit_debit_ratio": request.bank_data.credit_debit_ratio,
            }

        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyse/primary")
async def analyse_primary(company_name: str = Form(...)):
    """RAG-based primary analysis from uploaded documents."""
    try:
        queries = [
            f"{company_name} financial performance revenue profit",
            f"{company_name} risk factors litigation",
            f"{company_name} promoter background management",
            f"{company_name} collateral security assets",
            f"{company_name} industry sector outlook",
        ]
        context = doc_intelligence.multi_query_search(queries, company_name)
        return {"status": "success", "data": {"context": context, "company": company_name}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyse/secondary")
async def analyse_secondary(company_name: str = Form(...), promoter_names: str = Form(""), industry: str = Form("")):
    """Web research using Tavily."""
    try:
        promoters = [n.strip() for n in promoter_names.split(",") if n.strip()]
        research = orchestrator.research_agent.research(company_name, promoters, industry)
        return {"status": "success", "data": research}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyse/fraud")
async def analyse_fraud(company_name: str = Form(...)):
    """Run fraud graph analytics with sample data."""
    try:
        report = fraud_analyzer.build_sample_graph(company_name)
        from dataclasses import asdict
        return {"status": "success", "data": asdict(report)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/score")
async def ml_score(features: MLFeatures):
    """Run ML scoring only."""
    try:
        feature_dict = features.model_dump()
        risk_result = decision_engine.risk_model.predict(feature_dict)
        loan_amount = decision_engine.loan_model.predict(feature_dict)
        interest_rate = decision_engine.rate_model.predict(feature_dict)

        return {
            "status": "success",
            "data": {
                "default_probability": risk_result["default_probability"],
                "credit_score": risk_result["credit_score"],
                "loan_amount_cr": round(loan_amount, 2),
                "interest_rate_pct": interest_rate,
                "shap_explanation": risk_result["shap_explanation"],
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/decision")
async def full_decision(request: FullDecisionRequest):
    """Full pipeline — agents + ML + decision + memo."""
    try:
        ci = request.company_info
        fd = request.financial_data

        # Step 1: Financial analysis
        financial_dict = fd.model_dump()

        # Step 2: GST validation
        gst_validation = None
        if request.gst_data:
            bank_dict = None
            if request.bank_data:
                bank_dict = {"total_credits": request.bank_data.total_credits}
            gst_validation = ingestion.validate_gst(request.gst_data.model_dump(), bank_dict)

        # Step 3: Bank data
        bank_analysis = None
        if request.bank_data:
            bank_analysis = request.bank_data.model_dump()

        # Step 4: Run multi-agent analysis
        company_dict = ci.model_dump()
        agent_result = orchestrator.run_full_analysis(
            company_info=company_dict,
            financial_data=financial_dict,
            gst_validation=gst_validation,
            bank_analysis=bank_analysis,
            cashflow_data=request.cashflow_history,
            field_insights=request.field_insights.model_dump() if request.field_insights else None,
        )

        # Step 5: Build ML features
        ratios = agent_result.get("financial_analysis", {}).get("ratios", {})
        features = {
            **ratios,
            "promoter_contribution_pct": fd.promoter_contribution_pct,
            "gst_compliance_score": 7.0,
            "gstr_2a_3b_discrepancy_pct": gst_validation.get("discrepancy_pct", 5) if gst_validation else 5,
            "bank_credit_debit_ratio": request.bank_data.credit_debit_ratio if request.bank_data else 1.05,
            "emi_bounce_count": request.bank_data.emi_bounce_count if request.bank_data else 0,
            "avg_bank_utilisation_pct": request.bank_data.avg_utilisation_pct if request.bank_data else 65,
            "cibil_score": ci.cibil_score,
            "litigation_count": len(agent_result.get("research", {}).get("litigation_history", [])),
            "years_in_business": ci.years_in_business,
            "promoter_experience_yrs": ci.promoter_experience_yrs,
            "secondary_research_risk": {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(
                agent_result.get("research", {}).get("overall_risk", "LOW"), 0
            ),
            "sector_outlook_score": 6.0,
            "macro_risk_score": ci.macro_risk_score,
            "security_coverage": ci.security_coverage,
            "revenue_cr": fd.revenue_cr,
        }

        # Step 6: ML decision
        ml_decision = decision_engine.decide(features, agent_result)

        # Step 7: Generate CAM
        cam = cam_generator.generate(ml_decision, agent_result, company_dict)

        return {
            "status": "success",
            "data": {
                "decision": ml_decision,
                "agent_analysis": agent_result,
                "cam": cam,
            }
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memo")
async def generate_memo(request: FullDecisionRequest):
    """Generate CAM from existing decision data."""
    try:
        # Quick pipeline
        ci = request.company_info
        fd = request.financial_data
        financial_dict = fd.model_dump()

        agent_result = orchestrator.run_full_analysis(
            company_info=ci.model_dump(),
            financial_data=financial_dict,
        )

        ratios = agent_result.get("financial_analysis", {}).get("ratios", {})
        features = {
            **ratios,
            "cibil_score": ci.cibil_score,
            "years_in_business": ci.years_in_business,
            "promoter_experience_yrs": ci.promoter_experience_yrs,
            "security_coverage": ci.security_coverage,
            "gst_compliance_score": 7.0,
            "gstr_2a_3b_discrepancy_pct": 5,
            "bank_credit_debit_ratio": 1.05,
            "emi_bounce_count": 0,
            "avg_bank_utilisation_pct": 65,
            "litigation_count": 0,
            "secondary_research_risk": 0,
            "sector_outlook_score": 6.0,
            "macro_risk_score": ci.macro_risk_score,
            "promoter_contribution_pct": fd.promoter_contribution_pct,
            "revenue_cr": fd.revenue_cr,
        }

        ml_decision = decision_engine.decide(features, agent_result)
        cam = cam_generator.generate(ml_decision, agent_result, ci.model_dump())

        return {"status": "success", "data": cam}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Check if models are loaded."""
    if not decision_engine.risk_model.is_trained:
        print("⚠️  ML models not found. Run 'python scripts/train_model.py' to train.")
    else:
        print("✅ ML models loaded successfully.")
    print("🚀 Intelli-Credit API is ready.")
