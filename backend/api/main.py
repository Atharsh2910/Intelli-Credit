"""
Intelli-Credit: FastAPI Backend
9 endpoints for the credit decisioning engine.
"""

import os
import sys
import json
import logging
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Configure logging so we can see errors on Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
# Service Instances (initialized in startup event to avoid port-bind failures)
# ---------------------------------------------------------------------------

ingestion = None
orchestrator = None
decision_engine = None
cam_generator = None
doc_intelligence = None
fraud_analyzer = None
doc_classifier = None
schema_manager = None
swot_analyzer = None
genai_narrative = None
services_ready = False
model_dir = os.path.join(project_root, "backend", "ml", "models")


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str  # "healthy", "degraded", or "initializing"
    timestamp: str
    models_loaded: bool
    services_ready: bool = False
    services: Dict[str, bool] = {}
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
    models_loaded = False
    try:
        if decision_engine and hasattr(decision_engine, 'risk_model'):
            models_loaded = decision_engine.risk_model.is_trained
    except Exception:
        pass

    # Report exactly which services are up and which are down
    svc_status = {
        "ingestion": ingestion is not None,
        "orchestrator": orchestrator is not None,
        "decision_engine": decision_engine is not None,
        "cam_generator": cam_generator is not None,
        "doc_intelligence": doc_intelligence is not None,
        "fraud_analyzer": fraud_analyzer is not None,
    }

    if not services_ready:
        status = "initializing"
    elif all(svc_status.values()):
        status = "healthy"
    else:
        status = "degraded"

    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        models_loaded=models_loaded,
        services_ready=services_ready,
        services=svc_status,
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
        research = agent_result.get("research", {})
        
        features = {
            **ratios,
            "promoter_contribution_pct": fd.promoter_contribution_pct,
            "gst_compliance_score": 7.0,
            "gstr_2a_3b_discrepancy_pct": gst_validation.get("discrepancy_pct", 5) if gst_validation else 5,
            "bank_credit_debit_ratio": request.bank_data.credit_debit_ratio if request.bank_data else 1.05,
            "emi_bounce_count": request.bank_data.emi_bounce_count if request.bank_data else 0,
            "avg_bank_utilisation_pct": request.bank_data.avg_utilisation_pct if request.bank_data else 65,
            "cibil_score": ci.cibil_score,
            "litigation_count": len(research.get("litigation_history", [])),
            "years_in_business": ci.years_in_business,
            "promoter_experience_yrs": ci.promoter_experience_yrs,
            "secondary_research_risk": {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(
                research.get("overall_risk", "LOW"), 0
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

        # Step 8: SWOT Analysis
        swot = swot_analyzer.generate(agent_result, ml_decision, company_dict)

        # Step 9: GenAI Narrative
        narrative = genai_narrative.generate(agent_result, ml_decision, company_dict)

        return {
            "status": "success",
            "data": {
                "decision": ml_decision,
                "agent_analysis": agent_result,
                "cam": cam,
                "swot_analysis": swot,
                "genai_narrative": narrative,
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
# New Feature Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/classify")
async def classify_documents(
    files: List[UploadFile] = File(...),
):
    """Auto-classify uploaded documents with confidence scores."""
    try:
        classifications = []
        for file in files:
            file_bytes = await file.read()
            result = doc_classifier.classify_from_bytes(file_bytes, file.filename)
            from dataclasses import asdict
            classifications.append(asdict(result))
        return {"status": "success", "data": {"classifications": classifications, "doc_types": doc_classifier.get_doc_types()}}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class ClassificationConfirmation(BaseModel):
    filename: str
    confirmed_type: str


@app.post("/api/classify/confirm")
async def confirm_classifications(confirmations: List[ClassificationConfirmation]):
    """Accept user-confirmed or edited classifications."""
    try:
        confirmed = []
        for c in confirmations:
            confirmed.append({
                "filename": c.filename,
                "confirmed_type": c.confirmed_type,
            })
        return {"status": "success", "data": {"confirmed": confirmed}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/schema/defaults")
async def get_default_schemas():
    """Return default extraction schemas for all document types."""
    try:
        schemas = schema_manager.get_all_schemas()
        return {"status": "success", "data": schemas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CustomSchemaRequest(BaseModel):
    doc_type: str
    fields: List[Dict[str, Any]]


@app.post("/api/schema/custom")
async def set_custom_schema(request: CustomSchemaRequest):
    """Set a custom extraction schema for a document type."""
    try:
        result = schema_manager.set_custom_schema(request.doc_type, request.fields)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/report/download")
async def download_report(request: FullDecisionRequest):
    """Generate and download a PDF Credit Appraisal Memo."""
    try:
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
        pdf_bytes = cam_generator.generate_pdf(ml_decision, agent_result, ci.model_dump())

        import io
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=CAM_{ci.company_name.replace(' ', '_')}.pdf"
            },
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _init_services():
    """Initialize all services (runs in background thread to not block port binding)."""
    import time
    global ingestion, orchestrator, decision_engine, cam_generator
    global doc_intelligence, fraud_analyzer, services_ready
    global doc_classifier, schema_manager, swot_analyzer, genai_narrative

    t_total = time.time()
    logger.info("🔄 Initializing services...")

    # -- 1. IngestionPipeline (local only, should be fast) --
    try:
        t = time.time()
        from backend.ingestion.data_ingestion import IngestionPipeline
        logger.info(f"  IngestionPipeline: import took {time.time()-t:.1f}s")
        t = time.time()
        ingestion = IngestionPipeline()
        logger.info(f"✅ IngestionPipeline ready ({time.time()-t:.1f}s init)")
    except Exception as e:
        logger.error(f"❌ IngestionPipeline failed: {e}")
        traceback.print_exc()

    # -- 2. Orchestrator (imports langchain + credit agents, creates DocumentIntelligence → Pinecone) --
    try:
        t = time.time()
        from backend.agents.orchestrator import LangChainCreditOrchestrator
        logger.info(f"  Orchestrator: import took {time.time()-t:.1f}s")
        t = time.time()
        orchestrator = LangChainCreditOrchestrator()
        logger.info(f"✅ Orchestrator ready ({time.time()-t:.1f}s init)")
    except Exception as e:
        logger.error(f"❌ Orchestrator failed: {e}")
        traceback.print_exc()

    # -- 3. DecisionEngine (loads ML models from disk) --
    try:
        t = time.time()
        from backend.decision_engine.engine import CreditDecisionEngine
        logger.info(f"  DecisionEngine: import took {time.time()-t:.1f}s")
        t = time.time()
        decision_engine = CreditDecisionEngine(model_dir)
        logger.info(f"✅ DecisionEngine ready ({time.time()-t:.1f}s init)")
    except Exception as e:
        logger.error(f"❌ DecisionEngine failed: {e}")
        traceback.print_exc()

    # -- 4. CAMGenerator (lightweight, just creates OpenAI client) --
    try:
        t = time.time()
        from backend.cam_generator.memo_generator import CAMGenerator
        cam_generator = CAMGenerator()
        logger.info(f"✅ CAMGenerator ready ({time.time()-t:.1f}s)")
    except Exception as e:
        logger.error(f"❌ CAMGenerator failed: {e}")
        traceback.print_exc()

    # -- 5. DocumentIntelligence --
    # The Orchestrator already creates one internally (orchestrator.doc_intelligence).
    # Re-use it if available, otherwise create a fresh one.
    try:
        if orchestrator and hasattr(orchestrator, 'doc_intelligence') and orchestrator.doc_intelligence:
            doc_intelligence = orchestrator.doc_intelligence
            logger.info("✅ DocumentIntelligence reused from Orchestrator")
        else:
            t = time.time()
            from backend.rag.document_intelligence import DocumentIntelligence
            doc_intelligence = DocumentIntelligence()
            logger.info(f"✅ DocumentIntelligence ready ({time.time()-t:.1f}s)")
    except Exception as e:
        logger.error(f"❌ DocumentIntelligence failed: {e}")
        traceback.print_exc()

    # -- 6. FraudGraphAnalyzer (local only, NetworkX graph) --
    try:
        t = time.time()
        from backend.fraud_graph.graph_analytics import FraudGraphAnalyzer
        fraud_analyzer = FraudGraphAnalyzer()
        logger.info(f"✅ FraudGraphAnalyzer ready ({time.time()-t:.1f}s)")
    except Exception as e:
        logger.error(f"❌ FraudGraphAnalyzer failed: {e}")
        traceback.print_exc()

    # New feature services
    try:
        from backend.classification.classifier import DocumentClassifier
        doc_classifier = DocumentClassifier()
        logger.info("✅ DocumentClassifier ready")
    except Exception as e:
        logger.error(f"❌ DocumentClassifier failed: {e}")
        traceback.print_exc()

    try:
        from backend.schema.dynamic_schema import DynamicSchemaManager
        schema_manager = DynamicSchemaManager()
        logger.info("✅ DynamicSchemaManager ready")
    except Exception as e:
        logger.error(f"❌ DynamicSchemaManager failed: {e}")
        traceback.print_exc()

    try:
        from backend.analysis.swot_analysis import SWOTAnalyzer
        swot_analyzer = SWOTAnalyzer()
        logger.info("✅ SWOTAnalyzer ready")
    except Exception as e:
        logger.error(f"❌ SWOTAnalyzer failed: {e}")
        traceback.print_exc()

    try:
        from backend.analysis.genai_narrative import GenAINarrativeGenerator
        genai_narrative = GenAINarrativeGenerator()
        logger.info("✅ GenAINarrativeGenerator ready")
    except Exception as e:
        logger.error(f"❌ GenAINarrativeGenerator failed: {e}")
        traceback.print_exc()

    # Check ML models
    if decision_engine and hasattr(decision_engine, 'risk_model') and decision_engine.risk_model.is_trained:
        logger.info("✅ ML models loaded successfully.")
    else:
        logger.warning("⚠️  ML models not found. Run 'python scripts/train_model.py' to train.")

    services_ready = True
    logger.info(f"🚀 Intelli-Credit API is ready. Total init: {time.time()-t_total:.1f}s")


@app.on_event("startup")
async def startup():
    """
    Kick off service initialization in a background thread so that
    uvicorn can bind the port immediately.  Render requires the port
    to be open within ~60 s or it declares 'no open port found'.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _init_services)

