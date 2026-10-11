"""
Comprehensive test script for Intelli-Credit backend endpoints.
Tests all endpoints using FastAPI TestClient to ensure proper wiring and zero errors.
"""

import os
import sys
import json
import io
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from backend.api.main import app, _init_services

# Initialize services synchronously for the test
print("Initializing backend services...")
_init_services()
print("Services initialized.")

client = TestClient(app)

def test_health():
    print("\n--- 1. Testing GET /api/health ---")
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"Health check failed: {resp.status_code} {resp.text}"
    data = resp.json()
    print("Health Status:", data.get("status"), "Services:", data.get("services"))
    print("[OK] /api/health PASSED")

def test_schema_defaults():
    print("\n--- 2. Testing GET /api/schema/defaults ---")
    resp = client.get("/api/schema/defaults")
    assert resp.status_code == 200, f"Schema defaults failed: {resp.status_code} {resp.text}"
    data = resp.json()
    assert data["status"] == "success"
    print("Default schema types:", list(data["data"].keys()))
    print("[OK] /api/schema/defaults PASSED")

def test_schema_custom():
    print("\n--- 3. Testing POST /api/schema/custom ---")
    schema_payload = {
        "doc_type": "custom_audit",
        "fields": [
            {"name": "auditor_name", "label": "Auditor Name", "type": "text", "required": True}
        ]
    }
    resp = client.post("/api/schema/custom", json=schema_payload)
    assert resp.status_code == 200, f"Schema custom failed: {resp.status_code} {resp.text}"
    print("[OK] /api/schema/custom PASSED")

def test_classify():
    print("\n--- 4. Testing POST /api/classify ---")
    sample_content = b"Annual Report of Tata Motors Limited 2024. Balance Sheet, Director's Report."
    files = [("files", ("annual_report.pdf", sample_content, "application/pdf"))]
    resp = client.post("/api/classify", files=files)
    assert resp.status_code == 200, f"Classify failed: {resp.status_code} {resp.text}"
    data = resp.json()
    first_cls = data["data"]["classifications"][0]
    print("Classification result:", first_cls.get("suggested_type") or first_cls.get("doc_type"), "confidence:", first_cls["confidence"])
    print("[OK] /api/classify PASSED")

def test_classify_confirm():
    print("\n--- 5. Testing POST /api/classify/confirm ---")
    resp = client.post("/api/classify/confirm", json=[{"filename": "annual_report.pdf", "confirmed_type": "annual_report"}])
    assert resp.status_code == 200, f"Classify confirm failed: {resp.status_code} {resp.text}"
    print("[OK] /api/classify/confirm PASSED")

def test_ingest_structured():
    print("\n--- 6. Testing POST /api/ingest/structured ---")
    financial_payload = {
        "company_name": "Acme Engineering",
        "gst_data": {
            "company_name": "Acme Engineering",
            "period": "FY2023-24",
            "gstr_2a_itc": 8.0,
            "gstr_3b_itc": 8.2,
            "gst_turnover": 118.0
        },
        "bank_data": {
            "total_credits": 120.0,
            "avg_monthly_balance": 8.0,
            "emi_bounce_count": 0,
            "avg_utilisation_pct": 60.0,
            "credit_debit_ratio": 1.05
        }
    }
    resp = client.post("/api/ingest/structured", json=financial_payload)
    assert resp.status_code == 200, f"Ingest structured failed: {resp.status_code} {resp.text}"
    print("[OK] /api/ingest/structured PASSED")

def test_ingest_document():
    print("\n--- 7. Testing POST /api/ingest/document ---")
    # Minimal text PDF or text content
    sample_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n200\n%%EOF"
    files = {"file": ("report.pdf", sample_content, "application/pdf")}
    resp = client.post("/api/ingest/document", files=files, data={"company_name": "Tata Motors", "doc_type": "annual_report"})
    assert resp.status_code == 200, f"Ingest document failed: {resp.status_code} {resp.text}"
    print("[OK] /api/ingest/document PASSED")

def test_analyse_primary():
    print("\n--- 8. Testing POST /api/analyse/primary ---")
    resp = client.post("/api/analyse/primary", data={"company_name": "Tata Motors"})
    assert resp.status_code == 200, f"Analyse primary failed: {resp.status_code} {resp.text}"
    print("[OK] /api/analyse/primary PASSED")

def test_analyse_secondary():
    print("\n--- 9. Testing POST /api/analyse/secondary (SerpAPI) ---")
    resp = client.post("/api/analyse/secondary", data={"company_name": "Tata Motors", "promoter_names": "Ratan Tata", "industry": "Automotive"})
    assert resp.status_code == 200, f"Analyse secondary failed: {resp.status_code} {resp.text}"
    data = resp.json()["data"]
    print("Secondary news count:", len(data.get("company_news", [])))
    print("Secondary risk:", data.get("overall_risk"))
    print("[OK] /api/analyse/secondary PASSED")

def test_analyse_fraud():
    print("\n--- 10. Testing POST /api/analyse/fraud ---")
    resp = client.post("/api/analyse/fraud", data={"company_name": "Tata Motors"})
    assert resp.status_code == 200, f"Analyse fraud failed: {resp.status_code} {resp.text}"
    data = resp.json()["data"]
    print("Fraud risk score:", data.get("overall_risk_score"), "level:", data.get("risk_level"))
    print("[OK] /api/analyse/fraud PASSED")

def test_score():
    print("\n--- 11. Testing POST /api/score ---")
    features = {
        "dscr": 1.5,
        "debt_equity_ratio": 1.2,
        "current_ratio": 1.4,
        "ebitda_margin_pct": 16.0,
        "interest_coverage_ratio": 4.5,
        "promoter_contribution_pct": 40.0,
        "gst_compliance_score": 8.0,
        "cibil_score": 750,
        "litigation_count": 0,
        "years_in_business": 10,
    }
    resp = client.post("/api/score", json=features)
    assert resp.status_code == 200, f"Score failed: {resp.status_code} {resp.text}"
    data = resp.json()["data"]
    print("Default probability:", data["default_probability"], "Credit score:", data["credit_score"])
    print("[OK] /api/score PASSED")

def test_decision_and_memo():
    print("\n--- 12. Testing POST /api/decision (Full Pipeline with Groq & SerpAPI) ---")
    decision_request = {
        "company_info": {
            "company_name": "Acme Engineering Ltd",
            "promoter_names": ["Rajesh Sharma"],
            "industry": "Manufacturing",
            "years_in_business": 12,
            "cibil_score": 740,
            "macro_risk_score": 5,
            "security_coverage": 1.5,
        },
        "financial_data": {
            "revenue_cr": 120.0,
            "ebitda_cr": 18.0,
            "pat_cr": 8.5,
            "total_debt_cr": 35.0,
            "net_worth_cr": 50.0,
            "interest_expense_cr": 3.5,
            "depreciation_cr": 2.5,
            "current_assets_cr": 40.0,
            "current_liabilities_cr": 28.0,
            "inventory_cr": 15.0,
            "fixed_assets_cr": 45.0,
            "annual_debt_service_cr": 7.0,
            "promoter_contribution_pct": 45.0,
            "revenue_history": [95.0, 108.0, 120.0],
        },
        "gst_data": {
            "company_name": "Acme Engineering Ltd",
            "period": "FY2023-24",
            "gstr_2a_itc": 8.2,
            "gstr_3b_itc": 8.0,
            "gst_turnover": 118.0,
        },
        "bank_data": {
            "total_credits": 122.0,
            "total_debits": 112.0,
            "avg_monthly_balance": 8.5,
            "cheque_bounces": 0,
            "emi_bounce_count": 0,
            "credit_debit_ratio": 1.09,
            "avg_utilisation_pct": 60.0,
        }
    }
    resp = client.post("/api/decision", json=decision_request)
    assert resp.status_code == 200, f"Decision failed: {resp.status_code} {resp.text}"
    data = resp.json()["data"]
    print("Decision:", data["decision"].get("decision"))
    print("Credit Grade:", data["decision"].get("credit_grade"))
    print("Loan Amount (Cr):", data["decision"].get("final_loan_amount_cr"))
    print("CAM Generated:", bool(data.get("cam")))
    print("SWOT Generated:", bool(data.get("swot_analysis")))
    print("GenAI Narrative Generated:", bool(data.get("genai_narrative")))
    print("[OK] /api/decision PASSED")

    print("\n--- 13. Testing POST /api/memo ---")
    resp_memo = client.post("/api/memo", json=decision_request)
    assert resp_memo.status_code == 200, f"Memo failed: {resp_memo.status_code} {resp_memo.text}"
    print("[OK] /api/memo PASSED")

    print("\n--- 14. Testing POST /api/report/download (PDF generation) ---")
    resp_report = client.post("/api/report/download", json=decision_request)
    assert resp_report.status_code == 200, f"Report download failed: {resp_report.status_code} {resp_report.text}"
    assert len(resp_report.content) > 1000
    print("Report PDF byte size:", len(resp_report.content))
    print("[OK] /api/report/download PASSED")

if __name__ == "__main__":
    test_health()
    test_schema_defaults()
    test_schema_custom()
    test_classify()
    test_classify_confirm()
    test_ingest_structured()
    test_ingest_document()
    test_analyse_primary()
    test_analyse_secondary()
    test_analyse_fraud()
    test_score()
    test_decision_and_memo()
    print("\n==========================================")
    print("🎉 ALL 14 BACKEND ENDPOINTS PASSED CLEANLY!")
    print("==========================================")
