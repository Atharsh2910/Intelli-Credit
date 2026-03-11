"""
Intelli-Credit: Dynamic Schema Configuration
Allows users to define/customize extraction schemas for different document types.
"""

from typing import Dict, List, Any
from copy import deepcopy


# Default extraction schemas per document type
DEFAULT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "annual_report": {
        "label": "Annual Report",
        "fields": [
            {"name": "company_name", "label": "Company Name", "type": "text", "required": True},
            {"name": "financial_year", "label": "Financial Year", "type": "text", "required": True},
            {"name": "revenue", "label": "Revenue (₹ Cr)", "type": "number", "required": True},
            {"name": "ebitda", "label": "EBITDA (₹ Cr)", "type": "number", "required": True},
            {"name": "pat", "label": "PAT (₹ Cr)", "type": "number", "required": True},
            {"name": "total_debt", "label": "Total Debt (₹ Cr)", "type": "number", "required": True},
            {"name": "net_worth", "label": "Net Worth (₹ Cr)", "type": "number", "required": True},
            {"name": "total_assets", "label": "Total Assets (₹ Cr)", "type": "number", "required": False},
            {"name": "dividend_paid", "label": "Dividend Paid (₹ Cr)", "type": "number", "required": False},
            {"name": "employee_count", "label": "Employee Count", "type": "number", "required": False},
            {"name": "directors", "label": "Directors / Promoters", "type": "text", "required": False},
            {"name": "auditor_opinion", "label": "Auditor Opinion", "type": "text", "required": False},
        ],
    },
    "financial_statement": {
        "label": "Financial Statement",
        "fields": [
            {"name": "company_name", "label": "Company Name", "type": "text", "required": True},
            {"name": "period", "label": "Period", "type": "text", "required": True},
            {"name": "revenue", "label": "Revenue (₹ Cr)", "type": "number", "required": True},
            {"name": "cost_of_goods", "label": "Cost of Goods Sold (₹ Cr)", "type": "number", "required": False},
            {"name": "ebitda", "label": "EBITDA (₹ Cr)", "type": "number", "required": True},
            {"name": "depreciation", "label": "Depreciation (₹ Cr)", "type": "number", "required": True},
            {"name": "interest_expense", "label": "Interest Expense (₹ Cr)", "type": "number", "required": True},
            {"name": "pat", "label": "PAT (₹ Cr)", "type": "number", "required": True},
            {"name": "current_assets", "label": "Current Assets (₹ Cr)", "type": "number", "required": True},
            {"name": "current_liabilities", "label": "Current Liabilities (₹ Cr)", "type": "number", "required": True},
            {"name": "fixed_assets", "label": "Fixed Assets (₹ Cr)", "type": "number", "required": True},
            {"name": "inventory", "label": "Inventory (₹ Cr)", "type": "number", "required": False},
            {"name": "trade_receivables", "label": "Trade Receivables (₹ Cr)", "type": "number", "required": False},
            {"name": "trade_payables", "label": "Trade Payables (₹ Cr)", "type": "number", "required": False},
        ],
    },
    "gst_return": {
        "label": "GST Return",
        "fields": [
            {"name": "gstin", "label": "GSTIN", "type": "text", "required": True},
            {"name": "period", "label": "Return Period", "type": "text", "required": True},
            {"name": "gstr_type", "label": "GSTR Type (1/2A/3B)", "type": "text", "required": True},
            {"name": "taxable_value", "label": "Total Taxable Value (₹)", "type": "number", "required": True},
            {"name": "igst", "label": "IGST (₹)", "type": "number", "required": False},
            {"name": "cgst", "label": "CGST (₹)", "type": "number", "required": False},
            {"name": "sgst", "label": "SGST (₹)", "type": "number", "required": False},
            {"name": "itc_claimed", "label": "ITC Claimed (₹)", "type": "number", "required": True},
            {"name": "tax_payable", "label": "Tax Payable (₹)", "type": "number", "required": True},
        ],
    },
    "bank_statement": {
        "label": "Bank Statement",
        "fields": [
            {"name": "account_number", "label": "Account Number", "type": "text", "required": True},
            {"name": "bank_name", "label": "Bank Name", "type": "text", "required": True},
            {"name": "period", "label": "Statement Period", "type": "text", "required": True},
            {"name": "opening_balance", "label": "Opening Balance (₹)", "type": "number", "required": True},
            {"name": "closing_balance", "label": "Closing Balance (₹)", "type": "number", "required": True},
            {"name": "total_credits", "label": "Total Credits (₹)", "type": "number", "required": True},
            {"name": "total_debits", "label": "Total Debits (₹)", "type": "number", "required": True},
            {"name": "emi_bounces", "label": "EMI/ECS Bounces", "type": "number", "required": False},
            {"name": "avg_balance", "label": "Average Monthly Balance (₹)", "type": "number", "required": False},
        ],
    },
    "itr": {
        "label": "Income Tax Return",
        "fields": [
            {"name": "pan", "label": "PAN Number", "type": "text", "required": True},
            {"name": "assessment_year", "label": "Assessment Year", "type": "text", "required": True},
            {"name": "total_income", "label": "Total Income (₹)", "type": "number", "required": True},
            {"name": "tax_payable", "label": "Tax Payable (₹)", "type": "number", "required": True},
            {"name": "tax_paid", "label": "Tax Paid (₹)", "type": "number", "required": False},
            {"name": "business_income", "label": "Business Income (₹)", "type": "number", "required": False},
        ],
    },
    "legal": {
        "label": "Legal / Court Document",
        "fields": [
            {"name": "case_number", "label": "Case Number", "type": "text", "required": True},
            {"name": "court_name", "label": "Court / Tribunal", "type": "text", "required": True},
            {"name": "parties", "label": "Parties Involved", "type": "text", "required": True},
            {"name": "case_type", "label": "Case Type", "type": "text", "required": False},
            {"name": "status", "label": "Status", "type": "text", "required": False},
            {"name": "amount_disputed", "label": "Amount Disputed (₹)", "type": "number", "required": False},
        ],
    },
    "mca": {
        "label": "MCA Filing",
        "fields": [
            {"name": "cin", "label": "CIN", "type": "text", "required": True},
            {"name": "company_name", "label": "Company Name", "type": "text", "required": True},
            {"name": "form_type", "label": "Form Type", "type": "text", "required": False},
            {"name": "date_of_filing", "label": "Date of Filing", "type": "text", "required": False},
            {"name": "charges", "label": "Charges / Mortgages", "type": "text", "required": False},
        ],
    },
    "rating_report": {
        "label": "Credit Rating Report",
        "fields": [
            {"name": "agency", "label": "Rating Agency", "type": "text", "required": True},
            {"name": "rating", "label": "Rating Assigned", "type": "text", "required": True},
            {"name": "outlook", "label": "Outlook", "type": "text", "required": True},
            {"name": "facility_type", "label": "Facility Type", "type": "text", "required": False},
            {"name": "facility_amount", "label": "Facility Amount (₹ Cr)", "type": "number", "required": False},
            {"name": "rating_date", "label": "Rating Date", "type": "text", "required": False},
        ],
    },
    "other": {
        "label": "Other Document",
        "fields": [
            {"name": "title", "label": "Document Title", "type": "text", "required": False},
            {"name": "description", "label": "Description", "type": "text", "required": False},
            {"name": "key_data", "label": "Key Data Points", "type": "text", "required": False},
        ],
    },
}


class DynamicSchemaManager:
    """Manages extraction schemas with user customization support."""

    def __init__(self):
        # Start with deep copy of defaults
        self._custom_schemas: Dict[str, Dict[str, Any]] = {}

    def get_default_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Return the default schemas for all document types."""
        return deepcopy(DEFAULT_SCHEMAS)

    def get_schema(self, doc_type: str) -> Dict[str, Any]:
        """Return the active schema for a document type (custom if set, else default)."""
        if doc_type in self._custom_schemas:
            return deepcopy(self._custom_schemas[doc_type])
        return deepcopy(DEFAULT_SCHEMAS.get(doc_type, DEFAULT_SCHEMAS["other"]))

    def set_custom_schema(self, doc_type: str, fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Set a custom schema for a document type."""
        label = DEFAULT_SCHEMAS.get(doc_type, {}).get("label", doc_type.replace("_", " ").title())
        self._custom_schemas[doc_type] = {
            "label": label,
            "fields": fields,
            "is_custom": True,
        }
        return self._custom_schemas[doc_type]

    def reset_schema(self, doc_type: str) -> Dict[str, Any]:
        """Reset a document type back to its default schema."""
        if doc_type in self._custom_schemas:
            del self._custom_schemas[doc_type]
        return self.get_schema(doc_type)

    def get_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Return all schemas (custom overrides merged with defaults)."""
        result = deepcopy(DEFAULT_SCHEMAS)
        for doc_type, schema in self._custom_schemas.items():
            result[doc_type] = deepcopy(schema)
        return result
