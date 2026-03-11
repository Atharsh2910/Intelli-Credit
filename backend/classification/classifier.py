"""
Intelli-Credit: Document Classifier
Automatically classifies uploaded documents by type with confidence scoring.
Supports human-in-the-loop override.
"""

import re
from typing import Dict, List, Any
from dataclasses import dataclass, field, asdict


# Document types supported by the system
DOC_TYPES = [
    "annual_report",
    "gst_return",
    "bank_statement",
    "itr",
    "legal",
    "mca",
    "rating_report",
    "financial_statement",
    "other",
]

DOC_TYPE_LABELS = {
    "annual_report": "Annual Report",
    "gst_return": "GST Return",
    "bank_statement": "Bank Statement",
    "itr": "Income Tax Return",
    "legal": "Legal / Court Document",
    "mca": "MCA Filing",
    "rating_report": "Credit Rating Report",
    "financial_statement": "Financial Statement",
    "other": "Other Document",
}

# Keyword patterns with weights for classification
CLASSIFICATION_PATTERNS = {
    "annual_report": {
        "patterns": [
            "annual report", "directors' report", "director's report",
            "auditor's report", "auditor report", "balance sheet",
            "chairman's message", "corporate governance",
            "management discussion", "standalone financial",
        ],
        "filename_patterns": ["annual", "ar_", "annual_report"],
    },
    "gst_return": {
        "patterns": [
            "gstr", "gst return", "gstin", "goods and services tax",
            "gstr-1", "gstr-2a", "gstr-3b", "gst summary",
            "input tax credit", "output tax", "igst", "cgst", "sgst",
        ],
        "filename_patterns": ["gst", "gstr"],
    },
    "bank_statement": {
        "patterns": [
            "bank statement", "account statement", "opening balance",
            "closing balance", "transaction history", "passbook",
            "account number", "ifsc", "branch", "debit", "credit",
            "ecs return", "nach", "rtgs", "neft", "imps",
        ],
        "filename_patterns": ["bank", "statement", "passbook"],
    },
    "itr": {
        "patterns": [
            "income tax return", "itr", "assessment year",
            "total income", "tax payable", "form 26as",
            "pan number", "computation of income",
        ],
        "filename_patterns": ["itr", "income_tax", "tax_return"],
    },
    "legal": {
        "patterns": [
            "court", "tribunal", "nclt", "judgement", "judgment",
            "petition", "respondent", "petitioner", "advocate",
            "hon'ble", "drt", "arbitration", "writ",
        ],
        "filename_patterns": ["legal", "court", "nclt", "judgement"],
    },
    "mca": {
        "patterns": [
            "mca", "ministry of corporate affairs", "cin",
            "form 32", "form 20b", "charge", "roc",
            "registrar of companies", "incorporation",
        ],
        "filename_patterns": ["mca", "roc", "cin"],
    },
    "rating_report": {
        "patterns": [
            "crisil", "icra", "care", "rating rationale",
            "credit rating", "rating action", "brickwork",
            "acuite", "india ratings", "assigned rating",
        ],
        "filename_patterns": ["rating", "crisil", "icra", "care"],
    },
    "financial_statement": {
        "patterns": [
            "profit and loss", "cash flow statement",
            "notes to accounts", "schedule", "significant accounting",
            "balance sheet", "shareholders' funds", "reserves and surplus",
            "trade receivables", "trade payables",
        ],
        "filename_patterns": ["financial", "pl_statement", "bs_"],
    },
}


@dataclass
class ClassificationResult:
    """Result of auto-classification for a single document."""
    filename: str
    suggested_type: str
    suggested_label: str
    confidence: float  # 0.0 to 1.0
    all_scores: Dict[str, float] = field(default_factory=dict)
    confirmed: bool = False
    confirmed_type: str = ""


class DocumentClassifier:
    """
    Classifies uploaded documents by type using keyword pattern matching.
    Returns confidence scores for human review.
    """

    def classify_text(self, text: str, filename: str) -> ClassificationResult:
        """Classify a document based on its text content and filename."""
        text_lower = text.lower()
        filename_lower = filename.lower()

        scores: Dict[str, float] = {}

        for doc_type, config in CLASSIFICATION_PATTERNS.items():
            score = 0.0

            # Text pattern matching
            text_matches = 0
            for pattern in config["patterns"]:
                count = text_lower.count(pattern)
                if count > 0:
                    text_matches += min(count, 5)  # Cap per-pattern contribution

            # Normalize text score (max ~1.0 based on number of patterns)
            max_possible = len(config["patterns"]) * 3
            text_score = min(1.0, text_matches / max(max_possible, 1))
            score += text_score * 0.75  # Text content is 75% weight

            # Filename pattern matching
            filename_matches = sum(
                1 for p in config["filename_patterns"]
                if p in filename_lower
            )
            if filename_matches > 0:
                score += 0.25  # Filename is 25% weight

            scores[doc_type] = round(score, 3)

        # Sort by score and pick the best
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_type = sorted_types[0][0] if sorted_types[0][1] > 0.05 else "other"
        best_confidence = sorted_types[0][1] if best_type != "other" else 0.0

        # Normalize confidence to 0-1 range more meaningfully
        confidence = min(1.0, best_confidence * 2.5)  # Scale up since max raw is ~1.0

        return ClassificationResult(
            filename=filename,
            suggested_type=best_type,
            suggested_label=DOC_TYPE_LABELS.get(best_type, "Other Document"),
            confidence=round(confidence, 2),
            all_scores=scores,
        )

    def classify_from_bytes(self, file_bytes: bytes, filename: str) -> ClassificationResult:
        """Classify a document from file bytes (extracts text first)."""
        text = ""
        try:
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                text += page.get_text("text") + "\n"
            doc.close()
        except Exception:
            pass

        if len(text.strip()) < 50:
            # Fallback: classify by filename only
            return self.classify_text("", filename)

        return self.classify_text(text, filename)

    def get_doc_types(self) -> List[Dict[str, str]]:
        """Return all supported document types for the UI dropdown."""
        return [
            {"value": k, "label": v}
            for k, v in DOC_TYPE_LABELS.items()
        ]
