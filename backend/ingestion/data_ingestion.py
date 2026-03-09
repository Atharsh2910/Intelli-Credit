"""
Intelli-Credit: Data Ingestion Layer
Handles multi-format document processing for Indian corporate credit analysis.
- PDF parsing (PyMuPDF + pdfplumber fallback)
- GSTR-2A vs GSTR-3B cross-validation
- Bank statement analysis (EMI bounces, round-tripping, avg balance)
- Text chunking for RAG pipeline
"""

import os
import re
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ExtractedDocument:
    """Represents a parsed document."""
    doc_id: str
    filename: str
    doc_type: str  # annual_report, gst_return, bank_statement, legal, mca, other
    company_name: str
    text: str
    pages: int
    tables: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GSTValidationResult:
    """Result of GSTR-2A vs GSTR-3B cross-validation."""
    company_name: str
    period: str
    gstr_2a_itc: float
    gstr_3b_itc: float
    discrepancy_pct: float
    discrepancy_flag: bool  # True if > 10%
    gst_turnover: float
    bank_credits: float
    gst_bank_gap_pct: float
    gst_bank_flag: bool
    risk_signals: List[str] = field(default_factory=list)


@dataclass
class BankAnalysisResult:
    """Result of bank statement analysis."""
    company_name: str
    period: str
    total_credits: float
    total_debits: float
    avg_monthly_balance: float
    min_monthly_balance: float
    emi_bounce_count: int
    emi_bounce_months: List[str] = field(default_factory=list)
    round_trip_transactions: List[Dict] = field(default_factory=list)
    credit_debit_ratio: float = 0.0
    avg_utilisation_pct: float = 0.0
    risk_signals: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PDF Parsing Engine
# ---------------------------------------------------------------------------

class PDFParser:
    """Multi-strategy PDF parser with PyMuPDF primary + pdfplumber fallback."""

    def parse(self, file_path: str, company_name: str, doc_type: str = "other") -> ExtractedDocument:
        """Parse a PDF and return structured extraction."""
        doc_id = hashlib.md5(f"{file_path}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        filename = os.path.basename(file_path)

        # Try PyMuPDF first (faster)
        text, pages, tables = self._parse_pymupdf(file_path)

        # Fallback to pdfplumber if text is too short (likely scanned)
        if len(text.strip()) < 100:
            text_pb, pages_pb, tables_pb = self._parse_pdfplumber(file_path)
            if len(text_pb) > len(text):
                text, pages, tables = text_pb, pages_pb, tables_pb

        # Detect document type if not specified
        if doc_type == "other":
            doc_type = self._detect_doc_type(text, filename)

        doc = ExtractedDocument(
            doc_id=doc_id,
            filename=filename,
            doc_type=doc_type,
            company_name=company_name,
            text=text,
            pages=pages,
            tables=tables,
            metadata={
                "parsed_at": datetime.now().isoformat(),
                "file_path": file_path,
                "char_count": len(text),
            }
        )

        # Chunk the text for RAG
        doc.chunks = self._chunk_text(doc)

        return doc

    def parse_bytes(self, file_bytes: bytes, filename: str, company_name: str, doc_type: str = "other") -> ExtractedDocument:
        """Parse PDF from bytes (for API upload)."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            result = self.parse(tmp_path, company_name, doc_type)
            result.filename = filename
            return result
        finally:
            os.unlink(tmp_path)

    def _parse_pymupdf(self, file_path: str) -> Tuple[str, int, List[Dict]]:
        """Parse PDF using PyMuPDF."""
        doc = fitz.open(file_path)
        text_parts = []
        tables = []

        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            text_parts.append(f"\n--- Page {page_num + 1} ---\n{page_text}")

            # Extract tables from page (heuristic: look for tabular patterns)
            tab_data = self._extract_tables_from_text(page_text, page_num + 1)
            tables.extend(tab_data)

        pages = len(doc)
        doc.close()
        return "\n".join(text_parts), pages, tables

    def _parse_pdfplumber(self, file_path: str) -> Tuple[str, int, List[Dict]]:
        """Parse PDF using pdfplumber (better for tables and scanned docs)."""
        text_parts = []
        tables = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text_parts.append(f"\n--- Page {page_num + 1} ---\n{page_text}")

                # Extract tables
                page_tables = page.extract_tables() or []
                for t_idx, table in enumerate(page_tables):
                    if table and len(table) > 1:
                        headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]
                        rows = []
                        for row in table[1:]:
                            row_dict = {}
                            for i, cell in enumerate(row):
                                key = headers[i] if i < len(headers) else f"col_{i}"
                                row_dict[key] = str(cell).strip() if cell else ""
                            rows.append(row_dict)
                        tables.append({
                            "page": page_num + 1,
                            "table_index": t_idx,
                            "headers": headers,
                            "rows": rows,
                        })

            pages = len(pdf.pages)

        return "\n".join(text_parts), pages, tables

    def _extract_tables_from_text(self, text: str, page_num: int) -> List[Dict]:
        """Heuristic table extraction from text patterns."""
        tables = []
        lines = text.split("\n")
        table_lines = []
        in_table = False

        for line in lines:
            # Detect table rows by multiple spaces or tabs
            parts = re.split(r'\s{2,}|\t', line.strip())
            if len(parts) >= 3 and any(re.search(r'\d', p) for p in parts):
                in_table = True
                table_lines.append(parts)
            else:
                if in_table and len(table_lines) >= 3:
                    tables.append({
                        "page": page_num,
                        "table_index": len(tables),
                        "headers": table_lines[0],
                        "rows": [dict(zip(table_lines[0], row)) for row in table_lines[1:]],
                    })
                table_lines = []
                in_table = False

        return tables

    def _detect_doc_type(self, text: str, filename: str) -> str:
        """Detect document type from content and filename."""
        text_lower = text.lower()
        filename_lower = filename.lower()

        type_patterns = {
            "annual_report": ["annual report", "directors' report", "auditor's report", "balance sheet"],
            "gst_return": ["gstr", "gst return", "gstin", "goods and services tax"],
            "bank_statement": ["bank statement", "account statement", "opening balance", "closing balance"],
            "itr": ["income tax return", "itr", "assessment year", "total income"],
            "legal": ["court", "tribunal", "nclt", "judgement", "petition"],
            "mca": ["mca", "ministry of corporate affairs", "cin", "form 32", "charge"],
            "rating_report": ["crisil", "icra", "care", "rating rationale", "credit rating"],
            "financial_statement": ["profit and loss", "cash flow statement", "notes to accounts"],
        }

        for doc_type, patterns in type_patterns.items():
            matches = sum(1 for p in patterns if p in text_lower or p in filename_lower)
            if matches >= 2:
                return doc_type

        return "other"

    def _chunk_text(self, doc: ExtractedDocument, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Chunk document text for RAG with overlap."""
        text = doc.text
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind(".", start, end)
                last_newline = text.rfind("\n", start, end)
                break_point = max(last_period, last_newline)
                if break_point > start + chunk_size // 2:
                    end = break_point + 1

            chunk_text = text[start:end].strip()
            if len(chunk_text) > 50:  # Skip tiny chunks
                chunk_id = f"{doc.doc_id}_chunk_{len(chunks)}"
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "doc_id": doc.doc_id,
                    "doc_type": doc.doc_type,
                    "company": doc.company_name,
                    "chunk_index": len(chunks),
                    "char_start": start,
                    "char_end": end,
                })

            start = end - overlap if end < len(text) else end

        return chunks


# ---------------------------------------------------------------------------
# GST Cross-Validation Engine
# ---------------------------------------------------------------------------

class GSTValidator:
    """Validates GSTR-2A vs GSTR-3B and GST vs Bank credits."""

    DISCREPANCY_THRESHOLD = 10.0  # Percentage

    def validate(self, gst_data: Dict[str, Any], bank_data: Optional[Dict[str, Any]] = None) -> GSTValidationResult:
        """
        Cross-validate GST returns.

        gst_data should contain:
          - company_name: str
          - period: str (e.g., "FY2023-24")
          - gstr_2a_itc: float (ITC from GSTR-2A auto-populated)
          - gstr_3b_itc: float (ITC self-declared in GSTR-3B)
          - gst_turnover: float (total GST turnover)

        bank_data should contain:
          - total_credits: float (from bank statement)
        """
        company_name = gst_data.get("company_name", "Unknown")
        period = gst_data.get("period", "Unknown")
        gstr_2a_itc = float(gst_data.get("gstr_2a_itc", 0))
        gstr_3b_itc = float(gst_data.get("gstr_3b_itc", 0))
        gst_turnover = float(gst_data.get("gst_turnover", 0))

        # ITC discrepancy check
        if gstr_2a_itc > 0:
            discrepancy_pct = abs(gstr_3b_itc - gstr_2a_itc) / gstr_2a_itc * 100
        else:
            discrepancy_pct = 0.0 if gstr_3b_itc == 0 else 100.0

        discrepancy_flag = discrepancy_pct > self.DISCREPANCY_THRESHOLD

        # GST vs Bank credits gap
        bank_credits = float(bank_data.get("total_credits", 0)) if bank_data else 0
        if gst_turnover > 0 and bank_credits > 0:
            gst_bank_gap_pct = abs(gst_turnover - bank_credits) / gst_turnover * 100
        else:
            gst_bank_gap_pct = 0.0

        gst_bank_flag = gst_bank_gap_pct > 20.0  # >20% gap is a red flag

        # Build risk signals
        risk_signals = []
        if discrepancy_flag:
            risk_signals.append(
                f"GSTR-2A vs GSTR-3B ITC discrepancy of {discrepancy_pct:.1f}% "
                f"({gstr_2a_itc:.2f}L vs {gstr_3b_itc:.2f}L) — potential revenue manipulation or circular trading"
            )

        if gst_bank_flag:
            risk_signals.append(
                f"GST turnover vs bank credits gap of {gst_bank_gap_pct:.1f}% "
                f"({gst_turnover:.2f}L vs {bank_credits:.2f}L) — possible cash transactions or inflated sales"
            )

        if gstr_3b_itc > gstr_2a_itc * 1.5:
            risk_signals.append("GSTR-3B ITC significantly exceeds GSTR-2A — fake invoice risk")

        return GSTValidationResult(
            company_name=company_name,
            period=period,
            gstr_2a_itc=gstr_2a_itc,
            gstr_3b_itc=gstr_3b_itc,
            discrepancy_pct=round(discrepancy_pct, 2),
            discrepancy_flag=discrepancy_flag,
            gst_turnover=gst_turnover,
            bank_credits=bank_credits,
            gst_bank_gap_pct=round(gst_bank_gap_pct, 2),
            gst_bank_flag=gst_bank_flag,
            risk_signals=risk_signals,
        )

    def compute_compliance_score(self, gst_history: List[Dict]) -> float:
        """
        Compute GST compliance score (0-10) based on filing history.
        Factors: regularity, discrepancy rate, nil filings, late filings.
        """
        if not gst_history:
            return 5.0  # Neutral if no data

        score = 10.0
        total_months = len(gst_history)

        # Late filing penalty
        late_count = sum(1 for m in gst_history if m.get("late_filing", False))
        late_ratio = late_count / total_months if total_months > 0 else 0
        score -= late_ratio * 3.0  # Up to -3 for all late

        # Nil filing penalty (suspicious if frequent)
        nil_count = sum(1 for m in gst_history if m.get("nil_filing", False))
        nil_ratio = nil_count / total_months if total_months > 0 else 0
        if nil_ratio > 0.3:
            score -= (nil_ratio - 0.3) * 5.0

        # Discrepancy penalty
        avg_disc = np.mean([m.get("discrepancy_pct", 0) for m in gst_history])
        if avg_disc > 10:
            score -= min((avg_disc - 10) * 0.2, 3.0)

        return max(0.0, min(10.0, round(score, 1)))


# ---------------------------------------------------------------------------
# Bank Statement Analyzer
# ---------------------------------------------------------------------------

class BankStatementAnalyzer:
    """Analyze bank statements for credit signals and red flags."""

    def analyze(self, transactions: List[Dict], company_name: str,
                period: str = "Unknown", cc_limit: float = 0) -> BankAnalysisResult:
        """
        Analyze bank statement transactions.

        Each transaction should have:
          - date: str
          - description: str
          - debit: float
          - credit: float
          - balance: float
        """
        if not transactions:
            return BankAnalysisResult(
                company_name=company_name, period=period,
                total_credits=0, total_debits=0, avg_monthly_balance=0,
                min_monthly_balance=0, emi_bounce_count=0,
            )

        df = pd.DataFrame(transactions)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["debit"] = pd.to_numeric(df.get("debit", 0), errors="coerce").fillna(0)
        df["credit"] = pd.to_numeric(df.get("credit", 0), errors="coerce").fillna(0)
        df["balance"] = pd.to_numeric(df.get("balance", 0), errors="coerce").fillna(0)

        total_credits = float(df["credit"].sum())
        total_debits = float(df["debit"].sum())

        # Monthly balance analysis
        df["month"] = df["date"].dt.to_period("M")
        monthly = df.groupby("month").agg(
            avg_bal=("balance", "mean"),
            min_bal=("balance", "min"),
        )
        avg_monthly_balance = float(monthly["avg_bal"].mean())
        min_monthly_balance = float(monthly["min_bal"].min())

        # EMI bounce detection
        emi_patterns = [
            r"emi.*bounce", r"ecs.*return", r"nach.*return",
            r"dishonour", r"insufficient.*fund", r"emi.*unpaid",
        ]
        bounce_mask = df["description"].str.lower().apply(
            lambda x: any(re.search(p, str(x)) for p in emi_patterns)
        )
        emi_bounce_count = int(bounce_mask.sum())
        emi_bounce_months = df.loc[bounce_mask, "month"].astype(str).unique().tolist()

        # Round-tripping detection (same amount credited and debited within 3 days)
        round_trips = self._detect_round_tripping(df)

        # Credit/debit ratio
        credit_debit_ratio = total_credits / total_debits if total_debits > 0 else 0

        # CC utilisation
        avg_utilisation = 0.0
        if cc_limit > 0:
            avg_utilisation = (1 - avg_monthly_balance / cc_limit) * 100
            avg_utilisation = max(0, min(100, avg_utilisation))

        # Risk signals
        risk_signals = []
        if emi_bounce_count >= 3:
            risk_signals.append(
                f"{emi_bounce_count} EMI/ECS bounces detected — serious repayment concern"
            )
        elif emi_bounce_count > 0:
            risk_signals.append(
                f"{emi_bounce_count} EMI/ECS bounce(s) detected — monitor repayment behavior"
            )

        if len(round_trips) > 2:
            risk_signals.append(
                f"{len(round_trips)} potential round-tripping transactions detected"
            )

        if credit_debit_ratio < 0.8:
            risk_signals.append(
                f"Credit-to-debit ratio is {credit_debit_ratio:.2f} — business may be cash-negative"
            )

        if avg_utilisation > 90:
            risk_signals.append(
                f"Average CC utilisation is {avg_utilisation:.0f}% — near-full utilisation indicates stress"
            )

        if min_monthly_balance < 0:
            risk_signals.append("Negative balance observed — account overdrawn")

        return BankAnalysisResult(
            company_name=company_name,
            period=period,
            total_credits=round(total_credits, 2),
            total_debits=round(total_debits, 2),
            avg_monthly_balance=round(avg_monthly_balance, 2),
            min_monthly_balance=round(min_monthly_balance, 2),
            emi_bounce_count=emi_bounce_count,
            emi_bounce_months=emi_bounce_months,
            round_trip_transactions=round_trips,
            credit_debit_ratio=round(credit_debit_ratio, 2),
            avg_utilisation_pct=round(avg_utilisation, 1),
            risk_signals=risk_signals,
        )

    def _detect_round_tripping(self, df: pd.DataFrame, window_days: int = 3) -> List[Dict]:
        """Detect potential round-tripping: same amount in and out within window."""
        round_trips = []
        credits = df[df["credit"] > 0].copy()
        debits = df[df["debit"] > 0].copy()

        for _, credit_row in credits.iterrows():
            amount = credit_row["credit"]
            if amount < 100000:  # Only flag significant amounts (>1L)
                continue

            date = credit_row["date"]
            matching_debits = debits[
                (abs(debits["debit"] - amount) / amount < 0.01) &  # Within 1%
                (abs((debits["date"] - date).dt.days) <= window_days)
            ]

            if len(matching_debits) > 0:
                round_trips.append({
                    "credit_date": str(credit_row["date"]),
                    "debit_date": str(matching_debits.iloc[0]["date"]),
                    "amount": float(amount),
                    "description_in": str(credit_row.get("description", "")),
                    "description_out": str(matching_debits.iloc[0].get("description", "")),
                })

        return round_trips[:10]  # Cap at 10


# ---------------------------------------------------------------------------
# Unified Ingestion Pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """End-to-end ingestion: parse → validate → chunk → store."""

    def __init__(self):
        self.pdf_parser = PDFParser()
        self.gst_validator = GSTValidator()
        self.bank_analyzer = BankStatementAnalyzer()

    def ingest_document(self, file_path: str = None, file_bytes: bytes = None,
                        filename: str = "document.pdf", company_name: str = "Unknown",
                        doc_type: str = "other") -> Dict[str, Any]:
        """Ingest a document and return structured data."""
        if file_path:
            doc = self.pdf_parser.parse(file_path, company_name, doc_type)
        elif file_bytes:
            doc = self.pdf_parser.parse_bytes(file_bytes, filename, company_name, doc_type)
        else:
            raise ValueError("Either file_path or file_bytes must be provided")

        return {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "doc_type": doc.doc_type,
            "company_name": doc.company_name,
            "pages": doc.pages,
            "char_count": len(doc.text),
            "chunk_count": len(doc.chunks),
            "tables_found": len(doc.tables),
            "chunks": doc.chunks,
            "tables": doc.tables,
            "text_preview": doc.text[:500],
            "metadata": doc.metadata,
        }

    def validate_gst(self, gst_data: Dict, bank_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Run GST cross-validation."""
        result = self.gst_validator.validate(gst_data, bank_data)
        return asdict(result)

    def analyze_bank_statement(self, transactions: List[Dict], company_name: str,
                               period: str = "Unknown", cc_limit: float = 0) -> Dict[str, Any]:
        """Analyze bank statement transactions."""
        result = self.bank_analyzer.analyze(transactions, company_name, period, cc_limit)
        return asdict(result)

    def compute_gst_compliance_score(self, gst_history: List[Dict]) -> float:
        """Get GST compliance score."""
        return self.gst_validator.compute_compliance_score(gst_history)
