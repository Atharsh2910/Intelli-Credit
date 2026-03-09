"""
Intelli-Credit: RAG Document Intelligence
- Text chunking with overlap
- Embedding generation (Sentence-Transformers / OpenAI)
- Pinecone vector store integration
- Multi-query RAG retrieval
"""

import os
import hashlib
from typing import Dict, List, Optional, Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Optional imports
try:
    from pinecone import Pinecone
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class DocumentIntelligence:
    """RAG pipeline for document analysis."""

    def __init__(self):
        self.openai_client = None
        self.pinecone_index = None
        self._init_clients()
        self._local_store: List[Dict] = []  # Fallback in-memory store

    def _init_clients(self):
        if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if HAS_PINECONE and os.getenv("PINECONE_API_KEY"):
            try:
                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                index_name = os.getenv("PINECONE_INDEX_NAME", "intelli-credit")
                existing = [i.name for i in pc.list_indexes()]
                if index_name in existing:
                    self.pinecone_index = pc.Index(index_name)
            except Exception:
                pass

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI or fallback."""
        if self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small", input=texts
                )
                return [e.embedding for e in response.data]
            except Exception:
                pass
        # Fallback: simple hash-based pseudo-embeddings for MVP
        return [self._pseudo_embedding(t) for t in texts]

    def _pseudo_embedding(self, text: str, dim: int = 1536) -> List[float]:
        """Generate deterministic pseudo-embedding for fallback."""
        h = hashlib.sha256(text.encode()).hexdigest()
        np.random.seed(int(h[:8], 16) % (2**31))
        return np.random.randn(dim).tolist()

    def upsert_chunks(self, chunks: List[Dict], company_name: str) -> Dict[str, Any]:
        """Store document chunks in vector DB."""
        texts = [c["text"] for c in chunks]
        embeddings = self.generate_embeddings(texts)

        stored = 0
        if self.pinecone_index:
            vectors = []
            for chunk, emb in zip(chunks, embeddings):
                vectors.append({
                    "id": chunk["id"],
                    "values": emb,
                    "metadata": {
                        "text": chunk["text"][:1000],
                        "doc_type": chunk.get("doc_type", ""),
                        "company": company_name,
                        "chunk_index": chunk.get("chunk_index", 0),
                    }
                })
            # Batch upsert
            for i in range(0, len(vectors), 100):
                batch = vectors[i:i+100]
                self.pinecone_index.upsert(vectors=batch, namespace=company_name)
            stored = len(vectors)
        else:
            # Local fallback
            for chunk, emb in zip(chunks, embeddings):
                self._local_store.append({
                    "id": chunk["id"],
                    "embedding": emb,
                    "text": chunk["text"],
                    "doc_type": chunk.get("doc_type", ""),
                    "company": company_name,
                })
            stored = len(chunks)

        return {"stored": stored, "backend": "pinecone" if self.pinecone_index else "local"}

    def search(self, query: str, company_name: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant document chunks."""
        query_emb = self.generate_embeddings([query])[0]

        if self.pinecone_index:
            try:
                results = self.pinecone_index.query(
                    vector=query_emb, top_k=top_k, namespace=company_name,
                    include_metadata=True,
                )
                return [
                    {"text": m.metadata.get("text", ""), "score": float(m.score),
                     "doc_type": m.metadata.get("doc_type", ""), "id": m.id}
                    for m in results.matches
                ]
            except Exception:
                pass

        # Local search fallback using cosine similarity
        if not self._local_store:
            return []

        company_docs = [d for d in self._local_store if d["company"] == company_name]
        if not company_docs:
            return []

        scores = []
        qa = np.array(query_emb)
        for doc in company_docs:
            da = np.array(doc["embedding"])
            cos_sim = float(np.dot(qa, da) / (np.linalg.norm(qa) * np.linalg.norm(da) + 1e-8))
            scores.append((doc, cos_sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"text": d["text"], "score": s, "doc_type": d["doc_type"], "id": d["id"]}
            for d, s in scores[:top_k]
        ]

    def multi_query_search(self, queries: List[str], company_name: str, top_k: int = 5) -> List[Dict]:
        """Search with multiple queries and deduplicate results."""
        all_results = {}
        for q in queries:
            for r in self.search(q, company_name, top_k):
                rid = r["id"]
                if rid not in all_results or r["score"] > all_results[rid]["score"]:
                    all_results[rid] = r

        results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return results[:top_k * 2]

    def get_context(self, query: str, company_name: str, max_chars: int = 4000) -> str:
        """Get assembled context string for LLM consumption."""
        results = self.search(query, company_name, top_k=8)
        context_parts = []
        total_chars = 0
        for r in results:
            text = r["text"]
            if total_chars + len(text) > max_chars:
                break
            context_parts.append(f"[Source: {r['doc_type']}]\n{text}")
            total_chars += len(text)
        return "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."
