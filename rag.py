"""
rag.py
FAISS semantic retrieval for EV planning guidelines.
Falls back to keyword overlap scoring if FAISS/sentence-transformers unavailable.
"""

import os
import re

# ── Optional heavy imports — guarded for Streamlit Cloud cold-start ───────────
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    import numpy as np
    FAISS_AVAILABLE = False

KB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "knowledge_base",
    "ev_planning_guidelines.txt",
)

# Module-level singletons (lazy-loaded)
_chunks: list = []
_index = None
_model = None


def _load_chunks() -> list:
    """Split the guidelines file into paragraph-level chunks."""
    with open(KB_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [c.strip() for c in re.split(r"\n{2,}", text) if len(c.strip()) > 40]
    return chunks


def _build_index(chunks: list):
    """Embed chunks and build a FAISS flat inner-product index."""
    global _index, _model
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = _model.encode(chunks, normalize_embeddings=True)
    dim = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dim)
    _index.add(embeddings.astype(np.float32))


def _ensure_ready():
    """Lazily load chunks and (if FAISS available) build the index."""
    global _chunks
    if not _chunks:
        _chunks = _load_chunks()
    if FAISS_AVAILABLE and _index is None:
        try:
            _build_index(_chunks)
        except Exception:
            pass  # fall through to keyword fallback


def retrieve_context(query: str, k: int = 4) -> str:
    """
    Return top-k relevant passages from the knowledge base.
    Uses FAISS semantic search when available, else keyword fallback.
    """
    _ensure_ready()
    if FAISS_AVAILABLE and _index is not None:
        return _semantic_search(query, k)
    return _keyword_fallback(query, k)


def _semantic_search(query: str, k: int) -> str:
    q_vec = _model.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = _index.search(q_vec, k)
    results = [_chunks[i] for i in indices[0] if 0 <= i < len(_chunks)]
    return "\n\n---\n\n".join(results)


def _keyword_fallback(query: str, k: int) -> str:
    """Score chunks by keyword overlap with the query tokens."""
    query_words = set(re.findall(r"\w+", query.lower()))
    scores = []
    for i, chunk in enumerate(_chunks):
        chunk_words = set(re.findall(r"\w+", chunk.lower()))
        overlap = len(query_words & chunk_words)
        scores.append((overlap, i))
    scores.sort(reverse=True)
    top = [_chunks[i] for _, i in scores[:k]]
    return "\n\n---\n\n".join(top)