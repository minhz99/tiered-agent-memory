"""
Query Understanding Plane — Intent classification, query rewriting, cue expansion.

Con người hiếm khi truy trí nhớ bằng nguyên văn câu vừa nghe. Não tự diễn giải
lại câu hỏi thành các khái niệm gần kề. Agent cũng nên làm vậy.
"""
from __future__ import annotations
import re, hashlib, logging
from typing import List, Optional
import numpy as np
from tam.models import QueryContext, IntentCategory
from tam.config import TAMConfig

logger = logging.getLogger(__name__)

# Synonyms / concept expansion map (có thể mở rộng hoặc thay bằng LLM)
CONCEPT_MAP = {
    "bug": ["error", "issue", "defect", "crash", "exception"],
    "lỗi": ["bug", "sự cố", "crash", "exception", "vấn đề"],
    "deploy": ["deployment", "release", "ship", "production"],
    "database": ["DB", "SQL", "storage", "query", "schema"],
    "gym": ["fitness", "workout", "exercise", "health"],
    "diet": ["nutrition", "calories", "food", "meal plan"],
    "algorithm": ["thuật toán", "complexity", "data structure"],
    "nhớ": ["remember", "recall", "trước đây", "lần trước"],
}

class QueryPlane:
    def __init__(self, config: Optional[TAMConfig] = None):
        self.config = config or TAMConfig()
        self._embedder = None

    def analyze(self, raw_query: str) -> QueryContext:
        """Pipeline: classify intent → rewrite query → expand concepts → compute embedding."""
        ctx = QueryContext(raw_query=raw_query)
        ctx.intent = self._classify_intent(raw_query)
        ctx.complexity_score = self._estimate_complexity(raw_query)
        ctx.domain = self._detect_domain(raw_query)
        ctx.rewritten_query = self._rewrite(raw_query, ctx.intent)
        ctx.expanded_concepts = self._expand_concepts(ctx.rewritten_query)
        ctx.cue_strength = self._compute_cue_strength(ctx)
        ctx.allowed_memory_types = self._gate_by_intent(ctx.intent)
        ctx.embedding = self._embed(ctx.rewritten_query + " " + " ".join(ctx.expanded_concepts))
        return ctx

    def _classify_intent(self, query: str) -> IntentCategory:
        q_lower = query.lower()
        best, best_score = IntentCategory.UNKNOWN, 0
        for cat_name, keywords in self.config.intent_categories.items():
            score = sum(1 for kw in keywords if kw.lower() in q_lower)
            if score > best_score:
                best_score = score
                best = IntentCategory(cat_name)
        return best

    def _estimate_complexity(self, query: str) -> float:
        """Ước lượng độ phức tạp: dựa trên length, question words, etc."""
        words = query.split()
        length_factor = min(1.0, len(words) / 50)
        question_words = sum(1 for w in words if w.lower() in
                            ("why", "how", "compare", "analyze", "explain",
                             "tại sao", "làm sao", "so sánh", "phân tích"))
        question_factor = min(1.0, question_words / 3)
        return 0.5 * length_factor + 0.5 * question_factor

    def _detect_domain(self, query: str) -> str:
        q_lower = query.lower()
        if any(kw in q_lower for kw in ("code", "bug", "api", "server", "database", "function")):
            return "technical"
        if any(kw in q_lower for kw in ("gym", "diet", "health", "hobby")):
            return "personal"
        return "general"

    def _rewrite(self, query: str, intent: IntentCategory) -> str:
        """Diễn giải lại query — thêm context hints theo intent."""
        rewritten = query
        if intent == IntentCategory.RECALL:
            rewritten = f"Retrieve past experience: {query}"
        elif intent == IntentCategory.DECISION:
            rewritten = f"Decision analysis: {query}"
        elif intent == IntentCategory.TECHNICAL:
            rewritten = f"Technical query: {query}"
        return rewritten

    def _expand_concepts(self, query: str) -> List[str]:
        """Mở rộng query bằng concept map."""
        expanded = []
        q_lower = query.lower()
        for key, synonyms in CONCEPT_MAP.items():
            if key.lower() in q_lower:
                expanded.extend(synonyms)
        return list(set(expanded))[:10]

    def _compute_cue_strength(self, ctx: QueryContext) -> float:
        """Tính cường độ cue — quyết định có mở Latent hay không."""
        strength = 0.0
        # Explicit cue: user yêu cầu nhớ lại
        recall_words = ["remember", "nhớ", "trước đây", "lần trước", "hồi đó", "recall"]
        if any(w in ctx.raw_query.lower() for w in recall_words):
            strength += 0.5
        # Complexity cue: câu hỏi phức tạp → nên tìm thêm
        strength += ctx.complexity_score * 0.3
        # Concept expansion cue: nhiều concepts → topic rộng
        if len(ctx.expanded_concepts) > 3:
            strength += 0.2
        return min(1.0, strength)

    def _gate_by_intent(self, intent: IntentCategory) -> list:
        """Intent-aware gating: lọc loại memory phù hợp."""
        from tam.models import MemoryType
        all_types = list(MemoryType)
        if intent == IntentCategory.TECHNICAL:
            return [MemoryType.REASONING, MemoryType.SYSTEM, MemoryType.SEMANTIC, MemoryType.ABSTRACT]
        if intent == IntentCategory.PERSONAL:
            return [MemoryType.SEMANTIC, MemoryType.STYLE, MemoryType.EPISODIC, MemoryType.ABSTRACT]
        if intent == IntentCategory.RECALL:
            return all_types  # Mở hết khi recall
        return all_types

    def _embed(self, text: str) -> List[float]:
        """Tạo embedding — dùng simple hash-based cho demo, có thể swap sentence-transformers."""
        if self.config.use_transformer_embeddings:
            return self._embed_transformer(text)
        return self._embed_simple(text)

    def _embed_simple(self, text: str) -> List[float]:
        """Hash-based embedding (deterministic, cho demo)."""
        dim = self.config.embedding_dim
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], 'big'))
        vec = rng.randn(dim).astype(np.float32)
        # Add word-level features
        words = text.lower().split()
        for i, w in enumerate(words[:dim]):
            wh = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
            vec[wh % dim] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    def _embed_transformer(self, text: str) -> List[float]:
        """Sentence-transformers embedding (optional)."""
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            return self._embedder.encode(text).tolist()
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to simple embedding")
            return self._embed_simple(text)
