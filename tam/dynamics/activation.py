"""
Activation Engine — Tính điểm kích hoạt cho memory candidates.

score = w1*similarity + w2*importance + w3*recency + w4*confidence + w5*strategy_match

Activation không chỉ là cộng điểm — nó cần thêm thành phần competition.
Một memory được chọn không chỉ vì nó mạnh, mà còn vì nó thắng được các
memory gần trùng hoặc cạnh tranh trực tiếp.
"""
from __future__ import annotations
import math, time
from typing import List, Optional
from tam.models import MemoryRecord, RetrievalResult, QueryContext, MemoryType
from tam.config import ActivationWeights

class ActivationEngine:
    def __init__(self, weights: Optional[ActivationWeights] = None):
        self.w = weights or ActivationWeights()

    def score(self, record: MemoryRecord, similarity: float, query_ctx: Optional[QueryContext] = None) -> float:
        """Tính activation score tổng hợp."""
        # Recency: exponential decay theo thời gian từ lần dùng cuối
        recency = self._recency_score(record)
        # Strategy match: bonus nếu memory type phù hợp với intent
        strategy = self._strategy_match(record, query_ctx)

        total = (
            self.w.similarity * similarity
            + self.w.importance * record.importance
            + self.w.recency * recency
            + self.w.confidence * record.confidence
            + self.w.strategy_match * strategy
        )
        return max(0.0, min(1.0, total))

    def score_results(self, results: List[RetrievalResult], query_ctx: Optional[QueryContext] = None) -> List[RetrievalResult]:
        """Tính lại activation score cho danh sách results."""
        for r in results:
            r.activation_score = self.score(r.memory, r.activation_score, query_ctx)
        results.sort(key=lambda r: r.activation_score, reverse=True)
        return results

    def _recency_score(self, record: MemoryRecord) -> float:
        """Điểm recency: cao nếu mới dùng gần đây."""
        if record.last_used_at is None:
            return 0.3  # Chưa từng dùng → điểm trung bình thấp
        hours_ago = (time.time() - record.last_used_at) / 3600
        return math.exp(-0.01 * hours_ago)  # Half-life ≈ 70 giờ

    def _strategy_match(self, record: MemoryRecord, query_ctx: Optional[QueryContext]) -> float:
        """Bonus nếu memory type phù hợp với intent."""
        if query_ctx is None:
            return 0.5
        intent = query_ctx.intent.value
        mt = record.memory_type
        # Technical intent → ưu tiên reasoning & system
        if intent == "technical" and mt in (MemoryType.REASONING, MemoryType.SYSTEM):
            return 1.0
        # Personal intent → ưu tiên semantic & style
        if intent == "personal" and mt in (MemoryType.SEMANTIC, MemoryType.STYLE):
            return 1.0
        # Recall intent → ưu tiên episodic
        if intent == "recall" and mt == MemoryType.EPISODIC:
            return 1.0
        # Decision intent → ưu tiên reasoning
        if intent == "decision" and mt == MemoryType.REASONING:
            return 1.0
        return 0.5
