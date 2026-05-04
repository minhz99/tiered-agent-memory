"""
Reasoning Plane — ReasoningBank & strategy distillation.

Reasoning Memory lưu Generalizable Reasoning Strategies — không phải facts
mà là "chiến lược giải bài toán nào?". Strategy chỉ được promote vào
ReasoningBank khi đã qua validation: nhiều tình huống, scope rõ, có contrastive.
"""
from __future__ import annotations
import time, logging
from typing import List, Optional
import numpy as np
from tam.models import MemoryRecord, MemoryType, MemoryTier, ValidationStatus, RetrievalResult
from tam.config import TAMConfig

logger = logging.getLogger(__name__)

class ReasoningPlane:
    """ReasoningBank — sub-bank của Active/Abstract chứa strategies đã validation."""

    def __init__(self, config: Optional[TAMConfig] = None):
        self.config = config or TAMConfig()
        self._bank: List[MemoryRecord] = []

    def add_strategy(self, record: MemoryRecord):
        record.memory_type = MemoryType.REASONING
        self._bank.append(record)

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[RetrievalResult]:
        """Tìm chiến lược phù hợp với query hiện tại."""
        if not query_embedding or not self._bank:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0: return []
        q = q / qn

        results = []
        for rec in self._bank:
            if not rec.embedding or not rec.is_current:
                continue
            if rec.validation_status == ValidationStatus.INVALIDATED:
                continue
            e = np.array(rec.embedding, dtype=np.float32)
            en = np.linalg.norm(e)
            if en == 0: continue
            sim = float(np.dot(q, e / en))
            results.append(RetrievalResult(
                memory=rec, activation_score=sim, source_tier=MemoryTier.ACTIVE
            ))
        results.sort(key=lambda r: r.activation_score, reverse=True)
        return results[:top_k]

    def can_promote_to_bank(self, record: MemoryRecord) -> bool:
        """Kiểm tra strategy có đủ điều kiện vào ReasoningBank."""
        if not record.core_strategy:
            return False
        if len(record.success_case_ids) < self.config.evolution.min_traces_for_distill:
            return False
        if not record.applicability_scope:
            return False
        return True

    def distill_from_traces(self, success_traces: List[dict], failure_traces: List[dict]) -> Optional[MemoryRecord]:
        """Chắt lọc reasoning memory từ traces thành công & thất bại."""
        if len(success_traces) < 2:
            return None

        # Tạo reflection summary contrastive
        strategy = f"Strategy distilled from {len(success_traces)} successes"
        if failure_traces:
            strategy += f" and {len(failure_traces)} failures"

        record = MemoryRecord(
            memory_type=MemoryType.REASONING,
            tier=MemoryTier.ACTIVE,
            content=strategy,
            core_strategy=strategy,
            applicability_scope="general",
            source="distillation",
            confidence=0.6,
            importance=0.7,
            validation_status=ValidationStatus.NEEDS_REVIEW,
            success_case_ids=[str(t.get("timestamp", "")) for t in success_traces],
            failure_case_ids=[str(t.get("timestamp", "")) for t in failure_traces],
            created_at=time.time(),
        )
        return record

    def get_all(self) -> List[MemoryRecord]:
        return [r for r in self._bank if r.is_current]

    def count(self) -> int:
        return len([r for r in self._bank if r.is_current])

    def load_from_active(self, active_memories: List[MemoryRecord]):
        """Load reasoning memories từ Active tier vào bank."""
        for m in active_memories:
            if m.memory_type == MemoryType.REASONING and m.is_current:
                if m.id not in {r.id for r in self._bank}:
                    self._bank.append(m)
