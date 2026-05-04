"""
Reinforcement Engine — Củng cố memory + Contrastive Signals.

Reinforcement không chỉ thưởng cho success. Hệ phải học từ cả thành công lẫn
thất bại. Contrastive signals giúp phân biệt "chiến lược nào nên thử lại"
và "chiến lược nào nên tránh".
"""
from __future__ import annotations
import time, logging
from typing import List, Optional
from tam.models import MemoryRecord, PipelineTrace

logger = logging.getLogger(__name__)

class ReinforcementEngine:
    def __init__(self, positive_delta: float = 0.1, negative_delta: float = 0.05):
        self.positive_delta = positive_delta
        self.negative_delta = negative_delta

    def reinforce_positive(self, record: MemoryRecord, trace_id: Optional[str] = None):
        """Củng cố memory đã chứng minh hữu ích."""
        record.reinforcement_score += self.positive_delta
        record.importance = min(1.0, record.importance + self.positive_delta * 0.5)
        record.confidence = min(1.0, record.confidence + 0.02)
        record.last_confirmed_at = time.time()
        if trace_id:
            record.success_case_ids.append(trace_id)
        logger.debug(f"Positive reinforcement: {record.id[:8]} → score={record.reinforcement_score:.2f}")

    def reinforce_negative(self, record: MemoryRecord, reason: str = "", trace_id: Optional[str] = None):
        """Ghi nhận thất bại — contrastive signal."""
        record.reinforcement_score -= self.negative_delta
        record.confidence = max(0.0, record.confidence - 0.05)
        if trace_id:
            record.failure_case_ids.append(trace_id)
        if reason:
            record.failure_reasons.append(reason)
        logger.debug(f"Negative reinforcement: {record.id[:8]} → score={record.reinforcement_score:.2f}")

    def process_trace(self, trace: PipelineTrace, all_candidates: List[MemoryRecord]):
        """Xử lý reinforcement từ kết quả pipeline trace."""
        if trace.success is None:
            return
        winner_ids = set(trace.final_wm_ids)
        trace_id = str(trace.timestamp)
        for record in all_candidates:
            if record.id in winner_ids:
                if trace.success:
                    self.reinforce_positive(record, trace_id)
                else:
                    self.reinforce_negative(record, "Used but task failed", trace_id)
            else:
                # Memory không được chọn — nếu task thành công, không phạt
                if not trace.success:
                    pass  # Không phạt memory không được chọn

    def generate_reflection(self, trace: PipelineTrace) -> str:
        """Tạo reflection summary từ trace — cầu nối với Reasoning Memory."""
        parts = []
        if trace.success:
            parts.append("✅ Task thành công.")
            if trace.final_wm_ids:
                parts.append(f"   Memories sử dụng: {len(trace.final_wm_ids)}")
        else:
            parts.append("❌ Task thất bại.")
        if trace.matts_branches:
            parts.append(f"   MaTTS branches: {len(trace.matts_branches)}")
        parts.append(f"   Response confidence: {trace.response_confidence:.2f}")
        return "\n".join(parts)
