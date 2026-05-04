"""
Decay Engine — Suy giảm theo loại memory.

Không phải loại memory nào cũng quên giống nhau:
- episodic: decay nhanh
- semantic: decay chậm
- reasoning: decay chậm nhưng nhạy với phản ví dụ
- style: decay trung bình
- system: gần như không decay
"""
from __future__ import annotations
import math, time, logging
from typing import List
from tam.models import MemoryRecord, MemoryType
from tam.config import DecayProfiles

logger = logging.getLogger(__name__)

class DecayEngine:
    def __init__(self, profiles: DecayProfiles = None):
        self.profiles = profiles or DecayProfiles()

    def get_rate(self, record: MemoryRecord) -> float:
        """Lấy decay rate cho một memory dựa trên type."""
        if record.decay_rate_override is not None:
            return record.decay_rate_override
        mapping = {
            MemoryType.EPISODIC: self.profiles.episodic,
            MemoryType.SEMANTIC: self.profiles.semantic,
            MemoryType.REASONING: self.profiles.reasoning,
            MemoryType.STYLE: self.profiles.style,
            MemoryType.SYSTEM: self.profiles.system,
            MemoryType.ABSTRACT: self.profiles.abstract,
        }
        return mapping.get(record.memory_type, self.profiles.semantic)

    def compute_decay_factor(self, record: MemoryRecord) -> float:
        """Tính hệ số suy giảm hiện tại (0→đã quên, 1→còn nguyên)."""
        rate = self.get_rate(record)
        ref_time = record.last_used_at or record.last_confirmed_at or record.created_at
        if ref_time is None:
            return 1.0
        days_elapsed = (time.time() - ref_time) / 86400
        return math.exp(-rate * days_elapsed)

    def apply_decay(self, record: MemoryRecord) -> MemoryRecord:
        """Áp dụng decay: giảm importance và confidence."""
        factor = self.compute_decay_factor(record)
        record.importance *= factor
        record.recall_confidence *= factor
        # Reasoning memory: thêm penalty nếu có nhiều failure cases
        if record.memory_type == MemoryType.REASONING and record.failure_case_ids:
            failure_penalty = 0.05 * len(record.failure_case_ids)
            record.importance = max(0.0, record.importance - failure_penalty)
        return record

    def apply_batch(self, records: List[MemoryRecord]) -> List[MemoryRecord]:
        """Áp dụng decay cho batch — dùng trong background maintenance."""
        decayed = 0
        for r in records:
            old_imp = r.importance
            self.apply_decay(r)
            if r.importance < old_imp * 0.99:
                decayed += 1
        logger.info(f"Decay applied to {decayed}/{len(records)} memories")
        return records

    def should_demote(self, record: MemoryRecord, threshold: float = 0.1) -> bool:
        """Kiểm tra memory có nên bị hạ tầng (demote) không."""
        factor = self.compute_decay_factor(record)
        return factor < threshold and record.usage_count < 3
