"""
Abstraction Plane — Tạo abstract memory từ nhiều atomic memories.

Khi người dùng có nhiều memory nhỏ cùng xoay quanh một chủ đề, hệ cần một
đại diện mức cao để tóm "big picture" mà không phải nạp quá nhiều chi tiết.
"""
from __future__ import annotations
import time, logging
from collections import defaultdict
from typing import List, Optional
from tam.models import MemoryRecord, MemoryType, MemoryTier
from tam.config import TAMConfig

logger = logging.getLogger(__name__)

class AbstractionPlane:
    def __init__(self, config: Optional[TAMConfig] = None):
        self.config = config or TAMConfig()

    def find_abstraction_groups(self, memories: List[MemoryRecord]) -> dict:
        """Nhóm memories theo domain_tags chung."""
        groups = defaultdict(list)
        for m in memories:
            if m.memory_type == MemoryType.ABSTRACT:
                continue
            for tag in m.domain_tags:
                groups[tag].append(m)
        # Chỉ giữ nhóm có ≥ 2 thành viên
        return {k: v for k, v in groups.items() if len(v) >= 2}

    def synthesize(self, group_name: str, members: List[MemoryRecord]) -> MemoryRecord:
        """Tổng hợp nhiều atomic memories thành 1 abstract memory."""
        contents = [m.content for m in members if m.content]
        summary = f"Nhóm '{group_name}': tổng hợp từ {len(members)} memories. "
        summary += "Nội dung chính: " + "; ".join(c[:60] for c in contents[:5])
        if len(contents) > 5:
            summary += f" ... và {len(contents)-5} mục khác"

        # Lấy trung bình importance và confidence
        avg_importance = sum(m.importance for m in members) / len(members)
        avg_confidence = sum(m.confidence for m in members) / len(members)

        # Thu thập tất cả tags
        all_domain = list(set(t for m in members for t in m.domain_tags))
        all_intent = list(set(t for m in members for t in m.intent_tags))

        # Tính embedding trung bình
        embeddings = [m.embedding for m in members if m.embedding]
        avg_emb = None
        if embeddings:
            import numpy as np
            avg_emb = np.mean(embeddings, axis=0).tolist()

        return MemoryRecord(
            tier=MemoryTier.ACTIVE,
            memory_type=MemoryType.ABSTRACT,
            content=summary,
            summary=f"Abstract: {group_name}",
            abstraction_group=group_name,
            domain_tags=all_domain,
            intent_tags=all_intent,
            source="abstraction",
            confidence=avg_confidence,
            importance=avg_importance,
            embedding=avg_emb,
            created_at=time.time(),
            is_current=True,
        )

    def should_abstract(self, group_members: List[MemoryRecord], threshold: int = 3) -> bool:
        """Kiểm tra nhóm có đủ lớn để tạo abstract memory."""
        return len(group_members) >= threshold
