"""
State Plane — Conflict resolution + Temporal awareness.
"""
from __future__ import annotations
import time, logging
from typing import List, Tuple
from tam.models import MemoryRecord, ConflictStatus, RetrievalResult
from tam.config import TAMConfig

logger = logging.getLogger(__name__)

class StatePlane:
    def __init__(self, config=None):
        self.config = config or TAMConfig()

    def filter_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Pipeline: temporal check → conflict resolution."""
        valid = [r for r in results if r.memory.is_valid_at()]
        return self.resolve_conflicts(valid)

    def detect_conflicts(self, candidates: List[RetrievalResult]) -> List[Tuple[RetrievalResult, RetrievalResult]]:
        conflicts = []
        for i, a in enumerate(candidates):
            for b in candidates[i+1:]:
                if self._are_conflicting(a.memory, b.memory):
                    conflicts.append((a, b))
        return conflicts

    def resolve_conflicts(self, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        conflicts = self.detect_conflicts(candidates)
        demoted_ids = set()
        for a, b in conflicts:
            newer, older = (a, b) if a.memory.created_at > b.memory.created_at else (b, a)
            older.memory.conflict_status = ConflictStatus.SUPERSEDED
            older.memory.superseded_by = newer.memory.id
            older.memory.confidence *= 0.5
            older.activation_score *= 0.3
            demoted_ids.add(older.memory.id)
        return [c for c in candidates if c.memory.id not in demoted_ids]

    def _are_conflicting(self, a: MemoryRecord, b: MemoryRecord) -> bool:
        shared = set(a.domain_tags) & set(b.domain_tags)
        if not shared:
            return False
        if a.conflict_status in (ConflictStatus.OUTDATED, ConflictStatus.SUPERSEDED):
            return True
        if b.conflict_status in (ConflictStatus.OUTDATED, ConflictStatus.SUPERSEDED):
            return True
        if a.abstraction_group and a.abstraction_group == b.abstraction_group and a.content != b.content:
            return True
        return False
