"""
Competition Plane — Competition, inhibition, dedup và selection.

Việc lấy top-K theo điểm số là chưa đủ. Các memory phải cạnh tranh nhau để
được kích hoạt, và memory mạnh hơn phải ức chế memory na ná hoặc dư thừa.
Sử dụng MMR (Maximal Marginal Relevance) để đảm bảo diversity.
"""
from __future__ import annotations
import logging
from typing import List, Optional
import numpy as np
from tam.models import RetrievalResult, MemoryRecord, QueryContext
from tam.config import CompetitionConfig

logger = logging.getLogger(__name__)

class CompetitionPlane:
    def __init__(self, config: Optional[CompetitionConfig] = None):
        self.config = config or CompetitionConfig()

    def compete(self, candidates: List[RetrievalResult], query_ctx: Optional[QueryContext] = None) -> List[RetrievalResult]:
        """
        Pipeline cạnh tranh:
        1. Dedup: loại memory trùng content
        2. Competition: MMR-style diversity selection
        3. Inhibition: memory mạnh ức chế memory yếu gần trùng
        """
        if not candidates:
            return []
        # Step 1: Dedup
        deduped = self._dedup(candidates)
        # Step 2: MMR selection
        winners = self._mmr_select(deduped, self.config.max_winners)
        # Step 3: Mark inhibited
        winner_ids = {r.memory.id for r in winners}
        for r in deduped:
            if r.memory.id not in winner_ids:
                r.inhibited = True
                # Find the winner that inhibited this candidate
                r.inhibited_by = self._find_inhibitor(r, winners)

        inhibited = [r for r in deduped if r.inhibited]
        logger.info(f"Competition: {len(candidates)} candidates → "
                    f"{len(winners)} winners, {len(inhibited)} inhibited")
        return winners

    def _dedup(self, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """Loại bỏ memory có content gần trùng."""
        seen_contents = {}
        result = []
        for c in candidates:
            key = c.memory.content.strip().lower()[:200]
            if key not in seen_contents:
                seen_contents[key] = c
                result.append(c)
            else:
                # Giữ cái có score cao hơn
                existing = seen_contents[key]
                if c.activation_score > existing.activation_score:
                    result.remove(existing)
                    seen_contents[key] = c
                    result.append(c)
        return result

    def _mmr_select(self, candidates: List[RetrievalResult], k: int) -> List[RetrievalResult]:
        """Maximal Marginal Relevance — balance relevance vs diversity."""
        if len(candidates) <= k:
            return list(candidates)

        lam = self.config.diversity_lambda
        selected = []
        remaining = list(candidates)

        # Chọn candidate có score cao nhất trước
        remaining.sort(key=lambda r: r.activation_score, reverse=True)
        selected.append(remaining.pop(0))

        while len(selected) < k and remaining:
            best_mmr = -float('inf')
            best_idx = 0

            for i, cand in enumerate(remaining):
                relevance = cand.activation_score
                # Max similarity với bất kỳ memory đã chọn
                max_sim = max(
                    self._content_similarity(cand.memory, s.memory)
                    for s in selected
                )
                mmr = lam * relevance - (1 - lam) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _content_similarity(self, a: MemoryRecord, b: MemoryRecord) -> float:
        """Ước lượng similarity giữa 2 memory (embedding hoặc text overlap)."""
        if a.embedding and b.embedding:
            va = np.array(a.embedding)
            vb = np.array(b.embedding)
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na > 0 and nb > 0:
                return float(np.dot(va / na, vb / nb))
        # Fallback: word overlap
        words_a = set(a.content.lower().split())
        words_b = set(b.content.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / max(len(words_a | words_b), 1)

    def _find_inhibitor(self, candidate: RetrievalResult, winners: List[RetrievalResult]) -> Optional[str]:
        """Tìm winner nào ức chế candidate này (closest match)."""
        best_sim, best_id = 0.0, None
        for w in winners:
            sim = self._content_similarity(candidate.memory, w.memory)
            if sim > best_sim:
                best_sim = sim
                best_id = w.memory.id
        return best_id if best_sim > 0.3 else None
