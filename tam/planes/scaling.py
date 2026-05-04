"""
Scaling Plane — MaTTS: Memory-aware Test-Time Scaling.

Thay vì chỉ quyết định mở bao nhiêu tầng trí nhớ, MaTTS quyết định mức
scale của quá trình suy luận. Khi task khó, hệ bật suy luận đa nhánh.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from tam.models import QueryContext, RetrievalResult
from tam.config import MaTTSConfig

logger = logging.getLogger(__name__)

class ScalingPlane:
    def __init__(self, config: Optional[MaTTSConfig] = None):
        self.config = config or MaTTSConfig()

    def determine_budget(self, query_ctx: QueryContext) -> str:
        """Xác định MaTTS budget: fast | medium | deep."""
        score = query_ctx.complexity_score
        if score < self.config.fast_threshold:
            return "fast"
        elif score < self.config.deep_threshold:
            return "medium"
        return "deep"

    def should_branch(self, budget: str) -> bool:
        return budget == "deep"

    def create_branches(self, query_ctx: QueryContext, strategy_hints: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Tạo các nhánh suy luận — mỗi nhánh có strategy hint riêng."""
        budget = query_ctx.matts_budget
        if budget == "fast":
            return [{"branch_id": 0, "strategy": None, "mode": "direct"}]
        n_branches = min(self.config.max_branches, max(1, len(strategy_hints) + 1))
        branches = []
        # Branch 0: no strategy (baseline)
        branches.append({"branch_id": 0, "strategy": None, "mode": "baseline"})
        # Branch 1+: guided by strategy hints
        for i, hint in enumerate(strategy_hints[:n_branches - 1]):
            branches.append({
                "branch_id": i + 1,
                "strategy": hint.memory.core_strategy or hint.memory.content[:100],
                "mode": "strategy_guided",
            })
        logger.info(f"MaTTS: created {len(branches)} branches (budget={budget})")
        return branches

    def select_best_branch(self, branches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Chọn nhánh tốt nhất (stub — trong production dùng LLM judge)."""
        if not branches:
            return {"branch_id": 0, "strategy": None, "mode": "direct"}
        # Trong demo, chọn branch đầu tiên có strategy
        for b in branches:
            if b.get("strategy"):
                return b
        return branches[0]
