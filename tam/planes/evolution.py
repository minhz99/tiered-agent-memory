"""
Evolution Plane — Self-evolving background worker.

Worker chạy nền không chỉ dọn rác mà còn tự chắt lọc logs thành higher-quality
memory rồi tiêm ngược vào Active Memory / ReasoningBank. Đây là lúc hệ chuyển
từ maintenance sang self-curation.
"""
from __future__ import annotations
import logging, time
from typing import List, Optional
from tam.models import MemoryRecord, MemoryType, PipelineTrace
from tam.dynamics.decay import DecayEngine
from tam.planes.reasoning import ReasoningPlane
from tam.config import EvolutionConfig

logger = logging.getLogger(__name__)

class EvolutionPlane:
    def __init__(self, config: Optional[EvolutionConfig] = None):
        self.config = config or EvolutionConfig()
        self._last_synthesis = 0.0

    def should_run(self) -> bool:
        elapsed_hours = (time.time() - self._last_synthesis) / 3600
        return elapsed_hours >= self.config.synthesis_interval_hours

    def run_maintenance(self, all_active: List[MemoryRecord], decay_engine: DecayEngine) -> dict:
        """Maintenance cơ bản: apply decay, detect demotion candidates."""
        decayed = decay_engine.apply_batch(all_active)
        demote_candidates = [r for r in decayed if decay_engine.should_demote(r)]
        return {
            "total_processed": len(all_active),
            "demote_candidates": len(demote_candidates),
            "demote_ids": [r.id for r in demote_candidates],
        }

    def run_synthesis(self, traces: List[dict], reasoning_plane: ReasoningPlane) -> dict:
        """Self-curation: phân tích traces → distill reasoning memories."""
        self._last_synthesis = time.time()
        successes = [t for t in traces if t.get("success") == 1]
        failures = [t for t in traces if t.get("success") == 0]

        result = {
            "total_traces": len(traces),
            "successes": len(successes),
            "failures": len(failures),
            "new_strategies": 0,
        }

        if len(successes) >= self.config.min_traces_for_distill:
            new_strategy = reasoning_plane.distill_from_traces(successes, failures)
            if new_strategy:
                reasoning_plane.add_strategy(new_strategy)
                result["new_strategies"] = 1
                logger.info(f"Evolution: distilled new strategy from {len(successes)} traces")

        return result

    def generate_daily_report(self, maintenance_result: dict, synthesis_result: dict) -> str:
        """Tạo báo cáo tự tiến hóa."""
        lines = [
            "═══ TAM Evolution Report ═══",
            f"⏰ Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "── Maintenance ──",
            f"  Memories processed: {maintenance_result.get('total_processed', 0)}",
            f"  Demotion candidates: {maintenance_result.get('demote_candidates', 0)}",
            "",
            "── Synthesis ──",
            f"  Traces analyzed: {synthesis_result.get('total_traces', 0)}",
            f"  Successes: {synthesis_result.get('successes', 0)}",
            f"  Failures: {synthesis_result.get('failures', 0)}",
            f"  New strategies: {synthesis_result.get('new_strategies', 0)}",
            "═══════════════════════════",
        ]
        return "\n".join(lines)
