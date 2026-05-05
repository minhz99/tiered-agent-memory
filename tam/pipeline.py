"""
TAM Pipeline — 12-step end-to-end orchestrator.

Bước 1:  Phân tích intent, domain, complexity
Bước 2:  Thiết lập MaTTS budget
Bước 3:  Query rewriting / expansion
Bước 4:  Intent-aware gating
Bước 5:  Retrieval từ Active & ReasoningBank
Bước 6:  Phát hiện cue để mở Latent
Bước 7:  Retrieval mở rộng từ Latent
Bước 8:  Competition, inhibition, conflict filtering
Bước 9:  Xây dựng Working Memory
Bước 10: MaTTS suy luận đa nhánh
Bước 11: Chọn đáp án, ghi thành công/thất bại
Bước 12: Update, reflection, lịch tự tiến hóa
"""
from __future__ import annotations
import logging, time
from typing import Optional, Dict, Any

from tam.config import TAMConfig
from tam.models import (
    MemoryRecord, MemoryType, MemoryTier, QueryContext,
    RetrievalResult, PipelineTrace
)
from tam.tiers.working_memory import WorkingMemory
from tam.tiers.active_memory import ActiveMemory
from tam.tiers.latent_memory import LatentMemory
from tam.tiers.archive import Archive
from tam.dynamics.activation import ActivationEngine
from tam.dynamics.decay import DecayEngine
from tam.dynamics.reinforcement import ReinforcementEngine
from tam.planes.query_plane import QueryPlane
from tam.planes.competition import CompetitionPlane
from tam.planes.abstraction import AbstractionPlane
from tam.planes.reasoning import ReasoningPlane
from tam.planes.state_plane import StatePlane
from tam.planes.scaling import ScalingPlane
from tam.planes.evolution import EvolutionPlane

from tam.utils.ast_parser import ASTAnalyzer

logger = logging.getLogger(__name__)


class TAMPipeline:
    """
    Orchestrator chính — chạy 12-step pipeline cho mỗi truy vấn.
    """

    def __init__(self, config: Optional[TAMConfig] = None):
        self.config = config or TAMConfig()

        # Tiers
        self.wm = WorkingMemory(self.config.working_memory)
        self.active = ActiveMemory(self.config, db_path=self.config.db_path)
        self.latent = LatentMemory(self.config, db_path=self.config.db_path)
        self.archive = Archive(self.config, db_path=self.config.db_path)

        # Dynamics
        self.activation = ActivationEngine(self.config.activation_weights)
        self.decay = DecayEngine(self.config.decay_profiles)
        self.reinforcement = ReinforcementEngine()

        # Planes
        self.query_plane = QueryPlane(self.config)
        self.competition = CompetitionPlane(self.config.competition)
        self.abstraction = AbstractionPlane(self.config)
        self.reasoning = ReasoningPlane(self.config)
        self.state = StatePlane(self.config)
        self.scaling = ScalingPlane(self.config.matts)
        self.evolution = EvolutionPlane(self.config.evolution)

        # Load reasoning memories from Active into ReasoningBank
        self.reasoning.load_from_active(self.active.get_all_current())

    def process(self, raw_query: str) -> Dict[str, Any]:
        """
        Chạy 12-step pipeline cho một truy vấn.

        Returns dict với keys:
            query_context, wm_summary, trace, branches, response_confidence
        """
        trace = PipelineTrace(timestamp=time.time())
        self.wm.clear()

        # ══ BƯỚC 1: Phân tích intent, domain, complexity ══
        logger.info("Step 1: Query Understanding")
        query_ctx = self.query_plane.analyze(raw_query)
        trace.query_context = query_ctx

        # ══ BƯỚC 2: Thiết lập MaTTS budget ══
        logger.info("Step 2: MaTTS Budget")
        query_ctx.matts_budget = self.scaling.determine_budget(query_ctx)

        # ══ BƯỚC 3: Query rewriting / expansion ══
        # (Đã được thực hiện trong query_plane.analyze())
        logger.info(f"Step 3: Query rewritten → '{query_ctx.rewritten_query[:80]}'")
        logger.info(f"  Expanded concepts: {query_ctx.expanded_concepts}")

        # ══ BƯỚC 4: Intent-aware gating ══
        logger.info(f"Step 4: Intent gating → allowed types: "
                    f"{[t.value for t in query_ctx.allowed_memory_types]}")

        # ══ BƯỚC 5: Retrieval từ Active & ReasoningBank ══
        logger.info("Step 5: Active + ReasoningBank retrieval")
        active_results = []
        if query_ctx.embedding:
            active_results = self.active.search_vector(
                query_ctx.embedding,
                top_k=self.config.competition.max_candidates,
                memory_types=query_ctx.allowed_memory_types,
            )
        reasoning_results = self.reasoning.search(
            query_ctx.embedding, top_k=3
        ) if query_ctx.embedding else []

        # Score all results
        active_results = self.activation.score_results(active_results, query_ctx)
        reasoning_results = self.activation.score_results(reasoning_results, query_ctx)

        trace.candidates_active = active_results
        trace.candidates_reasoning = reasoning_results

        # ══ BƯỚC 6: Phát hiện cue để mở Latent ══
        logger.info(f"Step 6: Cue detection → strength={query_ctx.cue_strength:.2f}")

        # ══ BƯỚC 7: Retrieval mở rộng từ Latent ══
        latent_results = []
        if self.latent.should_activate(query_ctx):
            logger.info("Step 7: Latent ACTIVATED")
            latent_results = self.latent.search_with_expanded_query(query_ctx, top_k=5)
            latent_results = self.activation.score_results(latent_results, query_ctx)
        else:
            logger.info("Step 7: Latent NOT activated")
        trace.candidates_latent = latent_results

        # ══ BƯỚC 8: Competition, inhibition, conflict filtering ══
        logger.info("Step 8: Competition + Conflict filtering")
        all_candidates = active_results + reasoning_results + latent_results

        # State plane: temporal + conflict
        all_candidates = self.state.filter_results(all_candidates)

        # Competition plane: MMR diversity
        winners = self.competition.compete(all_candidates, query_ctx)
        trace.winners = winners
        trace.inhibited = [c for c in all_candidates
                          if c.memory.id not in {w.memory.id for w in winners}]

        # ══ BƯỚC 9: Xây dựng Working Memory ══
        logger.info("Step 9: Composing Working Memory")
        min_conf = min((w.memory.recall_confidence for w in winners), default=1.0)
        self.wm.set_control_params({
            "recall_confidence": min_conf,
            "matts_mode": query_ctx.matts_budget,
        })
        if reasoning_results:
            best_strategy = reasoning_results[0].memory.core_strategy
            if best_strategy:
                self.wm.set_control_params({"strategy_hint": best_strategy})

        injected = self.wm.inject(winners)
        trace.final_wm_ids = [s.memory.id for s in injected]

        # ══ BƯỚC 10: MaTTS suy luận đa nhánh ══
        logger.info("Step 10: MaTTS branching")
        branches = self.scaling.create_branches(query_ctx, reasoning_results)
        trace.matts_branches = branches

        # ══ BƯỚC 11: Chọn đáp án ══
        logger.info("Step 11: Select answer")
        best_branch = self.scaling.select_best_branch(branches)
        trace.response_confidence = min_conf

        # ══ BƯỚC 12: Update, reflection, evolution ══
        logger.info("Step 12: Post-processing")
        context = self.wm.compose_context(raw_query)

        # Log trace to archive
        self.archive.log_trace(trace)

        result = {
            "query_context": {
                "raw": query_ctx.raw_query,
                "rewritten": query_ctx.rewritten_query,
                "intent": query_ctx.intent.value,
                "complexity": round(query_ctx.complexity_score, 2),
                "cue_strength": round(query_ctx.cue_strength, 2),
                "matts_budget": query_ctx.matts_budget,
                "expanded_concepts": query_ctx.expanded_concepts,
            },
            "retrieval": {
                "active_candidates": len(active_results),
                "reasoning_candidates": len(reasoning_results),
                "latent_candidates": len(latent_results),
                "latent_activated": bool(latent_results),
                "winners": len(winners),
                "inhibited": len(trace.inhibited),
            },
            "wm_summary": self.wm.get_summary(),
            "branches": branches,
            "best_branch": best_branch,
            "response_confidence": round(min_conf, 2),
            "composed_context": context,
            "trace": trace,
        }
        return result

    def add_memory(self, content: str, memory_type: MemoryType = MemoryType.SEMANTIC,
                   tier: MemoryTier = MemoryTier.ACTIVE, **kwargs) -> MemoryRecord:
        """Thêm một memory mới vào hệ thống."""
        # AST-Aware: Trích xuất metadata nếu là code
        ast_meta = ASTAnalyzer.extract_metadata(content)
        if ast_meta["is_code"]:
            summary = ASTAnalyzer.get_context_summary(ast_meta)
            logger.info(f"AST-Aware: Detected code structure. Summary: {summary}")
            # Bổ sung summary vào để embedding hiểu ngữ cảnh code tốt hơn
            content_for_embedding = f"{content}\nContext: {summary}"
        else:
            content_for_embedding = content

        # Tạo embedding với is_query=False (dùng 'passage: ' prefix)
        embedding = self.query_plane._embed(content_for_embedding, is_query=False)

        record = MemoryRecord(
            content=content,
            memory_type=memory_type,
            tier=tier,
            embedding=embedding,
            metadata=ast_meta if ast_meta["is_code"] else kwargs.get("metadata", {}),
            **{k: v for k, v in kwargs.items() if k != "metadata"},
        )

        if tier == MemoryTier.ACTIVE:
            self.active.add(record)
            if memory_type == MemoryType.REASONING:
                self.reasoning.add_strategy(record)
        elif tier == MemoryTier.LATENT:
            self.latent.add(record)
        elif tier == MemoryTier.ARCHIVE:
            self.archive.archive_memory(record)

        logger.info(f"Added memory: {record.id[:8]} [{tier.value}/{memory_type.value}]")
        return record

    def mark_success(self, trace: PipelineTrace):
        """Đánh dấu lượt xử lý thành công."""
        trace.success = True
        all_candidates = [w.memory for w in trace.winners]
        self.reinforcement.process_trace(trace, all_candidates)

    def mark_failure(self, trace: PipelineTrace, reason: str = ""):
        """Đánh dấu lượt xử lý thất bại."""
        trace.success = False
        trace.reflection = reason
        all_candidates = [w.memory for w in trace.winners]
        self.reinforcement.process_trace(trace, all_candidates)

    def run_evolution(self) -> str:
        """Chạy background evolution worker."""
        # Maintenance
        all_active = self.active.get_all_current()
        # Sử dụng DecayEngine mới để tính toán suy giảm ký ức
        maint = self.decay.apply_decay(all_active)

        # Apply updates
        for r in all_active:
            self.active.update(r)
        
        # Di chuyển sang tầng Latent nếu bị hạ tầng
        for mid in maint["demote_to_latent"]:
            rec = self.active.get(mid)
            if rec:
                self.active.remove(mid)
                self.latent.add(rec)
                
        # Di chuyển sang Archive nếu bị hạ tầng từ Latent (nếu có logic chạy Latent maintenance)
        for mid in maint["demote_to_archive"]:
            # (Thực hiện tương tự cho latent -> archive)
            pass

        # Synthesis
        traces = self.archive.get_recent_traces(limit=self.config.evolution.max_logs_per_batch)
        synth = self.evolution.run_synthesis(traces, self.reasoning)

        return self.evolution.generate_daily_report(maint, synth)

    def get_stats(self) -> Dict[str, Any]:
        """Thống kê tổng quan hệ thống."""
        return {
            "active_memories": self.active.count(),
            "latent_memories": self.latent.count(),
            "archived_memories": self.archive.count_memories(),
            "archived_traces": self.archive.count_traces(),
            "reasoning_strategies": self.reasoning.count(),
        }

    def close(self):
        """Đóng tất cả kết nối."""
        self.active.close()
        self.latent.close()
        self.archive.close()
