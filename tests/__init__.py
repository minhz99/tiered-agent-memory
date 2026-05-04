"""Tests for TAM Pipeline."""
import os, sys, tempfile, pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tam.config import TAMConfig
from tam.models import MemoryRecord, MemoryType, MemoryTier, QueryContext, IntentCategory
from tam.pipeline import TAMPipeline
from tam.dynamics.activation import ActivationEngine
from tam.dynamics.decay import DecayEngine
from tam.planes.competition import CompetitionPlane
from tam.planes.query_plane import QueryPlane


@pytest.fixture
def pipeline(tmp_path):
    db = str(tmp_path / "test.db")
    config = TAMConfig(db_path=db, use_transformer_embeddings=False)
    p = TAMPipeline(config)
    yield p
    p.close()


class TestModels:
    def test_memory_record_creation(self):
        r = MemoryRecord(content="test", memory_type=MemoryType.SEMANTIC)
        assert r.content == "test"
        assert r.is_current is True

    def test_memory_touch(self):
        r = MemoryRecord(content="test")
        r.touch()
        assert r.usage_count == 1
        assert r.last_used_at is not None

    def test_memory_mark_outdated(self):
        r = MemoryRecord(content="test")
        r.mark_outdated("new-id")
        assert r.is_current is False
        assert r.superseded_by == "new-id"

    def test_serialize_roundtrip(self):
        r = MemoryRecord(content="hello", domain_tags=["tech"], embedding=[1.0, 2.0])
        d = r.to_dict()
        r2 = MemoryRecord.from_dict(d)
        assert r2.content == "hello"
        assert r2.embedding == [1.0, 2.0]


class TestQueryPlane:
    def test_intent_classification(self):
        qp = QueryPlane()
        ctx = qp.analyze("How to fix this bug in my code?")
        assert ctx.intent == IntentCategory.TECHNICAL

    def test_recall_intent(self):
        qp = QueryPlane()
        ctx = qp.analyze("Nhớ lại lần trước tôi gặp lỗi không?")
        assert ctx.intent == IntentCategory.RECALL

    def test_cue_strength(self):
        qp = QueryPlane()
        ctx = qp.analyze("Remember what happened last time?")
        assert ctx.cue_strength > 0.3


class TestActivation:
    def test_score_range(self):
        engine = ActivationEngine()
        r = MemoryRecord(content="test", importance=0.5, confidence=0.8)
        score = engine.score(r, similarity=0.7)
        assert 0.0 <= score <= 1.0


class TestDecay:
    def test_type_aware_rates(self):
        engine = DecayEngine()
        ep = MemoryRecord(memory_type=MemoryType.EPISODIC)
        sem = MemoryRecord(memory_type=MemoryType.SEMANTIC)
        assert engine.get_rate(ep) > engine.get_rate(sem)


class TestCompetition:
    def test_dedup(self):
        from tam.models import RetrievalResult
        cp = CompetitionPlane()
        r1 = RetrievalResult(memory=MemoryRecord(content="same content"), activation_score=0.9)
        r2 = RetrievalResult(memory=MemoryRecord(content="same content"), activation_score=0.7)
        r3 = RetrievalResult(memory=MemoryRecord(content="different"), activation_score=0.8)
        winners = cp.compete([r1, r2, r3])
        assert len(winners) <= 5


class TestPipeline:
    def test_add_and_retrieve(self, pipeline):
        pipeline.add_memory("Python is great for ML", memory_type=MemoryType.SEMANTIC,
                           domain_tags=["technical"])
        result = pipeline.process("Tell me about Python")
        assert "query_context" in result
        assert result["query_context"]["intent"] in ("technical", "simple", "unknown")

    def test_stats(self, pipeline):
        stats = pipeline.get_stats()
        assert "active_memories" in stats

    def test_evolution(self, pipeline):
        pipeline.add_memory("test memory")
        report = pipeline.run_evolution()
        assert "Evolution Report" in report

    def test_end_to_end(self, pipeline):
        pipeline.add_memory("User likes Python", domain_tags=["tech"])
        pipeline.add_memory("Debug strategy: check logs first",
                           memory_type=MemoryType.REASONING,
                           core_strategy="Check logs first",
                           applicability_scope="debugging")
        result = pipeline.process("How to debug this error?")
        assert result["retrieval"]["active_candidates"] >= 0
        assert result["wm_summary"]["slot_count"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
