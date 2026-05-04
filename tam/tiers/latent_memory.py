"""
Tầng 3: Latent Memory — Ký ức ngủ đông có điều kiện kích hoạt.
"""
from __future__ import annotations
import json, logging, sqlite3
from typing import List, Optional
import numpy as np
from tam.models import MemoryRecord, MemoryTier, MemoryType, RetrievalResult, QueryContext
from tam.config import TAMConfig

logger = logging.getLogger(__name__)

class LatentMemory:
    TABLE_NAME = "latent_memory"

    def __init__(self, config: Optional[TAMConfig] = None, db_path: Optional[str] = None):
        self.config = config or TAMConfig()
        self._db_path = db_path or self.config.db_path
        self._conn = None
        self._index: List[tuple] = []
        self._init_db()
        self._rebuild_index()

    def _init_db(self):
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                id TEXT PRIMARY KEY, tier TEXT DEFAULT 'latent', memory_type TEXT,
                content TEXT, summary TEXT, abstraction_group TEXT,
                core_strategy TEXT, applicability_scope TEXT,
                domain_tags TEXT, intent_tags TEXT, module_tags TEXT, task_family TEXT,
                source TEXT, confidence REAL DEFAULT 1.0, recall_confidence REAL DEFAULT 1.0,
                validation_status TEXT DEFAULT 'unverified',
                valid_from REAL, valid_to REAL, is_current INTEGER DEFAULT 1,
                created_at REAL, last_confirmed_at REAL, last_used_at REAL,
                usage_count INTEGER DEFAULT 0, reinforcement_score REAL DEFAULT 0.0,
                importance REAL DEFAULT 0.5, decay_rate_override REAL,
                success_case_ids TEXT DEFAULT '[]', failure_case_ids TEXT DEFAULT '[]',
                failure_reasons TEXT DEFAULT '[]', reflection_summary TEXT,
                conflict_status TEXT DEFAULT 'none', superseded_by TEXT, supersedes TEXT,
                embedding TEXT
            )
        """)
        self._conn.commit()

    def _rebuild_index(self):
        self._index.clear()
        cursor = self._conn.execute(
            f"SELECT id, embedding FROM {self.TABLE_NAME} WHERE embedding IS NOT NULL AND is_current = 1"
        )
        for row in cursor:
            emb = json.loads(row["embedding"])
            self._index.append((row["id"], np.array(emb, dtype=np.float32)))

    def should_activate(self, query_ctx: QueryContext) -> bool:
        """Kiểm tra cue có đủ mạnh để mở Latent."""
        return query_ctx.cue_strength >= self.config.latent_cue_threshold

    def add(self, record: MemoryRecord):
        record.tier = MemoryTier.LATENT
        d = record.to_dict()
        d["is_current"] = 1 if record.is_current else 0
        cols = ", ".join(d.keys())
        phs = ", ".join(["?"] * len(d))
        self._conn.execute(f"INSERT OR REPLACE INTO {self.TABLE_NAME} ({cols}) VALUES ({phs})", list(d.values()))
        self._conn.commit()
        if record.embedding:
            self._index = [(m, e) for m, e in self._index if m != record.id]
            self._index.append((record.id, np.array(record.embedding, dtype=np.float32)))

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        row = self._conn.execute(f"SELECT * FROM {self.TABLE_NAME} WHERE id = ?", (memory_id,)).fetchone()
        return MemoryRecord.from_dict(dict(row)) if row else None

    def remove(self, memory_id: str):
        self._conn.execute(f"DELETE FROM {self.TABLE_NAME} WHERE id = ?", (memory_id,))
        self._conn.commit()
        self._index = [(m, e) for m, e in self._index if m != memory_id]

    def search_with_expanded_query(self, query_ctx: QueryContext, top_k: int = 10) -> List[RetrievalResult]:
        """Two-pass retrieval pass 2: dùng expanded query, KHÔNG dùng raw query."""
        if not self.should_activate(query_ctx):
            return []
        if query_ctx.embedding is None or not self._index:
            return []
        logger.info("Latent ACTIVATED: cue_strength=%.2f", query_ctx.cue_strength)
        q = np.array(query_ctx.embedding, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0: return []
        q = q / qn
        scores = []
        for mid, emb in self._index:
            en = np.linalg.norm(emb)
            if en == 0: continue
            scores.append((mid, float(np.dot(q, emb / en))))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for mid, sim in scores[:top_k]:
            rec = self.get(mid)
            if rec and rec.is_current:
                results.append(RetrievalResult(memory=rec, activation_score=sim * 0.9, source_tier=MemoryTier.LATENT))
        return results

    def get_all(self) -> List[MemoryRecord]:
        rows = self._conn.execute(f"SELECT * FROM {self.TABLE_NAME} WHERE is_current = 1").fetchall()
        return [MemoryRecord.from_dict(dict(r)) for r in rows]

    def count(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) as cnt FROM {self.TABLE_NAME} WHERE is_current = 1").fetchone()["cnt"]

    def close(self):
        if self._conn: self._conn.close(); self._conn = None
