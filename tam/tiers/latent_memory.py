"""
Tầng 3: Latent Memory — Ký ức ngủ đông.
"""

from __future__ import annotations
import json
import sqlite3
from typing import List, Optional
import numpy as np

from tam.models import MemoryRecord, MemoryTier, RetrievalResult, QueryContext
from tam.config import TAMConfig

class LatentMemory:
    TABLE_NAME = "latent_memory"

    def __init__(self, config: Optional[TAMConfig] = None, db_path: Optional[str] = None):
        self.config = config or TAMConfig()
        self._db_path = db_path or self.config.db_path
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                id TEXT PRIMARY KEY,
                tier TEXT,
                memory_type TEXT,
                content TEXT,
                summary TEXT,
                abstraction_group TEXT,
                core_strategy TEXT,
                applicability_scope TEXT,
                domain_tags TEXT,
                intent_tags TEXT,
                module_tags TEXT,
                task_family TEXT,
                source TEXT,
                confidence REAL,
                recall_confidence REAL,
                validation_status TEXT,
                valid_from REAL,
                valid_to REAL,
                is_current INTEGER,
                created_at REAL,
                last_confirmed_at REAL,
                last_used_at REAL,
                usage_count INTEGER,
                reinforcement_score REAL,
                recency_score REAL,
                importance REAL,
                decay_rate_override REAL,
                success_case_ids TEXT,
                failure_case_ids TEXT,
                failure_reasons TEXT,
                reflection_summary TEXT,
                conflict_status TEXT,
                superseded_by TEXT,
                supersedes TEXT,
                metadata TEXT,
                embedding TEXT
            )
        """)
        self._conn.commit()

    def add(self, record: MemoryRecord) -> None:
        record.tier = MemoryTier.LATENT
        d = record.to_dict()
        for k, v in d.items():
            if isinstance(v, (list, dict)):
                d[k] = json.dumps(v)
        columns = ", ".join(d.keys())
        placeholders = ", ".join(["?"] * len(d))
        self._conn.execute(f"INSERT OR REPLACE INTO {self.TABLE_NAME} ({columns}) VALUES ({placeholders})", list(d.values()))
        self._conn.commit()

    def should_activate(self, ctx: QueryContext) -> bool:
        return ctx.cue_strength >= self.config.latent_cue_threshold

    def search_with_expanded_query(self, ctx: QueryContext, top_k: int = 5) -> List[RetrievalResult]:
        if not ctx.embedding: return []
        q_vec = np.array(ctx.embedding, dtype=np.float32)
        cursor = self._conn.execute(f"SELECT * FROM {self.TABLE_NAME} WHERE is_current = 1")
        results = []
        for row in cursor:
            record = MemoryRecord.from_dict(dict(row))
            if record.embedding:
                emb = np.array(record.embedding, dtype=np.float32)
                sim = float(np.dot(q_vec, emb) / (np.linalg.norm(q_vec) * np.linalg.norm(emb)))
                if sim > 0.5:
                    results.append(RetrievalResult(memory=record, activation_score=sim, source_tier=MemoryTier.LATENT))
        results.sort(key=lambda x: x.activation_score, reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) as cnt FROM {self.TABLE_NAME}").fetchone()["cnt"]

    def close(self) -> None: self._conn.close()
