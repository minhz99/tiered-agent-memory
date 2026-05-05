"""
Tầng 2: Active Memory — Tri thức ổn định, truy cập mặc định.
"""

from __future__ import annotations
import json
import logging
import sqlite3
import time
from typing import List, Optional, Tuple
import numpy as np

from tam.models import MemoryRecord, MemoryTier, MemoryType, RetrievalResult
from tam.config import TAMConfig

logger = logging.getLogger(__name__)

class ActiveMemory:
    TABLE_NAME = "active_memory"

    def __init__(self, config: Optional[TAMConfig] = None, db_path: Optional[str] = None):
        self.config = config or TAMConfig()
        self._db_path = db_path or self.config.db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._index: List[Tuple[str, np.ndarray]] = []
        self._init_db()
        self._rebuild_index()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
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

    def _rebuild_index(self) -> None:
        self._index.clear()
        cursor = self._conn.execute(
            f"SELECT id, embedding FROM {self.TABLE_NAME} WHERE embedding IS NOT NULL AND is_current = 1"
        )
        for row in cursor:
            emb = json.loads(row["embedding"])
            self._index.append((row["id"], np.array(emb, dtype=np.float32)))

    def add(self, record: MemoryRecord) -> None:
        record.tier = MemoryTier.ACTIVE
        d = record.to_dict()
        # Convert list/dict to JSON strings for SQLite
        for k, v in d.items():
            if isinstance(v, (list, dict)):
                d[k] = json.dumps(v)
        
        columns = ", ".join(d.keys())
        placeholders = ", ".join(["?"] * len(d))
        self._conn.execute(
            f"INSERT OR REPLACE INTO {self.TABLE_NAME} ({columns}) VALUES ({placeholders})",
            list(d.values()),
        )
        self._conn.commit()
        if record.embedding:
            self._index = [(mid, emb) for mid, emb in self._index if mid != record.id]
            self._index.append((record.id, np.array(record.embedding, dtype=np.float32)))

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        row = self._conn.execute(f"SELECT * FROM {self.TABLE_NAME} WHERE id = ?", (memory_id,)).fetchone()
        return MemoryRecord.from_dict(dict(row)) if row else None

    def update(self, record: MemoryRecord) -> None: self.add(record)
    def remove(self, memory_id: str) -> None:
        self._conn.execute(f"DELETE FROM {self.TABLE_NAME} WHERE id = ?", (memory_id,))
        self._conn.commit()
        self._index = [(mid, emb) for mid, emb in self._index if mid != memory_id]

    def search_vector(self, query_embedding: List[float], top_k: int = 10, memory_types: Optional[List[MemoryType]] = None) -> List[RetrievalResult]:
        if not self._index: return []
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0: return []
        q_vec = q_vec / q_norm
        scores = []
        for mid, emb in self._index:
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0: continue
            sim = float(np.dot(q_vec, emb / emb_norm))
            scores.append((mid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for mid, sim in scores[:top_k * 2]:
            record = self.get(mid)
            if record and (not memory_types or record.memory_type in memory_types) and record.is_current:
                results.append(RetrievalResult(memory=record, activation_score=sim, source_tier=MemoryTier.ACTIVE))
                if len(results) >= top_k: break
        return results

    def get_all_current(self) -> List[MemoryRecord]:
        rows = self._conn.execute(f"SELECT * FROM {self.TABLE_NAME} WHERE is_current = 1").fetchall()
        return [MemoryRecord.from_dict(dict(r)) for r in rows]

    def count(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) as cnt FROM {self.TABLE_NAME} WHERE is_current = 1").fetchone()["cnt"]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
