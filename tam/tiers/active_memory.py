"""
Tầng 2: Active Memory — Tri thức ổn định, truy cập mặc định.

Active Memory chứa tri thức có xác suất tái sử dụng cao: semantic memory,
style profile, system memory và các Reasoning Memory đã kiểm chứng.
Đây là nguồn retrieval mặc định — những gì Agent "mặc định nghĩ tới" trước tiên.

Storage: SQLite cho persistence + NumPy in-memory index cho vector search.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from tam.models import MemoryRecord, MemoryTier, MemoryType, RetrievalResult
from tam.config import TAMConfig

logger = logging.getLogger(__name__)


class ActiveMemory:
    """
    Tầng 2 — Active Memory.

    Cung cấp:
    - Persistent storage via SQLite
    - Vector search via in-memory numpy index
    - Metadata filtering (domain, intent, type)
    - Automatic indexing on add/update
    """

    TABLE_NAME = "active_memory"

    def __init__(self, config: Optional[TAMConfig] = None, db_path: Optional[str] = None):
        self.config = config or TAMConfig()
        self._db_path = db_path or self.config.db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._index: List[Tuple[str, np.ndarray]] = []  # (id, embedding)
        self._init_db()
        self._rebuild_index()

    def _init_db(self) -> None:
        """Khởi tạo bảng SQLite nếu chưa tồn tại."""
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                id TEXT PRIMARY KEY,
                tier TEXT DEFAULT 'active',
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
                confidence REAL DEFAULT 1.0,
                recall_confidence REAL DEFAULT 1.0,
                validation_status TEXT DEFAULT 'unverified',
                valid_from REAL,
                valid_to REAL,
                is_current INTEGER DEFAULT 1,
                created_at REAL,
                last_confirmed_at REAL,
                last_used_at REAL,
                usage_count INTEGER DEFAULT 0,
                reinforcement_score REAL DEFAULT 0.0,
                importance REAL DEFAULT 0.5,
                decay_rate_override REAL,
                success_case_ids TEXT DEFAULT '[]',
                failure_case_ids TEXT DEFAULT '[]',
                failure_reasons TEXT DEFAULT '[]',
                reflection_summary TEXT,
                conflict_status TEXT DEFAULT 'none',
                superseded_by TEXT,
                supersedes TEXT,
                embedding TEXT
            )
        """)
        self._conn.commit()

    def _rebuild_index(self) -> None:
        """Rebuild in-memory vector index từ SQLite."""
        self._index.clear()
        cursor = self._conn.execute(
            f"SELECT id, embedding FROM {self.TABLE_NAME} WHERE embedding IS NOT NULL AND is_current = 1"
        )
        for row in cursor:
            emb = json.loads(row["embedding"])
            self._index.append((row["id"], np.array(emb, dtype=np.float32)))
        logger.debug(f"Rebuilt vector index: {len(self._index)} entries")

    def add(self, record: MemoryRecord) -> None:
        """Thêm một memory vào Active tier."""
        record.tier = MemoryTier.ACTIVE
        d = record.to_dict()
        d["is_current"] = 1 if record.is_current else 0

        columns = ", ".join(d.keys())
        placeholders = ", ".join(["?"] * len(d))
        self._conn.execute(
            f"INSERT OR REPLACE INTO {self.TABLE_NAME} ({columns}) VALUES ({placeholders})",
            list(d.values()),
        )
        self._conn.commit()

        # Update vector index
        if record.embedding:
            self._index = [
                (mid, emb) for mid, emb in self._index if mid != record.id
            ]
            self._index.append((record.id, np.array(record.embedding, dtype=np.float32)))

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """Truy xuất một memory theo ID."""
        row = self._conn.execute(
            f"SELECT * FROM {self.TABLE_NAME} WHERE id = ?", (memory_id,)
        ).fetchone()
        if row:
            return MemoryRecord.from_dict(dict(row))
        return None

    def update(self, record: MemoryRecord) -> None:
        """Cập nhật một memory đã tồn tại."""
        self.add(record)  # INSERT OR REPLACE

    def remove(self, memory_id: str) -> None:
        """Xóa một memory khỏi Active."""
        self._conn.execute(
            f"DELETE FROM {self.TABLE_NAME} WHERE id = ?", (memory_id,)
        )
        self._conn.commit()
        self._index = [(mid, emb) for mid, emb in self._index if mid != memory_id]

    def search_vector(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
    ) -> List[RetrievalResult]:
        """
        Vector search — tìm memory gần nhất theo cosine similarity.

        Args:
            query_embedding: Vector truy vấn
            top_k: Số kết quả tối đa
            memory_types: Lọc theo loại memory (None = tất cả)

        Returns:
            Danh sách RetrievalResult sắp xếp theo similarity giảm dần
        """
        if not self._index:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = q_vec / q_norm

        scores = []
        for mid, emb in self._index:
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue
            sim = float(np.dot(q_vec, emb / emb_norm))
            scores.append((mid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for mid, sim in scores[:top_k * 2]:  # Fetch extra for filtering
            record = self.get(mid)
            if record is None:
                continue
            if memory_types and record.memory_type not in memory_types:
                continue
            if not record.is_current:
                continue
            results.append(RetrievalResult(
                memory=record,
                activation_score=sim,
                source_tier=MemoryTier.ACTIVE,
            ))
            if len(results) >= top_k:
                break

        return results

    def search_metadata(
        self,
        domain_tags: Optional[List[str]] = None,
        intent_tags: Optional[List[str]] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 20,
    ) -> List[MemoryRecord]:
        """Tìm memory theo metadata filters."""
        query = f"SELECT * FROM {self.TABLE_NAME} WHERE is_current = 1"
        params = []

        if memory_types:
            placeholders = ", ".join(["?"] * len(memory_types))
            query += f" AND memory_type IN ({placeholders})"
            params.extend([mt.value for mt in memory_types])

        query += " ORDER BY importance DESC, reinforcement_score DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        records = [MemoryRecord.from_dict(dict(r)) for r in rows]

        # Post-filter by tags (JSON arrays stored as strings)
        if domain_tags:
            records = [
                r for r in records
                if any(tag in r.domain_tags for tag in domain_tags)
            ]
        if intent_tags:
            records = [
                r for r in records
                if any(tag in r.intent_tags for tag in intent_tags)
            ]

        return records[:limit]

    def get_all_current(self) -> List[MemoryRecord]:
        """Trả về tất cả memory hiện hành."""
        rows = self._conn.execute(
            f"SELECT * FROM {self.TABLE_NAME} WHERE is_current = 1"
        ).fetchall()
        return [MemoryRecord.from_dict(dict(r)) for r in rows]

    def count(self) -> int:
        """Đếm số memory hiện hành."""
        row = self._conn.execute(
            f"SELECT COUNT(*) as cnt FROM {self.TABLE_NAME} WHERE is_current = 1"
        ).fetchone()
        return row["cnt"]

    def close(self) -> None:
        """Đóng kết nối database."""
        if self._conn:
            self._conn.close()
            self._conn = None
