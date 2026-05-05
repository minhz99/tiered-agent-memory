"""
Tầng 4: Archive — Lưu trữ lạnh và Audit logs.
"""

from __future__ import annotations
import json
import sqlite3
from typing import List, Optional
from tam.models import MemoryRecord, MemoryTier, PipelineTrace
from tam.config import TAMConfig

class Archive:
    MEM_TABLE = "archive_memories"
    TRACE_TABLE = "pipeline_traces"

    def __init__(self, config: Optional[TAMConfig] = None, db_path: Optional[str] = None):
        self.config = config or TAMConfig()
        self._db_path = db_path or self.config.db_path
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.MEM_TABLE} (
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
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TRACE_TABLE} (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                query_raw TEXT,
                intent TEXT,
                success INTEGER,
                trace_json TEXT
            )
        """)
        self._conn.commit()

    def archive_memory(self, record: MemoryRecord) -> None:
        record.tier = MemoryTier.ARCHIVE
        record.is_current = False
        d = record.to_dict()
        for k, v in d.items():
            if isinstance(v, (list, dict)):
                d[k] = json.dumps(v)
        columns = ", ".join(d.keys())
        placeholders = ", ".join(["?"] * len(d))
        self._conn.execute(f"INSERT OR REPLACE INTO {self.MEM_TABLE} ({columns}) VALUES ({placeholders})", list(d.values()))
        self._conn.commit()

    def log_trace(self, trace: PipelineTrace) -> None:
        import uuid
        trace_id = str(uuid.uuid4())
        self._conn.execute(
            f"INSERT INTO {self.TRACE_TABLE} (id, timestamp, query_raw, intent, success, trace_json) VALUES (?, ?, ?, ?, ?, ?)",
            (trace_id, trace.timestamp, trace.query_context.raw_query if trace.query_context else "",
             trace.query_context.intent.value if trace.query_context else "unknown",
             1 if trace.success else 0, "{}")
        )
        self._conn.commit()

    def count_memories(self) -> int: return self._conn.execute(f"SELECT COUNT(*) as cnt FROM {self.MEM_TABLE}").fetchone()["cnt"]
    def count_traces(self) -> int: return self._conn.execute(f"SELECT COUNT(*) as cnt FROM {self.TRACE_TABLE}").fetchone()["cnt"]
    def close(self) -> None: self._conn.close()
