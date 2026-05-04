"""
Tầng 4: Archive — Lưu trữ lạnh, phục vụ audit và truy xuất sâu.
"""
from __future__ import annotations
import json, logging, sqlite3, time
from typing import List, Optional
from tam.models import MemoryRecord, MemoryTier, PipelineTrace
from tam.config import TAMConfig

logger = logging.getLogger(__name__)

class Archive:
    MEM_TABLE = "archive_memory"
    TRACE_TABLE = "archive_traces"

    def __init__(self, config: Optional[TAMConfig] = None, db_path: Optional[str] = None):
        self.config = config or TAMConfig()
        self._db_path = db_path or self.config.db_path
        self._conn = None
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.MEM_TABLE} (
                id TEXT PRIMARY KEY, tier TEXT DEFAULT 'archive', memory_type TEXT,
                content TEXT, summary TEXT, abstraction_group TEXT,
                core_strategy TEXT, applicability_scope TEXT,
                domain_tags TEXT, intent_tags TEXT, module_tags TEXT, task_family TEXT,
                source TEXT, confidence REAL, recall_confidence REAL,
                validation_status TEXT, valid_from REAL, valid_to REAL,
                is_current INTEGER DEFAULT 0, created_at REAL,
                last_confirmed_at REAL, last_used_at REAL,
                usage_count INTEGER, reinforcement_score REAL, importance REAL,
                decay_rate_override REAL,
                success_case_ids TEXT, failure_case_ids TEXT, failure_reasons TEXT,
                reflection_summary TEXT, conflict_status TEXT,
                superseded_by TEXT, supersedes TEXT, embedding TEXT,
                archived_at REAL
            )
        """)
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TRACE_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL, raw_query TEXT, intent TEXT,
                matts_budget TEXT, num_candidates INTEGER,
                num_winners INTEGER, response_confidence REAL,
                reflection TEXT, success INTEGER,
                trace_json TEXT
            )
        """)
        self._conn.commit()

    def archive_memory(self, record: MemoryRecord):
        """Chuyển memory sang Archive (lưu trữ lạnh)."""
        record.tier = MemoryTier.ARCHIVE
        record.is_current = False
        d = record.to_dict()
        d["is_current"] = 0
        d["archived_at"] = time.time()
        cols = ", ".join(d.keys())
        phs = ", ".join(["?"] * len(d))
        self._conn.execute(f"INSERT OR REPLACE INTO {self.MEM_TABLE} ({cols}) VALUES ({phs})", list(d.values()))
        self._conn.commit()

    def log_trace(self, trace: PipelineTrace):
        """Lưu trace của một lượt pipeline — phục vụ evolution worker."""
        self._conn.execute(f"""
            INSERT INTO {self.TRACE_TABLE}
            (timestamp, raw_query, intent, matts_budget, num_candidates,
             num_winners, response_confidence, reflection, success, trace_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trace.timestamp,
            trace.query_context.raw_query if trace.query_context else "",
            trace.query_context.intent.value if trace.query_context else "unknown",
            trace.query_context.matts_budget if trace.query_context else "fast",
            len(trace.candidates_active) + len(trace.candidates_latent),
            len(trace.winners),
            trace.response_confidence,
            trace.reflection,
            1 if trace.success else (0 if trace.success is False else None),
            json.dumps({"winner_ids": trace.final_wm_ids, "branches": trace.matts_branches}),
        ))
        self._conn.commit()

    def get_recent_traces(self, limit: int = 50, success_only: Optional[bool] = None) -> List[dict]:
        """Lấy traces gần đây — cho evolution worker phân tích."""
        query = f"SELECT * FROM {self.TRACE_TABLE}"
        params = []
        if success_only is True:
            query += " WHERE success = 1"
        elif success_only is False:
            query += " WHERE success = 0"
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_archived_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        row = self._conn.execute(f"SELECT * FROM {self.MEM_TABLE} WHERE id = ?", (memory_id,)).fetchone()
        return MemoryRecord.from_dict(dict(row)) if row else None

    def search_archived(self, keyword: str, limit: int = 20) -> List[MemoryRecord]:
        """Tìm kiếm trong archive theo keyword (full-text search đơn giản)."""
        rows = self._conn.execute(
            f"SELECT * FROM {self.MEM_TABLE} WHERE content LIKE ? ORDER BY archived_at DESC LIMIT ?",
            (f"%{keyword}%", limit)
        ).fetchall()
        return [MemoryRecord.from_dict(dict(r)) for r in rows]

    def count_memories(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) as cnt FROM {self.MEM_TABLE}").fetchone()["cnt"]

    def count_traces(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) as cnt FROM {self.TRACE_TABLE}").fetchone()["cnt"]

    def close(self):
        if self._conn: self._conn.close(); self._conn = None
