"""
TAM Data Models — Schema đầy đủ cho một Memory Record.

Triển khai 8 nhóm thuộc tính từ đặc tả:
  1. Nhận dạng (Identification)
  2. Nội dung (Content)
  3. Ngữ cảnh (Context)
  4. Nguồn & Độ tin cậy (Source & Confidence)
  5. Thời gian (Temporal)
  6. Hành vi (Behavioral)
  7. Contrastive Evidence
  8. Xung đột (Conflict)
"""

from __future__ import annotations

import uuid
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ── Enums ──────────────────────────────────────────────────────────────────────

class MemoryType(str, Enum):
    """Loại memory — quyết định decay profile và retrieval strategy."""
    EPISODIC  = "episodic"    # Ký ức sự kiện, quên nhanh
    SEMANTIC  = "semantic"    # Tri thức ổn định, quên chậm
    REASONING = "reasoning"   # Chiến lược suy luận tái sử dụng
    STYLE     = "style"       # Sở thích phong cách
    SYSTEM    = "system"      # Policy, năng lực tool
    ABSTRACT  = "abstract"    # Tổng hợp nhiều atomic memories


class MemoryTier(str, Enum):
    """Tầng lưu trữ — xác định mức sẵn sàng truy cập."""
    WORKING  = "working"      # Context đang hoạt động
    ACTIVE   = "active"       # Truy cập mặc định
    LATENT   = "latent"       # Ngủ đông, cần cue
    ARCHIVE  = "archive"      # Lưu trữ lạnh


class ConflictStatus(str, Enum):
    """Trạng thái xung đột của memory."""
    NONE       = "none"        # Không xung đột
    DETECTED   = "detected"    # Phát hiện xung đột, chưa xử lý
    SUPERSEDED = "superseded"  # Đã bị thay thế bởi memory mới
    OUTDATED   = "outdated"    # Đã hết hiệu lực theo thời gian


class ValidationStatus(str, Enum):
    """Trạng thái kiểm chứng — đặc biệt quan trọng cho Reasoning Memory."""
    UNVERIFIED   = "unverified"
    VERIFIED     = "verified"
    INVALIDATED  = "invalidated"
    NEEDS_REVIEW = "needs_review"


class IntentCategory(str, Enum):
    """Phân loại ý định của truy vấn."""
    TECHNICAL = "technical"
    PERSONAL  = "personal"
    RECALL    = "recall"
    DECISION  = "decision"
    SIMPLE    = "simple"
    UNKNOWN   = "unknown"


# ── Core Data Model ───────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    """
    Một bản ghi memory hoàn chỉnh trong hệ thống TAM.

    Triển khai đầy đủ 8 nhóm thuộc tính từ đặc tả kiến trúc:
    nhận dạng, nội dung, ngữ cảnh, nguồn & độ tin cậy, thời gian,
    hành vi, contrastive evidence, và xung đột.
    """

    # ── 1. Nhận dạng ──
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tier: MemoryTier = MemoryTier.ACTIVE
    memory_type: MemoryType = MemoryType.SEMANTIC

    # ── 2. Nội dung ──
    content: str = ""
    summary: str = ""
    abstraction_group: Optional[str] = None
    # Chỉ dành cho reasoning memory:
    core_strategy: Optional[str] = None
    applicability_scope: Optional[str] = None

    # ── 3. Ngữ cảnh ──
    domain_tags: List[str] = field(default_factory=list)
    intent_tags: List[str] = field(default_factory=list)
    module_tags: List[str] = field(default_factory=list)
    task_family: Optional[str] = None

    # ── 4. Nguồn & Độ tin cậy ──
    source: str = "user"                          # user | system | reflection | distill
    confidence: float = 1.0                       # [0, 1]
    recall_confidence: float = 1.0                # [0, 1] — meta-memory
    validation_status: ValidationStatus = ValidationStatus.UNVERIFIED

    # ── 5. Thời gian ──
    valid_from: Optional[float] = None            # Unix timestamp
    valid_to: Optional[float] = None              # Unix timestamp (None = vô hạn)
    is_current: bool = True
    created_at: float = field(default_factory=time.time)
    last_confirmed_at: Optional[float] = None
    last_used_at: Optional[float] = None

    # ── 6. Hành vi ──
    usage_count: int = 0
    reinforcement_score: float = 0.0
    recency_score: float = 1.0                    # [0, 1] — Score suy giảm theo thời gian
    importance: float = 0.5                       # [0, 1]
    decay_rate_override: Optional[float] = None   # Override decay profile

    # ── 7. Contrastive Evidence ──
    success_case_ids: List[str] = field(default_factory=list)
    failure_case_ids: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    reflection_summary: Optional[str] = None

    # ── 8. Xung đột ──
    conflict_status: ConflictStatus = ConflictStatus.NONE
    superseded_by: Optional[str] = None           # ID of the replacing memory
    supersedes: Optional[str] = None              # ID of the memory it replaces

    # ── Extra Metadata ──
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Embedding (tính toán khi cần) ──
    embedding: Optional[List[float]] = field(default=None, repr=False)

    def touch(self) -> None:
        """Cập nhật thời gian sử dụng cuối và tăng usage count."""
        self.last_used_at = time.time()
        self.usage_count += 1

    def mark_outdated(self, superseded_by_id: Optional[str] = None) -> None:
        """Đánh dấu memory là lỗi thời."""
        self.is_current = False
        self.conflict_status = ConflictStatus.OUTDATED
        self.valid_to = time.time()
        if superseded_by_id:
            self.superseded_by = superseded_by_id
            self.conflict_status = ConflictStatus.SUPERSEDED

    def is_valid_at(self, timestamp: Optional[float] = None) -> bool:
        """Kiểm tra memory có hiệu lực tại thời điểm cho trước."""
        t = timestamp or time.time()
        if self.valid_from and t < self.valid_from:
            return False
        if self.valid_to and t > self.valid_to:
            return False
        return self.is_current

    def to_dict(self) -> Dict[str, Any]:
        """Serialize thành dict để lưu trữ."""
        import json
        d = {}
        for k, v in self.__dict__.items():
            if k == "embedding":
                d[k] = json.dumps(v) if v else None
            elif k == "metadata":
                d[k] = json.dumps(v) if v else "{}"
            elif isinstance(v, (Enum,)):
                d[k] = v.value
            elif isinstance(v, list):
                d[k] = json.dumps(v)
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryRecord":
        """Deserialize từ dict."""
        import json
        kwargs = {}
        for k, v in d.items():
            if k == "embedding":
                kwargs[k] = json.loads(v) if v else None
            elif k == "metadata":
                kwargs[k] = json.loads(v) if isinstance(v, str) else (v or {})
            elif k == "tier":
                kwargs[k] = MemoryTier(v) if v else MemoryTier.ACTIVE
            elif k == "memory_type":
                kwargs[k] = MemoryType(v) if v else MemoryType.SEMANTIC
            elif k == "conflict_status":
                kwargs[k] = ConflictStatus(v) if v else ConflictStatus.NONE
            elif k == "validation_status":
                kwargs[k] = ValidationStatus(v) if v else ValidationStatus.UNVERIFIED
            elif k in ("domain_tags", "intent_tags", "module_tags",
                        "success_case_ids", "failure_case_ids", "failure_reasons"):
                kwargs[k] = json.loads(v) if isinstance(v, str) else (v or [])
            else:
                kwargs[k] = v
        return cls(**kwargs)


# ── Helper structures ─────────────────────────────────────────────────────────

@dataclass
class QueryContext:
    """Ngữ cảnh truy vấn đã qua xử lý."""
    raw_query: str = ""
    rewritten_query: str = ""
    expanded_concepts: List[str] = field(default_factory=list)
    intent: IntentCategory = IntentCategory.UNKNOWN
    domain: str = "general"
    complexity_score: float = 0.5        # [0, 1]
    matts_budget: str = "fast"           # fast | medium | deep
    allowed_memory_types: List[MemoryType] = field(
        default_factory=lambda: list(MemoryType)
    )
    cue_strength: float = 0.0           # Cường độ cue để mở Latent
    embedding: Optional[List[float]] = field(default=None, repr=False)


@dataclass
class RetrievalResult:
    """Kết quả truy xuất từ một tầng, kèm metadata."""
    memory: MemoryRecord
    activation_score: float = 0.0
    source_tier: MemoryTier = MemoryTier.ACTIVE
    inhibited: bool = False
    inhibited_by: Optional[str] = None


@dataclass
class PipelineTrace:
    """Lưu vết toàn bộ một lượt xử lý pipeline — phục vụ logging & evolution."""
    query_context: Optional[QueryContext] = None
    candidates_active: List[RetrievalResult] = field(default_factory=list)
    candidates_latent: List[RetrievalResult] = field(default_factory=list)
    candidates_reasoning: List[RetrievalResult] = field(default_factory=list)
    winners: List[RetrievalResult] = field(default_factory=list)
    inhibited: List[RetrievalResult] = field(default_factory=list)
    final_wm_ids: List[str] = field(default_factory=list)
    matts_branches: List[Dict[str, Any]] = field(default_factory=list)
    response_confidence: float = 1.0
    reflection: Optional[str] = None
    success: Optional[bool] = None
    timestamp: float = field(default_factory=time.time)
