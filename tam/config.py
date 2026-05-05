"""
TAM Configuration — Tập trung tất cả siêu tham số và cấu hình hệ thống.

Mọi hằng số có thể điều chỉnh được (thresholds, weights, capacities) đều được
định nghĩa tại đây để dễ thay đổi khi thí nghiệm hoặc triển khai.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ActivationWeights:
    """Trọng số cho hàm tính điểm kích hoạt (Activation Score)."""
    similarity: float = 0.35
    importance: float = 0.20
    recency: float = 0.15
    confidence: float = 0.15
    strategy_match: float = 0.15


@dataclass
class DecayProfiles:
    """Hệ số suy giảm (decay rate) theo loại memory — đơn vị: phần trăm/ngày.

    Giá trị càng cao thì memory loại đó quên càng nhanh.
    """
    episodic: float = 0.10       # Quên nhanh
    semantic: float = 0.01       # Quên chậm
    reasoning: float = 0.02      # Chậm nhưng nhạy với phản ví dụ
    style: float = 0.05          # Trung bình
    system: float = 0.005        # Gần như không quên
    abstract: float = 0.02       # Chậm


@dataclass
class CompetitionConfig:
    """Cấu hình cho Competition Plane."""
    similarity_threshold: float = 0.85   # Ngưỡng gần trùng để trigger inhibition
    diversity_lambda: float = 0.5        # Lambda cho MMR (Maximal Marginal Relevance)
    max_candidates: int = 20             # Số candidate tối đa trước competition
    max_winners: int = 5                 # Số memory thắng competition


@dataclass
class MaTTSConfig:
    """Cấu hình cho Memory-aware Test-Time Scaling."""
    fast_threshold: float = 0.3          # Dưới ngưỡng này → fast path
    deep_threshold: float = 0.7          # Trên ngưỡng này → deep reasoning
    max_branches: int = 3                # Số nhánh suy luận tối đa
    branch_depth: int = 2                # Độ sâu mỗi nhánh


@dataclass
class WorkingMemoryConfig:
    """Cấu hình cho Working Memory."""
    budget_ratio: float = 0.3            # Tỷ lệ tối đa của WM dành cho memories
    max_memories: int = 7                # Số memory tối đa trong WM
    max_tokens: int = 2048               # Budget token tối đa cho memory section


@dataclass
class EvolutionConfig:
    """Cấu hình cho Background Worker."""
    synthesis_interval_hours: float = 24.0
    min_traces_for_distill: int = 3      # Cần ≥3 traces mới distill reasoning
    validation_threshold: float = 0.6     # Ngưỡng confidence để promote
    max_logs_per_batch: int = 100


@dataclass
class TAMConfig:
    """Cấu hình tổng thể của hệ thống TAM."""

    # --- Storage ---
    db_path: str = "tam_memory.db"

    # --- Sub-configs ---
    activation_weights: ActivationWeights = field(default_factory=ActivationWeights)
    decay_profiles: DecayProfiles = field(default_factory=DecayProfiles)
    competition: CompetitionConfig = field(default_factory=CompetitionConfig)
    matts: MaTTSConfig = field(default_factory=MaTTSConfig)
    working_memory: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)

    # --- Retrieval ---
    latent_cue_threshold: float = 0.4    # Ngưỡng cue để mở Latent
    recall_confidence_warn: float = 0.5  # Dưới ngưỡng này → cảnh báo bất định

    # --- Style ---
    style_user_weight: float = 0.7
    style_context_weight: float = 0.3

    # --- Embedding ---
    embedding_dim: int = 384             # Dimensionality of embeddings (E5-small is 384)
    use_transformer_embeddings: bool = True   # Default to high-quality embeddings
    embedding_model_name: str = "intfloat/multilingual-e5-small"

    # --- Intent categories ---
    intent_categories: Dict[str, list] = field(default_factory=lambda: {
        "technical": ["code", "bug", "error", "algorithm", "debug", "deploy", "API",
                      "function", "class", "module", "database", "SQL", "server"],
        "personal":  ["hobby", "prefer", "like", "favorite", "habit", "feel",
                      "gym", "diet", "health", "fitness"],
        "recall":    ["remember", "last time", "before", "previously", "earlier",
                      "nhớ", "trước", "lần trước", "hồi đó"],
        "decision":  ["should", "choose", "compare", "best", "recommend",
                      "nên", "chọn", "so sánh", "tốt nhất"],
        "simple":    ["what is", "define", "là gì", "nghĩa là"],
    })
