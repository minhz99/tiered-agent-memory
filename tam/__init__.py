"""
Tiered Agent Memory (TAM) — Kiến trúc bộ nhớ nhận thức đa tầng cho AI Agent.

TAM không chỉ là một kho chứa dữ liệu (RAG truyền thống), mà là một hệ thống
mô phỏng nhận thức. Một Agent dùng TAM không chỉ biết nhớ, mà còn biết bỏ qua
(lọc nhiễu), biết nghi ngờ (khi có mâu thuẫn), biết khái quát hóa (nhìn bức
tranh lớn) và biết tự học sau những lần vấp ngã.

Architecture:
    4 Tiers:  Working → Active → Latent → Archive
    3 Dynamics: Activation, Decay, Reinforcement
    7 Control Planes: Query, Competition, Abstraction, Reasoning, State, Scaling, Evolution
"""

__version__ = "0.1.0"
__author__ = "TAM Research Team"

from tam.models import MemoryRecord, MemoryType, MemoryTier, ConflictStatus
from tam.pipeline import TAMPipeline
from tam.config import TAMConfig

__all__ = [
    "MemoryRecord",
    "MemoryType",
    "MemoryTier",
    "ConflictStatus",
    "TAMPipeline",
    "TAMConfig",
]
