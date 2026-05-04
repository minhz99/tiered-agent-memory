"""
Tầng 1: Working Memory — Không gian suy nghĩ hiện tại.

Working Memory là phần context mà mô hình nhìn thấy trong lượt trả lời hiện tại.
Nó không phải nơi "lưu" mà là nơi **compose**. WM được dựng từ 5 nguồn chính:
  1. Yêu cầu hiện tại
  2. Lịch sử ngắn hạn
  3. Chỉ dẫn hệ thống
  4. Tập nhỏ memory đã qua competition/filtering
  5. Control parameters (recall confidence, strategy hints, MaTTS mode)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from tam.models import MemoryRecord, RetrievalResult, MemoryType
from tam.config import WorkingMemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemorySlot:
    """Một vị trí trong WM — chứa memory đã được chọn và metadata injection."""
    memory: MemoryRecord
    activation_score: float
    role: str = "context"  # context | strategy | abstract | meta


class WorkingMemory:
    """
    Tầng 1 — Working Memory.

    Quản lý không gian context hiện tại với các ràng buộc:
    - Budget cap: giới hạn số memory
    - Competition filter đã áp dụng trước khi inject
    - Hỗn hợp atomic + abstract + reasoning
    - Meta-memory awareness
    """

    def __init__(self, config: Optional[WorkingMemoryConfig] = None):
        self.config = config or WorkingMemoryConfig()
        self._slots: List[WorkingMemorySlot] = []
        self._system_instructions: List[str] = []
        self._conversation_history: List[Dict[str, str]] = []
        self._control_params: Dict[str, Any] = {}

    @property
    def slots(self) -> List[WorkingMemorySlot]:
        return list(self._slots)

    @property
    def memory_count(self) -> int:
        return len(self._slots)

    @property
    def is_full(self) -> bool:
        return len(self._slots) >= self.config.max_memories

    def clear(self) -> None:
        """Xóa toàn bộ WM — gọi ở đầu mỗi lượt xử lý."""
        self._slots.clear()
        self._control_params.clear()

    def set_system_instructions(self, instructions: List[str]) -> None:
        """Thiết lập chỉ dẫn hệ thống."""
        self._system_instructions = instructions

    def add_conversation_turn(self, role: str, content: str) -> None:
        """Thêm một lượt hội thoại vào lịch sử ngắn hạn."""
        self._conversation_history.append({"role": role, "content": content})
        # Giữ tối đa 10 lượt gần nhất
        if len(self._conversation_history) > 10:
            self._conversation_history = self._conversation_history[-10:]

    def set_control_params(self, params: Dict[str, Any]) -> None:
        """Thiết lập control parameters (recall confidence, strategy hints, etc.)."""
        self._control_params.update(params)

    def inject(self, results: List[RetrievalResult]) -> List[WorkingMemorySlot]:
        """
        Inject các memory đã thắng competition vào WM.

        Chiến lược inject:
        1. Ưu tiên hỗn hợp: atomic + abstract + reasoning
        2. Tôn trọng budget cap
        3. Gán role cho từng slot
        """
        injected = []
        remaining_budget = self.config.max_memories - len(self._slots)

        if remaining_budget <= 0:
            logger.warning("WM đã đầy, không thể inject thêm")
            return injected

        # Phân loại candidates theo type
        reasoning = [r for r in results if r.memory.memory_type == MemoryType.REASONING]
        abstract = [r for r in results if r.memory.memory_type == MemoryType.ABSTRACT]
        others = [r for r in results
                  if r.memory.memory_type not in (MemoryType.REASONING, MemoryType.ABSTRACT)]

        # Chiến lược phân bổ: 1 reasoning, 1 abstract, còn lại cho atomic/semantic
        allocation = self._allocate_slots(reasoning, abstract, others, remaining_budget)

        for result, role in allocation:
            slot = WorkingMemorySlot(
                memory=result.memory,
                activation_score=result.activation_score,
                role=role,
            )
            self._slots.append(slot)
            result.memory.touch()
            injected.append(slot)

        logger.info(f"Injected {len(injected)} memories into WM "
                    f"(total: {self.memory_count}/{self.config.max_memories})")
        return injected

    def _allocate_slots(
        self,
        reasoning: List[RetrievalResult],
        abstract: List[RetrievalResult],
        others: List[RetrievalResult],
        budget: int,
    ) -> List[tuple]:
        """Phân bổ slots theo chiến lược atomic + abstract + reasoning mix."""
        allocation = []

        # 1. Tối đa 1 reasoning memory (nếu có)
        if reasoning and budget > 0:
            allocation.append((reasoning[0], "strategy"))
            budget -= 1

        # 2. Tối đa 1 abstract memory (nếu có)
        if abstract and budget > 0:
            allocation.append((abstract[0], "abstract"))
            budget -= 1

        # 3. Phần còn lại cho các loại khác, sắp xếp theo activation score
        others_sorted = sorted(others, key=lambda r: r.activation_score, reverse=True)
        for r in others_sorted[:budget]:
            allocation.append((r, "context"))

        return allocation

    def compose_context(self, current_query: str) -> str:
        """
        Compose context string cuối cùng cho LLM.

        Kết hợp: system instructions + conversation history + injected memories
        + control parameters + current query.
        """
        parts = []

        # System instructions
        if self._system_instructions:
            parts.append("=== SYSTEM ===")
            parts.extend(self._system_instructions)

        # Injected memories
        if self._slots:
            parts.append("\n=== RETRIEVED MEMORIES ===")
            for slot in self._slots:
                m = slot.memory
                prefix = f"[{slot.role.upper()}]"
                parts.append(f"{prefix} ({m.memory_type.value}) {m.content}")

        # Meta-memory warning
        recall_conf = self._control_params.get("recall_confidence", 1.0)
        if recall_conf < 0.5:
            parts.append("\n⚠ Recall confidence thấp — các ký ức trên có thể không chính xác.")
        elif recall_conf < 0.7:
            parts.append("\nℹ Recall confidence trung bình — nên kiểm chứng lại nếu quan trọng.")

        # Strategy hints
        strategy = self._control_params.get("strategy_hint")
        if strategy:
            parts.append(f"\n=== STRATEGY HINT ===\n{strategy}")

        # MaTTS mode
        matts = self._control_params.get("matts_mode", "fast")
        if matts == "deep":
            parts.append("\n=== REASONING MODE: DEEP ===")
            parts.append("Hệ thống đang ở chế độ suy luận sâu. Hãy phân tích kỹ.")

        # Conversation history
        if self._conversation_history:
            parts.append("\n=== CONVERSATION ===")
            for turn in self._conversation_history[-5:]:
                parts.append(f"{turn['role']}: {turn['content']}")

        # Current query
        parts.append(f"\n=== CURRENT QUERY ===\n{current_query}")

        return "\n".join(parts)

    def get_summary(self) -> Dict[str, Any]:
        """Trả về tóm tắt trạng thái WM hiện tại."""
        return {
            "slot_count": self.memory_count,
            "max_slots": self.config.max_memories,
            "slots": [
                {
                    "id": s.memory.id[:8],
                    "type": s.memory.memory_type.value,
                    "role": s.role,
                    "score": round(s.activation_score, 3),
                    "content_preview": s.memory.content[:80],
                }
                for s in self._slots
            ],
            "recall_confidence": self._control_params.get("recall_confidence", 1.0),
            "matts_mode": self._control_params.get("matts_mode", "fast"),
        }
