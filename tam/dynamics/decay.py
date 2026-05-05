import time
import logging
from typing import List, Dict, Any
from tam.models import MemoryRecord, MemoryTier
from tam.config import DecayProfiles

logger = logging.getLogger(__name__)

class DecayEngine:
    """
    Điều khiển quá trình suy giảm (decay) của ký ức theo thời gian.
    Giúp hệ thống tự động lọc bỏ hoặc đẩy các ký ức ít dùng vào tầng sâu hơn.
    """
    def __init__(self, profiles: DecayProfiles):
        self.profiles = profiles

    def apply_decay(self, memories: List[MemoryRecord], current_time: float = None) -> Dict[str, List[str]]:
        """
        Tính toán lại recency_score và quyết định việc di chuyển tầng (Tier Demotion).
        
        Returns:
            Dict chứa danh sách IDs bị hạ tầng: {"demote_to_latent": [], "demote_to_archive": []}
        """
        if current_time is None:
            current_time = time.time()
            
        results = {
            "demote_to_latent": [],
            "demote_to_archive": []
        }

        for memory in memories:
            # 1. Tính khoảng thời gian từ lần cuối truy cập (hoặc từ lúc tạo)
            last_time = memory.last_accessed_at or memory.created_at
            delta_days = (current_time - last_time) / (24 * 3600)
            
            if delta_days <= 0:
                continue

            # 2. Lấy hệ số decay tương ứng với loại memory
            decay_rate = getattr(self.profiles, memory.memory_type.value, 0.02)
            
            # 3. Cập nhật recency_score (Exponential Decay)
            # score = initial * e^(-rate * time)
            memory.recency_score = memory.recency_score * (1.0 - decay_rate * delta_days)
            memory.recency_score = max(0.0, min(1.0, memory.recency_score))

            # 4. Quyết định hạ tầng (Demotion logic)
            if memory.tier == MemoryTier.ACTIVE and memory.recency_score < 0.3:
                memory.tier = MemoryTier.LATENT
                results["demote_to_latent"].append(memory.id)
                logger.info(f"Demoting {memory.id[:8]} to LATENT (Score: {memory.recency_score:.2f})")
                
            elif memory.tier == MemoryTier.LATENT and memory.recency_score < 0.1:
                memory.tier = MemoryTier.ARCHIVE
                results["demote_to_archive"].append(memory.id)
                logger.info(f"Archiving {memory.id[:8]} (Score: {memory.recency_score:.2f})")

        return results

    def refresh(self, memory: MemoryRecord):
        """Reset recency_score khi memory được sử dụng thành công."""
        memory.recency_score = 1.0
        memory.last_accessed_at = time.time()
        logger.debug(f"Memory {memory.id[:8]} refreshed.")
