"""
TAM Demo — Interactive CLI demo cho Tiered Agent Memory.

Chạy: python demo.py
"""
import os
import sys
import logging
import tempfile

from tam import TAMPipeline, TAMConfig, MemoryType, MemoryTier

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tam.demo")


def seed_memories(pipeline: TAMPipeline):
    """Nạp dữ liệu mẫu vào hệ thống để demo."""
    print("\n🌱 Đang nạp dữ liệu mẫu...")

    # Semantic memories
    pipeline.add_memory(
        "User thích lập trình Python và hay dùng FastAPI cho backend.",
        memory_type=MemoryType.SEMANTIC,
        domain_tags=["technical", "python"],
        intent_tags=["technical"],
        importance=0.8,
    )
    pipeline.add_memory(
        "User quan tâm đến Machine Learning, đặc biệt là NLP và LLM.",
        memory_type=MemoryType.SEMANTIC,
        domain_tags=["technical", "ml"],
        intent_tags=["technical"],
        importance=0.9,
    )
    pipeline.add_memory(
        "User thích tập gym buổi sáng và đang theo chế độ ăn low-carb.",
        memory_type=MemoryType.SEMANTIC,
        domain_tags=["personal", "health"],
        intent_tags=["personal"],
        importance=0.6,
    )

    # System memories
    pipeline.add_memory(
        "Hệ thống deploy trên AWS ECS với PostgreSQL RDS.",
        memory_type=MemoryType.SYSTEM,
        domain_tags=["technical", "infrastructure"],
        importance=0.7,
    )

    # Style memory
    pipeline.add_memory(
        "User thích câu trả lời ngắn gọn, có code example, tránh dài dòng.",
        memory_type=MemoryType.STYLE,
        domain_tags=["style"],
        importance=0.7,
    )

    # Reasoning memory
    pipeline.add_memory(
        "Khi gặp lỗi timeout trong API, nên kiểm tra connection pool trước, "
        "sau đó kiểm tra N+1 query, cuối cùng mới xem xét scale horizontal.",
        memory_type=MemoryType.REASONING,
        domain_tags=["technical", "debugging"],
        core_strategy="Debug timeout: pool → N+1 → scale",
        applicability_scope="API performance issues",
        importance=0.85,
    )

    # Episodic memories (Latent)
    pipeline.add_memory(
        "Tuần trước user gặp bug memory leak trong service notification, "
        "nguyên nhân là không close connection sau khi gửi webhook.",
        memory_type=MemoryType.EPISODIC,
        tier=MemoryTier.LATENT,
        domain_tags=["technical", "debugging"],
        importance=0.5,
    )
    pipeline.add_memory(
        "Tháng trước user hỏi về cách tối ưu Elasticsearch query, "
        "đã giải quyết bằng cách thêm filter context vào bool query.",
        memory_type=MemoryType.EPISODIC,
        tier=MemoryTier.LATENT,
        domain_tags=["technical", "database"],
        importance=0.4,
    )

    print(f"✅ Đã nạp dữ liệu mẫu!")
    stats = pipeline.get_stats()
    print(f"   Active: {stats['active_memories']} | "
          f"Latent: {stats['latent_memories']} | "
          f"Reasoning: {stats['reasoning_strategies']}")


def print_result(result: dict):
    """In kết quả pipeline đẹp."""
    qc = result["query_context"]
    ret = result["retrieval"]
    wm = result["wm_summary"]

    print("\n" + "═" * 60)
    print("  📊 KẾT QUẢ PIPELINE")
    print("═" * 60)

    print(f"\n🔍 Query Understanding:")
    print(f"   Intent:     {qc['intent']}")
    print(f"   Complexity: {qc['complexity']}")
    print(f"   Budget:     {qc['matts_budget']}")
    print(f"   Cue:        {qc['cue_strength']}")
    if qc["expanded_concepts"]:
        print(f"   Concepts:   {', '.join(qc['expanded_concepts'][:5])}")

    print(f"\n📦 Retrieval:")
    print(f"   Active candidates:    {ret['active_candidates']}")
    print(f"   Reasoning candidates: {ret['reasoning_candidates']}")
    print(f"   Latent activated:     {'✅' if ret['latent_activated'] else '❌'}")
    print(f"   Latent candidates:    {ret['latent_candidates']}")
    print(f"   Winners:              {ret['winners']}")
    print(f"   Inhibited:            {ret['inhibited']}")

    print(f"\n🧠 Working Memory ({wm['slot_count']}/{wm['max_slots']} slots):")
    for slot in wm["slots"]:
        print(f"   [{slot['role']:>8}] ({slot['type']:>9}) score={slot['score']:.3f} │ {slot['content_preview'][:50]}")

    print(f"\n🎯 Response confidence: {result['response_confidence']}")
    if result["response_confidence"] < 0.5:
        print("   ⚠️  CẢNH BÁO: Recall confidence thấp!")

    if len(result["branches"]) > 1:
        print(f"\n🌿 MaTTS Branches: {len(result['branches'])}")
        for b in result["branches"]:
            strategy = b.get("strategy", "none")
            if strategy and len(strategy) > 60:
                strategy = strategy[:60] + "..."
            print(f"   Branch {b['branch_id']}: {b['mode']} │ {strategy or '—'}")

    print("═" * 60)


def interactive_mode(pipeline: TAMPipeline):
    """Chế độ tương tác."""
    print("\n" + "═" * 60)
    print("  🧠 TAM — Tiered Agent Memory Demo")
    print("  Hệ điều hành trí nhớ nhận thức đa tầng cho AI Agent")
    print("═" * 60)
    print("\nLệnh đặc biệt:")
    print("  /stats     — Xem thống kê hệ thống")
    print("  /evolve    — Chạy evolution worker")
    print("  /add <text>— Thêm memory mới")
    print("  /quit      — Thoát")
    print()

    while True:
        try:
            query = input("❓ Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query == "/quit":
            break
        if query == "/stats":
            stats = pipeline.get_stats()
            print(f"\n📈 Thống kê:")
            for k, v in stats.items():
                print(f"   {k}: {v}")
            continue
        if query == "/evolve":
            report = pipeline.run_evolution()
            print(f"\n{report}")
            continue
        if query.startswith("/add "):
            text = query[5:].strip()
            if text:
                pipeline.add_memory(text)
                print(f"✅ Đã thêm memory: '{text[:50]}...'")
            continue

        result = pipeline.process(query)
        print_result(result)


def main():
    # Tạo database tạm cho demo
    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tam_demo.db"
    )

    config = TAMConfig(
        db_path=db_path,
        use_transformer_embeddings=False,  # Dùng simple embedding cho demo
    )

    pipeline = TAMPipeline(config)

    try:
        # Nạp dữ liệu mẫu nếu database mới
        if pipeline.active.count() == 0:
            seed_memories(pipeline)

        # Chế độ interactive hoặc chạy test queries
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            test_queries = [
                "Làm sao để fix bug timeout trong API?",
                "Nhớ lại lần trước tôi gặp memory leak không?",
                "Tôi nên chọn Redis hay Memcached cho caching?",
                "Hôm nay tập gym bài gì?",
            ]
            for q in test_queries:
                print(f"\n{'─'*60}")
                print(f"❓ Query: {q}")
                result = pipeline.process(q)
                print_result(result)
        else:
            interactive_mode(pipeline)
    finally:
        pipeline.close()
        print("\n👋 Tạm biệt!")


if __name__ == "__main__":
    main()
