[🇺🇸 English](#english) | [🇻🇳 Tiếng Việt](#tiếng-việt)

---

# <a id="english"></a> Tiered Agent Memory (TAM)

A production-ready, 4-tiered cognitive memory architecture for AI Agents featuring self-evolving reasoning, competitive memory selection, and contrastive learning.

## Architecture

| Layer | Purpose | Default Access |
|-------|---------|---------------|
| **Working Memory** | Current context (LLM sees this) | Always |
| **Active Memory** | Stable knowledge, strategies | Default retrieval |
| **Latent Memory** | Dormant episodes, cue-triggered | On strong cue only |
| **Archive** | Cold storage, audit, raw traces | Offline analysis |

**3 Dynamics**: Activation · Decay · Reinforcement

**7 Control Planes**: Query Understanding · Competition · Abstraction · Reasoning · State · Scaling (MaTTS) · Evolution

## Quick Start

```bash
pip install numpy
python demo.py           # Interactive mode
python demo.py --test    # Run test queries
```

## Usage

```python
from tam import TAMPipeline, TAMConfig, MemoryType

pipeline = TAMPipeline(TAMConfig(db_path="agent.db"))

# Add memories
pipeline.add_memory("User likes Python", memory_type=MemoryType.SEMANTIC)
pipeline.add_memory("Debug: check logs first", memory_type=MemoryType.REASONING,
                    core_strategy="Check logs → check config → scale")

# Process query (runs 12-step pipeline)
result = pipeline.process("How to fix timeout bug?")
print(result["query_context"]["intent"])   # "technical"
print(result["wm_summary"])                # Working Memory state
```

## Tests

```bash
python -m pytest tests/ -v
```

## Documentation

Full specification available in `docs/TAM_Specification-en.tex` / `docs/TAM_Specification-en.pdf` (27 pages).

## License

See [LICENSE](LICENSE).

---

# <a id="tiếng-việt"></a> Tiered Agent Memory (TAM)

> **Hệ điều hành trí nhớ nhận thức đa tầng cho AI Agent**

Một kiến trúc bộ nhớ nhận thức 4 tầng sẵn sàng cho sản xuất (production-ready) dành cho AI Agent, nổi bật với khả năng tự tiến hóa suy luận, lựa chọn bộ nhớ cạnh tranh và học tương phản.

## Kiến trúc

| Tầng | Mục đích | Truy xuất mặc định |
|-------|---------|---------------|
| **Working Memory** | Ngữ cảnh hiện tại (LLM có thể thấy) | Luôn luôn |
| **Active Memory** | Tri thức ổn định, chiến lược | Mặc định |
| **Latent Memory** | Ký ức ngủ đông, cần gợi nhớ (cue) | Chỉ khi có cue mạnh |
| **Archive** | Lưu trữ lạnh, kiểm toán, dữ liệu thô | Phân tích ngoại tuyến |

**3 Động lực học**: Kích hoạt (Activation) · Suy giảm (Decay) · Củng cố (Reinforcement)

**7 Lớp điều phối**: Hiểu truy vấn (Query Understanding) · Cạnh tranh (Competition) · Trừu tượng hóa (Abstraction) · Suy luận (Reasoning) · Trạng thái (State) · Mở rộng (Scaling - MaTTS) · Tiến hóa (Evolution)

## Bắt đầu nhanh

```bash
pip install numpy
python demo.py           # Chế độ tương tác
python demo.py --test    # Chạy các truy vấn thử nghiệm
```

## Cách sử dụng

```python
from tam import TAMPipeline, TAMConfig, MemoryType

pipeline = TAMPipeline(TAMConfig(db_path="agent.db"))

# Thêm bộ nhớ
pipeline.add_memory("Người dùng thích Python", memory_type=MemoryType.SEMANTIC)
pipeline.add_memory("Debug: kiểm tra log trước", memory_type=MemoryType.REASONING,
                    core_strategy="Kiểm tra log → kiểm tra config → scale")

# Xử lý truy vấn (chạy pipeline 12 bước)
result = pipeline.process("Làm sao để fix bug timeout?")
print(result["query_context"]["intent"])   # "technical"
print(result["wm_summary"])                # Trạng thái Working Memory
```

## Kiểm thử

```bash
python -m pytest tests/ -v
```

## Tài liệu

Bản đặc tả đầy đủ có tại `docs/TAM_Specification.tex` / `docs/TAM_Specification.pdf` (27 trang).

## Giấy phép

Xem [LICENSE](LICENSE).
