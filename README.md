# Tiered Agent Memory (TAM)

> **Hệ điều hành trí nhớ nhận thức đa tầng cho AI Agent**

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

Full specification available in `docs/TAM_Specification.tex` / `docs/TAM_Specification.pdf` (27 pages).

## License

See [LICENSE](LICENSE).
