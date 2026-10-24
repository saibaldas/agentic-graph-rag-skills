# notebooks_intermediate_simplified — tier 2 (intermediate)

Tier 2 of 3: [beginner](../notebooks_simplified) → [intermediate](.) → [advanced](../notebooks_advanced_simplified).

You have read a tier-1 notebook and know what a node, a relationship and a skill are. Tier 2 is where
you start **doing it yourself**: fewer hand-holds than tier 1, no heavy agent machinery yet (that is tier 3).
Real Neo4j, same chapters, same skills.

## What makes it "intermediate"

| Feature | Tier 1 | **Tier 2** | Tier 3 |
|---|---|---|---|
| Explanation depth | very high, every term | the *why* and trade-offs | assumes tier 2 |
| Skills per chapter | core ones | the ones the original uses (+ fitting extras) | **all** |
| Skill output | shown | **"anatomy of the output"** per skill | shown + benchmarked |
| Skill inputs | mostly literals | **derived from Cypher** on the graph | derived + written back |
| Orchestration | one call at a time | **hand-written pipeline** function (2–3 skills) | `MiniAgent` + trace graph |
| Practice | asserts | **guided "Your turn" exercises** with reference answers | experiments + pitfalls |
| Trust a skill | — | one `cli.py` call + one benchmark | benchmark of every skill |
| Vibe coding | — | **one prompt + acceptance check** | 2–3 prompts + worked helper |

## Setup

```powershell
docker compose -f docker-compose/docker-compose.yml up -d     # Neo4j at http://localhost:7474 (neo4j/password)
pip install neo4j jupyter ipykernel
jupyter lab notebooks_intermediate_simplified
```

Run **one notebook at a time** (each wipes and reloads the shared graph). Custom credentials:
`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.

## Reading order

`ch1-defining-agentic-ai` → `ch2-architecture-foundations` → `ch3-knowledge-representation` → `ch4-memory` →
`ch5-reasoning-planning` → `ch6-tool-orchestration` → `ch7-self-evolution` → `ch8-optimization`.
(The spike notebooks exist in tier 1 only.)

## Files

* [`int_helpers.py`](int_helpers.py) — reuses the tier-3 helpers and adds `your_turn(...)`.
* [`_src/`](_src) + [`_build.py`](_build.py) — notebook sources and the builder (`python _build.py <src.py>`);
  `python _build.py --banner <tier> <folder>` refreshes the learning-path banner cell in a tier's notebooks.
