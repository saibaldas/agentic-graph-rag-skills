# notebooks_advanced_simplified — tier 3 (advanced): skills, agents and vibe coding

Tier 3 of 3: [beginner](../notebooks_simplified) → [intermediate](../notebooks_intermediate_simplified) → [advanced](.).
Next step after [`../notebooks_intermediate_simplified`](../notebooks_intermediate_simplified). Same real Neo4j
(`docker compose -f docker-compose/docker-compose.yml up -d`), but closer to the original
[`../notebooks`](../notebooks) and much heavier on the repo's [`skills/`](../skills).

## What tier 3 (advanced) adds beyond tier 2 (intermediate)

| Tier 3 feature | Where |
|---|---|
| **Every skill of the chapter** is loaded and used (not just the ones the original used) | all notebooks |
| Skill inputs come from **Cypher**; skill outputs are **written back** as nodes/edges | all notebooks |
| Non-trivial Cypher (variable-length paths, `collect`, `OPTIONAL MATCH`, cycle checks), each clause explained | all notebooks |
| A **mini agent** (`MiniAgent`: plan → act → observe) whose tools are skills + Cypher; the trace is stored and queried in Neo4j | ch1–ch8 |
| **Skill CLI & benchmark** section: run each skill's `cli.py` and self-check | ch1–ch8 |
| **🎯 Vibe-coding corner**: copy-paste prompts, a worked AI-style helper plus its test | ch1–ch8 |
| Experiments ("change X, predict Y") and common pitfalls | ch1–ch8 |

## Reading order

| # | Notebook | Skills used |
|---|---|---|
| 1 | `ch1-defining-agentic-ai` | 5 (crisis) |
| 2 | `ch2-architecture-foundations` | 3 |
| 3 | `ch3-knowledge-representation` | 8 |
| 4 | `ch4-memory` | 8 |
| 5 | `ch5-reasoning-planning` | 6 |
| 6 | `ch6-tool-orchestration` | 9 |
| 7 | `ch7-self-evolution` | 7 |
| 8 | `ch8-optimization` | 5 |
| 9 | `ch9-vibe-coding-capstone` | all 51: skills-as-a-graph, run every benchmark, an 8-chapter agent, build a new skill the repo way |

## Setup

```powershell
pip install neo4j jupyter ipykernel      # nbformat nbclient only to rebuild
jupyter lab notebooks_advanced_simplified
```

Every notebook resets the shared graph at its start, so run **one notebook at a time**.
Custom credentials: set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.

## Files

* [`adv_helpers.py`](adv_helpers.py) — `check`, `skill_catalog`, `run_benchmark`, `AgentTrace`, `MiniAgent`; re-exports the tier-1 helpers.
* [`_src/`](_src) + [`_build.py`](_build.py) — notebook sources and the builder/executor.

## Honest limits

No LLM or API key is used: planners are deterministic, "semantic" retrieval is a keyword/trigram
stand-in for embeddings, and scores from skills are heuristics. Community Neo4j cannot enforce
role-based GRANT/DENY, so access control is simulated in Cypher. Swap those seams for real
models when you move beyond learning.
