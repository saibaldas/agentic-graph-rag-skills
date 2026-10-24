# notebooks_simplified — tier 1 (beginner) on a real Neo4j graph

Tier 1 of 3: [beginner](.) → [intermediate](../notebooks_intermediate_simplified) → [advanced](../notebooks_advanced_simplified).
One simplified notebook per original notebook in [`../notebooks`](../notebooks). Same chapters,
same skills, same asserts — but with much more explanation, and the data lives in a **real
Neo4j graph database** (queried with Cypher) instead of AWS mocks.

## 1. Start Neo4j

```powershell
docker compose -f docker-compose/docker-compose.yml up -d
```

Browser UI: <http://localhost:7474> (login `neo4j` / `password`). Wait ~30 s for first start.
Different credentials? Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` before launching Jupyter.

## 2. Install and open

```powershell
pip install neo4j jupyter ipykernel
jupyter lab notebooks_simplified
```

Run cells top to bottom. Each notebook wipes and reloads the graph at the start, so run one
notebook at a time.

## Reading order

| Notebook | Original | Topic |
|---|---|---|
| `ch1-defining-agentic-ai` | `ch1-…` | Why vector-RAG agents stall; agency; constraint triangle |
| `ch2-architecture-foundations` | `ch2-…` | Architecture & readiness |
| `ch3-knowledge-representation` | `ch3-…` | How to model knowledge in graphs |
| `ch4-memory` | `ch4-…` | Agent memory (bi-temporal, hierarchical, consolidation) |
| `ch5-reasoning-planning` | `ch5-…` | Planning and validating investigations |
| `ch6-tool-orchestration` | `ch6-…` | Choosing and governing tools |
| `ch7-self-evolution` | `ch7-…` | Agents that improve themselves |
| `ch8-optimization` | `ch8-…` | Cost, latency, access control |
| `spike-*` | `spike-*` | Small focused seam checks |

## Files

* [`graph_helpers.py`](graph_helpers.py) — connection, `run()`/`show()`, the shared DevOps incident graph, skill loader.
* [`_src/`](_src) — percent-format sources the `.ipynb` files were generated from; [`_build.py`](_build.py)
  rebuilds and executes them (`python notebooks_simplified/_build.py notebooks_simplified/_src/<name>.py`;
  needs `pip install nbformat nbclient`).
