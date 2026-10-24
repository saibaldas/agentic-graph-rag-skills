# docker-compose — the Docker stacks for this repo

Three independent stacks. Each has **its own Neo4j on its own ports and its own data folders under [`volumes/`](volumes)**,
so any combination can run at the same time without clashing. Run every command **from the repo root**.

| Stack | File | Compose project (shown in Docker Desktop) | Containers | Ports | Data on disk |
|---|---|---|---|---|---|
| **Just Neo4j** | [`docker-compose.yml`](docker-compose.yml) | `agentic-graph-rag-neo4j` | `neo4j-5.26-ent-agentic-graph-rag-skills` | Neo4j bolt **7687**, browser 7474 (and 7473) | `volumes/justneo4j/` |
| **Phoenix** | [`docker-compose.phoenix.yml`](docker-compose.phoenix.yml) | `agentic-graph-rag-obs-phoenix` | `obs-phoenix-neo4j-…`, `obs-phoenix-ui-…` | Neo4j bolt **7688**, browser 7475 · Phoenix UI **6006** (+ OTLP gRPC 4317) | `volumes/phoenix/` + Docker volume `agentic-graph-rag-obs-phoenix-data` |
| **DeepEval** | [`docker-compose.deepeval.yml`](docker-compose.deepeval.yml) | `agentic-graph-rag-obs-deepeval` | `obs-deepeval-neo4j-…`, `obs-deepeval-ui-…` | Neo4j bolt **7689**, browser 7476 · DeepEval dashboard **8765** | `volumes/deepeval/` + Docker volume `agentic-graph-rag-obs-deepeval-data` |

Neo4j login for all three: `neo4j` / `password` (Enterprise edition, needed for the real RBAC in `notebooks_advanced*`).

## Which stack for which notebooks

| Notebooks | Start |
|---|---|
| `notebooks_simplified`, `notebooks_intermediate_simplified`, `notebooks_advanced_simplified`, `notebooks_advanced` | **Just Neo4j** (7687, the default) |
| `notebooks_advanced_with_observability` with `OBSERVABILITY_BACKEND=phoenix` (or `both`) | **Phoenix** stack (notebooks use Neo4j on 7688) |
| `notebooks_advanced_with_observability` with `OBSERVABILITY_BACKEND=deepeval` | **DeepEval** stack (notebooks use Neo4j on 7689) |
| `notebooks_advanced_with_observability` with `OBSERVABILITY_BACKEND=none` | **Just Neo4j** |

The notebooks pick the matching Neo4j from `OBSERVABILITY_BACKEND` automatically; set `NEO4J_URI` in their `.env` to override.
See [`../notebooks_advanced_with_observability/README.md`](../notebooks_advanced_with_observability/README.md).

## Commands

```powershell
# Just Neo4j
docker compose -f docker-compose/docker-compose.yml up -d

# Phoenix + its own Neo4j
docker compose -f docker-compose/docker-compose.phoenix.yml up -d

# DeepEval + its own Neo4j  (--env-file hands your OPENAI_API_KEY to the DeepEval container; --build builds ./deepeval)
docker compose --env-file notebooks_advanced_with_observability/.env -f docker-compose/docker-compose.deepeval.yml up -d --build

# Stop a stack (data is kept)
docker compose -f docker-compose/docker-compose.phoenix.yml down
# ...and also delete its Docker volume (stored Phoenix traces / DeepEval history)
docker compose -f docker-compose/docker-compose.phoenix.yml down -v
```

The first start of a Neo4j container downloads the APOC and Graph Data Science plugins into its `volumes/<stack>/plugins`
folder, so allow about a minute before connecting. Check readiness with
`docker exec <neo4j-container> cypher-shell -u neo4j -p password "RETURN 1"`.

## Folder layout

```
docker-compose/
├── docker-compose.yml            Just Neo4j
├── docker-compose.phoenix.yml    Phoenix + Neo4j
├── docker-compose.deepeval.yml   DeepEval + Neo4j
├── deepeval/                     build context for the DeepEval container (Dockerfile, server.py)
└── volumes/                      git-ignored runtime data, one folder per stack
    ├── justneo4j/{data,logs,import,plugins}
    ├── phoenix/{data,logs,import,plugins}
    └── deepeval/{data,logs,import,plugins}
```

To wipe one stack's database completely: stop it, then delete its `volumes/<stack>/data` folder.

## Pinned image versions
Nothing floats to `latest`; upgrade by editing the tag on purpose.

| Image | Pinned to |
|---|---|
| Neo4j | `neo4j:5.26.32-enterprise` (all three stacks) |
| Phoenix | `arizephoenix/phoenix:version-20.26.0` |
| DeepEval | `deepeval==4.3.1` on `python:3.11.17-slim`, built as `agentic-graph-rag/deepeval-local:4.3.1` |

## DeepEval, honestly
DeepEval is a Python library; its polished hosted dashboard is the paid Confident AI cloud. The DeepEval stack therefore
builds a small container (`deepeval/server.py`) that wraps the open-source library (GEval *Correctness*, *AnswerRelevancy*)
in an HTTP service with a minimal local dashboard. Nothing is sent to Confident AI. DeepEval's metric calls use your OpenAI
key (`gpt-4o-mini` by default, `DEEPEVAL_MODEL` to change).

## Ports quick reference
7687/7474/7473 · 7688/7475 · 7689/7476 (Neo4j) · 6006, 4317 (Phoenix) · 8765 (DeepEval). If one is taken on your machine,
change the left-hand side of the `ports:` entry in the compose file (and `NEO4J_URI` for Neo4j).
