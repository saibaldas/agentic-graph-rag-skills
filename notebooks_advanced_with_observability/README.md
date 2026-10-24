# notebooks_advanced_with_observability — the `notebooks_advanced` notebooks, now observable

Same nine notebooks as [`notebooks_advanced`](../notebooks_advanced) (which is untouched), plus **observability** you
switch with one line in `.env`:

```
OBSERVABILITY_BACKEND=phoenix     # or: deepeval | both | none
```

| Backend | What it is | Docker stack (all files live in [`../docker-compose/`](../docker-compose)) | You see |
|---|---|---|---|
| **`phoenix`** (Arize Phoenix) | Open-source LLM tracing + evals UI | `docker-compose.phoenix.yml` = Phoenix + its own Neo4j | `http://localhost:6006` — a trace for every LLM / embedding call, agent run, tool call, judge verdict and eval case |
| **`deepeval`** | Open-source LLM evaluation library (GEval, AnswerRelevancy …) | `docker-compose.deepeval.yml` = DeepEval + its own Neo4j | `http://localhost:8765` — local dashboard: LLM calls, judge verdicts, eval runs, and **DeepEval scores next to the notebook's own judge scores** |
| `both` | both stacks; the notebooks use the Phoenix Neo4j | start both files | both UIs |
| `none` | identical behaviour to `notebooks_advanced`; uses the original Neo4j | `docker-compose.yml` | nothing |

Both are free and local; you only pay OpenAI for the model calls (DeepEval's metric calls add roughly a cent per notebook, `gpt-4o-mini`).

> **Honest note on DeepEval.** DeepEval is a *library*; its polished hosted dashboard is the paid Confident AI cloud.
> So the DeepEval stack builds a small container that wraps the open-source library in an HTTP service plus a simple local
> dashboard ([`../docker-compose/deepeval/server.py`](../docker-compose/deepeval/server.py)). The metrics are real DeepEval metrics; the dashboard is minimal, not Confident AI.
> Phoenix is the richer UI for tracing; DeepEval's strength here is the second-opinion metrics.

## The docker-compose folder (no clashes, everything named)

Every stack has **its own Neo4j on its own ports and its own data folders**, so any combination can run at once.

| File | Compose project (name in Docker Desktop) | Containers | Ports | Data on disk |
|---|---|---|---|---|
| `docker-compose.yml` | `agentic-graph-rag-neo4j` | `neo4j-5.26-ent-agentic-graph-rag-skills` | Neo4j 7687 / 7474 | `docker-compose/volumes/justneo4j/{data,logs,import,plugins}` |
| `docker-compose.phoenix.yml` | `agentic-graph-rag-obs-phoenix` | `obs-phoenix-neo4j-…`, `obs-phoenix-ui-…` | Neo4j **7688** / 7475, Phoenix 6006 (+4317) | `docker-compose/volumes/phoenix/{data,logs,import,plugins}`, volume `agentic-graph-rag-obs-phoenix-data` |
| `docker-compose.deepeval.yml` | `agentic-graph-rag-obs-deepeval` | `obs-deepeval-neo4j-…`, `obs-deepeval-ui-…` | Neo4j **7689** / 7476, DeepEval 8765 | `docker-compose/volumes/deepeval/{data,logs,import,plugins}`, volume `agentic-graph-rag-obs-deepeval-data` |

**Pinned versions** (nothing floats to `latest`): Neo4j `neo4j:5.26.32-enterprise`, Phoenix `arizephoenix/phoenix:version-20.26.0`,
DeepEval `deepeval==4.3.1` on `python:3.11.17-slim` (built into image `agentic-graph-rag/deepeval-local:4.3.1`). To upgrade, change the
tag in the compose file on purpose.

**Which Neo4j do the notebooks use?** The one that matches `OBSERVABILITY_BACKEND` — the helper sets `NEO4J_URI` for you
(`phoenix`/`both` → `bolt://localhost:7688`, `deepeval` → `7689`, `none` → `7687`). Set `NEO4J_URI` in `.env` to override.
The three Neo4j databases are separate, so the notebooks' "wipe the graph on start" only affects the one in use.

## Setup (run commands from the repo root)
1. `pip install -r requirements.txt` (adds `openinference-instrumentation-openai` and OpenTelemetry exporters).
2. Copy [`.env.example`](.env.example) to `notebooks_advanced_with_observability/.env`; set `OPENAI_API_KEY` and `OBSERVABILITY_BACKEND`.
3. Start the stack you chose:
   ```
   docker compose -f docker-compose/docker-compose.phoenix.yml up -d
   docker compose --env-file notebooks_advanced_with_observability/.env -f docker-compose/docker-compose.deepeval.yml up -d --build
   ```
   (`--env-file` passes your OpenAI key to the DeepEval container. The first start downloads the Neo4j APOC/GDS plugins, so give Neo4j about a minute.)
4. Open a notebook (kernel working directory = this folder), run it, then open the UI. **Restart the kernel after changing
   `OBSERVABILITY_BACKEND`.** If the chosen backend's container is not running, the notebook prints one warning and carries
   on — observability never breaks a run.
5. Stop: `docker compose -f docker-compose/docker-compose.phoenix.yml down` (add `-v` to delete the stored traces); same for deepeval.

## What gets recorded
Nothing in the notebooks' logic changed. `real_helpers.py` gained hooks and each notebook gained one `h.obs_start(...)`
line after its imports and a final "Observability" section calling `h.obs_finish()`.

| Event | Phoenix | DeepEval container |
|---|---|---|
| every chat / embedding call | auto-instrumented span (prompt, answer, tokens, latency) | logged call with tokens and latency |
| `h.ToolAgent.run` | **AGENT** span, one **TOOL** span per tool call, LLM calls nested inside | LLM calls with their tool-call names |
| `h.judge_rubric` / `h.judge_pairwise` | **EVALUATOR** spans with `eval.mean`, `eval.scores`, position consistency | verdicts listed |
| `h.run_eval` | one span per case with `eval.score`, `eval.passed`, `eval.version` | eval run listed |
| end of notebook | traces flushed, link printed | **DeepEval re-scores a sample** (default 6) of the judged answers with GEval *Correctness* and *AnswerRelevancy*, printed beside the notebook's own judge mean |

How to learn from it: in Phoenix open the agent span of a failing case and read the nested LLM/tool spans; sort by latency
or token count to find cost hot-spots; in DeepEval's table look for rows where the notebook judge says 5/5 but DeepEval's
correctness is low (or the reverse) — those are the verdicts to read by hand, the practical meaning of
"LLM-as-a-judge needs its own evaluation".

## Cost
Same as [`notebooks_advanced`](../notebooks_advanced/README.md#estimated-openai-cost-measured-24-october-2026) (about $1.14
for all nine). Phoenix adds nothing. DeepEval adds about 12 `gpt-4o-mini` calls per notebook (≈ $0.002–0.01).

## Housekeeping
- Dev tools: `_build.py` (rebuild `.ipynb` from `_src/`), `_run.py <stem> [--save]` (run for real, serialized by a lock).
- `.env` and `docker-compose/volumes/` are git-ignored.
