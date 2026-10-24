# notebooks_advanced — the "real" tier: real LLM, real embeddings, real RBAC

Tier 4 (after beginner → intermediate → advanced-simplified). These notebooks remove the "honest limits" of
[`notebooks_advanced_simplified`](../notebooks_advanced_simplified):

| Limit in the simplified tiers | Here |
|---|---|
| No LLM; deterministic planners | **OpenAI `gpt-4o`** for planning, extraction, judging; real function-calling agent (`h.ToolAgent`) |
| Keyword/trigram "semantic" search | **`text-embedding-3-large`** (3072 dims) + a **Neo4j vector index** |
| Heuristic skill scores only | Heuristics **and** LLM-as-judge side by side, with disagreements analysed |
| Access control simulated in Cypher | **Real Neo4j RBAC** (Enterprise): `CREATE USER/ROLE`, `GRANT`/`DENY`, queries run as another user |

There is **no offline fallback**: without a key the first cell fails with instructions.

## Setup
1. `docker compose -f docker-compose/docker-compose.yml up -d` (Neo4j **Enterprise** 5.26; needed for RBAC).
2. `pip install -r requirements.txt` (adds `openai`, `python-dotenv`).
3. Copy [`.env.example`](.env.example) to `.env` in this folder and set `OPENAI_API_KEY` (`.env` is git-ignored).
4. Open a notebook; the kernel's working directory must be this folder.

Defaults: `gpt-4o`, `text-embedding-3-large`. Override with `OPENAI_LLM_MODEL`, `OPENAI_EMBED_MODEL`, `OPENAI_EMBED_DIM`.

## Notebooks
ch1–ch8 mirror the chapters; `ch9-vibe-coding-capstone` embeds the repo's skills, runs an agent over them and lets an LLM
generate and self-repair a new skill (in a scratch folder, never in `skills/`). **Every notebook has been executed for
real** against OpenAI and Neo4j (0 error cells, 55–82 checks passing each). A full pass over all nine costs
**about $1.14** (measured), roughly $0.08–$0.21 per notebook; details [below](#estimated-openai-cost-measured-24-october-2026).

## Evaluation and LLM-as-a-Judge (the thread through every notebook)
Each notebook ends with an **"Evaluation and LLM-as-a-Judge — how you know it works"** section. The idea: an agent that
"seems to work" is not evidence. You need a measuring stick, and you need to measure the stick too.

1. **Cheap deterministic checks first** (schema, Cypher, token overlap) — fast, but blind to meaning.
2. **LLM-as-a-Judge** — `h.judge_rubric` (reason first, then score 1–5), `h.judge_pairwise` (A/B with the order swapped; an
   order-dependent verdict counts as a tie).
3. **Calibrate the judge against human labels** — `h.agreement` gives exact/within-1 agreement, Cohen's kappa, Spearman.
4. **Probe judge biases** — position bias, verbosity padding, self-consistency (`h.judge_repeat`), and a "fluent but false"
   sabotage control that must score low.
5. **Compare judge vs heuristic** and read the disagreements.
6. **Regression / CI gate** — `h.run_eval` stores `(:EvalRun)-[:HAS_RESULT]->(:EvalResult)` in Neo4j and `h.compare_runs`
   fails a new version that regresses beyond a tolerance.

Measured judge-vs-human agreement (Cohen's kappa; 0.4–0.6 moderate, 0.6–0.8 substantial, >0.8 near-perfect):
ch1 0.73 · ch2 0.86 · ch3 0.25 · ch4 0.50 · ch5 0.66 · ch6 0.48 · ch7 0.75 · ch8 0.45 · ch9 0.72. The token-overlap
heuristics scored 0.04–0.15 in most chapters, which is the point of using a judge. Every chapter's sabotage control (fluent
but false answer) scored 1.0–1.3 against 4.5–5.0 for the good answer, and several chapters flagged an unstable judge item
(e.g. scores 3, 1, 3) — judges are noisy, so measure them. Some judge cells print warnings rather than fail for this reason.

## Estimated OpenAI cost (measured 24 October 2026)

**Prices used** (OpenAI pricing page, checked 2026-10-24, standard tier, per 1M tokens):
`gpt-4o` $2.50 input / $10.00 output · `gpt-4o-mini` $0.15 / $0.60 · `text-embedding-3-large` $0.13.

| Notebook | LLM calls (of which judge) | Input / output tokens | Cost per run |
|---|---|---|---|
| ch1 Defining Agentic AI | 101 (40) | 27.0k / 5.3k | **$0.12** |
| ch2 Architecture Foundations | 114 (45) | 25.8k / 7.4k | **$0.14** |
| ch3 Knowledge Representation | 95 (30) | 26.1k / 6.2k | **$0.13** |
| ch4 Memory | 90 (43) | 18.4k / 5.6k | **$0.10** |
| ch5 Reasoning & Planning | 103 (42) | 17.2k / 6.6k | **$0.11** |
| ch6 Tool Orchestration | 98 (33) | 33.3k / 4.6k | **$0.13** |
| ch7 Self-Evolution | 87 (34) | 22.9k / 6.1k | **$0.12** |
| ch8 Optimization | 87 (29) | 16.7k / 4.1k | **$0.08** |
| ch9 Vibe-Coding Capstone | 43 (27) | 30.8k / 13.3k | **$0.21** |
| **All nine, one pass** | **818 (323)** | **~218k / ~59k** | **≈ $1.14** |

These are **measured** from real runs on 2026-10-24 (the figure `h.cost_report()` prints), and they include the
evaluation / LLM-as-a-Judge calls (about 40% of all calls, roughly $0.4 of the total). Embeddings are under $0.01 per
notebook. Neo4j, Docker and the skill CLIs cost nothing. An earlier code-reading estimate of about $8 was roughly 7×
too high, because loops and prompts are small. Expect run-to-run variation of perhaps ±30% (retries, agent step counts,
ch9's repair loop); the ceiling if everything loops to its limit is a few dollars, not tens.

**Tracking and controlling cost.**
- Every notebook ends by printing `h.cost_report()` (calls, judge calls, tokens, approx USD).
- Re-running costs about the same again; there is no caching.
- To cut cost ~15×, set `OPENAI_LLM_MODEL=gpt-4o-mini` in `.env` (judging and extraction quality will drop), or set a budget in the OpenAI dashboard first.
- Prices change; check <https://developers.openai.com/api/docs/pricing> before a big run.

## Status and caveats
- All nine notebooks were executed end to end with a real key (`python notebooks_advanced\_run.py <stem> [--save]` runs one
  and lists any failing cell; it takes a lock because all notebooks share one Neo4j and reset it on start — run one at a time).
- LLM output is non-deterministic, so checks are tolerant; a few judge-dependent checks print `WARNING` instead of failing.
- LLM-as-a-Judge has its own bias (including self-preference when judge and generator are the same model family); treat scores
  as a signal, calibrated against human labels, not ground truth.
- Notebooks are longer than the earlier tiers (about 55–90 cells each) because of the evaluation sections.

Sources live in `_src/` (percent format); rebuild with `python notebooks_advanced\_build.py notebooks_advanced\_src\<file>.py --check`
(rebuilding drops saved outputs; re-save with `_run.py --save`).
