# notebooks_advanced_with_observability_plus_changes — new data, a *different* problem, and how the notebooks evolve

This folder starts from [`notebooks_advanced_with_observability`](../notebooks_advanced_with_observability/README.md)
(real OpenAI, real Neo4j, LLM-as-a-Judge evaluations, Phoenix / DeepEval observability) and asks the question every real
project eventually faces:

> **The world changed. New kinds of data arrived and a new, different problem showed up. Do my notebooks, skills and
> evaluations still help me, and how do I evolve them?**

Nothing in the original folder was modified. Everything here is a copy plus the "changes".

---

## 1. The story

| | The original story (every other notebook folder) | **The new story (this folder)** |
|---|---|---|
| Incident | **INC-1001** — checkout is slow | **INC-2001** — *customers in the EU are being charged twice* |
| Kind of problem | **Performance** (latency) | **Correctness / money** (duplicate charges) |
| Root cause | `deploy-481` shrank a database connection pool | `deploy-490` turned on automatic payment retries **without an idempotency key**, while `cfg-310` cut the gateway timeout from 30 s to 3 s, and PayGate's EU region slowed to 4.2 s |
| Where the clues are | Mostly the service graph + a few metrics | Change events, a vendor's health, hourly metrics, costs, logs, tickets, runbooks |

**The mechanism, in plain words (read it twice).** The payment service calls an external vendor, *PayGate*. PayGate got slow
in `eu-west-1` (4.2 s) but still *succeeded*. At the same time a config change made our client give up after 3 s. So every
charge "timed out" on our side although PayGate completed it; a freshly enabled auto-retry feature then sent the charge
*again*, and with no idempotency key the vendor treated it as a new charge. Result: duplicates, a dead-letter-queue backlog and
a 3.5× spike in vendor spend.

**Why it is a good teaching problem.** *No single signal shows the cause.* You need a timeline across different kinds of
events, you must eliminate **red herrings**, and a plain LLM with no data cannot know any of it.

| Hypothesis | Verdict | How the graph kills or confirms it |
|---|---|---|
| **H1** `deploy-490` enabled retries without an idempotency key | ✅ cause | flag `payment-auto-retry` has `idempotency_key=false`; retries jump 150 → 9,800/h right after |
| **H2** `cfg-310` timeout 30 s → 3 s | ✅ cause (with H1 and H3) | 3 s < PayGate p95 of 4.2 s, so every slow success is seen as a failure |
| **H3** PayGate EU slowdown | ✅ trigger | `VendorStatus vs-1`: eu-west-1 p95 4,200 ms (us-east-1 is healthy) |
| **H4** secret rotation `sec-22` | ❌ red herring | auth check returned 200, no 401/403 in the logs |
| **H5** `deploy-489` (new checkout UI) | ❌ red herring | touches `cart-service`, unrelated to charging |
| **H6** `deploy-482` / orders-db pool | ❌ red herring | pool fixed hours earlier; healthy |
| **H7** `cfg-311` email batch size | ❌ red herring | affects `notification-service` only |

A **past analogue** exists: `INC-0650` had the same root cause, so semantic search over incident text should surface it.
The answer is **never stored in the graph** (no `ROOT_CAUSE` edge for INC-2001); the ground truth lives only in
[`incident2.py`](incident2.py) so the notebooks can grade themselves.

---

## 2. What data was added to Neo4j (11 new kinds)

Loaded by `incident2.load_changes()` (idempotent `MERGE`, about 90 new nodes and 120 new relationships):

| Wave | New labels | The question they answer |
|---|---|---|
| Topology | `Region`, `Vendor`, `Queue`, `Engineer` (+ 2 new `Service`s) | Where does it run? Who do we call? Who is on call? |
| Change events | `ConfigChange`, `FeatureFlag`, `SecretRotation` (+ new `Deployment`s) | What changed right before the problem? |
| Signals | `VendorStatus`, `MetricSample` (hourly series), `CostRecord` (USD / hour) | What did the outside world and the numbers do? |
| Documents | `Ticket` (customer complaints), `LogEvent`, `Runbook`, a past `Incident` | What do people and machines *say*? (free text → embeddings) |

Every event has an `at` timestamp so one Cypher query can build a **timeline across different node types**. Four vector
indexes are created over the documents: `ticket_text_idx`, `logevent_text_idx`, `runbook_text_idx`, `incident_text_idx`.

---

## 3. How the notebooks evolved

Every notebook is the *same* notebook as in the observability folder (Part 1, still passing) **plus**:

* a **banner** pointing here, and
* a self-contained **"Part 2"** section that re-solves INC-2001 with *that chapter's* technique.

A new **ch0** introduces the data and the problem with plain Cypher, then with an LLM. Start with ch0, then ch1 → ch9.
Measured on 2026-10-24 (`h.cost_report()` approximations at `gpt-4o` rates, Part 1 + Part 2):

| Notebook | What Part 2 teaches (the technique applied to INC-2001) | Cost |
|---|---|---|
| [ch0](ch0-whats-changed-new-problem.ipynb) | The story, the 11 new data kinds, a timeline query, hypothesis elimination, embeddings, an LLM investigation graded by checks + a judge, and a 3-version evaluation (`no-context` → `graph-context-v1` → `graph-context-v2-rich`) | $0.09 |
| [ch1](ch1-defining-agentic-ai.ipynb) Defining agentic AI | When is it an *agent* vs a pipeline? Failure-mode classification of the new incident's agent runs | $0.14 |
| [ch2](ch2-architecture-foundations.ipynb) Architecture | Routing questions to graph, text or hybrid retrieval over the new data; skill vs LLM routing | $0.15 |
| [ch3](ch3-knowledge-representation.ipynb) Knowledge representation | Extracting entities/regions/services from tickets and logs, alias resolution, evidence spans | $0.14 |
| [ch4](ch4-memory.ipynb) Memory | Remembering INC-0650 vs INC-2001, RRF fusion, comparing healthy vs degraded vendor intervals | $0.11 |
| [ch5](ch5-reasoning-planning.ipynb) Reasoning & planning | A constrained hypothesis DAG, parallel evidence tests, `no_evidence` vs `rejected`, early termination | $0.12 |
| [ch6](ch6-tool-orchestration.ipynb) Tool orchestration | Governed tools for payment safety: approval gates, RBAC, red-team of invented tools | $0.17 |
| [ch7](ch7-self-evolution.ipynb) Self-evolution | Prompt v1 → v2 evaluated with a **cost-tolerance gate**; judge calibration (κ) | $0.17 |
| [ch8](ch8-optimization.ipynb) Optimization | Selective intelligence: cheap vs strong models, compression and routing measured on money incidents | $0.11 |
| [ch9](ch9-vibe-coding-capstone.ipynb) Capstone | Vibe-coding a skill for the new incident, with a fair score floor | $0.30 |
| **Total** | | **about $1.49** |

### Evaluations and LLM-as-a-Judge, again the stars
Every Part 2 ends with an evaluation: deterministic keyword graders from `incident2.keyword_score` (cheap, strict, gameable)
**and** an LLM judge with a rubric (flexible, can be lenient). Where they disagree is where you read by hand. Findings
the runs reported (your numbers will vary slightly, LLMs are not deterministic):

* **ch0:** bare model 0.13 → naive graph dossier 0.75 → richer dossier 0.81 on an 8-question gold set. The misses were *retrieval
  gaps* (the dossier lacked region data), not model weakness. Evolving the *context*, not the model, fixed them.
* **ch7:** the richer prompt lifted keyword coverage 0.25 → 1.00 but used ~5× the tokens, so the cost gate rejected it.
* **ch2:** skill routing 100%, LLM routing 92% (it sent the region question to text search).
* **ch9:** gold QA 6/8; it still missed the secret-rotation evidence detail. We kept a *fair* floor instead of hiding the gap.

---

## 4. Setup and run (from the repo root)

1. `pip install -r requirements.txt`
2. Put `OPENAI_API_KEY` and `OBSERVABILITY_BACKEND` in `notebooks_advanced_with_observability_plus_changes/.env`
   (see [`.env.example`](.env.example); `.env` is git-ignored — never commit a key).
3. Start a stack (see [`docker-compose/README.md`](../docker-compose/README.md)):
   ```
   docker compose -f docker-compose/docker-compose.phoenix.yml up -d
   ```
4. Open a notebook with the kernel working directory set to **this folder**. Run **ch0 first**, then ch1–ch9. Each Part 2 loads
   the new data itself (idempotent), so any chapter also runs alone. **Restart the kernel after changing `OBSERVABILITY_BACKEND`.**
5. Stop: `docker compose -f docker-compose/docker-compose.phoenix.yml down`.

**Which Neo4j?** The one matching `OBSERVABILITY_BACKEND` (`phoenix`/`both` → 7688, `deepeval` → 7689, `none` → 7687); override with `NEO4J_URI`.

### Observability (unchanged from the previous folder)
Phoenix (http://localhost:6006) shows every LLM call, agent/tool span and judge verdict as traces; the DeepEval container
(http://localhost:8765) re-scores a sample of judged answers with GEval *Correctness* / *AnswerRelevancy*. Use Phoenix to ask
"why was this step slow / wrong / expensive", and DeepEval for "do two judges agree?".

---

## 5. Cost (measured 24 October 2026)
About **$1.49** for all ten notebooks (ch0 ≈ $0.09; chapters $0.11–$0.30; Part 2 itself costs ~$0.01–0.06 per chapter, the rest is
Part 1). That is about $0.35 more than the observability folder (~$1.14 for nine). Phoenix adds nothing; DeepEval adds a few
cents. Pricing changes; run `h.cost_report()` for your own numbers.

---

## 6. Honest limits
* The data is **synthetic** and small; the story is hand-built so it has a clear answer.
* Keyword graders are strict and can be gamed; LLM judges can be lenient. Treat both as signals.
* Models are non-deterministic: a re-run can score slightly differently. Thresholds were set with margin, but a flaky check is
  possible; re-run once before assuming a bug.
* Neo4j allows **one vector index per label + property**. Two older notebooks (ch5, ch8) therefore reuse the new
  `incident_text_idx` / `runbook_text_idx` instead of creating their own: a real-world consequence of evolving a schema.

## 7. Files
| File | Purpose |
|---|---|
| [`incident2.py`](incident2.py) | The new data, ground truth, 8-question gold set, keyword grader, named evidence queries (`scn.q(...)`), `load_changes()` / `embed_new_documents()` |
| `real_helpers.py`, `observability.py` | Shared helpers (OpenAI, Neo4j, judges, evals, tracing); unchanged |
| `_src/*.py` | Percent-format sources of the notebooks (edit these, never the `.ipynb`) |
| `_build.py`, `_run.py`, `_run_part2.py` | Rebuild notebooks; run one for real (serialized by `.run.lock`); run only Part 2 cheaply |
| `.env`, `.run.lock` | Local, git-ignored |
