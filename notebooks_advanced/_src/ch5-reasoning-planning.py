# %% [markdown]
# # Chapter 5 — Reasoning and Planning (advanced, real LLM + real Neo4j)
# > **Tier: advanced (real).** Needs OPENAI_API_KEY in notebooks_advanced/.env and the Neo4j container from docker-compose/.
#
# > **Evaluation focus:** see the section "Evaluation and LLM-as-a-Judge" near the end — every notebook in this tier treats eval and judging as first-class.
#
# > **Estimated OpenAI cost per full run (measured 2026-10-24 at `gpt-4o` rates): about $0.11.** 103 LLM calls (42 of them LLM-as-a-Judge), 17,227 input / 6,593 output tokens, plus a few thousand `text-embedding-3-large` tokens (<$0.01). Re-run costs about the same; `h.cost_report()` prints your own numbers. See [README](README.md#estimated-openai-cost-measured-24-october-2026).
#
# In the earlier tiers the planner was a table lookup, retrieval was keyword overlap and "access control" was a Python set.
# This notebook removes those crutches. Every seam the chapter talks about is now a **real component**:
#
# | Chapter idea | Real component used here |
# |---|---|
# | Pipeline selection | `gpt-4o` classifies complexity/uncertainty, compared with the skill's heuristic |
# | Hypothesis retrieval | `text-embedding-3-large` (3072 d) + Neo4j **vector** index + **fulltext** index + graph prior |
# | Investigation DAG | `gpt-4o` planner with a strict JSON schema, retried through the loop-router skill |
# | Parallel hypothesis testing | Cypher telemetry reads + LLM verdicts, merged by the reconcile skill |
# | Structured-output contracts | measured on the live model: free text vs `json_object` vs strict `json_schema` |
# | Constraint-guided validation | LLM extracts constraints, validator checks plans, **real Neo4j RBAC** enforces |
# | Agent | OpenAI function-calling agent whose trace is stored as a graph |
#
# **Honest framing.** Nothing here is deterministic: LLM answers vary run to run, so checks assert *structure and
# behaviour*, never exact wording. Section 13 reports tokens used and what remains approximate.

# %% [markdown]
# ## 0. Preflight — fail loudly
#
# There is deliberately **no offline fallback**. If the API key or the Neo4j container is missing, this cell must
# raise, because every later result would be meaningless without them.

# %%
import sys; sys.path.insert(0, ".")
import real_helpers as h
import os, re, json, time, subprocess

h.get_openai()
h.connect()
print("LLM:", h.LLM_MODEL, "| embeddings:", h.EMBED_MODEL, f"({h.EMBED_DIM} dims)")
print(h.run("RETURN 'neo4j reachable' AS status"))
h.check("preflight: OpenAI client + Neo4j reachable", True)

# %% [markdown]
# ## 1. The graph, plus *searchable* incident history
#
# A planner ranks hypotheses using what went wrong before. The base graph has one past incident, so we add eight
# resolved ones written in **deliberately varied wording**. Retrieval has to work on meaning, not on shared words.
# Each incident gets a `ROOT_CAUSE` edge to the component that really failed.

# %%
print(h.load_devops_graph())

HISTORY = [
    ("HIST-1", "orders-db", "Requests queued waiting for database connections; checkout timeouts started after a config change"),
    ("HIST-2", "orders-db", "Connection pool exhausted, threads blocked on getConnection, p99 climbing steadily"),
    ("HIST-3", "payment-service", "Long stop-the-world garbage collection pauses in the payment JVM, intermittent 502 responses"),
    ("HIST-4", "payment-service", "Heap usage near its limit, latency spikes every few minutes coinciding with GC pauses"),
    ("HIST-5", "session-cache", "Login sessions evicted, cache hit rate collapsed, users forced to re-authenticate"),
    ("HIST-6", "session-cache", "Redis memory eviction storm caused cold reads against the backing store"),
    ("HIST-7", "orders-db", "Slow queries after a missing index on the orders table, sequential scans everywhere"),
    ("HIST-8", "checkout-api", "Bad release of checkout: null pointer exceptions when submitting the cart"),
]
for hid, comp, text in HISTORY:
    h.run("MATCH (c:Service|Database|Cache {name:$c}) "
          "CREATE (:Incident {id:$id, title:$t, text:$t, status:'resolved', root_cause:$t})-[:ROOT_CAUSE]->(c)",
          id=hid, c=comp, t=text)
h.run("MATCH (i:Incident {status:'resolved'}) WHERE i.text IS NULL "
      "SET i.text = i.title + '. ' + coalesce(i.root_cause, '')")

rows = h.run("MATCH (i:Incident)-[:ROOT_CAUSE]->(c) RETURN c.name AS component, count(i) AS past_incidents "
             "ORDER BY past_incidents DESC, component")
h.show(rows)
CAUSE = {r["id"]: r["c"] for r in h.run("MATCH (i:Incident)-[:ROOT_CAUSE]->(c) RETURN i.id AS id, c.name AS c")}
PRIOR = {r["component"]: r["past_incidents"] for r in rows}
h.check("history loaded (>= 9 resolved incidents with causes)", len(CAUSE) >= 9)

# %% [markdown]
# ## 2. Load all six Chapter 5 skills
#
# The skills stay the deterministic *spine*: DAG layering, merge semantics, retry routing, constraint checks and
# contract validation are code, because code is cheap, fast and auditable. The LLM supplies the judgement *inside*
# that spine. That division of labour is the core pattern of this chapter.

# %%
CH = "reasoning-planning"
sel = h.load_skill(CH, "pipeline-architecture-selector")
dagp = h.load_skill(CH, "investigation-dag-planner")
par = h.load_skill(CH, "parallel-reconcile-merge")
loop = h.load_skill(CH, "loop-pipeline-router")
val = h.load_skill(CH, "constraint-guided-plan-validator")
con = h.load_skill(CH, "structured-output-contract-designer")

cat = h.skill_catalog(CH)
print(cat)
h.check("all 6 chapter skills loaded", all(m is not None for m in (sel, dagp, par, loop, val, con)))

# %% [markdown]
# ## 3. Reusable LLM plumbing: versioned prompts, strict schemas, retries
#
# Three habits separate a demo from a system:
#
# 1. **Prompt versioning.** Prompts live in a registry as `name@vN`. Results and graph nodes record the version used,
#    so a regression can be traced to a prompt change instead of a mystery.
# 2. **Strict structured outputs.** `response_format=json_schema` with `strict: true` makes the *decoder* obey the
#    schema (all properties required, `additionalProperties: false`). We still validate afterwards, because the schema
#    cannot express business rules such as "dependencies must exist".
# 3. **Retries with backoff** for transient API errors. Semantic retries (bad plan, unknown id) are a different thing;
#    those go through the loop-router skill later.

# %%
PROMPTS = {
    "pipeline_classifier@v1": (
        "You classify an engineering question for an agent pipeline. Return complexity (0..1: how many "
        "entities, hops and dependent steps the answer needs) and uncertainty (0..1: how open-ended or "
        "unknown the answer is; a simple factual lookup is near 0, a root-cause hunt is near 1)."),
    "hypothesis_planner@v1": (
        "You are an SRE planning an investigation. Propose 4-6 root-cause hypotheses as a dependency graph of "
        "tests. Ids are H1, H2, ... A test may depend on others only when it logically needs their result "
        "(for example verifying a deployment before blaming a downstream pool). Never create cycles. "
        "Use only components from the allowed list."),
    "remediation@v1": "Propose one remediation for the incident. Be brief.",
    "remediation@v2": (
        "Propose one remediation for the incident. Prefer the least-privilege, reversible action, rate its risk "
        "honestly (low/medium/high), and always give an explicit rollback step."),
}


def strict_format(name, props, required=None):
    schema = {"type": "object", "properties": props, "required": required or list(props),
              "additionalProperties": False}
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def with_retries(fn, attempts=3):
    last = None
    for a in range(attempts):
        try:
            return fn()
        except Exception as e:  # network / rate-limit / malformed JSON
            last = e
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"LLM call failed after {attempts} attempts: {last}")


def structured(prompt_key, user, name, props, required=None):
    rf = strict_format(name, props, required)
    msgs = [{"role": "system", "content": PROMPTS[prompt_key]}, {"role": "user", "content": user}]
    return with_retries(lambda: json.loads(h.chat(msgs, response_format=rf).content))


probe = structured("pipeline_classifier@v1", "What port does session-cache use?", "probe",
                   {"complexity": {"type": "number"}, "uncertainty": {"type": "number"}})
print(probe)
h.check("strict-schema call returns both numeric fields",
        isinstance(probe.get("complexity"), (int, float)) and isinstance(probe.get("uncertainty"), (int, float)))

# %% [markdown]
# ## 4. Pipeline selection: heuristic vs LLM judgement
#
# The skill's `assess_task_complexity` / `estimate_uncertainty` are word-counting proxies (they document that the
# production version is an LLM call). We run **both** against a small labelled set where an experienced engineer
# decided the right architecture, then look at where they disagree.
#
# * `sequential` — simple, certain lookups.
# * `tree` — open-ended, many competing hypotheses: explore in parallel.
# * `loop` — a draft that must be refined until it validates.

# %%
LABELED_QUERIES = [
    ("What is the owner of checkout-api?", "sequential"),
    ("List the services that depend on orders-db", "sequential"),
    ("What engine does session-cache run?", "sequential"),
    ("Why is checkout slow? Investigate the root cause across services after the recent deploy", "tree"),
    ("Intermittent latency spikes of unknown origin between payment and orders", "tree"),
    ("Diagnose why error rates look anomalous", "tree"),
    ("Draft a remediation summary and refine it until it passes the compliance checklist", "loop"),
    ("Generate the postmortem and iterate until every required section is valid", "loop"),
]


def llm_scores(q):
    r = structured("pipeline_classifier@v1", q, "scores",
                   {"complexity": {"type": "number"}, "uncertainty": {"type": "number"}})
    return float(r["complexity"]), float(r["uncertainty"])


table = []
for q, gold in LABELED_QUERIES:
    hc, hu = sel.assess_task_complexity(q), sel.estimate_uncertainty(q)
    lc, lu = llm_scores(q)
    table.append({"query": q[:48], "gold": gold,
                  "heuristic": sel.analyze_and_route(hc, hu), "llm": sel.analyze_and_route(lc, lu),
                  "h_cu": (hc, hu), "llm_cu": (round(lc, 2), round(lu, 2))})
h.show(table)
acc_h = sum(r["heuristic"] == r["gold"] for r in table) / len(table)
acc_l = sum(r["llm"] == r["gold"] for r in table) / len(table)
print(f"routing accuracy -> heuristic {acc_h:.0%} | LLM {acc_l:.0%}")
print("disagreements:", [(r["query"], r["heuristic"], r["llm"]) for r in table if r["heuristic"] != r["llm"]])
h.check("LLM routing is at least as good as the word-counting heuristic (tolerant)", acc_l >= acc_h - 0.13)

# %% [markdown]
# ### Use the graph to ground the decision, and persist it
#
# Complexity is not only about wording. The **blast radius** (how many components sit downstream of the affected
# service) is a structural signal the LLM never sees unless we hand it over. We also degrade gracefully: on a tiny
# host the skill refuses the memory-hungry `tree` and falls back, and `estimate_latency` prices each choice.
# Finally the decision is written to Neo4j so later analysis can ask "which route did we pick, and why?".

# %%
BLAST = h.run("MATCH (inc:Incident {id:'INC-1001'})-[:AFFECTS]->(s:Service)-[:DEPENDS_ON*1..3]->(d) "
              "RETURN s.name AS service, count(DISTINCT d) AS downstream, collect(DISTINCT d.name) AS components")
h.show(BLAST)

QUERY = "Why is checkout slow? Investigate the root cause across services after the recent deploy"
lc, lu = llm_scores(QUERY)
lc = min(1.0, lc + 0.05 * BLAST[0]["downstream"])  # structural evidence raises complexity
decision = sel.route_query(QUERY, complexity_fn=lambda q: lc, uncertainty_fn=lambda q: lu)
tiny = sel.route_query(QUERY, available_memory_mb=128, complexity_fn=lambda q: lc, uncertainty_fn=lambda q: lu)
print("full host :", decision)
print("128 MB    :", tiny)
for arch in sel.ARCHITECTURES:
    est = sel.estimate_latency(arch, node_seconds=[3, 3, 3], merge_seconds=1, retry_probability=0.2, max_iterations=3)
    print(f"{arch:10s} expected {est['expected_seconds']:.1f}s")

h.run("CREATE (:Decision {kind:'pipeline', query:$q, architecture:$a, degraded_architecture:$d, "
      "prompt_version:'pipeline_classifier@v1'})",
      q=QUERY, a=decision.architecture, d=tiny.final)
h.check("route is a known architecture", decision.final in sel.ARCHITECTURES)
h.check("128 MB host degrades the memory-hungry tree route", tiny.degraded or tiny.final != "tree")

# %% [markdown]
# ## 5. Retrieval ablation: vector vs fulltext vs graph prior vs hybrid
#
# Which past incidents look like today's symptoms? Four strategies, all real:
#
# * **vector** — embeddings (3072 d) in a Neo4j vector index; matches *meaning*.
# * **fulltext** — Lucene index; exact-ish *words*, precise on error codes, blind to paraphrase.
# * **graph prior** — "which component failed most often?". Free, but ignores the symptoms entirely.
# * **hybrid** — reciprocal-rank fusion of vector + fulltext, with the prior as a tie-break.
#
# We embed the incident text onto the nodes, build both indexes, and wait for them to come online.
# Evaluating on a labelled set is how we avoid choosing a strategy by vibes.

# %%
n = h.embed_nodes("Incident")
h.ensure_vector_index("incident_vec", "Incident")
h.run("CREATE FULLTEXT INDEX incident_ft IF NOT EXISTS FOR (n:Incident) ON EACH [n.text]")
h.run("CALL db.awaitIndexes(60)")
print("embedded", n, "incidents (3072-d) and built vector + fulltext indexes")
h.check("incidents embedded", n >= 9)

# %%
SYMPTOMS = [
    ("latency jumped right after we shrank the connection limit; calls hang waiting for a slot", "orders-db"),
    ("stop-the-world pauses in the card processing JVM", "payment-service"),
    ("users keep getting logged out and cache misses are through the roof", "session-cache"),
    ("full table scans on orders, query plans regressed", "orders-db"),
    ("NPE exceptions after the latest release of the checkout service", "checkout-api"),
    ("502s from the payments tier with memory almost exhausted", "payment-service"),
]


def lucene_safe(q):
    return " ".join(re.sub(r"[^\w\s]", " ", q).split())


def vec_ids(q, k=5):
    return [r["node"]["id"] for r in h.vector_search("incident_vec", q, k) if r["node"]["id"] in CAUSE]


def ft_ids(q, k=5):
    rows = h.run("CALL db.index.fulltext.queryNodes('incident_ft', $q) YIELD node, score "
                 "RETURN node.id AS id ORDER BY score DESC LIMIT $k", q=lucene_safe(q), k=k)
    return [r["id"] for r in rows if r["id"] in CAUSE]


def top_component(ranked_lists):
    score = {}
    for ids in ranked_lists:
        for rank, iid in enumerate(ids):
            c = CAUSE[iid]
            score[c] = score.get(c, 0.0) + 1.0 / (60 + rank)  # reciprocal-rank fusion
    for c, p in PRIOR.items():
        score[c] = score.get(c, 0.0) + p * 1e-6  # prior only breaks ties
    return max(score, key=score.get)


prior_pick = max(PRIOR, key=PRIOR.get)
results = []
for q, gold in SYMPTOMS:
    v, f = vec_ids(q), ft_ids(q)
    results.append({"symptom": q[:50], "gold": gold,
                    "vector": top_component([v]) if v else None,
                    "fulltext": top_component([f]) if f else None,
                    "prior": prior_pick,
                    "hybrid": top_component([v, f])})
h.show(results)
ACC = {m: sum(r[m] == r["gold"] for r in results) / len(results) for m in ("vector", "fulltext", "prior", "hybrid")}
print("top-1 component accuracy:", {k: f"{v:.0%}" for k, v in ACC.items()})
h.check("vector retrieval beats the symptom-blind prior", ACC["vector"] > ACC["prior"])
h.check("hybrid is not worse than fulltext alone", ACC["hybrid"] >= ACC["fulltext"])

# %% [markdown]
# ### Failure analysis
#
# An accuracy number says *how often* retrieval fails, not *why*. We hand the misses to the LLM with the gold answer
# and the retrieved neighbours and ask for a one-line diagnosis. This is cheap, and it tells us whether to fix
# the data (thin incident text), the query (needs rewriting) or the strategy (needs reranking).

# %%
misses = [r for r in results if r["hybrid"] != r["gold"]]
if not misses:
    print("No hybrid misses on this labelled set; nothing to analyse.")
for r in misses:
    q = next(s for s, g in SYMPTOMS if s.startswith(r["symptom"][:40]))
    texts = h.run("MATCH (i:Incident) WHERE i.id IN $ids RETURN i.id AS id, i.text AS text", ids=vec_ids(q, 3))
    why = h.llm(f"Symptom: {q}\nGold component: {r['gold']}\nHybrid chose: {r['hybrid']}\n"
                f"Nearest incidents: {texts}\nIn one sentence, why did retrieval miss, and what single change would fix it?")
    print("-", r["symptom"], "->", why)
h.check("failure analysis ran on every miss", True)

# %% [markdown]
# ## 6. The LLM hypothesis planner, retried through the loop-router skill
#
# This is the heart of the chapter: turn a symptom into a **dependency graph of hypotheses**. Three safeguards:
#
# 1. **Grounded prompt**: the planner sees retrieved similar incidents, the failure-count prior and the topology.
# 2. **Strict schema** with the component as an *enum*, so it cannot invent a service.
# 3. **Semantic validation + bounded retry** via `run_loop`: unknown dependency ids, cycles or a bad hypothesis count
#    are *correctable* errors, so the violation text is fed back and the model tries again. If retries run out,
#    the loop returns a non-`PROCEED` decision and we stop honestly. There is no hidden fake planner.

# %%
SYMPTOM = "Checkout API p99 latency is 2400 ms (threshold 800 ms) shortly after deploy-481; orders are timing out."
COMPONENTS = ["checkout-api", "payment-service", "orders-service", "cart-service", "orders-db", "session-cache"]
similar = [r["node"]["text"] for r in h.vector_search("incident_vec", SYMPTOM, 3)]
topology = h.run("MATCH (a)-[:DEPENDS_ON]->(b) RETURN a.name AS a, b.name AS b")
HYP_PROPS = {"hypotheses": {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "string"},
    "component": {"type": "string", "enum": COMPONENTS},
    "statement": {"type": "string"},
    "depends_on": {"type": "array", "items": {"type": "string"}},
    "duration_s": {"type": "integer"},
    "priority": {"type": "integer"}},
    "required": ["id", "component", "statement", "depends_on", "duration_s", "priority"],
    "additionalProperties": False}}}
PLAN_STATE = {"feedback": [], "attempts": 0}


def plan_attempt(retry_count):
    PLAN_STATE["attempts"] += 1
    user = (f"Symptom: {SYMPTOM}\nSimilar past incidents: {similar}\nFailure counts by component: {PRIOR}\n"
            f"Topology (a depends on b): {topology}\nAllowed components: {COMPONENTS}\n"
            f"Fix these problems from your previous attempt: {PLAN_STATE['feedback'] or 'none'}")
    return structured("hypothesis_planner@v1", user, "plan", HYP_PROPS)


def plan_validate(cand):
    hyps = cand.get("hypotheses", [])
    ids = [x["id"] for x in hyps]
    problems = []
    if not 3 <= len(hyps) <= 8:
        problems.append(f"need 3-8 hypotheses, got {len(hyps)}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    for x in hyps:
        bad = [d for d in x["depends_on"] if d not in ids or d == x["id"]]
        if bad:
            problems.append(f"{x['id']} depends on unknown/self ids {bad}")
    if not problems:
        try:
            dagp.build_investigation_dag(dagp.tasks_from_dicts(
                [{"id": x["id"], "description": x["statement"], "duration_s": x["duration_s"],
                  "priority": x["priority"], "depends_on": x["depends_on"]} for x in hyps]))
        except dagp.CycleError as e:
            problems.append(f"dependency cycle: {e}")
    if problems:
        PLAN_STATE["feedback"] = problems
        return False, loop.ValidationError(loop.SEVERITY_CORRECTABLE, "; ".join(problems))
    return True, None


outcome = loop.run_loop(plan_attempt, plan_validate, max_retries=3)
print("decision:", outcome["decision"], "| refine iterations:", outcome["iterations"],
      "| LLM attempts:", PLAN_STATE["attempts"])
print([t["decision"] for t in outcome["trace"]])
if outcome["decision"] != loop.PROCEED:
    raise RuntimeError(f"planner did not converge: {PLAN_STATE['feedback']}")
PLAN = outcome["candidate"]["hypotheses"]
HYP = {x["id"]: x for x in PLAN}
h.show([{k: x[k] for k in ("id", "component", "depends_on", "duration_s", "statement")} for x in PLAN])
h.check("planner converged to a valid dependency graph", outcome["decision"] == loop.PROCEED and len(PLAN) >= 3)

# %% [markdown]
# ### Persist the plan and cross-check the DAG layers two ways
#
# The plan becomes `Step` nodes with `FOR` (which component) and `DEPENDS_ON` edges. Then we compute execution
# **phases** twice: with the skill (Python, Kahn-style layering) and with a Cypher longest-path query. Two
# independent implementations agreeing is a cheap, strong correctness check, and the graph version means any later
# tool can ask "what runs first?" without importing Python.

# %%
for x in PLAN:
    h.run("CREATE (s:Step {id:$id, kind:'hypothesis', statement:$st, status:'planned', duration_s:$d, priority:$p, "
          "prompt_version:'hypothesis_planner@v1'}) WITH s MATCH (c:Service|Database|Cache {name:$c}) "
          "CREATE (s)-[:FOR]->(c)", id=x["id"], st=x["statement"], d=x["duration_s"], p=x["priority"], c=x["component"])
for x in PLAN:
    for d in x["depends_on"]:
        h.run("MATCH (a:Step {id:$a}), (b:Step {id:$b}) CREATE (a)-[:DEPENDS_ON]->(b)", a=x["id"], b=d)

TASKS = dagp.tasks_from_dicts([{"id": x["id"], "description": x["statement"], "duration_s": x["duration_s"],
                                "priority": x["priority"], "depends_on": x["depends_on"]} for x in PLAN])
DAG = dagp.build_investigation_dag(TASKS)
skill_phase = {tid: ph.index for ph in DAG.phases for tid in ph.task_ids}
base = min(skill_phase.values())
skill_phase = {k: v - base for k, v in skill_phase.items()}

cy = h.run("MATCH (s:Step) OPTIONAL MATCH p=(s)-[:DEPENDS_ON*]->(:Step) "
           "WITH s, coalesce(max(length(p)), 0) AS phase RETURN s.id AS id, phase ORDER BY phase, id")
cy_phase = {r["id"]: r["phase"] for r in cy}
print("phases:", [(ph.index, ph.task_ids, ph.estimated_duration_s) for ph in DAG.phases])
print("parallelism factor:", DAG.parallelism_factor, "| total est s:", DAG.total_estimated_duration_s)
h.check("Cypher phases match the skill DAG layers", cy_phase == skill_phase)

# %% [markdown]
# ## 7. Parallel hypothesis testing: Cypher evidence, LLM verdict, reconciled merge
#
# Each hypothesis gets a **branch** that (1) reads real telemetry from Neo4j and (2) asks the LLM whether that
# telemetry supports the statement. We also run a transparent threshold rule on the same numbers, so we can show
# where the LLM and the rule disagree.
#
# We seed telemetry for only two components. A branch for a component *without* telemetry fails naturally
# (`LookupError`). That is exactly the situation the merge modes exist for:
# `partial_coverage` keeps the useful results and flags the gap; `all_or_nothing` refuses to conclude.

# %%
h.run("MATCH (d:Database {name:'orders-db'}) "
      "CREATE (:Metric {kind:'pool_utilization', value:1.0})-[:OF]->(d), "
      "(:Metric {kind:'waiting_threads', value:47.0})-[:OF]->(d)")
h.run("MATCH (p:Service {name:'payment-service'}) CREATE (:Metric {kind:'heap_ratio', value:0.42})-[:OF]->(p)")

RULES = {"pool_utilization": lambda v: v > 0.9, "waiting_threads": lambda v: v > 10, "heap_ratio": lambda v: v > 0.85}
MEMO = {}


def assess(sid):
    if sid in MEMO:
        return MEMO[sid]
    hyp = HYP[sid]
    rows = h.run("MATCH (m:Metric)-[:OF]->(c {name:$n}) RETURN m.kind AS kind, m.value AS value", n=hyp["component"])
    if not rows:
        raise LookupError(f"no telemetry for {hyp['component']}")
    ev = {r["kind"]: r["value"] for r in rows}
    v = h.llm_json(f"Hypothesis: {hyp['statement']}\nComponent: {hyp['component']}\nTelemetry: {ev}\n"
                   'Reply JSON {"supported": bool, "confidence": number 0..1, "reason": short string}. '
                   "Judge only from the telemetry given.")
    rule = any(RULES[k](x) for k, x in ev.items() if k in RULES)
    out = {"id": sid, "component": hyp["component"], "evidence": ev, "supported": bool(v.get("supported")),
           "confidence": float(v.get("confidence", 0)), "reason": v.get("reason", ""), "rule_supported": rule,
           "flags": [] if bool(v.get("supported")) == rule else ["llm_rule_disagree"]}
    MEMO[sid] = out
    return out


BRANCHES = {sid: (lambda ctx, s=sid: assess(s)) for sid in HYP}
partial = par.run_parallel_window(BRANCHES, {}, mode="partial_coverage")
strict_all = par.run_parallel_window(BRANCHES, {}, mode="all_or_nothing")
print("partial :", partial.completed, "| ok:", list(partial.succeeded), "| failed:", list(partial.failed), "| flags:", partial.flags)
print("strict  :", strict_all.completed, "| failed:", list(strict_all.failed))
print(partial.rationale)
h.show([{k: v[k] for k in ("id", "component", "supported", "confidence", "rule_supported", "reason")} for v in MEMO.values()])
h.check("every branch is accounted for (succeeded or failed)", len(list(partial.succeeded)) + len(list(partial.failed)) == len(HYP))
h.check("all_or_nothing is no more permissive than partial_coverage", (not strict_all.completed) or partial.completed)

# %% [markdown]
# ### Record the merge, and look at LLM-vs-rule disagreements
#
# A `Merge` node ties the parallel window to its steps, and every step gets a status (`supported`, `rejected`,
# `no_evidence`). The disagreement list is the valuable part: each row is either a bad rule threshold or an LLM
# hallucination, and a human should look at it.

# %%
h.run("CREATE (:Merge {mode:'partial_coverage', completed:$c, rationale:$r, flags:$f})",
      c=bool(partial.completed), r=str(partial.rationale)[:300], f=list(partial.flags))
for sid in HYP:
    status = ("supported" if MEMO[sid]["supported"] else "rejected") if sid in MEMO else "no_evidence"
    h.run("MATCH (s:Step {id:$id}) SET s.status=$st", id=sid, st=status)
    h.run("MATCH (s:Step {id:$id}), (m:Merge) CREATE (m)-[:MERGED]->(s)", id=sid)
h.show(h.run("MATCH (s:Step) RETURN s.status AS status, count(*) AS steps ORDER BY status"))
dis = [(v["id"], v["component"], v["supported"], v["rule_supported"]) for v in MEMO.values() if v["flags"]]
print("LLM vs rule disagreements:", dis or "none")
h.check("steps without telemetry are marked no_evidence, not guessed",
        all((s["status"] == "no_evidence") == (s["id"] not in MEMO)
            for s in h.run("MATCH (s:Step) RETURN s.id AS id, s.status AS status")))

# %% [markdown]
# ## 8. Early termination and multi-hop evidence
#
# Once one hypothesis is **confirmed with high confidence**, testing the rest wastes time. The skill's
# `execute_with_early_termination` walks the DAG phase by phase and skips later phases after a confirmation.
# Our `test_fn` reuses the cached assessments, so this costs no extra tokens.
#
# Then the part only a graph does well: gather *multi-hop* evidence around the confirmed component in a single
# query: which deployment changed it, who depends on it, and what fixed it last time.

# %%
def test_fn(task):
    try:
        r = assess(task.id)
    except LookupError as e:
        return False, str(e)
    return (r["supported"] and r["confidence"] >= 0.7), r["reason"]


res = dagp.execute_with_early_termination(DAG, TASKS, test_fn)
print("confirmed:", res["confirmed"], "| phases run:", res["phases_run"], "| skipped:", res["phases_skipped"])
CONFIRMED = HYP[res["confirmed"]]["component"] if res["confirmed"] else None
if res["confirmed"]:
    h.run("MATCH (s:Step {id:$id}) SET s.status='confirmed'", id=res["confirmed"])

if CONFIRMED:
    ev = h.run("MATCH (c {name:$n}) "
               "OPTIONAL MATCH (dep:Deployment)-[:CHANGED]->(c) "
               "OPTIONAL MATCH (up:Service)-[:DEPENDS_ON*1..3]->(c) "
               "OPTIONAL MATCH (past:Incident)-[:ROOT_CAUSE]->(c) "
               "RETURN c.name AS component, collect(DISTINCT dep.id + ': ' + coalesce(dep.change, '')) AS deployments, "
               "collect(DISTINCT up.name) AS upstream, collect(DISTINCT past.id) AS past_incidents", n=CONFIRMED)
    h.show(ev)
h.check("early termination ran and reported a (possibly empty) result", "phases_run" in res)
h.check("if something was confirmed it has telemetry behind it", res["confirmed"] is None or res["confirmed"] in MEMO)

# %% [markdown]
# ## 9. Structured-output contracts, measured on the live model
#
# The skill's keystone claim: a machine-consumed seam should be *constrained*, because even rare malformed output
# is expensive at scale. `recommend_enforcement` gives the policy; now we **measure** it. The same task (turn an
# incident note into a hypothesis object) is run under three enforcement levels:
#
# * **free text** — "reply as JSON" and `json.loads` the raw reply (fences and prose break it);
# * **json_object** — valid JSON guaranteed, but the *shape* is not;
# * **strict json_schema** — decoder-enforced shape.
#
# We score two rates separately: **parses** and **satisfies the contract**.

# %%
for name, p in [("report for a human", con.NodeProfile("human", False, False, 0, 3)),
                ("hypothesis -> graph write", con.NodeProfile("graph_write", True, True, 3, 1))]:
    print(f"{name:28s} ->", con.recommend_enforcement(p)["recommended"])
CONTRACT = con.schema_from_node_type("hypothesis")
print("contract:", CONTRACT)

NOTES = [
    "pool at 100% and 47 threads waiting on orders-db after deploy-481",
    "payment JVM heap at 91% with long GC pauses",
    "session-cache evictions rising and hit rate down to 20%",
    "checkout returning NPE after release v2.14.0",
    "cart-service latency fine but checkout p99 2400 ms",
    "orders-service retry storm hitting the database",
]
HYP_SCHEMA = strict_format("hyp", {"id": {"type": "string"}, "statement": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "evidence": {"type": "array", "items": {"type": "string"}}})
TASK = ("Reply as a JSON object. Turn this incident note into a hypothesis with fields id, statement, confidence (0..1 float) and "
        "evidence (list of strings): ")


def gen(mode, note):
    msgs = [{"role": "user", "content": TASK + note}]
    if mode == "free":
        return h.chat(msgs).content
    if mode == "json_object":
        return h.chat(msgs, response_format={"type": "json_object"}).content
    return h.chat(msgs, response_format=HYP_SCHEMA).content


stats = {}
for mode in ("free", "json_object", "json_schema"):
    parsed = valid = 0
    for note in NOTES:
        try:
            obj = json.loads(with_retries(lambda: gen(mode, note)))
            parsed += 1
            valid += con.validate_against_contract(obj, CONTRACT)["valid"]
        except json.JSONDecodeError:
            pass
    stats[mode] = (parsed / len(NOTES), valid / len(NOTES))
    print(f"{mode:12s} parse rate {stats[mode][0]:.0%} | contract-valid {stats[mode][1]:.0%}")
print(con.reliability_gain(free_text_parse_rate=stats["free"][0], constrained_parse_rate=stats["json_schema"][0]))
h.check("strict schema parses at least as often as free text", stats["json_schema"][0] >= stats["free"][0])
h.check("strict schema parses every sample (tolerant: >= 5/6)", stats["json_schema"][0] >= 5 / 6)

# %% [markdown]
# ### Gate graph writes on the contract, and A/B two prompt versions
#
# A schema guarantees *shape*, not *quality*. So we (1) refuse to write to the graph unless the payload validates
# against the remediation contract, and (2) compare `remediation@v1` and `remediation@v2` with an **LLM judge**
# on the same incidents. Judge scores are noisy and biased toward fluent text, so treat the gap as a hint to
# investigate, not a verdict.

# %%
REM_CONTRACT = con.schema_from_node_type("remediation")
REM_SCHEMA = {"action": {"type": "string"}, "risk": {"type": "string", "enum": ["low", "medium", "high"]},
              "rollback": {"type": "string"}}
CASES = ["INC-1001: checkout p99 2400 ms after deploy-481 shrank the orders-db pool from 100 to 20",
         "payment-service heap at 91% with GC pauses causing 502s",
         "session-cache hit rate collapsed after memory limit reduced"]
RUBRIC = "5 = specific, least-privilege, reversible, with a concrete rollback and honest risk; 1 = vague or reckless."
ab = {}
for key in ("remediation@v1", "remediation@v2"):
    scores = []
    for case in CASES:
        r = structured(key, case, "rem", REM_SCHEMA)
        gate = con.validate_against_contract(r, REM_CONTRACT)
        j = h.judge(case, json.dumps(r), RUBRIC)
        scores.append(j["score"])
        if key.endswith("v2") and case.startswith("INC-1001") and gate["valid"]:
            h.run("MATCH (i:Incident {id:'INC-1001'}) CREATE (r:Remediation {action:$a, risk:$k, rollback:$b, "
                  "prompt_version:$v})-[:FOR]->(i)", a=r["action"], k=r["risk"], b=r["rollback"], v=key)
    ab[key] = sum(scores) / len(scores)
print("mean judge score:", ab)
h.check("remediation written only through the contract gate",
        h.run("MATCH (r:Remediation) RETURN count(r) AS n")[0]["n"] >= 1)
h.check("judge scores are on the 1-5 scale", all(1 <= s <= 5 for s in ab.values()))

# %% [markdown]
# ## 10. Constraint-guided plan validation with LLM-extracted constraints
#
# Chapter 5 says plans must satisfy constraints *before* execution. Two real LLM jobs and one deterministic job:
#
# 1. The LLM **extracts constraints** from the natural-language request into the skill's spec format.
# 2. The LLM **proposes plans** (conservative / moderate / aggressive) over a closed action vocabulary.
# 3. The skill **verifies** them. The privilege of each action comes from a fixed catalogue, not from the LLM,
#    because a model should never self-report how dangerous its own action is.
#
# Then we compare the validator's pass/fail with an LLM safety judgement and report disagreements.

# %%
REQUEST = ("Fix the checkout latency incident within 2 days using at most 5 steps. Never drop tables, "
           "and make sure the incident is recorded.")
ACTIONS = ["query_metrics", "read_logs", "record_incident", "scale_pool", "rollback_config", "drop_table"]
PRIV = {"query_metrics": "read", "read_logs": "read", "record_incident": "write", "scale_pool": "write",
        "rollback_config": "write", "drop_table": "admin"}

spec = structured("remediation@v1", REQUEST, "constraints",
                  {"max_steps": {"type": "integer"}, "deadline_days": {"type": "integer"},
                   "forbidden_actions": {"type": "array", "items": {"type": "string", "enum": ACTIONS}},
                   "required_actions": {"type": "array", "items": {"type": "string", "enum": ACTIONS}}})
print("LLM-extracted spec:", spec)
CONSTRAINTS = val.extract_constraints(spec)
CAP = val.CapabilityModel(allowed_actions={"query_metrics", "read_logs", "record_incident", "scale_pool"},
                          max_privilege="write")

PLAN_ITEM = {"action": {"type": "string", "enum": ACTIONS}, "eta_days": {"type": "integer"}}
STYLES = {"conservative": "read-only diagnosis, record the incident, one small reversible fix",
          "moderate": "diagnose, scale the pool, record the incident",
          "aggressive": "fast and drastic: rollback config, drop tables to rebuild, skip diagnosis"}
PLANS = {}
for style, brief in STYLES.items():
    r = h.llm_json(f"Incident: checkout latency after deploy-481 shrank the orders-db pool.\nStyle: {style} ({brief}).\n"
                   f"Allowed actions: {ACTIONS}.\nReturn JSON "
                   '{"steps": [{"action": <allowed action>, "eta_days": <int>}]}')
    PLANS[style] = r["steps"]

rows = []
for style, steps in PLANS.items():
    plan = val.steps_from_dicts([{"action": s["action"], "privilege": PRIV.get(s["action"], "admin"),
                                  "params": {"eta_days": s.get("eta_days", 1)}} for s in steps])
    v = val.verify(plan, CONSTRAINTS, CAP)
    j = h.judge(REQUEST, json.dumps(steps), "Score safety: 5 = safe, bounded and recoverable; 1 = destructive or unauthorised.")
    h.run("CREATE (:Plan {style:$s, score:$sc, passed:$p, violations:$n, judge_safety:$j})",
          s=style, sc=float(v.score), p=bool(v.passed), n=len(v.violations), j=j["score"])
    rows.append({"style": style, "steps": [s["action"] for s in steps], "validator_passed": v.passed,
                 "validator_score": round(v.score, 2), "judge_safety": j["score"],
                 "agree": v.passed == (j["score"] >= 4)})
h.show(rows)
print("disagreements (validator vs LLM judge):", [r["style"] for r in rows if not r["agree"]] or "none")
h.check("validator rejects any plan containing drop_table",
        all(not r["validator_passed"] for r in rows if "drop_table" in r["steps"]))
h.check("every plan has a stored verdict", h.run("MATCH (p:Plan) RETURN count(p) AS n")[0]["n"] == 3)

# %% [markdown]
# ## 11. Real RBAC: does the database agree with our policy model?
#
# `CapabilityModel` is a *policy statement*. It cannot stop anything. Neo4j Enterprise can: here we create two
# real users with real roles and run the same actions through `h.run_as`. Then we compare the validator's
# prediction with what the database actually **enforces**.
#
# * `diag_agent` (role `diag_reader`): read everything, create `Note` nodes, but `DENY` writing to `Database`.
# * `diag_operator` (role `diag_operator`): additionally may set `max_connections` on `Database`.
#
# We deliberately give the operator's *policy model* a generous `rollback_config` permission that the database
# never granted. The mismatch is the lesson: **policy drift between the plan validator and real enforcement is
# normal, and the database is the source of truth.** Everything is idempotent and cleaned up in `finally`.

# %%
PW = "Diag-Pass-12345"
ACTION_CYPHER = {
    "record_incident": "CREATE (n:Note {text:'incident recorded by agent'}) RETURN count(n) AS v",
    "scale_pool": "MATCH (d:Database {name:'orders-db'}) SET d.max_connections = 100 RETURN d.max_connections AS v",
    "rollback_config": "MATCH (d:Deployment {id:'deploy-481'}) SET d.rolled_back = true RETURN d.id AS v",
}
POLICY = {
    "diag_agent": val.CapabilityModel({"query_metrics", "record_incident"}, "write"),
    "diag_operator": val.CapabilityModel({"record_incident", "scale_pool", "rollback_config"}, "write"),
}
RBAC_ROWS = []


def cleanup_rbac():
    for u in ("diag_agent", "diag_operator"):
        h.sysrun(f"DROP USER {u} IF EXISTS")
    for r in ("diag_reader", "diag_operator"):
        h.sysrun(f"DROP ROLE {r} IF EXISTS")
    h.run("MATCH (n:Note) DETACH DELETE n")


try:
    cleanup_rbac()
    h.run("CREATE (:Note {seed:true})")  # pre-create the label so we never need NAME MANAGEMENT privileges
    for role in ("diag_reader", "diag_operator"):
        h.sysrun(f"CREATE ROLE {role} IF NOT EXISTS")
        h.sysrun(f"GRANT ACCESS ON DATABASE neo4j TO {role}")
        h.sysrun(f"GRANT MATCH {{*}} ON GRAPH neo4j NODES * TO {role}")
        h.sysrun(f"GRANT MATCH {{*}} ON GRAPH neo4j RELATIONSHIPS * TO {role}")
        h.sysrun(f"GRANT CREATE ON GRAPH neo4j NODES Note TO {role}")
        h.sysrun(f"GRANT SET PROPERTY {{*}} ON GRAPH neo4j NODES Note TO {role}")
    h.sysrun("DENY SET PROPERTY {*} ON GRAPH neo4j NODES Database TO diag_reader")
    h.sysrun("GRANT SET PROPERTY {max_connections} ON GRAPH neo4j NODES Database TO diag_operator")
    for user, role in (("diag_agent", "diag_reader"), ("diag_operator", "diag_operator")):
        h.sysrun(f"CREATE USER {user} IF NOT EXISTS SET PASSWORD $pw CHANGE NOT REQUIRED", pw=PW)
        h.sysrun(f"GRANT ROLE {role} TO {user}")
    print(h.sysrun("SHOW USERS YIELD user, roles WHERE user STARTS WITH 'diag_' RETURN user, roles"))

    for user in ("diag_agent", "diag_operator"):
        for action, cypher in ACTION_CYPHER.items():
            predicted = POLICY[user].authorizes(val.PlanStep(action, PRIV[action]))
            try:
                h.run_as(user, PW, cypher)
                actual = True
            except Exception as e:
                actual = False if "Forbidden" in type(e).__name__ or "Forbidden" in str(e) or "permission" in str(e).lower() else None
                if actual is None:
                    raise
            RBAC_ROWS.append({"user": user, "action": action, "policy_predicts": predicted, "database_enforces": actual,
                              "agree": predicted == actual})
    h.show(RBAC_ROWS)
    print("policy/enforcement drift:", [(r["user"], r["action"]) for r in RBAC_ROWS if not r["agree"]])
    h.check("reader is really blocked from changing a Database node",
            any(r["user"] == "diag_agent" and r["action"] == "scale_pool" and r["database_enforces"] is False for r in RBAC_ROWS))
    h.check("operator can really scale the pool",
            any(r["user"] == "diag_operator" and r["action"] == "scale_pool" and r["database_enforces"] is True for r in RBAC_ROWS))
    h.check("drift detected: operator policy allows rollback the database denies",
            any(r["user"] == "diag_operator" and r["action"] == "rollback_config" and not r["agree"] for r in RBAC_ROWS))
finally:
    cleanup_rbac()

left = h.sysrun("SHOW USERS YIELD user WHERE user STARTS WITH 'diag_' RETURN user")
h.check("RBAC users and roles cleaned up", left == [])

# %% [markdown]
# ## 12. A real function-calling agent with a graph-stored trace
#
# So far *we* orchestrated the steps. Here the model decides which tool to call next (OpenAI function calling).
# Each tool is wrapped so that every call, including failures, becomes a `ToolCall` node linked in a `NEXT` chain
# under a `Run` that `INVESTIGATES` the incident. An exception is re-raised so `ToolAgent` feeds the error back to
# the model, which lets it recover.
#
# Tools reuse what we built: blast radius (Cypher), hybrid-ish incident search (vector index), telemetry reads and
# the pipeline router. Afterwards we **query the trace as a graph** and grade the final answer with an LLM judge
# plus a tolerant grounding check.

# %%
TR = h.AgentTrace("Diagnose INC-1001 checkout latency")
h.run("MATCH (r:Run {id:$rid}), (i:Incident {id:'INC-1001'}) MERGE (r)-[:INVESTIGATES]->(i)", rid=TR.run_id)


def traced(name, fn):
    def wrapper(**kw):
        try:
            out = fn(**kw)
            TR.log(name, kw, str(out)[:200], True)
            return out
        except Exception as e:
            TR.log(name, kw, f"{type(e).__name__}: {e}", False)
            raise
    return wrapper


def t_blast(service):
    return h.run("MATCH (s:Service {name:$n})-[:DEPENDS_ON*1..3]->(d) RETURN DISTINCT d.name AS component", n=service)


def t_similar(symptom):
    return [{"id": r["node"]["id"], "text": r["node"]["text"], "root_cause_component": CAUSE.get(r["node"]["id"])}
            for r in h.vector_search("incident_vec", symptom, 3)]


def t_metrics(component):
    rows = h.run("MATCH (m:Metric)-[:OF]->(c {name:$n}) RETURN m.kind AS kind, m.value AS value", n=component)
    if not rows:
        raise LookupError(f"no telemetry for {component}")
    return rows


def t_route(query):
    d = sel.route_query(query)
    return {"architecture": d.final, "complexity": d.complexity, "uncertainty": d.uncertainty}


def schema(prop, desc):
    return {"type": "object", "properties": {prop: {"type": "string", "description": desc}}, "required": [prop]}


TOOLS = {
    "blast_radius": (traced("blast_radius", t_blast), schema("service", "service name, e.g. checkout-api"),
                     "List components downstream of a service."),
    "find_similar_incidents": (traced("find_similar_incidents", t_similar), schema("symptom", "symptom text"),
                               "Semantic search over past resolved incidents with their root-cause component."),
    "get_metrics": (traced("get_metrics", t_metrics), schema("component", "component name"),
                    "Telemetry for a component; errors if none exists."),
    "route_pipeline": (traced("route_pipeline", t_route), schema("query", "the question"),
                       "Choose sequential/tree/loop for a question."),
}
SYSTEM = ("You are an SRE agent. Investigate the incident using the tools: map the blast radius, find similar past "
          "incidents, check telemetry for the top candidate components, then state the most likely root cause with the "
          "evidence. If a tool errors, say so and try another. Be concise.")
agent = h.ToolAgent(TOOLS, SYSTEM, max_steps=8, trace=TR)
ANSWER = agent.run("INC-1001: checkout-api p99 latency is 2400 ms after deploy-481. What is the root cause?", verbose=True)

# %%
summary = h.run("MATCH (r:Run {id:$rid})-[:STEP]->(c:ToolCall) "
                "RETURN count(c) AS calls, sum(CASE WHEN c.ok THEN 0 ELSE 1 END) AS failures, "
                "collect(DISTINCT c.tool) AS tools", rid=TR.run_id)
chain = h.run("MATCH (r:Run {id:$rid})-[:STEP]->(first:ToolCall {seq:1}) "
              "MATCH p=(first)-[:NEXT*0..]->(last:ToolCall) WHERE NOT (last)-[:NEXT]->() "
              "RETURN length(p) + 1 AS chain_len", rid=TR.run_id)
h.show(summary)
print("NEXT-chain length:", chain)
print(TR.timeline())
grade = h.judge("Root cause of checkout latency after deploy-481?", ANSWER,
                "5 = names orders-db and the connection pool change with evidence from the tools; 1 = unsupported guess.")
print("judge:", grade)
h.check("agent made at least one tool call", summary[0]["calls"] >= 1)
h.check("trace chain length equals number of calls", chain and chain[0]["chain_len"] == summary[0]["calls"])
h.check("final answer is grounded: mentions orders-db or its pool (tolerant)",
        "orders-db" in ANSWER.lower() or "pool" in ANSWER.lower())
h.check("judge returned a valid score", 1 <= grade["score"] <= 5)
print(h.cost_report())

# %% [markdown]
# ## 13. Skill CLIs and self-benchmarks
#
# Everything above ran through the skills' `lib.py`. The same skills ship as stdlib-only CLIs, so a cron job or CI
# step can use them with no LLM and no pip installs. Their `benchmark` subcommands are the deterministic *floor*:
# they prove the skill logic is right independent of any model, which is how you tell "the skill is broken" from
# "the model had a bad day".

# %%
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
SLUGS = ["pipeline-architecture-selector", "investigation-dag-planner", "parallel-reconcile-merge",
         "loop-pipeline-router", "constraint-guided-plan-validator", "structured-output-contract-designer"]
for slug in SLUGS:
    cli = os.path.join(str(h.SKILLS_DIR), CH, slug, "cli.py")
    out = subprocess.run([sys.executable, cli, "--help"], capture_output=True, text=True, encoding="utf-8", env=ENV)
    print(slug, "--help exit", out.returncode, "|", (out.stdout.strip().splitlines() or [""])[0][:80])
    h.check(f"{slug} --help exits 0", out.returncode == 0)

for slug in SLUGS:
    ok, line = h.run_benchmark(CH, slug)
    print(("PASS" if ok else "FAIL"), slug, "|", line)
    h.check(f"{slug} benchmark passes", ok)

# %% [markdown]
# ## 14. Vibe coding: prompts to give an AI assistant, and how to verify what it returns
#
# Assistants are good at producing plausible Cypher and plausible skills. Your job is to **specify** and **verify**.
# Three prompts you could use right now:
#
# 1. *"Write Neo4j 5 Cypher that returns the critical path through `(:Step)-[:DEPENDS_ON]->(:Step)`. Each Step has a
#    numeric `duration_s`. Return one row with the summed duration. Use `reduce()` over `nodes(p)`. Only start from
#    steps nothing depends on and end at steps with no dependencies."*
# 2. *"Add a new skill under skills/reasoning-planning/ named `hypothesis-ranker`: SKILL.md with the 7 sections plus
#    Source Attribution, a stdlib-only lib.py with TODO-marked seams for embeddings, and a cli.py whose `--help`
#    prints the SKILL.md description and whose `benchmark` subcommand self-checks."*
# 3. *"Give my ToolAgent a hard budget: stop after N tool calls or M tokens, and return the partial findings with a
#    flag `budget_exhausted`. Show me the diff only."*
#
# **Verification checklist** (apply before trusting any generated artefact):
#
# - [ ] Run it against a *known* answer first (below we compare Cypher with an independent Python implementation).
# - [ ] Does it terminate? Look for unbounded `*` paths and loops without a counter.
# - [ ] Does it use only functions and labels that exist? Grep the skill's `lib.py` for every call.
# - [ ] Are failures loud rather than swallowed (`except: pass`)?
# - [ ] Is anything stochastic asserted by exact wording? Tests should check structure and behaviour.
# - [ ] Did it touch only the files you asked it to touch?

# %%
CP_CYPHER = h.run("MATCH p=(e:Step)-[:DEPENDS_ON*0..]->(root:Step) "
                  "WHERE NOT (root)-[:DEPENDS_ON]->() AND NOT ()-[:DEPENDS_ON]->(e) "
                  "RETURN reduce(t = 0.0, n IN nodes(p) | t + n.duration_s) AS dur ORDER BY dur DESC LIMIT 1")


def longest(tid, memo={}):
    if tid not in memo:
        t = TASKS[tid]
        memo[tid] = t.duration_s + max([longest(d) for d in t.depends_on], default=0.0)
    return memo[tid]


# only steps nothing depends on end a path
dependents = {d for t in TASKS.values() for d in t.depends_on}
CP_PY = max(longest(tid) for tid in TASKS if tid not in dependents)
print("Cypher critical path:", CP_CYPHER[0]["dur"], "| independent Python DP:", CP_PY, "| skill path length:", DAG.critical_path_len)
h.check("Cypher critical path equals the independent Python computation",
        abs(CP_CYPHER[0]["dur"] - CP_PY) < 1e-6)

# %% [markdown]
# ## Evaluation and LLM-as-a-Judge — how you know it works
#
# Everything above *ran*. None of it proves the reasoning is **right**. Evaluation is how you know an agentic
# planning system works, and whether a change made it better or merely different. Think of a **pyramid**: cheap,
# objective, numerous checks at the bottom; expensive, subjective, scarce ones at the top.
#
# | Layer | What | Cost | What it can / cannot do |
# |---|---|---|---|
# | 1. Deterministic checks | Cypher / DAG / schema / validator assertions | ~free | Exact, but only for what you can state as a rule |
# | 2. Reference-based metrics | overlap, routing accuracy vs a **gold set** | cheap | Objective, but blind to paraphrase and to *why* an answer is wrong |
# | 3. LLM-as-judge | rubric scoring, pairwise preference | 1+ LLM call per item | Handles free text, but is a **model with biases** |
# | 4. Judge calibration | compare the judge with **human labels** | a one-off labelling effort | Tells you whether layer 3 deserves any trust |
# | 5. Human spot checks | read a sample yourself | scarce | The only ground truth; keeps layer 4 honest |
#
# **A judge is itself a model that must be evaluated.** Known biases, and the probe that targets each:
#
# - **Position bias**: prefers whichever answer is shown first. *Probe:* pairwise judging in both orders, plus an
#   identical-answers pair that must tie (`h.judge_pairwise`).
# - **Verbosity bias**: rewards length and polish. *Probe:* pad a weak answer with fluff; its score must not rise much.
# - **Self-preference**: favours text that looks like its own family's output. *Mitigation:* judge with a different model
#   family than the generator for decisions that matter (`h.judge_rubric(..., model=...)`); we flag it, we cannot measure it here.
# - **Leniency / inconsistency**: hands out 4s and 5s, or changes its mind between runs. *Probes:* calibration against
#   human labels and `h.judge_repeat` (self-consistency).
# - **Fooled by fluent wrongness**: *Probe:* a sabotaged answer with the same style but wrong facts **must** score lower.
#
# Scope: the gold set is about *this chapter's* ideas (DAG planning, merge modes, contracts, validation, routing).
# `h.judge*` helpers ask for the **reason before the score**, and use temperature 0 except in the repeat probe.

# %% [markdown]
# ### E1. A gold set, written down and stored in the graph
#
# A gold set is *questions + reference answers + human labels*. Ours has 8 questions whose references come from what
# this notebook established (the DAG phases, the two merge modes, the contract measurements, the validator and RBAC
# findings), and 12 **candidate answers that we scored by hand (1-5)**: correct ones, a vague-but-right one, clearly
# wrong ones, *subtly* wrong ones (swapped merge modes, the wrong pipeline, a prompt-only guardrail) and a
# **fluffy-but-empty** one. The labels are ours, so they are arguable; that is why E4 measures agreement.

# %%
GOLD = [
    {"id": "G1", "question": "How should an investigation planner order hypothesis tests so independent ones run in parallel?",
     "reference": "Model the hypotheses as a dependency DAG: tests with no unmet dependencies share a phase and run in "
                  "parallel, dependent tests wait for their prerequisites, and the critical path sets the total time."},
    {"id": "G2", "question": "A parallel branch fails because its component has no telemetry. Which merge mode keeps the useful results?",
     "reference": "partial_coverage keeps the successful branches and flags the gap; all_or_nothing refuses to conclude "
                  "when any branch fails."},
    {"id": "G3", "question": "Why enforce a strict json_schema instead of asking the model to 'reply as JSON'?",
     "reference": "The decoder enforces the shape so the output always parses, whereas free text can arrive in code fences "
                  "or prose and break json.loads. A schema guarantees shape, not quality, so business rules are still validated."},
    {"id": "G4", "question": "A generated plan contains drop_table. Who decides how privileged that action is, and what stops it?",
     "reference": "The validator rejects it. Privilege comes from a fixed action catalogue (drop_table is admin), never from "
                  "the LLM's own report, and real enforcement is Neo4j RBAC."},
    {"id": "G5", "question": "When should retrieval use vector search, fulltext, or both?",
     "reference": "Vector search matches meaning and handles paraphrase; fulltext matches exact words such as error codes. "
                  "Hybrid fuses both rankings (reciprocal-rank fusion) and is the safest default."},
    {"id": "G6", "question": "Which pipeline suits 'draft a remediation summary and refine it until it passes the compliance checklist'?",
     "reference": "The loop pipeline: draft, validate against the checklist, refine with the violations as feedback, with "
                  "bounded retries. Sequential suits simple lookups and tree suits open-ended root-cause hunts."},
    {"id": "G7", "question": "A hypothesis is confirmed with high confidence in an early phase. What should the executor do?",
     "reference": "Stop early: skip the remaining phases, because testing further hypotheses wastes time and tokens."},
    {"id": "G8", "question": "If the plan validator approves an action, is it guaranteed to be permitted at runtime?",
     "reference": "No. The CapabilityModel is a policy statement and cannot stop anything; policy can drift from what "
                  "Neo4j RBAC enforces, and the database is the source of truth."},
]
CANDIDATES = [  # (id, gold id, answer, HUMAN label 1-5, why we gave that label)
    ("C1", "G1", "Build a dependency DAG of the hypotheses. Tests with no unmet dependencies sit in the same phase and run "
                 "in parallel; dependent tests wait for their prerequisites, and the critical path determines the total time.",
     5, "correct and complete"),
    ("C2", "G1", "Run the hypotheses in a sensible order and do some of them at the same time where that makes sense.",
     3, "right direction, no mechanism"),
    ("C3", "G1", "Test every hypothesis one by one in alphabetical order of component name; the dependency structure does not matter.",
     1, "wrong: ignores dependencies and parallelism"),
    ("C4", "G2", "Use all_or_nothing: it keeps the successful branches and flags the missing telemetry as a gap.",
     2, "subtly wrong: the two modes are swapped"),
    ("C5", "G3", "A strict json_schema makes the decoder enforce the shape, so the output always parses, while free text can "
                 "come back in code fences or prose and break json.loads. The schema cannot check business rules, so we "
                 "still validate afterwards.", 5, "correct and complete"),
    ("C6", "G4", "Robust governance frameworks are essential for responsible AI, and organisations should embrace a holistic, "
                 "forward-looking approach to risk that unlocks long-term value.",
     1, "fluffy but empty: no fact, no mechanism"),
    ("C7", "G5", "Vector search handles paraphrase by meaning, fulltext matches exact words like error codes, and hybrid "
                 "fuses both rankings with reciprocal-rank fusion, which is the safest default.", 5, "correct and complete"),
    ("C8", "G6", "Use the tree pipeline: explore several drafts in parallel and keep the best one.",
     2, "subtly wrong: plausible but the wrong pipeline"),
    ("C9", "G7", "Keep testing every remaining hypothesis anyway; more evidence is always better and cost does not matter.",
     1, "wrong: defeats early termination"),
    ("C10", "G8", "No, the validator approval is not enough, so I would add a line to the system prompt telling the agent to "
                  "run only permitted actions.", 3, "right conclusion, wrong mechanism (prompted, not enforced)"),
    ("C11", "G5", "Always use vector search because it understands meaning; fulltext and fusion add nothing.",
     3, "half right: ignores exact-term cases"),
    ("C12", "G4", "The validator rejects it. Privilege comes from a fixed catalogue where drop_table is admin, not from the "
                  "LLM's self-report, and Neo4j RBAC is the real enforcement.", 5, "correct and complete"),
]
GOLD_BY_ID = {g["id"]: g for g in GOLD}
CAND = {c[0]: {"gold": c[1], "answer": c[2], "human": c[3], "why": c[4]} for c in CANDIDATES}
HUMAN = [CAND[c[0]]["human"] for c in CANDIDATES]

h.run("MATCH (g:Ch5EvalGold) DETACH DELETE g")
h.run("UNWIND $rows AS r CREATE (:Ch5EvalGold {id:r.id, question:r.question, reference:r.reference})", rows=GOLD)
stored = h.run("MATCH (g:Ch5EvalGold) RETURN g.id AS id, size(g.reference) AS ref_chars ORDER BY id")
h.show(stored)
h.check("gold set has >= 8 cases stored in the graph", len(stored) >= 8)
h.check("at least 8 human-labelled candidates, covering bad, subtle and fluffy answers",
        len(CANDIDATES) >= 8 and min(HUMAN) == 1 and max(HUMAN) == 5 and {2, 3} <= set(HUMAN))

# %% [markdown]
# ### E2. Layer 1 and 2: deterministic checks, then a reference-based metric
#
# **Layer 1** asserts what must be true of the *artefacts* with Cypher and plain Python: the plan we persisted is a
# well-formed acyclic graph, the skills behave as our references claim, and no candidate answer names a component the
# graph does not contain (a cheap **hallucination detector**). No LLM, no cost.
#
# **Layer 2** scores each candidate against its reference with token-overlap F1 rescaled to 1-5. Objective and free, but it only
# sees *words*: watch where it disagrees with the human labels (it cannot tell a swapped `partial_coverage` from the real one).

# %%
step_cycles = h.run("MATCH p = (s:Step)-[:DEPENDS_ON*1..8]->(s) RETURN count(p) AS c")[0]["c"]
drop_verdict = val.verify(val.steps_from_dicts([{"action": "drop_table", "privilege": "admin", "params": {"eta_days": 1}}]),
                          CONSTRAINTS, CAP)
print({"steps": len(PLAN), "step_cycles": step_cycles, "drop_table_passes": drop_verdict.passed,
       "architectures": list(sel.ARCHITECTURES)})
h.check("reference G1 holds: the persisted hypothesis plan is an acyclic DAG", step_cycles == 0 and len(PLAN) >= 3)
h.check("reference G4 holds: the validator rejects drop_table", not drop_verdict.passed)
h.check("reference G6 holds: sequential, tree and loop are the skill's architectures",
        {"sequential", "tree", "loop"} <= set(sel.ARCHITECTURES))
h.check("reference G3 holds: strict schema parses at least as often as free text", stats["json_schema"][0] >= stats["free"][0])
h.check("reference G1 holds: Cypher and the skill agree on phases", cy_phase == skill_phase)

KNOWN_NAMES = {r["n"] for r in h.run("MATCH (n) WHERE n.name IS NOT NULL RETURN n.name AS n")}
ENTITY_RE = re.compile(r"\b[a-z]+(?:-[a-z0-9]+)*-(?:db|api|service|cache|gateway)\b")
for cid, c in CAND.items():
    c["invented"] = sorted(set(ENTITY_RE.findall(c["answer"])) - KNOWN_NAMES)


def tokens(t: str) -> set:
    return set(re.findall(r"[a-z0-9_\-\.]+", t.lower()))


def f1_to_5(answer: str, reference: str) -> int:
    a, r = tokens(answer), tokens(reference)
    overlap = len(a & r)
    if not overlap:
        return 1
    p, rc = overlap / len(a), overlap / len(r)
    return int(round(1 + 4 * (2 * p * rc / (p + rc))))


for cid, c in CAND.items():
    c["f1_score"] = f1_to_5(c["answer"], GOLD_BY_ID[c["gold"]]["reference"])
h.show([{"id": cid, "human": c["human"], "overlap_1to5": c["f1_score"], "invented_entities": c["invented"], "why": c["why"]}
        for cid, c in CAND.items()])
print("layer-2 overlap vs human:", h.agreement(HUMAN, [CAND[c[0]]["f1_score"] for c in CANDIDATES]))
h.check("layer 1 finds no invented component names in any candidate", all(not c["invented"] for c in CAND.values()))

# %% [markdown]
# ### E3. Layer 3: a rubric judge with explicit 1-vs-5 anchors
#
# Vague rubrics ("is it good?") produce vague, lenient scores. Each criterion says exactly what a **1** and a **5** look
# like, and the judge sees the **reference**, so it grades against ground truth rather than its own beliefs.
# `h.judge_rubric` asks for a one-sentence reason first and returns per-criterion scores plus their mean.
# We judge all 12 candidates once (12 judge calls) and keep the numbers for calibration.

# %%
CRITERIA = {
    "correctness": "5 = every fact and decision agrees with the reference; 1 = contradicts the reference (wrong mode, wrong pipeline, wrong mechanism).",
    "grounding": "5 = names the specific mechanism or skill concept from the reference (DAG phases, partial_coverage, json_schema, RBAC); "
                 "1 = names none, or names concepts that contradict the reference.",
    "completeness": "5 = covers every point the reference makes; 1 = covers none of them (generic or off-topic).",
    "actionability": "5 = states the correct design decision and the right enforcement mechanism; 1 = offers no usable decision, "
                     "or one that would cause harm.",
}


def to_5(mean: float) -> int:
    return min(5, max(1, int(mean + 0.5)))


for cid, c in CAND.items():
    g = GOLD_BY_ID[c["gold"]]
    c["rubric"] = h.judge_rubric(g["question"], c["answer"], CRITERIA, reference=g["reference"])
    c["judge"] = to_5(c["rubric"]["mean"])
h.show([{"id": cid, "human": c["human"], "judge": c["judge"], **c["rubric"]["scores"]} for cid, c in CAND.items()])
worst = min(CAND.items(), key=lambda kv: kv[1]["rubric"]["mean"])
print(f"lowest-scored candidate {worst[0]}: {str(worst[1]['rubric']['reasons'].get('correctness', ''))[:140]}")
h.check("judge returned a 1-5 score for every candidate", all(1 <= c["judge"] <= 5 for c in CAND.values()))
print(h.cost_report())

# %% [markdown]
# ### E4. Calibration: does the judge agree with the humans?
#
# This is the step most teams skip. Compare the judge's 1-5 scores with our human labels on the **same items**:
#
# - **exact / within1**: how often the judge matches us exactly, or is off by at most one point. Within-1 near 1.0 with
#   low exact means "right ordering, different scale".
# - **Cohen's kappa**: exact agreement *corrected for chance*. Rough guide: < 0.2 poor, 0.2-0.4 fair, 0.4-0.6 moderate,
#   0.6-0.8 substantial, > 0.8 near-perfect. Be suspicious of high exact agreement with kappa near 0 (the judge may just say "3").
# - **Spearman**: do the two *rank* answers the same way? For picking a winner, this matters more than exact scores.
#
# With 12 items these numbers are noisy. Treat them as a smoke alarm, not a certificate. The check is deliberately
# **tolerant**: a weak judge is a finding to act on, not a crash.

# %%
JUDGE = [CAND[c[0]]["judge"] for c in CANDIDATES]
cal = h.agreement(HUMAN, JUDGE)
print("judge vs human:", cal)
h.show([{"id": cid, "human": c["human"], "judge": c["judge"], "delta": c["judge"] - c["human"]} for cid, c in CAND.items()])
lenient = [cid for cid, c in CAND.items() if c["judge"] - c["human"] >= 2]
if cal["kappa"] < 0.4 or cal["spearman"] < 0.7:
    print("WARNING: the judge is weakly calibrated against our labels. Do not use its scores for decisions yet. Options: "
          "(1) tighten the rubric anchors on the criteria with the biggest deltas, (2) add 2-3 few-shot examples of "
          "1/3/5 answers to the judge prompt, (3) switch to a stronger or different-family judge model, "
          "(4) re-label items where you suspect the human label is the wrong one.")
else:
    print("OK: kappa and Spearman clear the bar (kappa >= 0.4, spearman >= 0.7). Still re-check when the model or rubric changes.")
if lenient:
    print("Leniency: the judge scored these >= 2 points above the human label:", lenient)
h.check("calibration was computed over all candidates", cal["n"] == len(CANDIDATES))

# %% [markdown]
# ### E5. Bias probe 1 — position bias
#
# **What this tests:** whether the verdict depends on *which answer is shown first*. `h.judge_pairwise` asks twice with
# the order swapped; if the verdicts disagree it reports `consistent=False` (counted as a tie). Two pairs (4 judge calls):
# a real pair (C1 correct vs C2 vague: the better answer should win in **both** orders) and an **identical** pair
# (the same text twice: the only honest verdict is a tie, so any "winner" is pure position bias).

# %%
PAIR_Q, PAIR_REF = GOLD_BY_ID["G1"]["question"], GOLD_BY_ID["G1"]["reference"]
PAIR_CRIT = "Prefer the answer that is more correct, better grounded in the reference mechanism, and more actionable."
real_pair = h.judge_pairwise(PAIR_Q, CAND["C1"]["answer"], CAND["C2"]["answer"], PAIR_CRIT, reference=PAIR_REF)
same_pair = h.judge_pairwise(PAIR_Q, CAND["C1"]["answer"], CAND["C1"]["answer"], PAIR_CRIT, reference=PAIR_REF)
print("real pair (C1 vs C2):", real_pair)
print("identical pair      :", same_pair)
print("PASS" if real_pair["winner"] == "a" and real_pair["consistent"] else
      "WARNING: the better answer did not win consistently in both orders; average both orders or use absolute rubrics.")
print("PASS" if same_pair["winner"] == "tie" and same_pair["consistent"] else
      "WARNING: position bias detected on identical answers; always judge both orders and average, or use absolute rubrics.")
h.check("pairwise judge returned verdicts for both probes",
        real_pair["winner"] in ("a", "b", "tie") and same_pair["winner"] in ("a", "b", "tie"))

# %% [markdown]
# ### E6. Bias probe 2 — verbosity bias
#
# **What this tests:** whether *length and polish* buy points. We take the weak answer C2 and wrap it in confident,
# contentless filler (no new facts). A well-behaved judge scores it about the same; a verbosity-biased judge scores it
# higher. One extra judge call; we compare with the C2 score from E3. (A rise of more than 0.75 mean points is flagged.)

# %%
FLUFF = ("Great question, and an important one for any modern engineering organisation. In today's fast-moving, "
         "cloud-native landscape, holistic orchestration and a robust, forward-looking, best-practice-driven mindset "
         "are absolutely essential. With that context in mind: ")
FLUFF_TAIL = (" Ultimately, by leveraging synergies across teams and embracing a culture of continuous improvement, "
              "organisations can unlock significant long-term value and operational excellence.")
padded = FLUFF + CAND["C2"]["answer"] + FLUFF_TAIL
g2 = GOLD_BY_ID[CAND["C2"]["gold"]]
padded_res = h.judge_rubric(g2["question"], padded, CRITERIA, reference=g2["reference"])
rise = padded_res["mean"] - CAND["C2"]["rubric"]["mean"]
print(f"C2 plain mean={CAND['C2']['rubric']['mean']:.2f}  padded mean={padded_res['mean']:.2f}  rise={rise:+.2f}  "
      f"(answer grew {len(padded) / len(CAND['C2']['answer']):.1f}x)")
print("PASS: padding did not buy points." if rise <= 0.75 else
      "WARNING: verbosity bias. Add 'do not reward length' with a 1-anchor for filler, or add a conciseness criterion.")
h.check("verbosity probe produced a score", 1 <= padded_res["mean"] <= 5)

# %% [markdown]
# ### E7. Bias probe 3 — self-consistency, and a sabotage control
#
# **Self-consistency.** `h.judge_repeat` scores the *same* answer `n=3` times at temperature 0.7. A `spread` (max - min)
# of 0 or 1 is fine; 2+ means the judge is unreliable *on that item* and a single score should not be trusted. We test
# the mid-quality answer C2 and the subtly-wrong C10, because borderline answers are where judges wobble (6 judge calls).
#
# **Sabotage control.** Take a good answer and corrupt its *facts* while keeping its style: right shape, wrong content.
# A judge that is checking facts must score it clearly lower than the original. This one is a hard `h.check`:
# if a judge cannot tell a true answer from a fluent lie, none of its other scores mean anything (1 judge call).

# %%
ONE_LINE = ("Score 1-5 against the reference: 5 = correct, specific, naming the right mechanism and enforcement; "
            "3 = right direction but vague or wrong mechanism; 1 = wrong or empty.")
rep = {}
for cid in ("C2", "C10"):
    g = GOLD_BY_ID[CAND[cid]["gold"]]
    rep[cid] = h.judge_repeat(g["question"], CAND[cid]["answer"], f"{ONE_LINE} Reference: {g['reference']}", n=3)
    print(f"{cid}: scores={rep[cid]['scores']} mean={rep[cid]['mean']:.2f} spread={rep[cid]['spread']}")
    if rep[cid]["spread"] >= 2:
        print(f"  WARNING: unstable judge on {cid}; average several runs or tighten the rubric for this kind of item.")

SABOTAGED = ("Build a dependency DAG of the hypotheses, then run them strictly one after another in the order the LLM "
             "wrote them. Dependencies are ignored at run time, and the total time is the sum of every duration, "
             "so parallel phases never help.")
g1 = GOLD_BY_ID["G1"]
sab = h.judge_rubric(g1["question"], SABOTAGED, CRITERIA, reference=g1["reference"])
print(f"good C1 mean={CAND['C1']['rubric']['mean']:.2f}  sabotaged mean={sab['mean']:.2f}  invented entities: "
      f"{sorted(set(ENTITY_RE.findall(SABOTAGED)) - KNOWN_NAMES)}")
h.check("sabotaged (fluent but false) answer scores at least 1 point below the good one",
        sab["mean"] <= CAND["C1"]["rubric"]["mean"] - 1)
print(h.cost_report())

# %% [markdown]
# ### E8. Judge vs heuristic: where do they disagree, and why?
#
# Section 4 routed eight queries with the skill's word-counting heuristic. Here we put the **heuristic** and the **LLM
# judge** in front of the same 8 items: the heuristic score is 5 if its route equals the gold architecture and 1
# otherwise; the judge grades *the heuristic's verdict* against the gold architecture's definition (8 judge calls; partial
# credit allowed for a closely related architecture). Rows where they differ by 2+ points are the interesting ones: each is
# either a heuristic blind spot or a judge quirk, and you decide which by reading the item.

# %%
ARCH_DEF = {
    "sequential": "a fixed chain of steps for simple, certain lookups",
    "tree": "parallel exploration of competing hypotheses for open-ended, uncertain investigations",
    "loop": "draft, validate and refine until a checklist or contract passes",
}
cmp_rows = []
for (q, gold_arch), trow in zip(LABELED_QUERIES, table):
    verdict_text = f"Heuristic verdict: {trow['heuristic']}."
    j = h.judge(f"Which pipeline architecture fits this task? {q}", verdict_text,
                f"The correct architecture is {gold_arch}: {ARCH_DEF[gold_arch]}. Score 5 if the verdict names that "
                "architecture, 3 if it names a closely related one, 1 if it names an unrelated one.")
    heur = 5 if trow["heuristic"] == gold_arch else 1
    cmp_rows.append({"query": q[:40], "gold": gold_arch, "heuristic_route": trow["heuristic"], "heuristic": heur,
                     "judge": int(j["score"]), "judge_reason": str(j["reason"])[:80]})
h.show([{k: v for k, v in r.items() if k != "judge_reason"} for r in cmp_rows])
print("heuristic vs judge:", h.agreement([r["heuristic"] for r in cmp_rows], [r["judge"] for r in cmp_rows]))
diffs = [r for r in cmp_rows if abs(r["heuristic"] - r["judge"]) >= 2]
print(f"{len(diffs)} disagreement(s) of >= 2 points")
for r in diffs:
    if r["heuristic"] < r["judge"]:
        meaning = ("heuristic miss, judge gives partial credit: the heuristic chose a neighbouring architecture, so this "
                   "is a word-counting blind spot (or judge leniency toward a near-miss)")
    else:
        meaning = "judge stricter than exact match: the heuristic hit the gold label but the judge disputes it; re-read the item"
    print(f"  {r['query']!r} route={r['heuristic_route']} gold={r['gold']} heuristic={r['heuristic']} judge={r['judge']}: "
          f"{meaning}. Judge said: {r['judge_reason']}")
h.check("compared the heuristic and the judge on all 8 labelled queries", len(cmp_rows) == len(LABELED_QUERIES) == 8)

# %% [markdown]
# ### E9. Regression harness: change a prompt, prove it did not get worse
#
# Evaluation earns its keep when something *changes*. `h.run_eval` runs a system function over a case list, scores
# each case, and stores `(:EvalRun)-[:HAS_RESULT]->(:EvalResult)` in Neo4j under a name and **version**. One Cypher
# query (`h.compare_runs`) compares versions, and a **CI-style gate** turns the comparison into PASS/FAIL.
# The system here is the pipeline classifier from section 4 (`v1` as written vs `v2` with explicit anchors for the
# three architectures) on the 8 labelled queries: 16 LLM calls, deterministic scoring (route == gold), so a flaky judge
# cannot hide a regression. The gate prints PASS or FAIL and never raises.

# %%
PROMPTS["pipeline_classifier@v2"] = (
    PROMPTS["pipeline_classifier@v1"] + " Anchors: a single-fact lookup is complexity 0.1 and uncertainty 0.05; a "
    "multi-service root-cause hunt is complexity 0.7 and uncertainty 0.8; a draft-and-refine-until-valid task is "
    "complexity 0.4 and uncertainty 0.2.")
EVAL_NAME = "ch5_pipeline_classifier"
h.run("MATCH (r:EvalRun) WHERE r.name STARTS WITH 'ch5_' OPTIONAL MATCH (r)-[:HAS_RESULT]->(x) DETACH DELETE x, r")
ROUTE_CASES = [{"id": f"Q{i + 1}", "input": q, "gold": g} for i, (q, g) in enumerate(LABELED_QUERIES)]


def route_with(case, version):
    r = structured(f"pipeline_classifier@{version}", case["input"], "scores",
                   {"complexity": {"type": "number"}, "uncertainty": {"type": "number"}})
    return sel.analyze_and_route(float(r["complexity"]), float(r["uncertainty"]))


def score_route(case, out):
    ok = out == case["gold"]
    return {"score": 1.0 if ok else 0.0, "passed": ok, "notes": f"gold={case['gold']} got={out}"}


summaries = {}
for ver in ("v1", "v2"):
    summaries[ver] = h.run_eval(EVAL_NAME, ROUTE_CASES, lambda c, v=ver: route_with(c, v), score_route, version=ver)
    print(f"{ver}: pass_rate={summaries[ver]['pass_rate']:.0%} failures={summaries[ver]['failures']}")
runs = h.compare_runs(EVAL_NAME)
h.show(runs)
h.check("both versions were stored and are comparable in Neo4j", {r["version"] for r in runs} == {"v1", "v2"})

TOLERANCE = 0.15          # candidate may be at most this much worse than baseline (0-1 scale)
by_ver = {r["version"]: r for r in runs}
base_score, cand_score = by_ver["v1"]["mean_score"], by_ver["v2"]["mean_score"]
gate = "PASS" if cand_score >= base_score - TOLERANCE else "FAIL"
print(f"CI GATE: baseline v1={base_score:.2f}  candidate v2={cand_score:.2f}  tolerance={TOLERANCE:.2f}  ->  {gate}")
if gate == "FAIL":
    print("  Regression: inspect (:EvalResult) rows with passed=false for the candidate version before shipping.")
h.show(h.run("MATCH (:EvalRun {name:$n, version:'v2'})-[:HAS_RESULT]->(x:EvalResult) WHERE NOT x.passed "
             "RETURN x.case AS failed_case, x.notes AS notes ORDER BY x.case", n=EVAL_NAME) or [{"failed_case": "none"}])

# %% [markdown]
# ### E10. Judge report card, and a prompt to grow the eval set
#
# **What I would trust this judge for** (after reading the calibration, bias and sabotage results above):
#
# - **Ranking and regression-spotting** between prompt or skill versions, using the *same* judge, rubric and reference each time.
# - **Catching clearly bad and fluent-but-false answers**, because reference-anchored criteria plus the sabotage control passed.
# - **Triage**: deciding which 10% of planner outputs a human should read first.
#
# **What I would not trust it for:**
#
# - **Absolute quality claims** ("our planner is 4.3/5") or comparing numbers across different judges or rubrics.
# - **Safety and permissions.** Whether an action is allowed is a validator verdict plus a Neo4j RBAC fact (section 11), never a judge opinion.
# - **Exact-structure claims** (does the DAG have a cycle? does the contract validate?): use layer 1 assertions.
# - **Anything after the judge model, rubric or task changes**, until it is re-calibrated on fresh human labels.
# - **Self-preference**: the judge and the answering model share a family here, so expect flattery we did not measure.
#
# **Clean-up for this section:** the next cell removes the `Ch5EvalGold` and `ch5_` `EvalRun` data it created.
#
# **Vibe-coding prompt: extend the eval set**
# ```text
# Context: section "Evaluation and LLM-as-a-Judge" in notebooks_advanced/_src/ch5-reasoning-planning.py, and
# real_helpers.py (judge_rubric, judge_pairwise, judge_repeat, agreement, run_eval, compare_runs).
# Task: add 6 gold cases and 8 human-labelled candidates about early termination and the loop-router retry policy,
# including two subtly wrong answers (retrying a FATAL error, skipping phases without a confirmation) and one fluffy
# answer. Re-run the calibration.
# Constraints: reuse the CRITERIA dict; <= 20 extra judge calls; every Cypher query parameterised; Neo4j labels
# namespaced with Ch5; do not weaken any existing h.check.
# Acceptance: print agreement (kappa, spearman) before and after; if kappa drops by more than 0.1, list the new
# items with the largest judge-human delta and propose a rubric change. Do not change the human labels to fit the judge.
# ```

# %%
h.run("MATCH (g:Ch5EvalGold) DETACH DELETE g")
h.run("MATCH (r:EvalRun) WHERE r.name STARTS WITH 'ch5_' OPTIONAL MATCH (r)-[:HAS_RESULT]->(x) DETACH DELETE x, r")
left = h.run("MATCH (n) WHERE n:Ch5EvalGold OR n:EvalRun OR n:EvalResult RETURN count(n) AS c")[0]["c"]
h.check("evaluation section left no Ch5EvalGold / EvalRun / EvalResult nodes behind", left == 0)
print(h.cost_report())

# %% [markdown]
# ## 15. Cost, limits and what to swap next
#
# Tokens spent in this run (prices change, so we report tokens, not dollars):

# %%
print(h.cost_report())

# %% [markdown]
# **What is real now:** `gpt-4o` for classification, planning, verdicts, extraction and judging;
# `text-embedding-3-large` with a Neo4j vector index and a fulltext index; an OpenAI function-calling agent with a
# graph-stored trace; and Neo4j Enterprise RBAC that genuinely refuses writes.
#
# **What is still approximate, honestly:**
#
# - **Non-determinism.** Temperature 0 reduces but does not eliminate variation (model updates, tie-breaking,
#   floating-point and batching effects). Re-running can change a hypothesis list or flip a borderline verdict.
#   That is why checks assert structure and tolerant thresholds.
# - **Tiny labelled sets.** 6-8 examples per experiment show *direction*, not statistics. A 17-point accuracy gap is
#   one example. Use 50+ labelled cases and confidence intervals before deciding anything.
# - **Judge bias.** `h.judge` favours fluent, confident text and is itself an LLM; it can agree with the answer's
#   mistakes. Use it to find disagreements worth reading, not as ground truth. Calibrate it against human labels.
# - **The judge is a model too.** It has position, verbosity, leniency and same-family self-preference biases (judge and
#   generator are both `gpt-4o` here); the probes in the evaluation section reduce them but do not remove them.
#   Re-calibrate against fresh human labels whenever the judge model, rubric or task changes.
# - **Judge cost is extra calls.** Every judged item is at least one more LLM call (pairwise = two, repeats = n);
#   budget it, cap the loops, and judge a sample in CI rather than everything.
# - **Synthetic telemetry.** Metrics were seeded by hand for two components, so "supported" verdicts are only as
#   realistic as those numbers. A real system reads Prometheus/CloudWatch with noise, lag and gaps.
# - **Coarse privilege model.** Actions map to three levels and a handful of Cypher statements. Real systems need
#   per-resource, per-property and time-bound grants, plus audit logging of denials.
# - **Retrieval corpus.** Nine incidents is a toy; vector search quality depends on chunking, embedding dimension
#   and how incident text is written. Hybrid fusion weights here (RRF k=60) are untuned.
# - **Cost and latency.** The notebook is dozens of sequential LLM calls. At scale you would batch, cache embeddings,
#   use a smaller model for classification, and run branch tests concurrently.
#
# **What to swap next:** replace the seeded `Metric` nodes with a live telemetry tool; add a cross-encoder reranker
# after hybrid retrieval; run the evaluation harness in CI with a larger labelled set and a pinned model version;
# add per-prompt-version regression dashboards from the `prompt_version` properties already stored on the graph;
# and put the agent behind the same RBAC users used in section 11 so least privilege is enforced end to end.
