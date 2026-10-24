# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > [Beginner (tier 1/3)](../notebooks_simplified/ch8-optimization.ipynb)  →  [Intermediate (tier 2/3)](../notebooks_intermediate_simplified/ch8-optimization.ipynb)  →  **Advanced (tier 3/3) ← you are here**

# %% [markdown]
# # Chapter 8 (Advanced, explained) — Optimization: cheaper, faster, governed, evolvable
#
# **Tier 3 of 3 (advanced).** The beginner notebook (`notebooks_simplified/ch8-optimization.ipynb`) showed
# *what* each Chapter 8 skill returns. Here we go deeper: **the chapter's data lives in a real Neo4j graph**, every
# skill is fed by **Cypher queries** (not hard-coded dicts), skill outputs are **written back as nodes/edges**, and a
# tiny **agent** wires skills + Cypher together as tools and investigates the *"checkout-api is slow"* incident.
#
# **The 5 skills of Chapter 8** (all used below):
#
# | Skill | Book idea | Question it answers |
# |---|---|---|
# | `model-routing-selector` | Selective intelligence | *Which model should each workflow node use?* |
# | `cost-performance-scorer` | Measuring cost–performance | *Is the cheap model really cheaper per **successful** answer?* |
# | `kv-cache-latency-budgeter` | Hardware acceleration | *What bounds concurrency and end-to-end latency?* |
# | `subgraph-access-control` | Data governance | *Who may see which part of the graph? How do we erase PII?* |
# | `schema-evolution-migrator` | Production maintenance | *How does a live graph change without breaking the agent?* |
#
# **How to read it.** Markdown says *why*; code says *how*; every `check(...)` line is a self-test. No LLM or API key
# is needed — all skills are deterministic, stdlib-only Python.

# %%
import adv_helpers as a
import json, subprocess, sys, time

a.connect()
print("graph loaded:", a.load_devops_graph())      # clean slate + the shared DevOps incident graph

OPT = a.SKILLS_DIR / "optimization"
routing  = a.load_skill("optimization", "model-routing-selector",    "routing_lib")
costperf = a.load_skill("optimization", "cost-performance-scorer",   "costperf_lib")
budget   = a.load_skill("optimization", "kv-cache-latency-budgeter", "budget_lib")
govern   = a.load_skill("optimization", "subgraph-access-control",   "govern_lib")
schema   = a.load_skill("optimization", "schema-evolution-migrator", "schema_lib")

print("\nSkills in this chapter (read from each SKILL.md frontmatter):")
for s in a.skill_catalog("optimization"):
    print(f"  - {s['slug']:<28} {s['description'][:75]}...")

# %% [markdown]
# ## 1. Model the chapter as a graph
#
# **Why a graph?** The book's *horizontal workflow graph* is literally a graph: nodes are reasoning steps
# (`AlertClassifier`, `CausalAttributionNode`, …) and edges say which output feeds which input. If it lives in
# Neo4j we can ask structural questions (what is the critical path? which nodes have no traffic yet?) instead of
# re-reading Python dicts.
#
# We create three kinds of nodes: `WorkflowNode` (a pipeline step + its quality bar), `Model` (with cost relative
# to the frontier model), and `FEEDS` edges forming a **DAG** — a *directed acyclic graph*: arrows have a direction
# and you can never follow them back to where you started, so there is a well-defined execution order and
# independent branches can run in parallel.
#
# **Cypher clauses introduced here**
# - `UNWIND $rows AS r` — turns a Python list into one row per element so one query can write many nodes.
# - `MERGE (n:Label {key: …})` — "find it, or create it": safe to re-run (idempotent).
# - `SET n.prop = …` — write properties on what we matched.

# %%
wf_rows = [dict(name=n.name, task_class=n.task_class, q=n.required_quality, var=n.variable_difficulty)
           for n in routing.NODES.values()]
a.run("""
UNWIND $rows AS r
MERGE (w:WorkflowNode {name: r.name})
SET w.task_class = r.task_class, w.required_quality = r.q, w.variable_difficulty = r.var
""", rows=wf_rows)

model_rows = [dict(id=k, tier=v["tier"], cost=v["cost_vs_frontier"], cap=v["base_capability"])
              for k, v in routing.MODELS.items()]
a.run("""
UNWIND $rows AS m
MERGE (x:Model {id: m.id})
SET x.tier = m.tier, x.cost_vs_frontier = m.cost, x.base_capability = m.cap
""", rows=model_rows)

# Three independent "front-door" nodes feed DependencyAnalyzer; then a straight chain to the synthesis node.
feeds = [("AlertClassifier", "DependencyAnalyzer"), ("LogParser", "DependencyAnalyzer"),
         ("QueryAnalyst", "DependencyAnalyzer"), ("DependencyAnalyzer", "CausalAttributionNode"),
         ("CausalAttributionNode", "PredictionSynthesis")]
a.run("""
UNWIND $rows AS e
MATCH (src:WorkflowNode {name: e[0]}), (dst:WorkflowNode {name: e[1]})
MERGE (src)-[:FEEDS]->(dst)
""", rows=[list(e) for e in feeds])

a.show(a.run("MATCH (w:WorkflowNode) RETURN w.name AS node, w.task_class AS task, w.required_quality AS bar ORDER BY bar DESC"))

# %% [markdown]
# **Query 1 — critical path with a variable-length path.** `-[:FEEDS*]->` means "follow FEEDS edges **any number of
# times**" (`*1..3` would bound it). A *path* `p` is the whole chain of nodes+edges; `length(p)` counts its edges.
# `NOT EXISTS { … }` keeps only start nodes with no incoming edge (sources) and end nodes with no outgoing edge
# (sinks).
#
# The **critical path** is the longest chain: nodes on it *must* run one after another. Everything else can run
# **in parallel**, so `total nodes / nodes on critical path` is a ready-made *parallel factor* for the latency
# budget in section 4 — a number we derive from the graph rather than guess.

# %%
paths = a.run("""
MATCH p = (src:WorkflowNode)-[:FEEDS*]->(sink:WorkflowNode)
WHERE NOT EXISTS { ()-[:FEEDS]->(src) } AND NOT EXISTS { (sink)-[:FEEDS]->() }
RETURN src.name AS source, sink.name AS sink, length(p) + 1 AS nodes_on_path
ORDER BY nodes_on_path DESC, source
""")
a.show(paths)
critical = paths[0]["nodes_on_path"]
total_nodes = a.run("MATCH (w:WorkflowNode) RETURN count(w) AS c")[0]["c"]
PARALLEL_FACTOR = round(total_nodes / critical, 2)
print(f"\nnodes={total_nodes}  critical path={critical} nodes  ->  parallel factor = {PARALLEL_FACTOR}")
a.check("critical path is 4 nodes (a source, DependencyAnalyzer, Causal, Synthesis)", critical == 4)
a.check("parallel factor derived from the DAG is 1.5", PARALLEL_FACTOR == 1.5)

# %% [markdown]
# ## 2. Selective intelligence — route each node to the cheapest model that clears its bar
#
# **Idea (book: "Selective Intelligence").** Do not send every node through one frontier model. A fine-tuned 3B
# model reaches 0.99 on a *narrow* task like alert classification; only open-ended synthesis needs the frontier.
# `cheapest_meeting_bar` re-derives the book's assignment from first principles: sort models by cost, take the first
# whose *effective quality on that task class* meets the node's bar.
#
# **Three strategies:** *static* (fixed per node type — start here), *cascade* (try cheap, escalate on low
# confidence — FrugalGPT), *learned* (a trained router decides per query — RouteLLM).
#
# **The Cypher→skill→Cypher loop:** the node list and bars come **from the graph**, the skill decides, and we write
# the decision back as `(:WorkflowNode)-[:ROUTED_TO]->(:Model)` (plus an `ESCALATES_TO` edge for cascades).

# %%
nodes_from_graph = a.run("MATCH (w:WorkflowNode) RETURN w.name AS name ORDER BY name")
decisions = {}
for row in nodes_from_graph:
    d = routing.cheapest_meeting_bar(row["name"])
    decisions[row["name"]] = d
    base = d["model"].split(":")[-1].split("->")[0]          # 'cascade:8b->frontier' -> '8b'
    a.run("""
    MATCH (w:WorkflowNode {name: $n}), (m:Model {id: $m})
    MERGE (w)-[r:ROUTED_TO]->(m)
    SET r.strategy = $strategy, r.effective_quality = $q
    """, n=row["name"], m=base, strategy=d["strategy"], q=d.get("effective_quality"))
    if d["strategy"] == "cascade":
        a.run("""
        MATCH (w:WorkflowNode {name: $n}), (f:Model {id: 'claude-sonnet'})
        MERGE (w)-[e:ESCALATES_TO]->(f) SET e.when_confidence_below = $t
        """, n=row["name"], t=routing.CAUSAL_ESCALATE_BELOW)

a.show([{"node": n, "model": d["model"], "strategy": d["strategy"], "cost_vs_frontier": d.get("cost_vs_frontier")}
        for n, d in decisions.items()])
match_book = all(routing.static_route(n)["model"] == d["model"] for n, d in decisions.items())
a.check("first-principles routing reproduces the book's DEVOPS_MODEL_CONFIG for all 6 nodes", match_book)

# %% [markdown]
# **Query 2 — aggregation with `collect` and `count`.** `RETURN m.tier, count(w), collect(w.name)` groups rows by
# the non-aggregated column (`m.tier`): `count` tallies how many nodes land on each tier and `collect` gathers their
# names into a list. This is the "how much of my pipeline runs on the cheap tier?" dashboard in one query.

# %%
tiers = a.run("""
MATCH (w:WorkflowNode)-[r:ROUTED_TO]->(m:Model)
RETURN m.tier AS tier, count(w) AS nodes, collect(w.name) AS served_nodes
ORDER BY nodes DESC
""")
a.show(tiers)
a.check("only ONE node needs the frontier model outright (PredictionSynthesis)",
        [t for t in tiers if t["tier"] == "frontier"][0]["served_nodes"] == ["PredictionSynthesis"])

# %% [markdown]
# **Cascade and learned routing.** A *cascade* serves the answer from the cheapest model whose self-reported
# confidence clears a threshold, escalating otherwise. Lower the thresholds and more traffic stays cheap — at the
# price of more wrong answers; that dial is calibrated with the cost-performance scorer next. *Learned* routing
# replaces the thresholds with a trained classifier (here a stub — the production seam is marked `TODO` in `lib.py`).
#
# `pipeline_cost` then blends it all: frontier-everywhere vs selective, given how often the causal node escalates.

# %%
for conf in (0.95, 0.80, 0.50):
    c = routing.cascade_route("CausalAttributionNode", conf)
    print(f"confidence {conf:.2f} -> served by {c['served_by']:<14} ({c['tier']}, {c['cost_vs_frontier']}x)")
for thr in (0.3, 0.7):
    l = routing.learned_route("CausalAttributionNode", thr)
    print(f"learned router threshold {thr} -> {l['served_by']}")

pc = routing.pipeline_cost(0.30)
print(f"\nfrontier-everywhere = {pc['frontier_everywhere_units']} units, selective = {pc['selective_units']} units "
      f"-> {pc['reduction_pct']}% cheaper (equal token weights; the book's ~80% uses real token counts)")
a.check("high confidence is served by the 3B model, low by the frontier",
        routing.cascade_route("x", 0.95)["tier"] == "cheap" and routing.cascade_route("x", 0.5)["tier"] == "frontier")
a.check("selective intelligence cuts blended cost by > 60%", pc["reduction_pct"] > 60)

# %% [markdown]
# ## 3. Cost–performance — measure cost per *successful* completion
#
# **Idea.** Price per call is a trap: *"a cheap model that fails 40% of the time is not cheaper than an expensive
# one that succeeds first time."* The right metric is `total cost ÷ successes`. Two more concepts:
# - **Quality parity threshold** — the minimum accuracy a node must keep (0.99 for alerts, 0.90 for query analysis).
# - **Failure weights** — not all errors cost the same: calling a P1 outage "P3" is **10×** worse than the reverse.
#
# We load the sample invocation log into the graph as `(:WorkflowNode)-[:INVOKED]->(:Invocation)-[:USED_MODEL]->(:Model)`.

# %%
inv_rows = json.loads((OPT / "cost-performance-scorer" / "sample-invocations.json").read_text())["invocations"]
a.run("""
UNWIND range(0, size($rows) - 1) AS i
WITH i, $rows[i] AS r
MATCH (w:WorkflowNode {name: r.node_name}), (m:Model {id: r.model_id})
CREATE (inv:Invocation {id: i, input_tokens: r.input_tokens, output_tokens: r.output_tokens,
                        cost_usd: r.cost_usd, latency_ms: r.latency_ms, success: r.success})
CREATE (w)-[:INVOKED]->(inv)-[:USED_MODEL]->(m)
""", rows=inv_rows)
print("invocations stored:", a.run("MATCH (i:Invocation) RETURN count(i) AS c")[0]["c"])

# %% [markdown]
# **Query 3 — `OPTIONAL MATCH`.** A plain `MATCH` drops a node that has no matching neighbours. `OPTIONAL MATCH`
# behaves like a SQL *LEFT JOIN*: the node stays, and the missing parts are `null` (so `count` = 0). That is exactly
# what we want to spot a workflow node that has **no traffic yet** (`QueryAnalyst`). `CASE WHEN … THEN 1 ELSE 0 END`
# turns a boolean into a countable 0/1.

# %%
agg = a.run("""
MATCH (w:WorkflowNode)
OPTIONAL MATCH (w)-[:INVOKED]->(i:Invocation)
RETURN w.name AS node, count(i) AS calls,
       sum(CASE WHEN i.success THEN 1 ELSE 0 END) AS successes,
       round(sum(i.cost_usd) * 1000000) / 1000000 AS cost_usd
ORDER BY node
""")
a.show(agg)
a.check("OPTIONAL MATCH keeps the node with zero invocations",
        any(r["node"] == "QueryAnalyst" and r["calls"] == 0 for r in agg))

# %% [markdown]
# **Now feed the skill from Cypher.** We read the invocations back as rows, rebuild the skill's `CostTracker`, and
# cross-check: the skill's `cost_per_success` must equal what plain Cypher arithmetic gives. Two independent
# implementations agreeing is a cheap, strong test. Then we **write the result back** onto each node so later
# queries (and the agent) can use it.

# %%
rows = a.run("""
MATCH (w:WorkflowNode)-[:INVOKED]->(i:Invocation)-[:USED_MODEL]->(m:Model)
RETURN w.name AS node_name, m.id AS model_id, i.input_tokens AS input_tokens,
       i.output_tokens AS output_tokens, i.cost_usd AS cost_usd, i.latency_ms AS latency_ms,
       i.success AS success
ORDER BY i.id
""")
tracker = costperf.CostTracker()
for r in rows:
    tracker.log(costperf.NodeInvocation(**r))

for r in agg:
    if r["calls"]:
        rep = tracker.node_report(r["node"])
        a.run("MATCH (w:WorkflowNode {name:$n}) SET w.cost_per_success=$c, w.success_rate=$s, w.p95_ms=$p",
              n=r["node"], c=rep["cost_per_success"], s=rep["success_rate"], p=rep["p95_latency_ms"])

cypher_cps = {r["node"]: r["cost_usd"] / max(r["successes"], 1) for r in agg if r["calls"]}
a.check("skill cost_per_success == Cypher sum(cost)/successes for every node",
        all(abs(tracker.cost_per_success(n) - v) < 1e-6 for n, v in cypher_cps.items()))

print("Observed success rate vs the quality bar (tiny illustrative sample!):")
a.show(a.run("""
MATCH (w:WorkflowNode)-[:ROUTED_TO]->(m:Model)
WHERE w.success_rate IS NOT NULL
RETURN w.name AS node, m.id AS routed_to, w.success_rate AS observed, w.required_quality AS bar,
       w.success_rate >= w.required_quality AS meets_bar
ORDER BY node
"""))

# %% [markdown]
# **Cheap ≠ cheaper.** `compare_cost_per_success` reproduces the book's warning: a model 33% cheaper per call that
# only succeeds 60% of the time is *more expensive* per success than the reliable one.
#
# **Domain-weighted evaluation.** We store 200 labelled alert predictions for two candidate classifiers as
# `(:EvalCase)` nodes, read them back with Cypher, and score them with `evaluate_candidate`. Candidate *v1* makes
# 3 dangerous `P1_as_P3` errors; *v2* only 1 harmless `P3_as_P1`.

# %%
cmp = costperf.compare_cost_per_success(0.0001, 0.60, 0.00015, 0.98)
print("per call cheaper:", cmp["cheaper_per_call"], "| per success cheaper:", cmp["cheaper_per_success"])
a.check("cheaper per call is NOT cheaper per success", cmp["cheaper_per_call"] != cmp["cheaper_per_success"])

def make_cases(cand, p1_as_p3, p3_as_p1, n_p1=40, n_p3=160):
    cases = [("P1", "P3")] * p1_as_p3 + [("P1", "P1")] * (n_p1 - p1_as_p3)
    cases += [("P3", "P1")] * p3_as_p1 + [("P3", "P3")] * (n_p3 - p3_as_p1)
    return [dict(candidate=cand, gold=g, pred=p) for g, p in cases]

a.run("UNWIND $rows AS r CREATE (:EvalCase {candidate: r.candidate, node: 'AlertClassifier', gold: r.gold, pred: r.pred})",
      rows=make_cases("v1", 3, 1) + make_cases("v2", 0, 1))

results = {}
for cand in ("v1", "v2"):
    preds = [(r["gold"], r["pred"]) for r in a.run(
        "MATCH (c:EvalCase {candidate:$c}) RETURN c.gold AS gold, c.pred AS pred", c=cand)]
    results[cand] = costperf.evaluate_candidate(preds, costperf.EVAL_SETS["AlertClassifier"])
    print(cand, results[cand])
a.check("v1 fails the 0.99 bar and has a heavy weighted error", not results["v1"]["meets_threshold"]
        and results["v1"]["weighted_error_rate"] > 0.1)
a.check("v2 passes with a tiny weighted error", results["v2"]["meets_threshold"]
        and results["v2"]["weighted_error_rate"] < 0.01)

# %% [markdown]
# ## 4. Schema evolution — change a live graph without breaking the agent
#
# **Idea ("Production Systems and Maintenance").** A production graph keeps changing. Four disciplines:
# - **Migrations** (like Flyway for databases): versioned scripts with a **rollback**. **N-1 compatible** means the
#   *previous* agent version still works after the migration, so schema/data/code can ship in separate steps.
# - **Temporal invalidation**: never delete a changed fact; stamp it with an end time. This is the idea behind
#   **bi-temporal** models (Graphiti/Zep): *valid time* (when the fact was true in the world: `t_valid` → `t_invalid`)
#   versus *transaction time* (when the system learned it). Here we implement the valid-time half, which is what lets
#   you ask "what did checkout depend on during the June incident?".
# - **Incremental updates**: `MERGE … ON CREATE / ON MATCH` touches only what changed; cost ∝ new data, not graph size.
# - **Staged rollout**: schema → backfill → agent canary → cleanup, each depending on the previous one.
#
# **Step A — migrate `alert_config` (a text property) into real `MONITORED_BY` relationships.**
# `MERGE` on a relationship means "create it unless that exact edge exists", which is why re-applying is safe.

# %%
a.run("""
UNWIND $rows AS r
MATCH (s:Service {name: r.name}) SET s.alert_config = r.alert
""", rows=[{"name": "checkout-api", "alert": "p99-latency-alarm"},
           {"name": "payment-service", "alert": "payment-error-rate"},
           {"name": "orders-service", "alert": "orders-p99"}])

mig = schema.monitored_by_migration()
print(f"{mig.version}: {mig.description} (N-1 compatible: {mig.n1_compatible})")

def statements(script):
    """Split a .cypher migration into statements (drop // comment lines)."""
    lines = [l for l in script.split("\n") if not l.strip().startswith("//")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]

def monitored_count():
    return a.run("MATCH (:Service)-[r:MONITORED_BY]->(:AlertRule) RETURN count(r) AS c")[0]["c"]

for st in statements(mig.cypher):
    a.run(st)
applied = monitored_count()
for st in statements(mig.rollback):
    a.run(st)
rolled_back = monitored_count()
for st in statements(mig.cypher):
    a.run(st)
reapplied = monitored_count()
print(f"MONITORED_BY edges: applied={applied}, after rollback={rolled_back}, re-applied={reapplied}")
a.check("migration applies, rolls back cleanly, and re-applies idempotently", (applied, rolled_back, reapplied) == (3, 0, 3))

print("\nCoverage gap (services nobody monitors) — OPTIONAL MATCH + 'WITH ... WHERE ar IS NULL':")
a.show(a.run("""
MATCH (s:Service)
OPTIONAL MATCH (s)-[:MONITORED_BY]->(ar:AlertRule)
WITH s, ar WHERE ar IS NULL
RETURN s.name AS unmonitored_service
"""))

# %% [markdown]
# **Step B — temporal invalidation.** First we seed history: `checkout-api` has depended on `stripe-python 3.2.1`
# since May (valid-from `2026-05-01`). Then a deployment upgrades it to 3.3.0. The skill's Cypher closes the old edge
# (`SET old.t_invalid = datetime()`) and adds a new one.
#
# **Known issue (explained, not hidden):** the skill's generated Cypher does `MERGE (a)-[:DEPENDS_ON {…, t_invalid: null}]`.
# Neo4j refuses `MERGE` on a relationship whose pattern property is `null`. A missing property already means "still
# valid", so we catch the error, drop that one line and retry. In production you would fix the generator — a nice
# vibe-coding exercise (see the corner below).

# %%
a.run("""
MATCH (s:Service {name:'checkout-api'})
MERGE (l:Library {name:'stripe-python', version:'3.2.1'})
MERGE (s)-[:DEPENDS_ON {version:'3.2.1', t_valid: datetime('2026-05-01T00:00:00Z')}]->(l)
""")
inval = schema.temporal_invalidate_cypher("checkout-api", "stripe-python", "3.2.1", "3.3.0")
a.check("generated Cypher invalidates (SET t_invalid) and never DELETEs", "t_invalid = datetime()" in inval and "DELETE" not in inval.upper())
try:
    a.run(inval)
    print("skill Cypher ran as-is")
except Exception as exc:
    print("Neo4j rejected the skill's MERGE:", str(exc).splitlines()[0][:100])
    a.run(inval.replace("    t_invalid: null,\n", ""))
    print("-> retried without the null property")

# %% [markdown]
# **Point-in-time query.** Because nothing was deleted, we can "travel" to any instant: an edge was *valid at t* if
# `t_valid <= t` and (`t_invalid IS NULL` or `t_invalid > t`). `datetime($t)` parses a string parameter.

# %%
def deps_as_of(t):
    return [r["version"] for r in a.run("""
        MATCH (:Service {name:'checkout-api'})-[d:DEPENDS_ON]->(l:Library {name:'stripe-python'})
        WHERE d.t_valid <= datetime($t) AND (d.t_invalid IS NULL OR d.t_invalid > datetime($t))
        RETURN l.version AS version""", t=t)]

june = deps_as_of("2026-06-02T11:00:00Z")
now = deps_as_of("2999-01-01T00:00:00Z")
print("during the June incident:", june, "| today:", now)
a.check("history is preserved: June sees 3.2.1, today sees 3.3.0", june == ["3.2.1"] and now == ["3.3.0"])

# %% [markdown]
# **Step C — incremental merge for a whole deployment event, plus the staged rollout manifest.** The skill returns
# three statements (upsert service → upsert dependencies → invalidate vanished ones). We bind parameters from the
# event JSON. Then `validate_manifest` enforces the four-phase order and the canary/promotion rules, and the
# **retention policy** reminds us snapshots grow fast (200 resources × every 5 min = 57,600 nodes/day).

# %%
event = json.loads((OPT / "schema-evolution-migrator" / "sample-deployment-event.json").read_text())
event["service"] = "checkout-api"            # the sample says checkout-service; our graph calls it checkout-api
s1, s2, s3 = schema.incremental_merge_cypher(event)
a.run(s1, service_name=event["service"], env=event["environment"], new_version=event["version"])
for dep in event["dependencies"]:
    a.run(s2, lib_name=dep["name"], lib_version=dep["version"], service_name=event["service"])
a.run(s3, service_name=event["service"], current_libs=[d["name"] for d in event["dependencies"]])

a.show(a.run("""
MATCH (:Service {name:'checkout-api'})-[d:DEPENDS_ON]->(l:Library)
RETURN l.name AS library, l.version AS version, d.t_invalid IS NULL AS is_current ORDER BY library, version
"""))
manifest = schema.staged_rollout_manifest("2026-10-24-causal-attribution-v2")
verdict = schema.validate_manifest(manifest)
bad = json.loads(json.dumps(manifest)); bad["phases"][2].pop("canary_percent")
print("manifest valid:", verdict["valid"], "| broken manifest problems:", schema.validate_manifest(bad)["problems"])
a.check("manifest validates; removing the canary is caught", verdict["valid"] and not schema.validate_manifest(bad)["valid"])
a.check("snapshot growth for 200 resources @ 5 min = 57,600/day", schema.snapshot_growth_per_day(200, 5) == 57600)
a.check("incident records are retained longer than snapshots", "retain_long" in schema.RETENTION_POLICY["incident"])

# %% [markdown]
# ## 5. Governance — subgraph access control, PII, and the execution graph as an audit trail
#
# **Why it matters.** In a graph *everything is reachable from everything* — that is its power and its risk. Three
# defences from the chapter:
# 1. **Subgraph access control**: a persona role may traverse only certain labels/relationship types and read
#    certain properties. With *security transparency*, a forbidden node is **invisible** (not "access denied"), so an
#    agent cannot even probe for what exists. (Real enforcement is a Neo4j **Enterprise** feature — `GRANT/DENY`;
#    the skill *emits* that policy text, and below we *simulate* enforcement in Cypher on Community Edition.)
# 2. **Privacy by architecture**: the graph stores only UUIDs; names/emails live in a separate *identity store* joined by
#    UUID. GDPR Art. 17 "erasure" is then a one-row update (soft) or a `DETACH DELETE` (hard).
# 3. **Execution graph as compliance artifact**: every agent decision carries governance metadata, so an auditor's
#    question is just a query.
#
# We add cost data to services and two on-call `Employee` nodes that hold **only a UUID**.

# %%
a.run("""
UNWIND [['checkout-api',4.2,'BC-100'],['payment-service',3.1,'BC-100'],['orders-service',2.4,'BC-200'],['cart-service',0.9,'BC-300']] AS r
MATCH (s:Service {name:r[0]}) SET s.cost_per_hour = r[1], s.budget_code = r[2], s.status = 'up', s.team = s.owner
""")
a.run("""
MATCH (c:Service {name:'checkout-api'})
CREATE (:Employee {uuid:'u-1001'})-[:ON_CALL_FOR]->(c)
CREATE (:Employee {uuid:'u-1002'})-[:ON_CALL_FOR]->(c)
""")
IDENTITY_STORE = {"u-1001": {"name": "Ada Lovelace", "email": "ada@example.com"},
                  "u-1002": {"name": "Alan Turing", "email": "alan@example.com"}}   # stand-in for a separate SQL DB

print(govern.generate_policy("sre_oncall"))

# %% [markdown]
# **Derive what the investigation needs from the graph.** Instead of guessing which labels an SRE must traverse, we
# ask Cypher: starting at the incident, follow only the *diagnostic* relationship types up to 4 hops and list the
# labels we meet (`A|B|C*1..4` = any of these types, 1–4 hops). Then `audit_access` tests each role against that
# real query pattern — the book's pitfall: *"governance policies that block the agent's reasoning."*

# %%
def needed_labels():
    rows = a.run("""
    MATCH (i:Incident {id:'INC-1001'})-[:AFFECTS|TRIGGERED_BY|FIRED_ON|DEPENDS_ON|CHANGED|DEPLOYED_TO|MONITORED_BY*1..4]-(n)
    UNWIND labels(n) AS label
    RETURN DISTINCT label ORDER BY label
    """)
    return [r["label"] for r in rows if r["label"] in govern.NODE_LABELS]

need = needed_labels()
print("labels the investigation must traverse:", need)
audits = {r: govern.audit_access(r, need, {"Service": ["name", "status", "cost_per_hour"]}) for r in govern.ROLES}
for r, au in audits.items():
    print(f"{r:<16} invisible={au['invisible_labels']} masked={au['masked_properties']} complete={au['functionally_complete']}")
a.check("sre_oncall can run the whole investigation; finance and infra_analyst cannot",
        audits["sre_oncall"]["functionally_complete"] and not audits["finance_analyst"]["functionally_complete"]
        and not audits["infra_analyst"]["functionally_complete"])

# %% [markdown]
# **Simulated enforcement.** `scoped_view(role)` runs a Cypher query that only returns nodes whose labels the role may
# traverse, then drops properties the role may not read. Count what each role sees versus what exists. Notice
# `Employee` is invisible to *every* role: not denied — just absent from the world as far as the agent can tell.

# %%
def scoped_view(role):
    allowed = [l for l in govern.NODE_LABELS if govern.can_traverse(role, l)]
    rows = a.run("MATCH (n) WHERE any(l IN labels(n) WHERE l IN $allowed) "
                 "RETURN labels(n)[0] AS label, properties(n) AS props", allowed=allowed)
    return [{"label": r["label"],
             "props": {k: v for k, v in r["props"].items() if govern.can_read(role, r["label"], k)}} for r in rows]

total_by_label = {r["label"]: r["c"] for r in a.run("MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS c")}
views = {r: scoped_view(r) for r in govern.ROLES}
table = []
for r, v in views.items():
    labels_seen = sorted({x["label"] for x in v})
    table.append({"role": r, "nodes_visible": len(v), "labels_visible": ",".join(labels_seen),
                  "sees_cost": any("cost_per_hour" in x["props"] for x in v)})
a.show(table)
print("total nodes in graph:", sum(total_by_label.values()))
a.check("SRE never sees cost data", not table[0]["sees_cost"])
a.check("finance sees cost but no Deployment/Library nodes",
        table[1]["sees_cost"] and "Deployment" not in table[1]["labels_visible"])
a.check("Employee nodes are invisible to every role", all("Employee" not in t["labels_visible"] for t in table))

# %% [markdown]
# **Execution graph as compliance artifact.** Each agent decision becomes a `(:Decision)` node whose properties are the
# skill's `governance_metadata` (data sources, role, PII flag, model id/version, confidence, retention). Linking it
# `-[:MADE_BY]->(:Model)` ties the audit trail to the routing graph from section 2. The auditor's question *"which
# decisions touched PII, and which tier of model made them?"* is then one query — compared against the skill's
# own `audit_query` filter.

# %%
decision_specs = [
    ("dec-1", "sre_oncall", "llama-3.1-8b", False, 0.82),
    ("dec-2", "sre_oncall", "claude-sonnet", False, 0.93),
    ("dec-3", "finance_analyst", "llama-3.1-3b", False, 0.97),
    ("dec-4", "infra_analyst", "llama-3.1-8b", True, 0.71),     # touched the on-call roster
]
payload = [{"id": i, "model": m, "meta": govern.governance_metadata(
                ["infrastructure-graph"], role, m, "2026-10", conf, pii_accessed=pii, human_review_required=pii)}
           for i, role, m, pii, conf in decision_specs]
a.run("""
UNWIND $rows AS r
CREATE (d:Decision {id: r.id}) SET d += r.meta
WITH d, r MATCH (m:Model {id: r.model}) CREATE (d)-[:MADE_BY]->(m)
""", rows=payload)

a.show(a.run("""
MATCH (d:Decision)-[:MADE_BY]->(m:Model)
RETURN d.pii_accessed AS pii, count(d) AS decisions, collect(d.id) AS ids, collect(DISTINCT m.tier) AS tiers
ORDER BY pii DESC
"""))
records = [r["p"] for r in a.run("MATCH (d:Decision) RETURN properties(d) AS p")]
via_skill = sorted(r["id"] for r in govern.audit_query(records, pii_accessed=True))
via_cypher = sorted(r["id"] for r in a.run("MATCH (d:Decision) WHERE d.pii_accessed RETURN d.id AS id"))
a.check("PII audit: skill filter and Cypher WHERE agree (dec-4)", via_skill == via_cypher == ["dec-4"])

# %% [markdown]
# **GDPR erasure, both flavours.** *Soft* (user `u-1001`): null the identity mapping; the UUID node stays as an
# anonymous orphan so aggregate analysis still works. *Hard* (user `u-1002`): run the skill's `DETACH DELETE`
# (delete the node **and** its relationships). The identity-store update is a single row, so it is honoured in
# real time; graph cleanup can lag.

# %%
soft = govern.gdpr_erase("u-1001", "soft")
IDENTITY_STORE["u-1001"] = None                      # soft: mask the mapping, graph untouched
hard = govern.gdpr_erase("u-1002", "hard")
a.run(hard["graph_action"])                         # hard: the skill's Cypher, executed for real
IDENTITY_STORE.pop("u-1002")
left = {r["uuid"]: r["edges"] for r in a.run(
    "MATCH (e:Employee) OPTIONAL MATCH (e)-[r]->() RETURN e.uuid AS uuid, count(r) AS edges")}
print("Employee nodes left:", left)
a.check("soft erase keeps the anonymous orphan with its edge; hard erase removes node + edges",
        left == {"u-1001": 1} and IDENTITY_STORE["u-1001"] is None and soft["aggregate_analysis_preserved"]
        and not hard["aggregate_analysis_preserved"])

# %% [markdown]
# ## 6. Hardware & latency — what really bounds speed and concurrency
#
# **Two bottlenecks.** (1) *Graph analytics* (PageRank, betweenness) are memory-bandwidth bound; GPUs (cuGraph) give
# 80×–485× published speedups. (2) *Inference* concurrency is bound by the **KV cache** — the per-user attention
# memory a transformer keeps while generating. Quantizing weights does not move that ceiling; bounding the cache does
# (the book's MEMENTO recipe). The book redacts the exact compression factors, so the skill asks **you to measure**
# them — the numbers in the KV cell below are illustrative inputs, not book figures.
#
# **Blast radius** = everything that would be affected if one component fails (or changes). It is a transitive
# traversal: here, every service that depends — directly or through other services — on `orders-db`, where deployment
# `deploy-481` shrank the connection pool. We compute it with a variable-length path and time it.

# %%
t0 = time.perf_counter()
blast = a.run("""
MATCH p = (s:Service)-[:DEPENDS_ON*1..4]->(db:Database {name: 'orders-db'})
RETURN s.name AS service, s.tier AS tier, min(length(p)) AS hops
ORDER BY hops, service
""")
cypher_ms = (time.perf_counter() - t0) * 1000
a.show(blast)
a.run("MATCH (db:Database {name:'orders-db'}) SET db.blast_radius = $n", n=len(blast))   # persist the result
print(f"\nblast radius of orders-db = {len(blast)} services (Cypher took {cypher_ms:.1f} ms on this toy graph)")
a.check("blast radius = orders, payment (1 hop), checkout (2), cart (3)",
        [(r["service"], r["hops"]) for r in blast] ==
        [("orders-service", 1), ("payment-service", 1), ("checkout-api", 2), ("cart-service", 3)])

gpu = budget.estimate_gpu_time(4.0, "pagerank")
print("full-scale estimate: 4 s CPU PageRank ->", gpu["gpu_ms"], "ms with cuGraph;", budget.blast_radius_report()["gpu_result_ms"], "ms in the book's Example 8-14")

# %% [markdown]
# **End-to-end latency budget.** `budget_pipeline` sums per-call ranges (SLM 50–200 ms, graph op 5–50 ms) and divides
# by the **parallel factor** — the DAG-derived 1.5 from section 1. The book's target is the *whole task* under 2 s.
# We store the result on the incident as a `(:LatencyBudget)` node.
#
# **KV-bound concurrency.** `kv_bound_concurrency = floor((GPU memory − weights − overhead) / peak KV per user)`.
# Halving the weights barely helps when KV dominates; compressing KV multiplies users.

# %%
lb = budget.budget_pipeline(n_slm_calls=total_nodes, n_graph_ops=1, parallel_factor=PARALLEL_FACTOR)
seq = budget.budget_pipeline(n_slm_calls=total_nodes, n_graph_ops=1, parallel_factor=1.0)
a.run("""
MATCH (i:Incident {id:'INC-1001'})
MERGE (i)-[:HAS_BUDGET]->(b:LatencyBudget {kind:'pipeline'})
SET b.est_ms_lo = $lo, b.est_ms_hi = $hi, b.within_2s = $ok, b.parallel_factor = $pf
""", lo=lb["estimated_ms"][0], hi=lb["estimated_ms"][1], ok=lb["within_2s_target"], pf=PARALLEL_FACTOR)
print("sequential:", seq["estimated_ms"], "ms | parallel (DAG):", lb["estimated_ms"], "ms | within 2 s:", lb["within_2s_target"])

kv = budget.weight_quant_vs_kv_compression(80, 16, 12, weight_quant_factor=2, kv_compression_factor=4)
print({k: kv[k] for k in ("baseline_concurrency", "after_weight_quant", "after_kv_compression", "verdict")})
a.check("exploiting the DAG shortens the budget", lb["estimated_ms"][1] < seq["estimated_ms"][1])
a.check("KV compression beats weight quantization for concurrency", kv["verdict"] == "kv_compression"
        and kv["kv_compression_gain"] > kv["weight_quant_gain"])

# %% [markdown]
# ## 7. The agent — plan → act → observe → log
#
# Now everything becomes a **tool**. `MiniAgent` is a ~30-line loop: a *planner* picks the next tool from what the
# agent has already observed, the agent *acts* (calls it), *observes* the result, and `AgentTrace` stores every
# call in Neo4j as `(:Run)-[:STEP]->(:ToolCall)-[:NEXT]->(:ToolCall)`. Swap the rule-based planner for an LLM and
# you have a real agent; the tools and the trace stay identical.
#
# **Goal:** *"checkout-api is slow."* Tools: a Cypher suspect finder (multi-hop), a Cypher past-incident lookup,
# the routing skill, the access-control skill, the latency skill, and a decision writer. The first audit
# deliberately uses a **wrong role name** to show failure being recorded and recovered from.

# %%
agent = a.MiniAgent("INC-1001: checkout-api is slow — find the cause, pick a model, check governance, budget latency")

@agent.tool("find_suspects")
def find_suspects(service):
    return a.run("""
    MATCH p = (s:Service {name: $svc})-[:DEPENDS_ON*1..3]->(x)<-[:CHANGED]-(d:Deployment)
    RETURN d.id AS deployment, d.change AS change, x.name AS changed_node, length(p) AS hops
    ORDER BY hops""", svc=service)

@agent.tool("similar_incident")
def similar_incident(node):
    return a.run("""
    MATCH (old:Incident {status:'resolved'})-[:ROOT_CAUSE]->(x {name: $n})
    RETURN old.id AS incident, old.root_cause AS root_cause, old.fix AS fix""", n=node)

@agent.tool("route_node")
def route_node(node_name):
    return routing.cheapest_meeting_bar(node_name)

@agent.tool("audit_role")
def audit_role(role):
    return govern.audit_access(role, needed_labels())     # KeyError for an unknown role

@agent.tool("latency_budget")
def latency_budget(n_calls):
    return budget.budget_pipeline(n_calls, 1, PARALLEL_FACTOR)

@agent.tool("record_decision")
def record_decision(model_id, role, confidence, summary):
    meta = govern.governance_metadata(["infrastructure-graph"], role, model_id, "2026-10", confidence)
    a.run("""
    MATCH (r:Run {id: $rid})
    CREATE (r)-[:PRODUCED]->(d:Decision {id: $did, summary: $summary}) SET d += $meta""",
          rid=agent.trace.run_id, did=agent.trace.run_id + "-decision", summary=summary, meta=meta)
    return "decision stored"

def plan(mem, failed):
    """Rule-based planner: look at what we know, return the next (tool, args) or None when done."""
    if "find_suspects" not in mem:
        return "find_suspects", {"service": "checkout-api"}
    if "similar_incident" not in mem:
        return "similar_incident", {"node": mem["find_suspects"][0]["changed_node"]}
    if "route_node" not in mem:
        return "route_node", {"node_name": "CausalAttributionNode"}
    if "audit_role" not in mem:
        return "audit_role", {"role": "sre_oncall" if "audit_role" in failed else "sre"}   # first try is a typo
    if "latency_budget" not in mem:
        return "latency_budget", {"n_calls": total_nodes}
    if "record_decision" not in mem:
        cause = mem["similar_incident"][0]["root_cause"]
        return "record_decision", {"model_id": "llama-3.1-8b", "role": "sre_oncall", "confidence": 0.88,
                                   "summary": f"{mem['find_suspects'][0]['deployment']} -> {cause}"}
    return None

failed = []
for _ in range(12):
    step = plan(agent.memory, failed)
    if step is None:
        break
    tool, args = step
    try:
        out = agent.act(tool, **args)
        print(f"ACT {tool:<16} OK    -> {str(out)[:90]}")
    except Exception as exc:
        failed.append(tool)
        print(f"ACT {tool:<16} FAIL  -> {type(exc).__name__}: {exc}  (planner will retry with a fix)")

# %% [markdown]
# **Read the trace back from the graph.** The run is now data. `OPTIONAL MATCH (c)-[:NEXT]->(nx)` lists each call with
# its successor; a variable-length `[:NEXT*]` path measures the whole chain; `sum(CASE …)` counts failures; and the
# `PRODUCED` edge links the run to the governance-stamped decision. This is the same "execution graph as compliance
# artifact" idea as section 5, now for the agent's own behaviour.

# %%
rid = agent.trace.run_id
a.show(a.run("""
MATCH (:Run {id: $rid})-[:STEP]->(c:ToolCall)
OPTIONAL MATCH (c)-[:NEXT]->(nx:ToolCall)
RETURN c.seq AS seq, c.tool AS tool, c.ok AS ok, nx.tool AS next_tool ORDER BY seq""", rid=rid))
chain = a.run("""
MATCH (:Run {id: $rid})-[:STEP]->(first:ToolCall {seq: 1})
MATCH p = (first)-[:NEXT*]->(last:ToolCall)
RETURN max(length(p)) + 1 AS steps_in_chain""", rid=rid)[0]["steps_in_chain"]
stats = a.run("""
MATCH (r:Run {id: $rid})-[:STEP]->(c:ToolCall)
RETURN count(c) AS calls, sum(CASE WHEN c.ok THEN 0 ELSE 1 END) AS failures""", rid=rid)[0]
dec = a.run("MATCH (:Run {id:$rid})-[:PRODUCED]->(d:Decision) RETURN d.summary AS summary, d.model_id AS model", rid=rid)
print("stats:", stats, "| decision:", dec)
a.check("trace has 7 calls (6 tools + 1 recovered failure), chained by NEXT", stats["calls"] == 7 and chain == 7)
a.check("exactly one failure was recorded and recovered", stats["failures"] == 1 and "audit_role" in failed)
a.check("agent identified deploy-481 / pool change as the cause",
        dec and "deploy-481" in dec[0]["summary"] and "pool" in dec[0]["summary"])

# %% [markdown]
# ## 8. Skill CLI & benchmark — the same skills, no notebook required
#
# Each skill ships `cli.py` (argparse, stdlib-only) so cron jobs, CI or another agent can call it. Two contracts are
# tested: `--help` prints the SKILL.md description, and `benchmark` runs the skill's built-in self-checks and exits 0.
# `a.run_benchmark` wraps the subprocess call.

# %%
results = []
for s in a.skill_catalog("optimization"):
    ok, first = a.run_benchmark("optimization", s["slug"])
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {s['slug']:<28} {first[:70]}")
a.check("all 5 optimization skills pass their own benchmark", len(results) == 5 and all(results))

cli = OPT / "model-routing-selector" / "cli.py"
helptxt = subprocess.run([sys.executable, str(cli), "--help"], capture_output=True, text=True, encoding="utf-8")
print("\n$ python cli.py --help   (first lines)")
print("\n".join(helptxt.stdout.splitlines()[:6]))
route = subprocess.run([sys.executable, str(cli), "route", "CausalAttributionNode"], capture_output=True, text=True, encoding="utf-8")
print("\n$ python cli.py route CausalAttributionNode")
print(route.stdout.strip()[:400])
a.check("--help exits 0 and shows the skill description",
        helptxt.returncode == 0 and "Match model capability" in " ".join(helptxt.stdout.split()))
a.check("the CLI agrees with the library on the causal node", route.returncode == 0 and "cascade" in route.stdout)

# %% [markdown]
# ## 9. 🎯 Vibe-coding corner — extend this with an AI assistant
#
# **A good prompt has four parts:** (1) **context** — name the repo, folder, files and the data model; (2) **task** —
# one small, concrete change; (3) **constraints** — stdlib-only, no new dependencies, don't touch other folders, keep
# Cypher parameterized; (4) **acceptance check** — a command or assertion that proves it works. Then **you verify**:
# run the check yourself, read the diff, try one adversarial input.
#
# **Prompt 1 — a new Cypher query.**
# > *In `agentic-graph-rag-skills`, the graph has `(:WorkflowNode)-[:INVOKED]->(:Invocation)-[:USED_MODEL]->(:Model)`
# > with `cost_usd`, `success`, `latency_ms`. Write a parameterized Cypher query returning, per node and model, calls,
# > success rate and p95-ish latency (`percentileCont(latency_ms, 0.95)`), keeping nodes with no calls via
# > `OPTIONAL MATCH`. Wrap it in a Python function `node_model_stats()` using `adv_helpers.run`. Acceptance: for
# > `CausalAttributionNode` it returns two models, and the sum of calls equals 3.*
# > **Verify:** run the acceptance assertion; run the query in Neo4j Browser; check an empty node returns 0, not an error.
#
# **Prompt 2 — a new skill (follow the repo anatomy).**
# > *Create `skills/optimization/retention-pruner/` following the anatomy of `skills/optimization/schema-evolution-migrator/`:
# > `SKILL.md` with the 7 sections + Source Attribution to Ch8, a stdlib-only `lib.py` with
# > `prune_candidates(snapshots, policy)` that applies `RETENTION_POLICY`, and an argparse `cli.py` whose `--help` prints
# > the SKILL.md description and whose `benchmark` subcommand asserts ≥3 behaviours and exits non-zero on failure.
# > Do not add pip dependencies.* **Verify:** `python skills/optimization/retention-pruner/cli.py benchmark` exits 0
# > and `a.run_benchmark("optimization","retention-pruner")` prints PASS.
#
# **Prompt 3 — fix a real bug.**
# > *`temporal_invalidate_cypher` in `skills/optimization/schema-evolution-migrator/lib.py` emits
# > `MERGE … {t_invalid: null}` which Neo4j rejects. Change it to `MERGE` on `{version}` then `ON CREATE SET`
# > `t_valid = datetime()` and leave `t_invalid` unset. Keep the benchmark green and add a test that runs the Cypher
# > against Neo4j without the try/except from the notebook.* **Verify:** the benchmark, plus the notebook cell without its fallback.
#
# **Worked example — a helper "written by the assistant", plus the test you write.** Prompt: *"Write
# `rank_models_for_node(node)` returning `[(model_id, cost_per_success)]` sorted ascending, from the Invocation graph;
# return `[]` for a node with no traffic."* Review it like a PR: is the Cypher parameterized? Is division by zero
# handled? Does it agree with the skill's own `CostTracker`?

# %%
def rank_models_for_node(node):
    rows = a.run("""
    MATCH (:WorkflowNode {name: $node})-[:INVOKED]->(i:Invocation)-[:USED_MODEL]->(m:Model)
    WITH m.id AS model, sum(i.cost_usd) AS cost, sum(CASE WHEN i.success THEN 1 ELSE 0 END) AS ok
    RETURN model, cost / CASE WHEN ok = 0 THEN 1 ELSE ok END AS cps
    ORDER BY cps""", node=node)
    return [(r["model"], r["cps"]) for r in rows]

ranked = rank_models_for_node("CausalAttributionNode")
print(ranked)
a.check("ranking is ascending and the 8B model beats the frontier per success", [m for m, _ in ranked] == ["llama-3.1-8b", "claude-sonnet"])
a.check("matches CostTracker on the frontier-only node", abs(rank_models_for_node("PredictionSynthesis")[0][1]
        - tracker.cost_per_success("PredictionSynthesis")) < 1e-9)
a.check("no traffic -> empty list, not an error", rank_models_for_node("QueryAnalyst") == [])
a.check("unknown node -> empty list", rank_models_for_node("nope") == [])

# %% [markdown]
# **Read the generated code critically.** This helper ranks purely by cost per success — so it would happily pick the
# 8B model for the causal node even though that node's bar needs the frontier on hard cases. *Cost per success is
# necessary but not sufficient; the quality bar from section 2 still gates the choice.* Ask your assistant to add a
# `min_success_rate` argument and a test for it — a perfect follow-up prompt.

# %% [markdown]
# ## 10. 🧪 Experiments — change one thing, predict, then run
#
# **Experiment 1 — escalation rate.** `pipeline_cost(escalation_rate)` blends 8B and frontier cost for the causal node.
# *Predict:* if escalation goes from 10% to 90%, does the reduction shrink a little or a lot? (Hint: only one of five
# nodes is affected.)

# %%
sweep = {er: routing.pipeline_cost(er)["reduction_pct"] for er in (0.1, 0.3, 0.6, 0.9)}
print(sweep)
a.check("reduction falls monotonically as escalation rises", list(sweep.values()) == sorted(sweep.values(), reverse=True))
a.check("even at 90% escalation the pipeline is still cheaper than frontier-everywhere", sweep[0.9] > 50)

# %% [markdown]
# **Experiment 2 — KV cache per user.** Change `peak_kv_per_user_gb` from 12 to 2 and then to 30 in the call below.
# *Predict:* at what peak-KV value does KV compression stop mattering (concurrency already ≥ many users)? And at 30 GB,
# how many users fit at all? Rerun with your own measured value — that is what the skill asks of you.

# %%
for peak in (2, 12, 30):
    r = budget.weight_quant_vs_kv_compression(80, 16, peak, weight_quant_factor=2, kv_compression_factor=4)
    print(f"peak KV/user={peak:>2} GB -> base={r['baseline_concurrency']:>2}, "
          f"weights/2={r['after_weight_quant']:>2}, KV/4={r['after_kv_compression']:>2}")
a.check("at 30 GB/user only 2 users fit; KV/4 more than doubles it",
        budget.kv_bound_concurrency(80, 16, 30) == 2 and budget.kv_bound_concurrency(80, 16, 30 / 4) == 8)

# %% [markdown]
# **Experiment 3 — cascade thresholds.** Lower `cheap_threshold` from 0.85 to 0.6 for a confidence of 0.7.
# *Predict:* which tier serves it before and after, and what does that do to cost? Then think about the quality risk
# you just accepted — which skill would you use to measure it?

# %%
before = routing.cascade_route("CausalAttributionNode", 0.70)
after = routing.cascade_route("CausalAttributionNode", 0.70, cheap_threshold=0.6)
print("before:", before["served_by"], before["cost_vs_frontier"], "| after:", after["served_by"], after["cost_vs_frontier"])
a.check("a looser threshold sends the same query to a cheaper model", after["cost_vs_frontier"] < before["cost_vs_frontier"])

# %% [markdown]
# ## 11. ⚠️ Common pitfalls
#
# - **Optimizing price per call, not per success.** A cheap model with a low success rate can cost more (section 3).
# - **One global quality bar.** Alerts need 0.99, query analysis 0.90; also weight errors by consequence (P1→P3 is 10×).
# - **Over-restrictive governance.** A role that cannot traverse the labels an investigation needs silently cripples the
#   agent. Always `audit_access` against the *real* query pattern (we saw finance and infra_analyst fail).
# - **Deleting history instead of invalidating it.** You lose "what was true during the incident?" (point-in-time queries).
# - **Breaking N-1.** Dropping a property in the same release that stops using it; use the staged rollout and a delay
#   before cleanup.
# - **Over-pruning.** Keep incident records far longer than telemetry snapshots — incidents are training signal.
# - **Trusting illustrative numbers.** Capability scores, KV sizes and the 4 s → 100 ms figures are inputs or book
#   quotes, not measurements of your system. Measure first.
# - **Skill null-property bug.** `temporal_invalidate_cypher` needs the workaround shown in section 4 on real Neo4j.
# - **Leaving test constraints behind.** A shared database keeps constraints between runs; drop what you create.

# %% [markdown]
# ## 12. Verification battery

# %%
print("-- graph state --")
a.check("workflow graph: 6 WorkflowNodes, 5 FEEDS, 6 ROUTED_TO",
        (lambda c: (c["w"], c["f"], c["r"]) == (6, 5, 6))(a.run(
            "MATCH (w:WorkflowNode) OPTIONAL MATCH (w)-[r:ROUTED_TO]->() "
            "WITH count(DISTINCT w) AS w, count(r) AS r "
            "MATCH ()-[f:FEEDS]->() RETURN w, count(f) AS f, r")[0]))
a.check("14 invocations and 400 eval cases stored",
        a.run("MATCH (i:Invocation) RETURN count(i) AS c")[0]["c"] == 14
        and a.run("MATCH (e:EvalCase) RETURN count(e) AS c")[0]["c"] == 400)
a.check("cascade node has an ESCALATES_TO edge to the frontier model",
        a.run("MATCH (:WorkflowNode {name:'CausalAttributionNode'})-[:ESCALATES_TO]->(m) RETURN m.id AS m")[0]["m"] == "claude-sonnet")
a.check("incident INC-1001 has a latency budget and the blast radius is persisted",
        a.run("MATCH (:Incident {id:'INC-1001'})-[:HAS_BUDGET]->(b) RETURN b.within_2s AS ok")[0]["ok"] is True
        and a.run("MATCH (d:Database {name:'orders-db'}) RETURN d.blast_radius AS n")[0]["n"] == 4)
a.check("every one of the 5 skills was used (loaded modules)",
        all(m is not None for m in (routing, costperf, budget, govern, schema)))

print("-- cleanup: drop the constraint the migration created --")
a.run("DROP CONSTRAINT alertrule_name_unique IF EXISTS")
left_over = a.run("SHOW CONSTRAINTS YIELD name WHERE name = 'alertrule_name_unique' RETURN count(*) AS c")[0]["c"]
a.check("constraint dropped, shared DB left clean of our schema objects", left_over == 0)

# %% [markdown]
# ## 13. Recap
#
# - **Routing** (`model-routing-selector`): re-derived the book's model assignment from the quality bar, stored as
#   `ROUTED_TO` / `ESCALATES_TO` edges; only synthesis needs the frontier; blended cost falls > 60%.
# - **Measurement** (`cost-performance-scorer`): cost *per success*, per-node bars and failure-weighted evaluation,
#   fed from an `Invocation` graph and cross-checked against Cypher arithmetic.
# - **Speed** (`kv-cache-latency-budgeter`): blast radius as a variable-length traversal; DAG-derived parallel factor;
#   KV cache — not weights — bounds concurrency.
# - **Governance** (`subgraph-access-control`): role audits against the real traversal pattern, invisible-not-denied
#   enforcement, PII by UUID separation, soft/hard erasure, and the audit trail as a graph query.
# - **Evolution** (`schema-evolution-migrator`): reversible migrations, point-in-time history via `t_valid/t_invalid`,
#   incremental `MERGE`, staged rollouts — and an honest note on a null-property bug.
# - **Agent + CLI**: skills and Cypher became tools; the run is a queryable graph; every skill passes its CLI benchmark.
#
# **Next:** this is the last chapter. Revisit `notebooks_advanced_simplified/` chapters 1–7 with this chapter's lens
# (what would each primitive cost, who may see it, how does it evolve?), or take a vibe-coding prompt from section 9 and
# ship your own skill into `skills/optimization/`.
