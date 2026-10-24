# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > [Beginner (tier 1/3)](../notebooks_simplified/ch8-optimization.ipynb)  →  **Intermediate (tier 2/3) ← you are here**  →  [Advanced (tier 3/3)](../notebooks_advanced_simplified/ch8-optimization.ipynb)

# %% [markdown]
# # Chapter 8 — Optimization (intermediate tier)
#
# **Tier 2 of 3.** Tier 1 told you *what* each optimization primitive is; tier 3 runs every skill, a mini agent and
# all the benchmarks. Here you learn **why and how** — you read each skill's inputs and outputs, drive the skills from
# Cypher on a real Neo4j graph, chain a few of them by hand, and solve two exercises yourself.
#
# **The three forces of production (Ch8):** *cost* (frontier model everywhere is unaffordable), *governance*
# (one unscoped query can reach employee records) and *performance* (a traversal that is fast in dev is slow at scale).
#
# **Skills used (all in `skills/optimization/`):** `model-routing-selector`, `cost-performance-scorer`,
# `kv-cache-latency-budgeter`, `subgraph-access-control`, `schema-evolution-migrator`.
#
# **Running scenario:** the *checkout-api is slow* incident (`INC-1001`) in the shared DevOps graph. No API keys;
# everything is stdlib + a local Neo4j.

# %%
import int_helpers as h
import inspect, json, subprocess, sys, os

h.connect()
print("graph loaded:", h.load_devops_graph())      # clean slate + the shared DevOps incident graph

OPT = h.SKILLS_DIR / "optimization"
routing  = h.load_skill("optimization", "model-routing-selector",    "routing_lib")
costperf = h.load_skill("optimization", "cost-performance-scorer",   "costperf_lib")
budget   = h.load_skill("optimization", "kv-cache-latency-budgeter", "budget_lib")
govern   = h.load_skill("optimization", "subgraph-access-control",   "govern_lib")
schema   = h.load_skill("optimization", "schema-evolution-migrator", "schema_lib")

for s in h.skill_catalog("optimization"):
    print(f"  - {s['slug']:<28} {s['description'][:70]}...")

# %% [markdown]
# ## 0. Reading a skill's contract
#
# Every skill is a plain Python module: **inputs** are function arguments (usually derived from your graph),
# **outputs** are dicts you can store or branch on. Before calling anything, read the signature — it is the contract.

# %%
for fn in (routing.cheapest_meeting_bar, routing.cascade_route, costperf.compare_cost_per_success,
           budget.budget_pipeline, govern.audit_access, schema.incremental_merge_cypher):
    print(f"{fn.__module__.split('_')[0]:<9}{fn.__name__}{inspect.signature(fn)}")

# %% [markdown]
# ## 1. Model the workflow as a graph
#
# The agent's *horizontal workflow* is a **DAG** (directed acyclic graph): `WorkflowNode`s linked by `FEEDS` edges.
# Putting it in Neo4j lets us ask structural questions instead of re-reading Python dicts. We add it **next to** the
# incident graph. Node names and quality bars come straight from the routing skill's `NODES` table.
#
# Clauses used: `UNWIND $rows AS r` (one row per list item, so one query writes many nodes), `MERGE` (find-or-create,
# safe to re-run), `SET` (write properties).

# %%
h.run("""
UNWIND $rows AS r
MERGE (w:WorkflowNode {name: r.name})
SET w.task_class = r.task_class, w.required_quality = r.q
""", rows=[dict(name=n.name, task_class=n.task_class, q=n.required_quality) for n in routing.NODES.values()])

h.run("""
UNWIND $rows AS m
MERGE (x:Model {id: m.id})
SET x.tier = m.tier, x.cost_vs_frontier = m.cost
""", rows=[dict(id=k, tier=v["tier"], cost=v["cost_vs_frontier"]) for k, v in routing.MODELS.items()])

feeds = [("AlertClassifier", "DependencyAnalyzer"), ("LogParser", "DependencyAnalyzer"),
         ("QueryAnalyst", "DependencyAnalyzer"), ("DependencyAnalyzer", "CausalAttributionNode"),
         ("CausalAttributionNode", "PredictionSynthesis")]
h.run("""
UNWIND $rows AS e
MATCH (a:WorkflowNode {name: e[0]}), (b:WorkflowNode {name: e[1]})
MERGE (a)-[:FEEDS]->(b)
""", rows=[list(e) for e in feeds])
h.show(h.run("MATCH (w:WorkflowNode) RETURN w.name AS node, w.task_class AS task, w.required_quality AS bar ORDER BY bar DESC"))

# %% [markdown]
# **Query 1 — multi-hop / variable-length.** `-[:FEEDS*]->` follows FEEDS edges *any number of times* (`*1..3` would
# bound it); `p` is the whole path and `length(p)` counts its edges. `NOT EXISTS { … }` keeps only sources (no incoming
# edge) and sinks (no outgoing edge). The longest source→sink chain is the **critical path**: those nodes must run
# sequentially, everything else can run in parallel — so `total nodes / critical path` is a **parallel factor** we
# *derive from the graph* and feed to the latency budget later.

# %%
paths = h.run("""
MATCH p = (src:WorkflowNode)-[:FEEDS*]->(sink:WorkflowNode)
WHERE NOT EXISTS { ()-[:FEEDS]->(src) } AND NOT EXISTS { (sink)-[:FEEDS]->() }
RETURN src.name AS source, sink.name AS sink, length(p) + 1 AS nodes_on_path
ORDER BY nodes_on_path DESC, source
""")
h.show(paths)
critical = paths[0]["nodes_on_path"]
total_nodes = h.run("MATCH (w:WorkflowNode) RETURN count(w) AS c")[0]["c"]
PARALLEL_FACTOR = round(total_nodes / critical, 2)
print(f"nodes={total_nodes} critical path={critical} -> parallel factor {PARALLEL_FACTOR}")
h.check("critical path is 4 nodes", critical == 4)
h.check("parallel factor derived from the DAG is 1.5", PARALLEL_FACTOR == 1.5)

# %% [markdown]
# ## 2. Selective intelligence — `model-routing-selector`
#
# **Why.** Sending every node to a frontier model works and is wildly expensive. A fine-tuned 3B model reaches 0.99 on
# a *narrow* task such as alert classification; only open-ended synthesis needs the frontier.
# `cheapest_meeting_bar(node)` sorts models by cost and returns the first whose *effective quality on that task class*
# meets the node's bar.
#
# **Anatomy of the output:** `model` (what to call; `cascade:A->B` means try A, escalate to B) · `strategy`
# (`static` fixed, `cascade` escalate on low confidence, FrugalGPT-style) · `effective_quality` vs
# `required_quality` (the margin that justified the choice) · `cost_vs_frontier` (1.0 = frontier price).
#
# **Cypher → skill → Cypher:** node list from the graph, decision from the skill, result written back as
# `(:WorkflowNode)-[:ROUTED_TO]->(:Model)`.

# %%
decisions = {}
for row in h.run("MATCH (w:WorkflowNode) RETURN w.name AS name ORDER BY name"):
    d = routing.cheapest_meeting_bar(row["name"])
    decisions[row["name"]] = d
    primary = d["model"].split(":")[-1].split("->")[0]          # 'cascade:8b->frontier' -> '8b'
    h.run("""
    MATCH (w:WorkflowNode {name: $n}), (m:Model {id: $m})
    MERGE (w)-[r:ROUTED_TO]->(m)
    SET r.strategy = $strategy, r.effective_quality = $q
    """, n=row["name"], m=primary, strategy=d["strategy"], q=d["effective_quality"])

print(json.dumps(decisions["CausalAttributionNode"], indent=1)[:420], "...")
print("cascade at confidence 0.9 ->", routing.cascade_route("CausalAttributionNode", 0.9)["served_by"],
      "| at 0.6 ->", routing.cascade_route("CausalAttributionNode", 0.6)["served_by"])
h.check("router reproduces the book's static assignment for all 6 nodes",
        all(routing.static_route(n)["model"] == d["model"] for n, d in decisions.items()))

# %% [markdown]
# **Query 2 — aggregation.** `RETURN m.tier, count(w), collect(w.name)` groups by the non-aggregated column:
# `count` tallies nodes per tier and `collect` gathers their names. This is the "how much of my pipeline runs on the
# cheap tier?" dashboard, read from the routing decisions we just wrote.

# %%
tiers = h.run("""
MATCH (w:WorkflowNode)-[r:ROUTED_TO]->(m:Model)
RETURN m.tier AS tier, count(w) AS nodes, collect(w.name) AS which, round(avg(m.cost_vs_frontier), 3) AS avg_cost
ORDER BY avg_cost
""")
h.show(tiers)
pc = routing.pipeline_cost(0.30)
print(f"blended cost vs frontier-everywhere: -{pc['reduction_pct']}% (30% escalation rate)")
h.check("cost reduction >= 65%", pc["reduction_pct"] >= 65.0)

# %% [markdown]
# ## 3. Cost per *successful* completion — `cost-performance-scorer`
#
# **Why.** Cost per token misleads: a cheap model that fails often can cost more per *completed* task.
# You log each call as a `NodeInvocation(node_name, model_id, input_tokens, output_tokens, cost_usd, latency_ms,
# success)` into a `CostTracker`; `score_policy` then returns per-node `success_rate`, `cost_per_success`,
# `p95_latency_ms` and a `pipeline_cost_per_success`. `compare_cost_per_success(cheap_cost, cheap_rate, reliable_cost,
# reliable_rate)` shows the inversion in one call.

# %%
sample = json.loads((OPT / "cost-performance-scorer" / "sample-invocations.json").read_text())
tracker = costperf.CostTracker()
for row in sample["invocations"]:
    tracker.log(costperf.NodeInvocation(**row))
report = costperf.score_policy(tracker)
h.show([{k: r[k] for k in ("node", "n_calls", "success_rate", "cost_per_success", "p95_latency_ms")} for r in report["nodes"]])
inv = costperf.compare_cost_per_success(0.007, 0.60, 0.010, 0.98)
print(f"pipeline cost per success: ${report['pipeline_cost_per_success']}")
print(f"cheaper per call: {inv['cheaper_per_call']} | cheaper per success: {inv['cheaper_per_success']}")
h.check("SLM node is cheaper per success than the frontier node",
        tracker.node_report("AlertClassifier")["cost_per_success"] < tracker.node_report("PredictionSynthesis")["cost_per_success"])
h.check("cheaper-per-call model loses on cost-per-success", inv["cheaper_per_success"] == "reliable")

# %% [markdown]
# ## 4. Latency and memory — `kv-cache-latency-budgeter`
#
# **Why.** Incident response is the latency-critical path. `budget_pipeline(n_slm_calls, n_graph_ops, parallel_factor)`
# returns `estimated_ms` as a **(low, high) range** and `within_2s_target`; we pass the parallel factor *measured from
# the graph* in section 1. `estimate_gpu_time(cpu_seconds, algo)` models a cuGraph speed-up for blast-radius analytics.
# At the inference layer the binding constraint is **peak KV-cache per user, not weight size**:
# `weight_quant_vs_kv_compression` shows which lever multiplies concurrency.

# %%
bp = budget.budget_pipeline(n_slm_calls=5, n_graph_ops=1, parallel_factor=PARALLEL_FACTOR)
est = budget.estimate_gpu_time(35, "betweenness_centrality")
kv = budget.weight_quant_vs_kv_compression(gpu_memory_gb=80, model_weights_gb=16, peak_kv_per_user_gb=8,
                                           weight_quant_factor=2, kv_compression_factor=2)
print(f"pipeline latency {bp['estimated_ms']} ms (parallel {PARALLEL_FACTOR}x) within 2s: {bp['within_2s_target']}")
print(f"blast radius: 35 s CPU -> {est['gpu_ms']} ms GPU ({est['speedup']}x)")
print(f"concurrency on 80 GB: baseline {kv['baseline_concurrency']} | weight-quant {kv['after_weight_quant']} "
      f"| KV-compress {kv['after_kv_compression']} -> winner: {kv['verdict']}")
h.check("pipeline within the 2 s budget", bp["within_2s_target"])
h.check("KV compression beats weight quantization", kv["verdict"] == "kv_compression")

# %% [markdown]
# ## 5. Governance — `subgraph-access-control`
#
# **Why.** Cost and speed make the agent viable; governance makes it *safe*. SRE needs topology but not cost; finance
# needs cost but not topology. The skill's API: `generate_policy(role)` (Neo4j GRANT/DENY text),
# `can_traverse(role, label)` / `can_read(role, label, prop)` (booleans), `audit_access(role, needed_labels)`
# (`reachable_labels`, `invisible_labels`, `functionally_complete` — does least privilege starve the agent?) and
# `governance_metadata(...)` (a record attributing a decision to a model). Out-of-scope data is *invisible*, not
# "access denied" — the agent never learns it exists.

# %%
audit = govern.audit_access("sre_oncall", ["Service", "Deployment"])
print({k: audit[k] for k in ("reachable_labels", "invisible_labels", "functionally_complete")})
print("sre reads cost_per_hour:", govern.can_read("sre_oncall", "Service", "cost_per_hour"),
      "| finance reads it:", govern.can_read("finance_analyst", "Service", "cost_per_hour"))
print("finance traverses Library:", govern.can_traverse("finance_analyst", "Library"))
rec = govern.governance_metadata(["infrastructure_graph"], "sre_on_call", "llama-3.1-8b-instruct", "v2.3.1", 0.87,
                                 pii_accessed=False)
print("record:", rec)
records = [rec, govern.governance_metadata(["identity_store"], "billing_agent", "claude-sonnet", "v1.0", 0.9, pii_accessed=True)]
print("PII decisions:", len(govern.audit_query(records, pii_accessed=True)))
h.check("SRE cannot read cost data, finance can", not govern.can_read("sre_oncall", "Service", "cost_per_hour")
        and govern.can_read("finance_analyst", "Service", "cost_per_hour"))
h.check("SRE reasoning is not starved", audit["functionally_complete"])

# %% [markdown]
# ## 6. Production maintenance — `schema-evolution-migrator`
#
# **Why.** A live graph evolves. `monitored_by_migration()` returns a `Migration` (`version`, `cypher`, `rollback`,
# `n1_compatible` — old code keeps working). We apply its Cypher to the real graph, read the new edges back, and
# **roll back**, always dropping the constraint it creates (the database is shared).
#
# `temporal_invalidate_cypher(service, library, old, new)` closes the old dependency edge with `t_invalid` instead
# of deleting it, so history stays queryable.
#
# **Known issue (explained, not hidden):** that generated Cypher does `MERGE (a)-[:DEPENDS_ON {…, t_invalid: null}]`.
# Neo4j refuses to `MERGE` a relationship with a null pattern property. A missing property already means "still valid",
# so we catch the error and retry without that one line. (A great fix to hand to an AI assistant — see the vibe tip.)

# %%
mig = schema.monitored_by_migration()
print(f"{mig.version}: {mig.description} | N-1 compatible: {mig.n1_compatible}")
h.run("MATCH (s:Service) WHERE s.name IN ['checkout-api','payment-service'] SET s.alert_config = s.name + '-latency-alert'")

def apply(script):
    for stmt in [x.strip() for x in script.split(";") if x.strip()]:
        h.run(stmt)

apply(mig.cypher)
monitored = h.run("MATCH (s:Service)-[:MONITORED_BY]->(ar:AlertRule) RETURN s.name AS service, ar.name AS rule ORDER BY service")
h.show(monitored)
apply(mig.rollback)
after = h.run("MATCH (:Service)-[r:MONITORED_BY]->() RETURN count(r) AS c")[0]["c"]
left = h.run("SHOW CONSTRAINTS YIELD name WHERE name = 'alertrule_name_unique' RETURN count(*) AS c")[0]["c"]
h.check("migration created 2 edges; rollback removed them and the constraint", len(monitored) == 2 and after == 0 and left == 0)

# %%
h.run("""
MATCH (s:Service {name:'checkout-api'})
MERGE (l:Library {name:'stripe-python', version:'3.2.1'})
MERGE (s)-[:DEPENDS_ON {version:'3.2.1', t_valid: datetime('2026-05-01T00:00:00Z')}]->(l)
""")
inval = schema.temporal_invalidate_cypher("checkout-api", "stripe-python", "3.2.1", "3.3.0")
h.check("generated Cypher invalidates and never DELETEs", "t_invalid = datetime()" in inval and "DELETE" not in inval.upper())
try:
    h.run(inval)
    print("skill Cypher ran as-is")
except Exception as exc:
    print("Neo4j rejected the skill's MERGE:", str(exc).splitlines()[0][:90])
    h.run(inval.replace("    t_invalid: null,\n", ""))
    print("-> retried without the null property")

def versions_at(t):
    return [r["v"] for r in h.run("""
        MATCH (:Service {name:'checkout-api'})-[d:DEPENDS_ON]->(l:Library {name:'stripe-python'})
        WHERE d.t_valid <= datetime($t) AND (d.t_invalid IS NULL OR d.t_invalid > datetime($t))
        RETURN l.version AS v""", t=t)]

june, today = versions_at("2026-06-02T11:00:00Z"), versions_at("2999-01-01T00:00:00Z")
print("in June:", june, "| today:", today)
h.check("history preserved: June sees 3.2.1, today 3.3.0", june == ["3.2.1"] and today == ["3.3.0"])

# %% [markdown]
# ## 7. A hand-written pipeline for the incident
#
# No agent framework: a plain function that chains three skills, with the graph supplying the inputs.
# Order matters — **govern first** (can the agent see the data?), **then route** (which model per step?),
# **then budget** (does it fit the latency target?). Tier 3 replaces this function with a `MiniAgent`.

# %%
def investigate(incident_id: str) -> dict:
    # graph facts: affected service + services in the blast radius of what the deployment changed
    blast = h.run("""
    MATCH (i:Incident {id: $id})-[:AFFECTS]->(svc:Service)
    MATCH (dep:Deployment)-[:CHANGED]->(x)<-[:DEPENDS_ON*1..3]-(s:Service)
    RETURN svc.name AS affected, dep.id AS deployment, x.name AS changed, collect(DISTINCT s.name) AS blast_radius
    """, id=incident_id)[0]
    # 1) govern, 2) route, 3) budget
    gate = govern.audit_access("sre_oncall", ["Service", "Deployment"])
    steps = ["AlertClassifier", "DependencyAnalyzer", "CausalAttributionNode", "PredictionSynthesis"]
    routes = {n: routing.cheapest_meeting_bar(n)["model"] for n in steps}
    lat = budget.budget_pipeline(len(steps), 1, PARALLEL_FACTOR)
    go = gate["functionally_complete"] and lat["within_2s_target"]
    return {**blast, "access_ok": gate["functionally_complete"], "routes": routes,
            "latency_ms": lat["estimated_ms"], "decision": "RUN on routed models" if go else "BLOCK"}

result = investigate("INC-1001")
print(f"INCIDENT INC-1001: {result['affected']} is slow")
print(f"  suspect : {result['deployment']} changed {result['changed']}")
print(f"  blast   : {sorted(result['blast_radius'])}")
print(f"  access  : SRE role functionally complete = {result['access_ok']}")
for n, m in result["routes"].items():
    print(f"  route   : {n:<22} -> {m}")
print(f"  latency : {result['latency_ms']} ms | decision: {result['decision']}")
h.check("pipeline decides to run", result["decision"].startswith("RUN"))
h.check("blast radius includes checkout-api", "checkout-api" in result["blast_radius"])

# %% [markdown]
# ## 8. Your turn
#
# `h.your_turn(title, hint, solution, expected)` prints the task and hint, runs a reference solution and checks it.
# **Try your own version first**, then compare with the reference.
#
# **Exercise 1 (Cypher).** How many distinct services depend — directly or transitively — on `orders-db`?

# %%
def ex1():
    return h.run("""
    MATCH (s:Service)-[:DEPENDS_ON*1..3]->(:Database {name: 'orders-db'})
    RETURN count(DISTINCT s) AS n
    """)[0]["n"]

h.your_turn("services depending on orders-db", "variable-length DEPENDS_ON*1..3 and count(DISTINCT ...)", ex1, 4)

# %% [markdown]
# **Exercise 2 (change a skill's inputs).** With 80 GB of GPU memory and 16 GB of weights, how many concurrent users
# fit if peak KV per user drops from 8 GB to 4 GB? (`kv_bound_concurrency(gpu_memory_gb, model_weights_gb, peak_kv_per_user_gb)`)

# %%
def ex2():
    return budget.kv_bound_concurrency(80, 16, 4)

h.your_turn("concurrency with 4 GB KV per user", "call kv_bound_concurrency with the new KV size", ex2, 15)

# %% [markdown]
# ## 9. Trusting a skill — CLI and benchmark
#
# Every skill ships a `cli.py` (usable from cron, CI or any harness with no imports) and a `benchmark` self-check.
# Here: one CLI call and one benchmark. Tier 3 benchmarks every skill.

# %%
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
p = subprocess.run([sys.executable, str(OPT / "model-routing-selector" / "cli.py"), "route", "CausalAttributionNode"],
                   capture_output=True, text=True, encoding="utf-8", env=env, timeout=60)
cli_out = json.loads(p.stdout)
print("CLI route ->", cli_out["model"], "| same as library:", cli_out["model"] == decisions["CausalAttributionNode"]["model"])
ok, first = h.run_benchmark("optimization", "kv-cache-latency-budgeter")
print("benchmark kv-cache-latency-budgeter:", "PASS" if ok else "FAIL", "-", first)
h.check("CLI agrees with the library and the benchmark passes", cli_out["model"] == decisions["CausalAttributionNode"]["model"] and ok)

# %% [markdown]
# ## 🎯 Vibe-coding tip
#
# Hand the known bug to an AI assistant, with the acceptance check attached:
#
# > *In `skills/optimization/schema-evolution-migrator/lib.py`, `temporal_invalidate_cypher` emits
# > `MERGE (a)-[:DEPENDS_ON {…, t_invalid: null}]`, which Neo4j rejects. Change it to MERGE on `{version}` and use
# > `ON CREATE SET t_valid = datetime()`, leaving `t_invalid` unset. Stdlib only; touch no other file; keep
# > `python cli.py benchmark` green.*
#
# **Acceptance check:** run `python skills/optimization/schema-evolution-migrator/cli.py benchmark`, then run the
# new Cypher against Neo4j *without* the `.replace(...)` workaround and re-check the June/today history queries.

# %% [markdown]
# ## Why this matters, and common mistakes
#
# - **Cost vs quality:** routing trades a little quality margin for ~70% lower cost; cascades add latency on escalation.
# - **Governance vs capability:** tighter roles are safer but can starve the agent — that is why `audit_access` reports `functionally_complete`.
# - **Derive, don't guess:** the parallel factor came from the DAG, not a constant.
#
# **Common mistakes:** judging models by cost per call instead of per success · bounding weights when the KV cache is
# the real limit · deleting changed facts instead of invalidating them · leaving constraints behind in a shared DB ·
# trusting generated Cypher without running it.

# %%
h.check("5 skills loaded", all(m is not None for m in (routing, costperf, budget, govern, schema)))
h.check("routing decisions written to the graph", h.run("MATCH ()-[r:ROUTED_TO]->() RETURN count(r) AS c")[0]["c"] == 6)
h.check("temporal history kept: 2 stripe-python edges", h.run("MATCH (:Service {name:'checkout-api'})-[d:DEPENDS_ON]->(:Library {name:'stripe-python'}) RETURN count(d) AS c")[0]["c"] == 2)
h.check("no leftover constraint", h.run("SHOW CONSTRAINTS YIELD name WHERE name = 'alertrule_name_unique' RETURN count(*) AS c")[0]["c"] == 0)
print("all checks passed")

# %% [markdown]
# ## Recap and next
#
# - **Route** each node to the cheapest model that clears its bar (`model-routing-selector`) and **measure** cost per success.
# - **Budget** latency with a graph-derived parallel factor; bound the KV cache, not the weights.
# - **Govern** with per-role subgraphs and attribute every decision; **evolve** the schema with reversible, N-1 migrations and temporal invalidation.
# - Skill inputs came from Cypher, results went back to the graph, and a plain function chained them.
#
# **Next:** tier 3 adds every skill's benchmark, a `MiniAgent`, vibe-coding corners and experiments:
# [`../notebooks_advanced_simplified/ch8-optimization.ipynb`](../notebooks_advanced_simplified/ch8-optimization.ipynb).
