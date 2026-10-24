# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > **Beginner (tier 1/3) ← you are here**  →  [Intermediate (tier 2/3)](../notebooks_intermediate_simplified/ch8-optimization.ipynb)  →  [Advanced (tier 3/3)](../notebooks_advanced_simplified/ch8-optimization.ipynb)

# %% [markdown]
# # Chapter 8 (Simplified) — Optimization: from prototype to production, with a real Neo4j graph
#
# **Who this is for:** you are new to GraphRAG and agents. Every step is explained in plain language.
#
# **What you will learn**
# 1. **Cost** — why you should not run every step of an agent on the biggest, priciest AI model, and how to *route* each step to the cheapest model that is good enough.
# 2. **Measuring** — the metric that really matters: *cost per successful completion*.
# 3. **Speed** — how to budget latency, and why GPU memory (the "KV-cache") limits how many users you can serve.
# 4. **Governance** — how to make sure each person/role only sees the part of the graph they are allowed to see.
# 5. **Maintenance** — how to change a live graph's schema safely and update it incrementally without deleting history.
#
# **The story:** it is 08:42 and *"the checkout API is slow"*. In earlier chapters the DevOps agent learned to
# find the cause. Now we make it fit for production: cheap, safe, fast and maintainable.
#
# **What is new compared with the original notebook:** no AWS mocks. All data (workflow steps, model costs,
# call logs, decisions, schema changes) lives in a **real Neo4j graph**, and we use real Cypher to query it and
# to feed the five skills in `skills/optimization/`.
#
# > **Before running:** start Neo4j once with
# > `docker compose -f docker-compose/docker-compose.yml up -d`, then run cells top to bottom.

# %% [markdown]
# ## 0. Setup: connect to Neo4j and load the incident graph
#
# **Quick vocabulary**
# * **Node** = a thing (a service, a database, an incident). **Relationship** = a typed arrow between nodes
#   (`checkout-api -[:DEPENDS_ON]-> orders-service`). Both can carry **properties** (key/value facts).
# * **Cypher** = the query language for graphs. `(a)-[:DEPENDS_ON]->(b)` reads "a depends on b".
# * `graph_helpers.py` gives us `g.run(cypher, **params)` (run a query, get rows) and `g.show(rows)` (print a table).
#
# We load the usual incident graph, then **enrich** it with the extra things this chapter needs:
# cost/billing properties on services (finance cares about these), a library dependency, and an on-call employee
# whose compensation must stay private.

# %%
import json
import time
import graph_helpers as g

counts = g.load_devops_graph()          # wipes the DB, then loads the story
print("Loaded into Neo4j:", counts)

OPT = g.SKILLS_DIR / "optimization"

# Cost / billing / status facts on each service (UNWIND turns a Python list into one row per item)
g.run("""
UNWIND $rows AS r
MATCH (s:Service {name: r.name})
SET s.status = r.status, s.version = r.version, s.team = r.team,
    s.cost_per_hour = r.cost, s.budget_code = r.budget, s.billing_account = r.billing
""", rows=[
    {"name": "checkout-api",    "status": "degraded", "version": "v2.14.0", "team": "payments", "cost": 4.20, "budget": "BC-100", "billing": "ACC-1"},
    {"name": "payment-service", "status": "healthy",  "version": "v1.9.2",  "team": "payments", "cost": 6.80, "budget": "BC-100", "billing": "ACC-1"},
    {"name": "orders-service",  "status": "healthy",  "version": "v3.0.1",  "team": "orders",   "cost": 3.10, "budget": "BC-200", "billing": "ACC-2"},
    {"name": "cart-service",    "status": "healthy",  "version": "v0.8.5",  "team": "orders",   "cost": 1.40, "budget": "BC-200", "billing": "ACC-2"},
])

# A library dependency (with a validity window), a service-to-service CALLS edge,
# and an on-call employee identified ONLY by a UUID plus a private salary record.
g.run("""
MATCH (c:Service {name:'checkout-api'}), (cart:Service {name:'cart-service'})
CREATE (lib:Library {name:'stripe-python', version:'3.2.1'})
CREATE (c)-[:DEPENDS_ON {version:'3.2.1', t_valid: datetime()}]->(lib)
CREATE (cart)-[:CALLS]->(c)
CREATE (e:Employee {uuid:'u-7f3a'})
CREATE (e)-[:ON_CALL_FOR]->(c)
CREATE (e)-[:PAID]->(:Compensation {salary: 150000})
""")

g.show(g.run("MATCH (n) RETURN labels(n)[0] AS node_type, count(*) AS how_many ORDER BY node_type"))

# %% [markdown]
# 💡 **Tip:** open <http://localhost:7474> (login `neo4j` / `password`) and run `MATCH (n)-[r]->(m) RETURN n,r,m`
# to see this graph as a picture.
#
# ## 1. Load the five optimization skills
#
# A **skill** is a small Python file (`lib.py`) implementing one idea from the book. Every skill file is called
# `lib.py`, so `g.load_skill(chapter_folder, slug)` loads each one from its own path under a unique name.

# %%
routing  = g.load_skill("optimization", "model-routing-selector",    "routing_lib")
costperf = g.load_skill("optimization", "cost-performance-scorer",   "costperf_lib")
govern   = g.load_skill("optimization", "subgraph-access-control",   "govern_lib")
schema   = g.load_skill("optimization", "schema-evolution-migrator", "schema_lib")
budget   = g.load_skill("optimization", "kv-cache-latency-budgeter", "budget_lib")

print("Loaded 5 optimization skills from", OPT)

# %% [markdown]
# ## 2. Selective intelligence — route each step to the model it actually needs
#
# **The problem.** The agent is a **workflow graph**: a chain of small steps (*nodes*), such as "classify the
# alert", "parse the logs", "work out the root cause", "write the summary". Many teams run every step on one big
# "frontier" model. It works, but it is very expensive.
#
# **The idea (selective intelligence).** Different steps need different brainpower:
# * Narrow, repetitive jobs (classify an alert) → a **small language model (SLM)**, ~30x cheaper, once fine-tuned.
# * Multi-hop graph reasoning → a **mid-size** model (~10x cheaper).
# * Open-ended writing and hard causal reasoning → the **frontier** model.
# * When difficulty varies per request, use a **cascade**: try the cheap model first and escalate only when it is unsure.
#
# **Graph angle:** the workflow steps and the models are themselves data. We store them in Neo4j as
# `(:WorkflowNode)` and `(:Model)` nodes, ask the skill for each step's best model, and record the answer as a
# `[:ROUTED_TO]` relationship. Then a Cypher query shows the whole routing table.

# %%
# Store the agent's workflow steps and the model catalog in the graph.
g.run("""
UNWIND $nodes AS n
MERGE (w:WorkflowNode {name: n.name})
SET w.task_class = n.task_class, w.required_quality = n.required_quality
""", nodes=[{"name": n.name, "task_class": n.task_class, "required_quality": n.required_quality}
            for n in routing.NODES.values()])
g.run("""
UNWIND $models AS m
MERGE (x:Model {id: m.id})
SET x.tier = m.tier, x.cost_vs_frontier = m.cost
""", models=[{"id": k, "tier": v["tier"], "cost": v["cost_vs_frontier"]} for k, v in routing.MODELS.items()])

# Read the steps BACK from the graph and ask the skill for each one's route.
pipeline_nodes = g.run("""
MATCH (w:WorkflowNode) RETURN w.name AS name, w.task_class AS task_class, w.required_quality AS bar
ORDER BY w.name
""")
g.show(pipeline_nodes)

# %% [markdown]
# **How to read this:** `bar` is the minimum quality the step must reach. `cheapest_meeting_bar` walks the models
# from cheapest to dearest and stops at the first one that clears the bar for that step's task type.

# %%
nodes = ["AlertClassifier", "QueryAnalyst", "LogParser",
         "DependencyAnalyzer", "CausalAttributionNode", "PredictionSynthesis"]
assert {r["name"] for r in pipeline_nodes} == set(nodes)   # graph and skill agree on the steps

for r in pipeline_nodes:
    assert routing.NODES[r["name"]].required_quality == r["bar"]   # the bar came from the graph's data

for n in nodes:
    r = routing.cheapest_meeting_bar(n)
    base = r["model"].split(":")[-1].split("->")[0]            # cascade starts on the 8B model
    g.run("""
    MATCH (w:WorkflowNode {name: $node}), (m:Model {id: $model})
    MERGE (w)-[r:ROUTED_TO]->(m)
    SET r.strategy = $strategy
    """, node=n, model=base, strategy=r["strategy"])

print("Routing table, straight from the graph:")
g.show(g.run("""
MATCH (w:WorkflowNode)-[r:ROUTED_TO]->(m:Model)
RETURN w.name AS step, r.strategy AS strategy, m.id AS starts_on, m.tier AS tier,
       round(m.cost_vs_frontier * 1000) / 1000.0 AS cost_vs_frontier
ORDER BY cost_vs_frontier, step
"""))

pc = routing.pipeline_cost(0.30)
print(f"\nBlended pipeline cost reduction vs frontier-everywhere: {pc['reduction_pct']}%")
print("(the book reports ~80% under token-weighting; equal-weight nodes here land near 70%)")

# The same decision through the command line, proving harness portability
import subprocess, sys
out = subprocess.run(
    [sys.executable, str(OPT / "model-routing-selector" / "cli.py"), "route", "CausalAttributionNode"],
    capture_output=True, text=True)
print("\nCLI route CausalAttributionNode ->", json.loads(out.stdout)["model"])

# %% [markdown]
# **Two more routing strategies** the skill offers:
# * **Cascade by confidence** — the cheap model reports how sure it is; low confidence escalates up the ladder.
# * **Learned routing** — a trained router decides strong-vs-weak per request (stand-in here).

# %%
for conf in (0.95, 0.80, 0.50):
    r = routing.cascade_route("CausalAttributionNode", conf)
    print(f"confidence {conf:.2f} -> served by {r['served_by']:<14} ({r['tier']})")
print("learned route @0.3 ->", routing.learned_route("LogParser", 0.3)["served_by"])
print("static route        ->", routing.static_route("LogParser")["model"])

# %%
# Verification gate: the router reproduces the book's assignment and the blended cost falls substantially.
assert routing.cheapest_meeting_bar("AlertClassifier")["model"] == "llama-3.1-3b"
assert routing.cheapest_meeting_bar("CausalAttributionNode")["strategy"] == "cascade"
assert routing.cheapest_meeting_bar("PredictionSynthesis")["model"] == "claude-sonnet"
assert pc["reduction_pct"] >= 65.0
n_routes = g.run("MATCH (:WorkflowNode)-[r:ROUTED_TO]->(:Model) RETURN count(r) AS c")[0]["c"]
assert n_routes == 6
print("PASS: AlertClassifier on a 3B SLM, CausalAttributionNode cascaded, synthesis on the frontier, "
      f"blended cost down {pc['reduction_pct']}%. {n_routes} routes stored in Neo4j.")

# %% [markdown]
# ## 3. Measuring cost-performance — cost per *successful* completion
#
# **Why.** Routing is only as good as the numbers behind it. The right metric is **not** cost per call or per
# token, but **cost per successful completion**: total money spent ÷ tasks done *correctly*. A cheap model that
# fails 40% of the time can cost more per finished task than a pricier reliable one.
#
# **Graph angle:** every model call is logged as an `(:Invocation)` node linked to the workflow step that made
# it (`[:RAN_AS]`). Instead of AWS CloudWatch, **Cypher is our metrics store**: we aggregate costs with a query,
# then also feed the same rows into the skill's `CostTracker` and check the two agree.

# %%
sample = json.loads((OPT / "cost-performance-scorer" / "sample-invocations.json").read_text())

g.run("""
UNWIND range(0, size($rows) - 1) AS i
WITH i, $rows[i] AS r
MATCH (w:WorkflowNode {name: r.node_name})
CREATE (inv:Invocation {batch: 'sample', seq: i, model_id: r.model_id,
        input_tokens: r.input_tokens, output_tokens: r.output_tokens,
        cost_usd: r.cost_usd, latency_ms: r.latency_ms, success: r.success})
CREATE (inv)-[:RAN_AS]->(w)
""", rows=sample["invocations"])

print("Cost per success computed purely in Cypher:")
g.show(g.run("""
MATCH (i:Invocation {batch:'sample'})-[:RAN_AS]->(w:WorkflowNode)
WITH w.name AS step, count(i) AS calls,
     sum(CASE WHEN i.success THEN 1 ELSE 0 END) AS successes, sum(i.cost_usd) AS total_cost
RETURN step, calls, successes, round(total_cost * 1000000) / 1000000.0 AS total_cost_usd,
       round(total_cost / successes * 1000000) / 1000000.0 AS cost_per_success
ORDER BY cost_per_success
"""))

# %% [markdown]
# **How to read this:** `cost_per_success` divides *all* spend (including failed calls) by the number of
# successes. Now feed the same rows, read back from the graph, into the skill's `CostTracker`.

# %%
rows = g.run("""
MATCH (i:Invocation {batch:'sample'})-[:RAN_AS]->(w:WorkflowNode)
RETURN w.name AS node_name, i.model_id AS model_id, i.input_tokens AS input_tokens,
       i.output_tokens AS output_tokens, i.cost_usd AS cost_usd, i.latency_ms AS latency_ms,
       i.success AS success
ORDER BY i.seq
""")
tracker = costperf.CostTracker()
for row in rows:
    tracker.log(costperf.NodeInvocation(**row))

report = costperf.score_policy(tracker)
print(f"{'node':<24} {'$/success':>11} {'succ%':>6} {'p95 ms':>8}")
print("-" * 52)
for r in report["nodes"]:
    print(f"{r['node']:<24} {r['cost_per_success']:>11.6f} {r['success_rate']*100:>5.0f}% {r['p95_latency_ms']:>8.1f}")

graph_total = g.run("MATCH (i:Invocation {batch:'sample'}) RETURN sum(i.cost_usd) AS t")[0]["t"]
print(f"\nPipeline cost per success: ${report['pipeline_cost_per_success']}")
print(f"Neo4j (Cypher) total cost read back: ${round(graph_total, 6)}")

inv_cmp = costperf.compare_cost_per_success(0.007, 0.60, 0.010, 0.98)
print(f"Cost-per-success inversion: cheaper per call = {inv_cmp['cheaper_per_call']}, "
      f"cheaper per success = {inv_cmp['cheaper_per_success']}")

# %% [markdown]
# **Quality bars with weighted errors.** A generic accuracy score hides that some mistakes hurt more: calling a
# P1 (most urgent) alert "P3" is 10x worse than the reverse. `evaluate_candidate` applies those weights.

# %%
predictions = [("P1", "P1")] * 97 + [("P1", "P3")] * 2 + [("P3", "P1")] * 1
ev = costperf.evaluate_candidate(predictions, costperf.EVAL_SETS["AlertClassifier"])
print(ev)
print("Meets the 0.99 bar?", ev["meets_threshold"], "-> weighted error is", ev["weighted_error_rate"],
      "(accuracy alone looks fine at", ev["accuracy"], "but the P1-as-P3 misses are costly)")

# %%
# Verification gate: cost-per-success computed, SLM cheaper than frontier per success, graph totals
# agree with the tracker, and the inversion holds.
ac = tracker.node_report("AlertClassifier")
ps = tracker.node_report("PredictionSynthesis")
assert ac["cost_per_success"] > 0
assert ac["cost_per_success"] < ps["cost_per_success"]
assert abs(graph_total - report["pipeline_total_cost_usd"]) < 1e-5
assert inv_cmp["cheaper_per_success"] == "reliable"
assert not ev["meets_threshold"]
print("PASS: cost-per-success measured; the SLM node is far cheaper per success than the frontier node; "
      "the Neo4j total matches the tracker; the cheaper-per-call model loses on cost-per-success.")

# %% [markdown]
# ## 4. Hardware acceleration — dependency analysis within the latency budget
#
# **Why.** Incident response is the most time-sensitive path: an engineer wants the **blast radius** (what else
# breaks?) before they finish reading the alert. That is a graph computation over the dependency graph.
#
# First, the real thing on our small graph: *which services are affected if `orders-db` fails, and which
# services are the most depended-on?* (A simple stand-in for centrality algorithms like PageRank.)

# %%
t0 = time.perf_counter()
blast = g.run("""
MATCH (s:Service)-[:DEPENDS_ON*1..4]->(:Database {name:'orders-db'})
RETURN DISTINCT s.name AS affected_service, s.tier AS tier
ORDER BY tier, affected_service
""")
elapsed_ms = (time.perf_counter() - t0) * 1000
g.show(blast)
print(f"\nBlast-radius query on {counts['nodes']}+ nodes took {elapsed_ms:.1f} ms (tiny graph, easy).")

print("\nMost depended-on things (incoming DEPENDS_ON count):")
g.show(g.run("""
MATCH (x)<-[:DEPENDS_ON]-(d:Service)
RETURN coalesce(x.name, x.id) AS thing, labels(x)[0] AS type, count(DISTINCT d) AS dependents
ORDER BY dependents DESC, thing LIMIT 5
"""))

# %% [markdown]
# **Scaling up.** At production size (millions of nodes) this same traversal can take seconds on a CPU, and
# **GPU graph libraries (cuGraph)** cut it by 100x or more. The `kv-cache-latency-budgeter` skill holds the
# published speed-ups and checks whether a whole pipeline fits the **<2 second** target.
#
# **Latency budget.** The number of model calls in the pipeline comes from our graph; independent steps can run in
# parallel (`parallel_factor`).

# %%
br = budget.blast_radius_report()
est = budget.estimate_gpu_time(35, "betweenness_centrality")
print(f"Blast radius (Example 8-14): {br['cpu_seconds']} s CPU -> {br['gpu_result_ms']} ms GPU")
print(f"cuGraph estimate: 35 s CPU @ {est['speedup']}x -> {est['gpu_ms']} ms GPU")

pipeline = ["AlertClassifier", "LogParser", "DependencyAnalyzer", "CausalAttributionNode", "PredictionSynthesis"]
n_calls = g.run("MATCH (w:WorkflowNode) WHERE w.name IN $p RETURN count(w) AS c", p=pipeline)[0]["c"]
bp = budget.budget_pipeline(n_slm_calls=n_calls, n_graph_ops=1, parallel_factor=2.0)
print(f"\nPipeline latency ({n_calls} model calls + 1 graph op, 2x parallel): "
      f"{bp['estimated_ms']} ms; within 2s target: {bp['within_2s_target']}")

# %% [markdown]
# **What is a KV-cache?** When a language model reads a prompt and writes an answer, it keeps a working memory of
# every token so far, called the **KV-cache** (keys and values of attention). It lives in the GPU's memory and
# **grows with each user's conversation/reasoning**. So the real limit on how many users one GPU can serve at once
# is *peak KV-cache per user*, not model size:
#
# `users ≈ (GPU memory − model weights − overhead) ÷ KV per user`
#
# Shrinking the **weights** (quantization) frees little. Bounding the **cache** (the MEMENTO idea: summarise
# finished reasoning blocks and drop the originals) multiplies the number of users.

# %%
kv = budget.weight_quant_vs_kv_compression(gpu_memory_gb=80, model_weights_gb=16,
                                           peak_kv_per_user_gb=8,
                                           weight_quant_factor=2, kv_compression_factor=2)
print("Concurrency on an 80 GB H100:")
print(f"  baseline            : {kv['baseline_concurrency']} users")
print(f"  + weight quant (2x) : {kv['after_weight_quant']} users  (gain {kv['weight_quant_gain']})")
print(f"  + KV compress  (2x) : {kv['after_kv_compression']} users  (gain {kv['kv_compression_gain']})")
print(f"  winner: {kv['verdict']}  ->  bound the cache, not the weights")

# %%
# Verification gate: blast radius is sub-100ms on GPU, the pipeline fits the 2s budget, and KV compression
# beats weight quantization for concurrency.
assert est["gpu_ms"] < 100
assert bp["within_2s_target"]
assert kv["verdict"] == "kv_compression"
assert kv["kv_compression_gain"] > kv["weight_quant_gain"]
assert {r["affected_service"] for r in blast} == {"checkout-api", "orders-service", "payment-service", "cart-service"}
print("PASS: blast radius sub-100ms on GPU; the pipeline lands within the sub-2s budget; "
      "KV compression multiplies concurrency where weight quantization barely moves it.")

# %% [markdown]
# ## 5. Governance — scope what each role sees in the graph
#
# **The danger.** In a graph *everything is reachable from everything else* — that is the power, and the risk.
# One unscoped query can walk from a public service to employee salaries. Look:

# %%
print("An UNSCOPED query (no role limits) walking from a service to a private salary:")
g.show(g.run("""
MATCH (:Service {name:'checkout-api'})<-[:ON_CALL_FOR]-(e:Employee)-[:PAID]->(c:Compensation)
RETURN e.uuid AS employee, c.salary AS salary
"""))

# %% [markdown]
# **The fix: role-based, subgraph-level access control.** Each role may *traverse* only certain node types and
# relationship types, and *read* only certain properties. Two principles from the book:
# * **Security transparency** — a node you may not see is *invisible*, not "access denied". The agent cannot tell
#   "doesn't exist" from "not allowed", so it cannot probe the boundaries.
# * **Least privilege** — SRE (on-call engineers) see topology but not cost; Finance sees cost but not topology.
#
# The skill generates the real Neo4j Enterprise `GRANT/DENY` policy text (Community edition has no RBAC, so we
# *enforce the same rules in Cypher* below to show the effect).

# %%
print("SRE role policy (Neo4j GRANT/DENY):")
print(govern.generate_policy("sre_oncall"))

audit = govern.audit_access("sre_oncall", ["Service", "Library", "Metric"])
print(f"\nSRE dependency query functionally complete: {audit['functionally_complete']}")
print(f"SRE can read cost_per_hour:      {govern.can_read('sre_oncall', 'Service', 'cost_per_hour')}")
print(f"finance can read cost_per_hour:  {govern.can_read('finance_analyst', 'Service', 'cost_per_hour')}")
print(f"finance can traverse Library:    {govern.can_traverse('finance_analyst', 'Library')}")

# %% [markdown]
# **Enforcing the policy on real query results.** `role_view(role)` builds the role's *view* of the graph:
# * only nodes whose label the role may traverse (filtered **inside Cypher**),
# * only relationships of allowed types between visible nodes,
# * only properties the role may read (masked otherwise).
#
# Anything the policy does not mention (e.g. `Incident`, `Database`) is invisible by default: **deny by default**.

# %%
def role_view(role):
    pol = govern.ROLES[role]
    labels = [r["label"] for r in g.run("CALL db.labels() YIELD label RETURN label")]
    allowed = [l for l in labels if govern.can_traverse(role, l)]
    nodes = g.run("""
        MATCH (n) WHERE any(l IN labels(n) WHERE l IN $allowed)
        RETURN elementId(n) AS id, labels(n)[0] AS label, properties(n) AS props
    """, allowed=allowed)
    ids = [n["id"] for n in nodes]
    rels = g.run("""
        MATCH (a)-[r]->(b)
        WHERE type(r) IN $rels AND elementId(a) IN $ids AND elementId(b) IN $ids
        RETURN labels(a)[0] AS from_type, type(r) AS rel, labels(b)[0] AS to_type, count(*) AS n
        ORDER BY rel, from_type
    """, rels=sorted(pol.traverse_rels), ids=ids)
    for n in nodes:   # property masking
        n["props"] = {k: v for k, v in n["props"].items() if govern.can_read(role, n["label"], k)}
    return nodes, rels

total_nodes = g.run("MATCH (n) RETURN count(n) AS c")[0]["c"]
views = {role: role_view(role) for role in ("sre_oncall", "finance_analyst", "infra_analyst")}

for role, (nodes_v, rels_v) in views.items():
    by_label = {}
    for n in nodes_v:
        by_label[n["label"]] = by_label.get(n["label"], 0) + 1
    print(f"\n=== {role}: sees {len(nodes_v)} of {total_nodes} nodes -> {by_label}")
    print("    service rows as this role sees them:")
    g.show([{col: n["props"].get(col, "<masked>") for col in ("name", "status", "cost_per_hour", "budget_code")}
            for n in nodes_v if n["label"] == "Service"])
    print("    relationships it may follow:")
    g.show(rels_v)

# %% [markdown]
# **How to read this:** the SRE sees status but `<masked>` costs, and can follow `DEPENDS_ON`/`CALLS`/`DEPLOYED_TO`.
# Finance sees costs but no status and **zero relationships** — it cannot traverse the topology at all. Nobody sees
# `Employee` or `Compensation` — they are invisible, not "forbidden".
#
# > ⚠️ A policy can be *too* strict and starve the agent's reasoning. `audit_access` checks that the labels the
# > agent needs are reachable (`functionally_complete`). Notice our `Database` and `Incident` labels are not in
# > the book's policy, so a production setup would add them to the SRE role.

# %% [markdown]
# **Privacy by architecture (GDPR).** The graph stores only an opaque UUID (`u-7f3a`); the person's name and email
# live in a *separate* identity store, joined by UUID. "Right to be forgotten" is then easy. A **hard** erase
# deletes the UUID node and its relationships from the graph — we run the skill's generated Cypher for real:

# %%
identity_store = {"u-7f3a": {"name": "Dana Reyes", "email": "dana@example.com"}}   # stand-in relational DB
print("Employee nodes before erase:", g.run("MATCH (e:Employee) RETURN count(e) AS c")[0]["c"])
er = govern.gdpr_erase("u-7f3a", "hard")
print("identity store action :", er["identity_store_action"])
print("graph action          :", er["graph_action"])
g.run(er["graph_action"])
identity_store.pop("u-7f3a")
print("Employee nodes after erase :", g.run("MATCH (e:Employee) RETURN count(e) AS c")[0]["c"])
print("soft erase would instead:", govern.gdpr_erase("u-7f3a", "soft")["graph_action"])

# %% [markdown]
# **The execution graph as a compliance artifact.** Every agent decision is stored as a `(:Decision)` node with
# governance metadata: which data sources, which role, whether PII was touched, **which model** produced it.
# An auditor's question — *"which decisions accessed personal data last quarter?"* — becomes one Cypher query.

# %%
rec = govern.governance_metadata(
    data_sources_accessed=["infrastructure_graph", "metrics_api"],
    access_role="sre_on_call", model_id="llama-3.1-8b-instruct",
    model_version="v2.3.1", decision_confidence=0.87, pii_accessed=False)
rec2 = govern.governance_metadata(
    ["identity_store"], "billing_agent", "claude-sonnet", "v1.0", 0.90, pii_accessed=True)
print("Execution-graph governance record (Example 8-6):")
print(json.dumps(rec, indent=2))

for i, r in enumerate((rec, rec2), start=1):
    g.run("""
    MATCH (inc:Incident {id:'INC-1001'})
    CREATE (d:Decision {id: $id}) SET d += $props
    CREATE (d)-[:ABOUT]->(inc)
    """, id=f"dec-{i}", props=r)

print("\nAuditor query in Cypher: decisions that accessed PII")
pii_rows = g.run("""
MATCH (d:Decision)-[:ABOUT]->(i:Incident)
WHERE d.pii_accessed = true
RETURN d.id AS decision, d.access_role AS role, d.model_id AS model, i.id AS incident
""")
g.show(pii_rows)

# The skill's own filter, applied to records read back from the graph, gives the same answer.
records = g.run("MATCH (d:Decision) RETURN properties(d) AS p")
records = [r["p"] for r in records]
pii_hits = govern.audit_query(records, pii_accessed=True)
print(f"\nSkill audit_query over graph records: {len(pii_hits)} decision(s)")

# %%
# Verification gate: least privilege enforced both ways, SRE reasoning not starved, the decision is
# attributed to a model, and the PII audit works.
assert not govern.can_read("sre_oncall", "Service", "cost_per_hour")
assert govern.can_read("finance_analyst", "Service", "cost_per_hour")
assert not govern.can_traverse("finance_analyst", "Library")
assert not govern.can_traverse("sre_oncall", "Employee")
assert audit["functionally_complete"]
assert rec["model_id"] == "llama-3.1-8b-instruct"
assert len(pii_hits) == 1 and len(pii_rows) == 1
sre_nodes, sre_rels = views["sre_oncall"]
fin_nodes, fin_rels = views["finance_analyst"]
assert all("cost_per_hour" not in n["props"] for n in sre_nodes)
assert any("cost_per_hour" in n["props"] for n in fin_nodes)
assert not any(n["label"] in ("Employee", "Compensation") for v in views.values() for n in v[0])
assert not any(n["label"] == "Library" for n in fin_nodes) and fin_rels == []
print("PASS: SRE cannot see cost data; finance cannot see deployment topology; Employee is invisible to "
      "everyone; SRE reasoning is not starved; every decision is attributed to its model; "
      "the PII audit is a graph query.")

# %% [markdown]
# ## 6. Production maintenance — schema evolution and incremental updates
#
# **Why.** A production graph is never finished. Two jobs keep it healthy:
# 1. **Schema evolution** — add a new relationship type (`MONITORED_BY`) without breaking the running agent.
#    The key rule is **N-1 compatibility**: the *previous* version of the agent code must keep working after the
#    migration, so you can deploy schema, data and code in separate, reversible stages.
# 2. **Incremental updates** — when a deployment happens, update only what changed (a `MERGE` upsert), and
#    *invalidate* old facts (set an end time) instead of deleting them, so history remains for causal reasoning.
#
# **Step A — a real migration.** Today each service keeps its alert as a plain text property `alert_config`.
# The migration turns it into a proper `(:Service)-[:MONITORED_BY]->(:AlertRule)` relationship (plus a uniqueness
# constraint so no duplicate rules exist).

# %%
g.run("""
UNWIND $rows AS r
MATCH (s:Service {name: r.name}) SET s.alert_config = r.alert
""", rows=[{"name": "checkout-api", "alert": "p99-latency-alarm"},
           {"name": "payment-service", "alert": "payment-error-rate"},
           {"name": "orders-service", "alert": "orders-p99"}])

mig = schema.monitored_by_migration()
print(f"Migration {mig.version}: {mig.description}  (N-1 compatible: {mig.n1_compatible})")
print(mig.cypher)

def statements(script):
    """Split a .cypher migration script into runnable statements (drop // comment lines)."""
    lines = [l for l in script.split("\n") if not l.strip().startswith("//")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]

for stmt in statements(mig.cypher):
    g.run(stmt)

print("\nAfter migration:")
g.show(g.run("""
MATCH (s:Service)-[:MONITORED_BY]->(a:AlertRule)
RETURN s.name AS service, a.name AS alert_rule ORDER BY service
"""))
monitored = g.run("MATCH (:Service)-[r:MONITORED_BY]->(:AlertRule) RETURN count(r) AS c")[0]["c"]
print("MONITORED_BY relationships:", monitored)

# %% [markdown]
# **Rollback works too.** Every migration ships its undo script. We run it, confirm the new relationships are
# gone (old code never noticed either way — N-1!), then re-apply the migration. `MERGE` makes re-applying safe.

# %%
for stmt in statements(mig.rollback):
    g.run(stmt)
after_rollback = g.run("MATCH (:Service)-[r:MONITORED_BY]->() RETURN count(r) AS c")[0]["c"]
print("MONITORED_BY after rollback:", after_rollback)
for stmt in statements(mig.cypher):
    g.run(stmt)
reapplied = g.run("MATCH (:Service)-[r:MONITORED_BY]->() RETURN count(r) AS c")[0]["c"]
print("MONITORED_BY after re-apply:", reapplied)

# %% [markdown]
# **Step B — the staged rollout manifest.** Schema, data and code are deployed in four ordered phases, each
# depending on the previous one; the agent-code phase is a *canary* (5% of traffic) with promotion criteria.
# `validate_manifest` checks the discipline.

# %%
manifest = schema.staged_rollout_manifest("2024-03-18-causal-attribution-v2")
v = schema.validate_manifest(manifest)
print(f"Deployment manifest phases: {[p['name'] for p in manifest['phases']]}")
print(f"Manifest valid: {v['valid']}")

# %% [markdown]
# **Step C — temporal invalidation (never delete history).** A deployment upgrades `stripe-python` from 3.2.1 to
# 3.3.0 on `checkout-api`. We stamp the old `DEPENDS_ON` edge with an end time (`t_invalid`) and add a new edge.
# Later you can still ask "what did checkout depend on *during the incident in June*?".

# %%
inval = schema.temporal_invalidate_cypher("checkout-api", "stripe-python", "3.2.1", "3.3.0")
sets_invalid = "t_invalid = datetime()" in inval and "DELETE" not in inval.upper()
print(inval)
try:
    g.run(inval)
except Exception as exc:
    # Neo4j refuses MERGE on a relationship whose property value is null; drop that line and set it afterwards.
    print("\nNeo4j rejected the skill's MERGE (", str(exc).splitlines()[0][:90], ")")
    print("-> retrying without the null property (a missing property means the same thing).")
    g.run(inval.replace("    t_invalid: null,\n", ""))

print("\nstripe-python history on checkout-api (a missing/NULL t_invalid means 'still current'):")
g.show(g.run("""
MATCH (:Service {name:'checkout-api'})-[d:DEPENDS_ON]->(l:Library {name:'stripe-python'})
RETURN l.version AS version, d.t_invalid IS NULL AS is_current,
       toString(d.t_valid) AS valid_from, toString(d.t_invalid) AS invalidated_at
ORDER BY version
"""))

# %% [markdown]
# **Step D — the incremental merge for the whole deployment event.** The skill returns three Cypher statements:
# 1. upsert the service (`ON CREATE` / `ON MATCH`), 2. upsert each library dependency with timestamps,
# 3. invalidate dependencies no longer listed. The cost is proportional to the *new data*, not the whole graph.
# We run them for real. (The sample event names `checkout-service`; we point it at our `checkout-api`.)

# %%
event = json.loads((OPT / "schema-evolution-migrator" / "sample-deployment-event.json").read_text())
event["service"] = "checkout-api"
stmts = schema.incremental_merge_cypher(event)
print(f"Incremental merge for {event['service']} "
      f"(deps {', '.join(d['name'] for d in event['dependencies'])}): {len(stmts)} statements")

def apply_event(event):
    s1, s2, s3 = schema.incremental_merge_cypher(event)
    g.run(s1, service_name=event["service"], env=event["environment"], new_version=event["version"])
    for dep in event["dependencies"]:
        g.run(s2, lib_name=dep["name"], lib_version=dep["version"], service_name=event["service"])
    g.run(s3, service_name=event["service"], current_libs=[d["name"] for d in event["dependencies"]])

apply_event(event)
print("\nCurrent dependencies of checkout-api after the deployment event:")
g.show(g.run("""
MATCH (s:Service {name:'checkout-api'})-[d:DEPENDS_ON]->(l:Library)
RETURN l.name AS library, l.version AS version, d.t_invalid IS NULL AS is_current
ORDER BY library, version
"""))

growth = schema.snapshot_growth_per_day(200, 5)
print(f"Snapshot growth (Fischer): 200 resources @ 5-min = {growth} nodes/day -> plan retention policies:")
for k, vv in schema.RETENTION_POLICY.items():
    print(f"  {k:<9} {vv}")

# %%
# Verification gate: valid N-1 rollout, incremental upsert semantics, and the Fischer growth figure.
assert mig.n1_compatible
assert v["valid"]
joined = "\n".join(stmts)
assert "ON CREATE SET" in joined and "ON MATCH SET" in joined
assert "NOT l.name IN $current_libs" in joined
assert growth == 57600
assert sets_invalid
assert monitored == 3 and after_rollback == 0 and reapplied == 3
old_edge = g.run("""
MATCH (:Service {name:'checkout-api'})-[d:DEPENDS_ON]->(:Library {name:'stripe-python', version:'3.2.1'})
RETURN d.t_invalid IS NOT NULL AS invalidated
""")
assert old_edge and old_edge[0]["invalidated"]          # old edge still exists, but is closed
new_edge = g.run("""
MATCH (:Service {name:'checkout-api'})-[d:DEPENDS_ON]->(:Library {name:'stripe-python', version:'3.3.0'})
WHERE d.t_invalid IS NULL RETURN count(d) AS c
""")[0]["c"]
assert new_edge == 1
print("PASS: N-1 compatible migration applied, rolled back and re-applied in Neo4j; a valid four-phase "
      "staged rollout; the incremental merge upserts and invalidates-not-deletes; Fischer growth is "
      "57,600 nodes/day for one small cluster.")

# %% [markdown]
# ## 7. The completed agent — one deployment event, all five skills
#
# Now we trace a single deployment event through the optimized pipeline:
# 1. **Route** each step to the right model (selective intelligence).
# 2. **Govern** — the agent acts under the SRE role, so cost data is out of scope.
# 3. **Remember** — merge the event into the graph incrementally.
# 4. **Measure** — log per-step cost/latency as `Invocation` nodes and total them with Cypher.
# 5. **Budget** — check end-to-end latency.
# 6. **Record** — write a `Decision` node with governance metadata.
#
# We deliver the *same* event a second time to show the merge is **idempotent** (re-delivery changes nothing).

# %%
def handle_deployment_event(event):
    pipeline = ["AlertClassifier", "LogParser", "DependencyAnalyzer",
                "CausalAttributionNode", "PredictionSynthesis"]

    # 1. Selective intelligence — route each node
    routes = {n: routing.cheapest_meeting_bar(n) for n in pipeline}

    # 2. Governance — the agent runs as SRE; cost data is out of scope
    assert not govern.can_read("sre_oncall", "Service", "cost_per_hour")

    # 3. Memory — incremental merge of the event into the real graph
    before = g.run("MATCH (:Service {name:$s})-[d:DEPENDS_ON]->(:Library) RETURN count(d) AS c", s=event["service"])[0]["c"]
    apply_event(event)
    after = g.run("MATCH (:Service {name:$s})-[d:DEPENDS_ON]->(:Library) RETURN count(d) AS c", s=event["service"])[0]["c"]

    # 4. Cost signals per node -> Invocation nodes in the graph
    node_signals = {
        "AlertClassifier":       (0.00007, 41.0),
        "LogParser":             (0.00021, 88.0),
        "DependencyAnalyzer":    (0.00030, 180.0),
        "CausalAttributionNode": (0.00050, 220.0),
        "PredictionSynthesis":   (0.00100, 710.0),
    }
    batch = f"event-{event['version']}-{int(time.time())}"
    g.run("""
    UNWIND $rows AS r
    MATCH (w:WorkflowNode {name: r.node})
    CREATE (i:Invocation {batch: $batch, model_id: r.model, cost_usd: r.cost, latency_ms: r.lat, success: true})
    CREATE (i)-[:RAN_AS]->(w)
    """, batch=batch, rows=[{"node": n, "model": routes[n]["model"], "cost": c, "lat": l}
                            for n, (c, l) in node_signals.items()])
    event_cost = g.run("MATCH (i:Invocation {batch:$b}) RETURN sum(i.cost_usd) AS t", b=batch)[0]["t"]

    # 5. Latency budget
    lat = budget.budget_pipeline(n_slm_calls=5, n_graph_ops=1, parallel_factor=2.0)

    # 6. Governance record in the execution graph
    rec = govern.governance_metadata(
        ["infrastructure_graph", "metrics_api"], "sre_on_call",
        "llama-3.1-8b-instruct", "v2.3.1", 0.87, pii_accessed=False)
    g.run("""
    MATCH (inc:Incident {id:'INC-1001'})
    CREATE (d:Decision {id: $id}) SET d += $props
    CREATE (d)-[:ABOUT]->(inc)
    """, id=f"dec-{batch}", props=rec)

    return {"routes": routes, "edges_before": before, "edges_after": after,
            "latency": lat, "governance": rec, "event_cost_usd": event_cost}

result = handle_deployment_event(event)
print("Pipeline routing:")
for n in ["AlertClassifier", "LogParser", "DependencyAnalyzer", "CausalAttributionNode", "PredictionSynthesis"]:
    print(f"  {n:<24} -> {result['routes'][n]['model']}")
print(f"\nDependency edges before/after re-delivering the event: {result['edges_before']} / {result['edges_after']} "
      "(unchanged = idempotent)")
print(f"Event cost (summed in Neo4j):     ${round(result['event_cost_usd'], 5)} "
      "(book target ~$0.002 vs ~$0.01 frontier-everywhere)")
print(f"End-to-end latency estimate:      {result['latency']['estimated_ms']} ms "
      f"(within 2s: {result['latency']['within_2s_target']})")
print(f"Governance role:                  {result['governance']['access_role']}, "
      f"pii_accessed={result['governance']['pii_accessed']}")

print("\nAll agent decisions now recorded in the graph:")
g.show(g.run("""
MATCH (d:Decision)-[:ABOUT]->(i:Incident)
RETURN d.access_role AS role, d.model_id AS model, d.pii_accessed AS pii, i.id AS incident
ORDER BY role
"""))

# %%
# Closing verdict: the production-hardened agent is routed, governed, within latency budget, and affordable.
pc = routing.pipeline_cost(0.30)
routed = pc["reduction_pct"] >= 65.0
governed = (not govern.can_read("sre_oncall", "Service", "cost_per_hour")
            and result["governance"]["pii_accessed"] is False)
within_budget = result["latency"]["within_2s_target"]
affordable = result["event_cost_usd"] < 0.01
idempotent = result["edges_before"] == result["edges_after"]

assert routed and governed and within_budget and affordable and idempotent

print("=" * 68)
print("The production-hardened DevOps agent:")
print(f"  ROUTED     - blended cost {pc['reduction_pct']}% below frontier-everywhere")
print("  GOVERNED   - SRE role cannot read cost data; every decision is logged")
print(f"  IN BUDGET  - {result['latency']['estimated_ms']} ms end-to-end, within the sub-2s target")
print(f"  AFFORDABLE - ${round(result['event_cost_usd'], 5)} per event")
print("=" * 68)

# Leave the shared database tidy: drop the constraint the migration created.
g.run("DROP CONSTRAINT alertrule_name_unique IF EXISTS")

# %% [markdown]
# ## Recap
#
# | Idea | What you did in Neo4j |
# |---|---|
# | **Selective intelligence** | Stored workflow steps + models as nodes; routed each step to the cheapest model meeting its quality bar |
# | **Cost per success** | Logged calls as `Invocation` nodes; aggregated cost-per-success in Cypher and with the skill |
# | **Latency & KV-cache** | Ran a blast-radius traversal; budgeted the pipeline; saw why the KV-cache, not weights, limits users |
# | **Governance** | Role-filtered Cypher views, GDPR hard-erase, and `Decision` nodes making audits a one-line query |
# | **Maintenance** | Applied, rolled back and re-applied a real schema migration; invalidated (not deleted) old edges; idempotent incremental merge |
#
# The DevOps agent that started in Chapter 1 as "the checkout API is slow" is now routed, governed, in budget and
# affordable.
#
# **Next:** this was the final chapter. Head back to the repo `README.md` for the full chapter-by-chapter
# skill list and to try the skills from the command line (`cli.py`) in your own projects.
