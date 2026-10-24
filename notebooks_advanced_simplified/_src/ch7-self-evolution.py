# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > [Beginner (tier 1/3)](../notebooks_simplified/ch7-self-evolution.ipynb)  →  [Intermediate (tier 2/3)](../notebooks_intermediate_simplified/ch7-self-evolution.ipynb)  →  **Advanced (tier 3/3) ← you are here**

# %% [markdown]
# # Chapter 7 (Advanced) — Self-Evolution on a Real Graph
#
# **Tier 3 of 3 (advanced).** The beginner notebook (`notebooks_simplified/`) showed each Chapter 7 skill once.
# Here we wire **all 7 skills** into one *graph-native* loop on a live Neo4j database:
#
# `execution graph -> 4-layer diagnosis -> taxonomy -> intervention -> semantic backprop -> graduated validation -> XSkill memory`
#
# **What is different from the beginner tier**
# * Skill inputs are **derived from Cypher** (not hard-coded dicts) and skill outputs are **written back** as nodes/edges.
# * 3+ non-trivial Cypher queries (variable-length paths, `collect`/`count`, `OPTIONAL MATCH`), each explained on first use.
# * A **MiniAgent** runs plan -> act -> observe on the *checkout-api-is-slow* incident, and its trace lives in Neo4j.
# * Skill **CLI + benchmark** section, **vibe-coding corner**, **experiments**, **pitfalls** and a final **verification battery**.
#
# **Scenario** (fictional AWS account `123456789012`, no LLM calls, no API keys): a prediction agent said
# *"stripe-python 3.3.0 will break checkout-api through an API contract violation"*. The real cause was a
# **connection-pool exhaustion** triggered by a timeout change. We use the skills to learn from that miss.

# %%
import adv_helpers as a
import json, subprocess, sys
from datetime import date

a.load_devops_graph()          # wipes + reloads the shared DevOps graph: the expected start of every notebook

exec_graph = a.load_skill("self-evolution", "execution-graph")
cascade    = a.load_skill("self-evolution", "four-layer-eval-cascade")
taxonomy   = a.load_skill("self-evolution", "evolution-taxonomy-classifier")
selector   = a.load_skill("self-evolution", "intervention-selector")
backprop   = a.load_skill("self-evolution", "semantic-backprop-attributor")
gvp        = a.load_skill("self-evolution", "graduated-validation-protocol")
xskill     = a.load_skill("self-evolution", "xskill-self-improving-object")

print("Chapter 7 skills available:")
a.show([{"skill": s["slug"], "what it does": s["description"][:70] + "..."} for s in a.skill_catalog("self-evolution")])
a.check("7 chapter skills catalogued", len(a.skill_catalog("self-evolution")) == 7)

# %% [markdown]
# ## 1. Model the evidence as a graph
#
# `load_devops_graph()` gave us services, a database, a deployment and an incident. We add three things the
# diagnosis needs: the **library** that changed (with its changelog entries), **log events** emitted by `checkout-api`,
# and a few **learnings** the agent previously stored (used later for garbage collection).
#
# **Why a graph?** The question "which change, on which dependency path, produced which symptom?" is a *path*
# question. In a relational table it needs several joins; in a graph it is one pattern.
#
# Cypher notes (first use): `UNWIND $list AS x` turns a Python list into one row per element;
# `MATCH` finds a pattern; `CREATE` adds nodes/edges; `$name` is a safe query parameter (never string-format values into Cypher).

# %%
changes = [
    {"id": "c1", "kind": "api",    "text": "deprecated sync batch_charge API"},
    {"id": "c2", "kind": "api",    "text": "added async payment methods"},
    {"id": "c3", "kind": "config", "text": "default connection timeout reduced 30 to 10 seconds"},
]
events = [
    {"id": "r001", "level": "INFO",  "ms": 41,   "pool": False, "msg": "ok"},
    {"id": "d001", "level": "INFO",  "ms": 0,    "pool": False, "msg": "deploy stripe-python 3.2.1->3.3.0 timeout=10s"},
    {"id": "r107", "level": "WARN",  "ms": 2810, "pool": False, "msg": "db_pool_wait_ms=2700"},
    {"id": "r113", "level": "ERROR", "ms": 5012, "pool": True,  "msg": "status=504"},
    {"id": "r119", "level": "ERROR", "ms": 4998, "pool": True,  "msg": "status=504"},
]
a.run("""
MATCH (c:Service {name:'checkout-api'})
CREATE (c)-[:USES {methods:['payments.charge','payments.retrieve']}]->(l:Library {name:'stripe-python', from_v:'3.2.1', to_v:'3.3.0'})
WITH l
UNWIND $changes AS ch
CREATE (l)-[:HAS_CHANGE]->(:Change {id:ch.id, kind:ch.kind, text:ch.text})
""", changes=changes)
a.run("""
MATCH (c:Service {name:'checkout-api'})
UNWIND $events AS e
CREATE (c)-[:EMITTED]->(:LogEvent {id:e.id, level:e.level, duration_ms:e.ms, pool_exhausted:e.pool, msg:e.msg})
""", events=events)
a.run("""
UNWIND $ls AS l
CREATE (:Learning {id:l.id, confidence:l.conf, last_accessed_days:l.days, resolved_promoted:l.promoted, contradicted:l.contra})
""", ls=[
    {"id": "premature-closure-fix",        "conf": 0.9, "days": 2,  "promoted": True,  "contra": False},
    {"id": "stale-api-hypothesis",         "conf": 0.6, "days": 45, "promoted": False, "contra": False},
    {"id": "contradicted-guess",           "conf": 0.4, "days": 5,  "promoted": False, "contra": True},
    {"id": "fresh-pool-exhaustion-pattern","conf": 0.8, "days": 1,  "promoted": False, "contra": False},
])
n = a.run("MATCH (n) RETURN count(n) AS nodes")[0]["nodes"]
a.check("graph has Library, Change, LogEvent, Learning nodes", n >= 20)

# %% [markdown]
# ## 2. Cypher query #1 — blast radius (variable-length path)
#
# **Blast radius** = the set of components that can be affected when one thing changes. If `checkout-api` is
# degraded, who suffers (upstream callers) and who is it leaning on (downstream dependencies)?
#
# Clause by clause:
# * `-[:DEPENDS_ON*1..3]->` is a **variable-length path**: follow `DEPENDS_ON` edges 1 to 3 hops. `*` is "repeat".
# * `DISTINCT` removes duplicates when several paths reach the same node.
# * `collect(...)` is an **aggregation** that folds many rows into one list; `count(...)` counts rows.
# * `length(p)` is the number of hops of path `p` (we bind the whole path with `p = ...`).

# %%
upstream = a.run("""
MATCH p = (d:Service)-[:DEPENDS_ON*1..3]->(:Service {name:'checkout-api'})
RETURN collect(DISTINCT d.name) AS callers_hurt, count(DISTINCT d) AS n
""")[0]
downstream = a.run("""
MATCH p = (:Service {name:'checkout-api'})-[:DEPENDS_ON*1..3]->(x)
RETURN x.name AS dependency, min(length(p)) AS hops
ORDER BY hops, dependency
""")
print("Callers hurt if checkout-api degrades:", upstream["callers_hurt"])
a.show(downstream)
a.check("cart-service is in the upstream blast radius", "cart-service" in upstream["callers_hurt"])
a.check("orders-db is reachable within 2 hops", any(r["dependency"] == "orders-db" and r["hops"] == 2 for r in downstream))

# %% [markdown]
# ## 3. Cypher query #2 — evidence per service with `OPTIONAL MATCH`
#
# A plain `MATCH` drops a service that has *no* log events. `OPTIONAL MATCH` keeps the row and fills `null`
# instead (like a SQL `LEFT JOIN`). That matters for diagnosis: *absence* of errors on `payment-service` is evidence too.
#
# `WITH` passes the aggregated values to the next clause; `CASE WHEN ... THEN 1 ELSE 0 END` is an inline if/else
# used to count only pool-exhaustion events.

# %%
evidence = a.run("""
MATCH (s:Service)
OPTIONAL MATCH (s)-[:EMITTED]->(e:LogEvent)
WITH s, count(e) AS events,
     sum(CASE WHEN e.pool_exhausted THEN 1 ELSE 0 END) AS pool_hits,
     max(e.duration_ms) AS worst_ms
RETURN s.name AS service, events, pool_hits, worst_ms
ORDER BY pool_hits DESC, service
""")
a.show(evidence)
a.check("checkout-api has 2 pool-exhaustion events", evidence[0]["service"] == "checkout-api" and evidence[0]["pool_hits"] == 2)
a.check("silent services are kept by OPTIONAL MATCH", any(r["events"] == 0 for r in evidence))

# %% [markdown]
# ## 4. The agent's autobiography: execution graph (skill 1)
#
# An **execution graph** is a **DAG** (directed acyclic graph: arrows, never a loop back) of every step the agent
# took. Each node is an atomic operation with latency / tokens / cost. It uses a **two-phase write**:
# `begin_node` records structure *before* the step runs (so a crash still leaves a trace) and `complete_node`
# fills in results afterwards.
#
# We store the 5 recorded steps in Neo4j first, then **read them back with Cypher** to drive the skill — the
# database is the source of truth, the skill is the analyser. `OPTIONAL MATCH (p)-[:NEXT]->(s)` finds each
# step's parent (the root has none).

# %%
steps = [
    ("KnowledgeGraphQuery",   "Retrieval", 98.7,  0,   0.0,   {"deps": "checkout-api uses payments.charge/retrieve; no batch_charge usage"}),
    ("ChangelogRetrieval",    "Retrieval", 120.4, 0,   0.0,   {"changelog": [c["text"] for c in changes]}),
    ("ChangeAnalyzer",        "LLM_Call",  610.2, 420, 0.006, {"classified": "API deprecations present, configuration changes present"}),
    ("CausalAttributionNode", "LLM_Call",  1400.0, 847, 0.013, {"failure_mode": "API contract violation", "confidence": 0.78}),
    ("PredictionSynthesis",   "LLM_Call",  305.1, 210, 0.004, {"prediction": "HIGH impact; review checkout-api for batch_charge usage"}),
]
a.run("CREATE (:Execution {id:'pred_7f3a9c', query:'Will stripe-python 3.2.1 -> 3.3.0 break checkout-api?', verdict:'partially correct'})")
a.run("""
UNWIND $steps AS s
MATCH (e:Execution {id:'pred_7f3a9c'})
CREATE (e)-[:HAS_STEP]->(:Step {execution:'pred_7f3a9c', seq:s.seq, label:s.label, type:s.type,
        latency_ms:s.lat, tokens:s.tok, cost:s.cost, output:s.out})
""", steps=[{"seq": i, "label": l, "type": t, "lat": lat, "tok": tok, "cost": c, "out": json.dumps(o)}
            for i, (l, t, lat, tok, c, o) in enumerate(steps)])
a.run("""
MATCH (a:Step {execution:'pred_7f3a9c'}), (b:Step {execution:'pred_7f3a9c'})
WHERE b.seq = a.seq + 1
CREATE (a)-[:NEXT]->(b)
""")

rows = a.run("""
MATCH (:Execution {id:'pred_7f3a9c'})-[:HAS_STEP]->(s:Step)
OPTIONAL MATCH (p:Step)-[:NEXT]->(s)
RETURN s.seq AS seq, s.label AS label, s.type AS type, p.label AS parent,
       s.latency_ms AS lat, s.tokens AS tok, s.cost AS cost, s.output AS out
ORDER BY seq
""")
eg = exec_graph.ExecutionGraph(execution_id="pred_7f3a9c")
ids = {}
for r in rows:
    nid = eg.begin_node(r["type"], input_payload={"node": r["label"]}, parent_id=ids.get(r["parent"]))
    eg.complete_node(nid, output_payload=json.loads(r["out"]), latency_ms=r["lat"], token_count=r["tok"], cost_usd=r["cost"])
    ids[r["label"]] = nid
summary = eg.summary()
chain = [n.input_payload["node"] for n in eg.causal_chain(ids["CausalAttributionNode"])]
print("summary:", json.dumps(summary, default=str))
print("causal chain (root first):", " -> ".join(chain))
a.check("execution graph has 5 nodes, 0 failed", summary["node_count"] == 5 and summary["failed_node_count"] == 0)
a.check("causal chain ends at the faulty node", chain[-1] == "CausalAttributionNode" and chain[0] == "KnowledgeGraphQuery")

# %% [markdown]
# ## 5. Cypher query #3 — the causal path, multi-hop, and a cost hot-spot
#
# The same chain the skill computed in memory can be asked of Neo4j directly. `[:NEXT*]` walks any number of hops;
# `nodes(p)` returns the nodes along path `p` and `UNWIND` flattens them to rows. Then an aggregation answers a
# different question: *where does the money and time go?*

# %%
path_rows = a.run("""
MATCH p = (:Step {execution:'pred_7f3a9c', label:'KnowledgeGraphQuery'})-[:NEXT*]->(:Step {label:'CausalAttributionNode'})
UNWIND nodes(p) AS n
RETURN n.label AS step
""")
cost = a.run("""
MATCH (:Execution {id:'pred_7f3a9c'})-[:HAS_STEP]->(s:Step)
WITH s.type AS type, count(s) AS steps, sum(s.latency_ms) AS total_ms, sum(s.cost) AS usd, collect(s.label) AS labels
RETURN type, steps, round(total_ms) AS total_ms, usd, labels ORDER BY total_ms DESC
""")
a.show(cost)
a.check("Neo4j path equals the skill's causal chain", [r["step"] for r in path_rows] == chain)
a.check("LLM calls dominate cost", cost[0]["type"] == "LLM_Call" and abs(cost[0]["usd"] - 0.023) < 1e-9)

# %% [markdown]
# ## 6. Four-layer evaluation cascade (skill 2)
#
# The cascade asks questions from cheapest/most general to most specific and **stops at the first failing layer**:
# * **L0** hallucination gate — is the answer grounded in the retrieved text?
# * **L1** context evaluator — did the agent *have* the facts it needed?
# * **L2** cognitive fault isolator — **KNOWLEDGE** failure (wrong facts, `Knowledge Index` below 0.8) or
#   **REASONING** failure (right facts, bad connections)? **InfoGain** is how much each step increased confidence in
#   the right answer; steps below 0.05 are "low-information" — the reasoning stalled.
# * **L3** TIR-judge — verify a quantitative claim.
#
# The premise text comes from the `Change` nodes in Neo4j, and the faulty node from the step with the worst latency-per-token profile.
# We write the verdict back as a `Diagnosis` node that `TARGETS` the step.

# %%
changelog = [r["t"] for r in a.run("MATCH (:Library)-[:HAS_CHANGE]->(c:Change) RETURN c.text AS t ORDER BY c.id")]
a.run("MATCH (e:Execution {id:'pred_7f3a9c'}) SET e.infogain_trace=$t, e.knowledge_index=0.91",
      t=[0.34, 0.29, 0.22, 0.03, -0.01, 0.19])
ex = a.run("MATCH (e:Execution {id:'pred_7f3a9c'}) RETURN e.query AS q, e.infogain_trace AS t, e.knowledge_index AS ki")[0]
fault = a.run("""
MATCH (:Execution {id:'pred_7f3a9c'})-[:HAS_STEP]->(s:Step {type:'LLM_Call'})
RETURN s.label AS node_id, s.type AS node_type ORDER BY s.latency_ms DESC LIMIT 1
""")[0]
premise = ("Knowledge graph and changelog for stripe-python 3.3.0. " + ". ".join(changelog) + ". "
           "checkout-service uses payments charge and payments retrieve. "
           "API contract violation is one candidate. Connection pool exhaustion is another candidate.")
execution = {
    "execution_id": "pred_7f3a9c", "query": ex["q"],
    "answer": "API contract violation from deprecated batch_charge",
    "context_premise": premise,
    "required_claims": ["deprecated batch_charge", "connection timeout reduced 30 to 10"],
    "infogain_trace": ex["t"], "knowledge_index": ex["ki"],
    "fault_node": {"node_id": fault["node_id"], "node_type": fault["node_type"], "description": fault["node_id"]},
    "diagnosis": "Premature commitment to the API hypothesis at step 4; the timeout change was never explored.",
}
report = cascade.run_cascade(execution)
cog = report["layer_2_cognitive"]
print(json.dumps({k: report[k] for k in ("overall_verdict", "stopped_at_layer", "recommended_intervention", "target_nodes")}, indent=2))
print("L2:", cog["failure_type"], "low steps:", cog["low_infogain_steps"], "KI:", cog["knowledge_index"])
a.run("""
MATCH (s:Step {execution:'pred_7f3a9c', label:$node})
CREATE (:Diagnosis {id:'dx-1', failure_type:$ft, layer:$layer, low_steps:$low, knowledge_index:$ki})-[:TARGETS]->(s)
""", node=report["target_nodes"][0], ft=cog["failure_type"], layer=report["stopped_at_layer"],
      low=cog["low_infogain_steps"], ki=cog["knowledge_index"])
a.check("cascade stops at layer 2 with REASONING", report["stopped_at_layer"] == 2 and cog["failure_type"] == "REASONING")
a.check("low-InfoGain steps are [4, 5]", cog["low_infogain_steps"] == [4, 5])
a.check("target is CausalAttributionNode", report["target_nodes"] == ["CausalAttributionNode"])

# %% [markdown]
# **Probing the cascade (what-if).** A good diagnostic is *discriminating*: different evidence must route to different
# layers. We change one input at a time and watch where it stops. (This is the **blast radius of the diagnosis itself**.)

# %%
def stop_layer(**over):
    r = cascade.run_cascade({**execution, **over})
    return r["stopped_at_layer"], r["recommended_intervention"]

what_if = [
    {"scenario": "baseline", "result": stop_layer()},
    {"scenario": "missing timeout fact in context", "result": stop_layer(context_premise=premise.replace("connection timeout reduced 30 to 10 seconds", "nothing"))},
    {"scenario": "answer invents 'orders-v9.9' API", "result": stop_layer(answer="orders-v9.9 invalid_cursor_7 regression")},
    {"scenario": "low knowledge index 0.5", "result": stop_layer(knowledge_index=0.5)},
]
a.show([{"scenario": w["scenario"], "stopped_at_layer": w["result"][0], "intervention": w["result"][1]} for w in what_if])
a.check("different evidence -> different routes", len({w["result"] for w in what_if}) >= 3)
a.check("KI 0.5 forces FINE_TUNE", what_if[3]["result"][1] == "FINE_TUNE")

# %% [markdown]
# ## 7. Evolution taxonomy (skill 3)
#
# Before changing the system, ask four questions: **what** evolves (model / context / tool / architecture),
# **when** (`intra_test_time` = during the request, `inter_test_time` = between requests), **how** it learns
# (reward-, imitation- or population-based), **where** it applies (general vs domain-specialised).
# `route_failure` maps a failure type to an axis automatically, and `budget_check` answers a **latency-budget**
# question: if the evolution step fires *in the request path*, does it still fit the SLA?

# %%
classification = taxonomy.classify_from_signals(
    what="model", when="inter_test_time", how="reward_based", where="domain_specialized",
    notes="Prompt refinement for CausalAttributionNode after premature-closure diagnosis.")
print(json.dumps(classification.to_dict(), indent=2, default=str)[:600])
routes = [{"failure": f, "recurring": rec, **{k: v for k, v in taxonomy.route_failure(f, recurring=rec).items() if k != "rationale"}}
          for f, rec in [("REASONING", False), ("REASONING", True), ("KNOWLEDGE", False), ("FORMAT", False)]]
a.show(routes)
budgets = [taxonomy.budget_check(when_fires="intra_test_time", measured_op_ms=ms, request_budget_ms=2000) for ms in (300, 2500)]
print("budget verdicts:", [b["verdict"] for b in budgets])
a.check("taxonomy has all four axes", set(classification.graph_rationale) == {"what", "when", "how", "where"})
a.check("recurring REASONING escalates to fine-tune", routes[1]["mechanism"] != routes[0]["mechanism"])
a.check("slow in-path op is moved off the request path", budgets[1]["verdict"] == "MOVE_TO_INTER")

# %% [markdown]
# ## 8. Intervention selector (skill 4) — deterministic routing, tunable thresholds
#
# Four branches in strict order: insufficient context -> `RETRIEVAL_FIX`; `FORMAT_VIOLATION` -> `STRUCTURAL_CONSTRAINT`;
# localized reasoning failure with intact knowledge -> `PROMPT_REFINEMENT`; else -> `FINE_TUNE`.
# **Why deterministic?** Auditable and reproducible: the same report always yields the same fix.
# **Trade-off:** thresholds (`low_step_max=2`, `ki_floor=0.8`) are policy; tune them on your own history.
# We write the result as `Diagnosis -[:RESULTS_IN]-> Intervention -[:MODIFIES]-> Step`.

# %%
intervention = selector.select_intervention(report)
print(selector.explain(report))
a.run("""
MATCH (d:Diagnosis {id:'dx-1'})-[:TARGETS]->(s:Step {label:$node})
CREATE (d)-[:RESULTS_IN]->(:Intervention {type:$t, action:$act, risk_rank:$r})-[:MODIFIES]->(s)
""", node=intervention.target[0], t=intervention.type, act=intervention.action, r=selector.risk_rank(intervention.type))
tight = selector.select_intervention(report, ki_floor=0.95)
table = [{"type": t, **{k: v for k, v in selector.intervention_intensity(t).items() if k in ("tier", "risk", "reversibility")}}
         for t in ("PROMPT_REFINEMENT", "FINE_TUNE", "CODE_MODIFICATION")]
a.show(table)
print("With ki_floor=0.95 the same report becomes:", tight.type)
a.check("intervention is PROMPT_REFINEMENT on the faulty node", intervention.type == "PROMPT_REFINEMENT" and intervention.target == ["CausalAttributionNode"])
a.check("stricter ki_floor changes the decision", tight.type == "FINE_TUNE")
a.check("risk ordering prompt < fine-tune < code", selector.risk_rank("PROMPT_REFINEMENT") < selector.risk_rank("FINE_TUNE") < selector.risk_rank("CODE_MODIFICATION"))

# %% [markdown]
# ## 9. Semantic backpropagation (skill 5)
#
# In neural nets, gradients flow backwards numerically. Here, **textual feedback** flows backwards through the
# execution DAG. The key insight: the node where the error *surfaced* may not be the node *responsible*.
# We (a) read edges and outputs from Neo4j, (b) attribute blame, (c) include **neighbor context** — what the sibling
# nodes fed into the same successor — so feedback is grounded rather than "do better".
#
# Demo 1 (classic): a `Validator` complains, but the `CurrencyConverter` upstream used the wrong 2022 rate.
# Demo 2 (our incident): the neighbors of the faulty node were sound, so blame **stays** on `CausalAttributionNode`.

# %%
a.run("""
UNWIND [['Extractor','CurrencyConverter'],['CurrencyConverter','Validator'],['DateChecker','Validator']] AS e
MERGE (x:Component {id:e[0], demo:'currency'}) MERGE (y:Component {id:e[1], demo:'currency'})
MERGE (x)-[:FEEDS]->(y)
""")
outs_cur = {"Extractor": "Revenue: $10M", "CurrencyConverter": "EUR 9.5M", "DateChecker": "Date: 2022", "Validator": "conversion error flagged"}
for k, v in outs_cur.items():
    a.run("MATCH (c:Component {id:$k, demo:'currency'}) SET c.output=$v", k=k, v=v)
edges_cur = [(r["a"], r["b"]) for r in a.run("MATCH (x:Component {demo:'currency'})-[:FEEDS]->(y) RETURN x.id AS a, y.id AS b ORDER BY a, b")]
outs_read = {r["k"]: r["v"] for r in a.run("MATCH (c:Component {demo:'currency'}) RETURN c.id AS k, c.output AS v")}
fb1 = backprop.backprop(edges_cur, outs_read, failure_node="Validator", predicted="EUR 9.5M",
                        actual="EUR 9.0M; 2022 rate was 0.9 not 0.95")
print("demo 1 target:", fb1.target_node, "| neighbor keys:", list(fb1.neighbor_context))

edges7 = [(r["a"], r["b"]) for r in a.run("MATCH (p:Step {execution:'pred_7f3a9c'})-[:NEXT]->(q) RETURN p.label AS a, q.label AS b, p.seq AS sq ORDER BY sq")]
outs7 = {r["l"]: r["o"] for r in a.run("MATCH (s:Step {execution:'pred_7f3a9c'}) RETURN s.label AS l, s.output AS o")}
fb7 = backprop.backprop(edges7, outs7, failure_node="CausalAttributionNode",
                        predicted="API contract violation (confidence 0.78)",
                        actual="Connection pool exhaustion after timeout 30s->10s")
print("demo 2 target:", fb7.target_node)
a.run("""
MATCH (s:Step {execution:'pred_7f3a9c', label:$n})
CREATE (:Feedback {id:'fb-1', text:$t})-[:ATTRIBUTED_TO]->(s)
""", n=fb7.target_node, t=fb7.feedback)
a.check("blame moves upstream to CurrencyConverter", fb1.target_node == "CurrencyConverter" and "DateChecker_output" in fb1.neighbor_context)
a.check("blame stays on CausalAttributionNode when inputs were sound", fb7.target_node == "CausalAttributionNode")

# %% [markdown]
# ## 10. Graduated validation (skill 6) + entropy-collapse guard
#
# Self-modification needs gates proportional to risk:
# * **Tier 1 canary** (prompt/hyperparameter): small traffic slice; needs positive lift, `p < 0.05`, no KPI regression.
# * **Tier 2 staging** (fine-tune/tool integration): weighted utility `w_a*accuracy + w_c*(1-cost) + w_s*safety`
#   must beat a baseline, and any safety regression fails.
# * **Tier 3 airlock** (safety / new tools / novel): a human must decide; no decision = *pending*, not pass.
#
# A **provenance signature** (SHA-256 of the candidate) is the audit trail for rollbacks. The **entropy-collapse
# guard** (Kepler dual-store GC) prunes stale/contradicted learnings so memory doesn't fill with noise.

# %%
cand1 = {"intervention_type": "prompt_refinement", "target_node": "CausalAttributionNode",
         "metrics": {"target_lift": 0.12, "target_pvalue": 0.01, "kpi_deltas": {"latency": 0.0, "accuracy_other": 0.01}}}
cand1_bad = {**cand1, "metrics": {"target_lift": 0.12, "target_pvalue": 0.2, "kpi_deltas": {"latency": 0.0}}}
cand2 = {"intervention_type": "fine_tune",
         "scores": {"accuracy": 0.9, "cost": 0.3, "safety_score": 0.95},
         "weights": {"w_accuracy": 0.5, "w_cost": 0.2, "w_safety": 0.3},
         "baseline": {"accuracy": 0.8, "cost": 0.3, "safety_score": 0.95}}
cand3 = {"intervention_type": "tool_integration", "new_tool": True, "risk_reward": {"risk": "high", "reward": "medium"}}
results = {"canary": gvp.graduated_validation(cand1), "canary p=0.2": gvp.graduated_validation(cand1_bad),
           "staging": gvp.graduated_validation(cand2), "airlock (no human)": gvp.graduated_validation(cand3),
           "airlock (approved)": gvp.graduated_validation({**cand3, "human_decision": "approve"})}
a.show([{"case": k, "tier": v.tier, "passed": v.passed, "needs_human": v.requires_human} for k, v in results.items()])
sig = gvp.provenance_signature(cand1)
a.run("""
MATCH (s:Step {execution:'pred_7f3a9c', label:'CausalAttributionNode'})
CREATE (v:PromptVersion {id:'prompt-v2', tier:$tier, passed:$ok, signature:$sig, status:'candidate'})-[:REFINES]->(s)
""", tier=results["canary"].tier, ok=results["canary"].passed, sig=sig)

rows = a.run("MATCH (l:Learning) RETURN l.id AS id, l.confidence AS c, l.last_accessed_days AS d, l.resolved_promoted AS p, l.contradicted AS x ORDER BY id")
learnings = [gvp.Learning(r["id"], confidence=r["c"], last_accessed_days=r["d"], resolved_promoted=r["p"],
                          contradicted_by_higher_conf=r["x"]) for r in rows]
kept, removed = gvp.garbage_collect(learnings)
a.run("MATCH (l:Learning) WHERE l.id IN $ids DETACH DELETE l", ids=[l.id for l in removed])
rate = gvp.promotion_rate(total_learnings=len(learnings), promoted=1)
print("kept:", [l.id for l in kept], "| removed:", [l.id for l in removed])
print(f"promotion rate {rate:.0%} -> {gvp.promotion_health(rate)}")
a.check("canary passes, weak p-value fails", results["canary"].passed and not results["canary p=0.2"].passed)
a.check("staging passes with positive net utility", results["staging"].tier == "TIER2_STAGING" and results["staging"].passed)
a.check("airlock waits for a human, then passes on approve",
        results["airlock (no human)"].requires_human and not results["airlock (no human)"].passed and results["airlock (approved)"].passed)
a.check("GC keeps only the fresh learning", [l.id for l in kept] == ["fresh-pool-exhaustion-pattern"])
a.check("provenance signature is a SHA-256 hex", len(sig) == 64)

# %% [markdown]
# ## 11. XSkill: experiences, skill extraction, routing, decay (skill 7)
#
# What was learned must not evaporate. XSkill turns *action-level experiences* ("in context X, doing Y led to Z")
# into reusable **skills**, tracks each skill's track record, and routes new tasks by **demonstrated success**
# rather than description similarity. Old experiences get **temporal decay** (half weight after 90 days) and skills
# whose recent success rate falls under 60% get flagged **stale**; `amendify` rewrites a failing skill (new version).
#
# We keep skill track records as `(:Skill)-[:RAN]->(:Outcome)` in Neo4j and rebuild the skill graph from a query.

# %%
a.run("""
MATCH (:Execution {id:'pred_7f3a9c'})-[:HAS_STEP]->(s:Step)
WHERE s.label IN ['CausalAttributionNode','PredictionSynthesis']
SET s.task_type='deployment_prediction',
    s.action = CASE s.label WHEN 'CausalAttributionNode' THEN 'select primary failure mode' ELSE 'emit structured prediction' END,
    s.context = CASE s.label WHEN 'CausalAttributionNode' THEN 'changelog has API + config changes' ELSE 'single-hypothesis input' END,
    s.outcome = CASE s.label WHEN 'CausalAttributionNode' THEN 'diverged' ELSE 'success' END,
    s.caused_task_failure = (s.label = 'CausalAttributionNode')
""")
nodes = [{"id": r["id"], "task_type": r["tt"], "action": r["a"], "context": r["c"], "outcome": r["o"],
          "caused_task_failure": r["f"], "neighbors": r["nb"]} for r in a.run("""
    MATCH (s:Step {execution:'pred_7f3a9c'}) WHERE s.action IS NOT NULL
    OPTIONAL MATCH (s)-[:NEXT]-(n:Step)
    RETURN s.label AS id, s.task_type AS tt, s.action AS a, s.context AS c, s.outcome AS o,
           s.caused_task_failure AS f, collect(DISTINCT n.label) AS nb ORDER BY id""")]
experiences = xskill.extract_experiences(nodes)
print("experiences:", [(e.outcome, e.action_taken) for e in experiences])
a.run("""
MATCH (s:Step {execution:'pred_7f3a9c', label:'CausalAttributionNode'})
CREATE (:Experience {outcome:$o, action:$act, lesson:$l})-[:DISTILLED_FROM]->(s)
""", o=experiences[0].outcome, act=experiences[0].action_taken, l=experiences[0].lesson)

# skill extraction needs >= min_support successful executions with the same task type
successes = [{"id": f"ok-{i}", "task_type": "canary_rollout", "steps": ["canary 1 percent", "monitor kpis", "promote"],
              "preconditions": ["pool checked"]} for i in range(3)]
skills_out = xskill.extract_skills(successes, min_support=3)
print("extracted skills:", [(s.task_type, s.steps, s.success_rate) for s in skills_out])

# temporal decay: one experience from today, one from 200 days ago
old = xskill.AgentExperience(**{**experiences[0].__dict__, "source_node_id": "old-step", "timestamp": "2024-01-01T00:00:00"})
fresh = xskill.AgentExperience(**{**experiences[0].__dict__, "timestamp": date.today().isoformat() + "T00:00:00"})
decay = xskill.retire_experiences([fresh, old], now_days=date.today().toordinal())
a.show(decay)

# %% [markdown]
# **Routing by track record.** `Skill`/`Outcome` nodes store history. The aggregation below (`count`, `sum(CASE ...)`)
# is the success rate; the skill graph reads the same rows and `route()` picks the winner. A vague-but-similar skill
# (`deployment`) must lose to the specialised one that actually works (`canary-deployment`).

# %%
a.run("""
CREATE (:Skill {id:'deployment', description:'general deployment skill for services'})
CREATE (:Skill {id:'canary-deployment', description:'canary progressive rollout with pool checks'})
""")
for sid, ok, bad in (("deployment", 2, 3), ("canary-deployment", 7, 1)):
    a.run("""
    MATCH (s:Skill {id:$sid})
    UNWIND $results AS r
    CREATE (s)-[:RAN {pattern:'canary'}]->(:Outcome {result:r})
    """, sid=sid, results=["success"] * ok + ["failure"] * bad)
a.show(a.run("""
MATCH (s:Skill)-[:RAN {pattern:'canary'}]->(o:Outcome)
RETURN s.id AS skill, count(o) AS runs, sum(CASE o.result WHEN 'success' THEN 1 ELSE 0 END) AS wins,
       round(100.0 * sum(CASE o.result WHEN 'success' THEN 1 ELSE 0 END) / count(o)) / 100 AS rate ORDER BY skill"""))
sg = xskill.SkillGraph()
sg.add("deployment", {"name": "deployment", "description": "general deployment skill for services", "steps": ["check changelog", "deploy"]})
sg.add("canary-deployment", {"name": "canary deployment", "description": "canary progressive rollout with pool checks",
                             "steps": ["canary 1 percent", "monitor kpis", "promote"]})
for r in a.run("MATCH (s:Skill)-[:RAN {pattern:'canary'}]->(o:Outcome) RETURN s.id AS sid, o.result AS res ORDER BY sid"):
    sg.learn(r["sid"], {"pattern": "canary", "description": "canary deployment"}, r["res"])
chosen = sg.route({"description": "canary deployment for checkout-api", "pattern": "canary"})
sg.cognify("canary-deployment")
stale = xskill.flag_stale_skills(list(sg.skills.values()), recent=10, floor=0.6)
print("route() ->", chosen.skill_id, "| stale skills:", stale)
a.run("MATCH (s:Skill {id:$id}) SET s.routed_for_canary=true", id=chosen.skill_id)

a.check("exactly one diverged experience (the faulty step)", len(experiences) == 1 and experiences[0].outcome == "failure")
a.check("a skill is extracted from 3 successes", len(skills_out) == 1)
a.check("old experience is half-weighted", [d["weight"] for d in decay] == [1.0, 0.5])
a.check("route() picks by demonstrated success", chosen.skill_id == "canary-deployment")
a.check("the 40% success skill is flagged stale", stale == ["deployment"])

# %% [markdown]
# ## 12. The agent loop: plan -> act -> observe on *checkout-api is slow*
#
# An **agent** here is just a loop: **plan** (pick the next tool), **act** (call it), **observe** (read the result,
# decide what next). `MiniAgent` registers tools, logs every call, and `AgentTrace` persists them as
# `(:Run)-[:STEP]->(:ToolCall)-[:NEXT]->(:ToolCall)`. Tools here are a mix of **Cypher** and **skills** — that
# is the point of this chapter's design: skills are the agent's *verbs*, the graph is its *memory and world model*.
#
# The policy is deliberately deterministic (no LLM): after each observation a rule picks the next tool. Swap the rule
# for an LLM later; the tools and trace stay identical.

# %%
agent = a.MiniAgent("checkout-api is slow: find the cause and the safest fix")

@agent.tool("cypher_pool_evidence")
def t_pool_evidence():
    return a.run("MATCH (:Service {name:'checkout-api'})-[:EMITTED]->(e:LogEvent) "
                 "RETURN count(e) AS events, sum(CASE WHEN e.pool_exhausted THEN 1 ELSE 0 END) AS pool_hits")[0]

@agent.tool("skill_diagnose")
def t_diagnose():
    r = cascade.run_cascade(execution)
    return {"stopped_at_layer": r["stopped_at_layer"], "failure_type": r["layer_2_cognitive"]["failure_type"], "report": r}

@agent.tool("skill_select_intervention")
def t_select(report):
    return selector.select_intervention(report).to_dict()

@agent.tool("skill_validate")
def t_validate(candidate):
    g = gvp.graduated_validation(candidate)
    return {"tier": g.tier, "passed": g.passed}

obs = agent.act("cypher_pool_evidence")                       # step 1: gather evidence
plan_log = ["gather evidence"]
if obs["pool_hits"] >= 2:                                     # observe -> decide
    plan_log.append("pool exhaustion confirmed -> diagnose reasoning path")
    dx = agent.act("skill_diagnose")
    plan_log.append(f"diagnosis stopped at layer {dx['stopped_at_layer']} ({dx['failure_type']}) -> pick fix")
    fix = agent.act("skill_select_intervention", report=dx["report"])
    plan_log.append(f"fix={fix['type']} -> validate through gate")
    verdict = agent.act("skill_validate", candidate=cand1)
    plan_log.append(f"gate: {verdict['tier']} passed={verdict['passed']}")
for i, p in enumerate(plan_log, 1):
    print(f"  {i}. {p}")
a.check("agent reached a validated fix", fix["type"] == "PROMPT_REFINEMENT" and verdict["passed"])

# %% [markdown]
# **Query the trace back from Neo4j.** The agent's own behaviour is now data. `collect` gathers the tool order;
# the variable-length `[:NEXT*]` path proves the calls form one unbroken chain from first to last.

# %%
tl = agent.trace.timeline()
a.show(tl)
summary_row = a.run("""
MATCH (r:Run {id:$rid})-[:STEP]->(c:ToolCall)
WITH r, c ORDER BY c.seq
RETURN r.goal AS goal, count(c) AS calls, collect(c.tool) AS tools, sum(CASE WHEN c.ok THEN 0 ELSE 1 END) AS failures
""", rid=agent.trace.run_id)[0]
chain_len = a.run("""
MATCH (r:Run {id:$rid})-[:STEP]->(first:ToolCall {seq:1})
MATCH p = (first)-[:NEXT*]->(last:ToolCall)
RETURN max(length(p)) AS hops
""", rid=agent.trace.run_id)[0]["hops"]
print(summary_row["tools"], "| failures:", summary_row["failures"], "| chain hops:", chain_len)
a.check("trace has 4 ordered tool calls", summary_row["calls"] == 4 and summary_row["tools"][0] == "cypher_pool_evidence")
a.check("trace is one NEXT-chain of 3 hops", chain_len == 3)
a.check("no failed tool calls", summary_row["failures"] == 0)

# %% [markdown]
# ## 13. Skill CLI & benchmark
#
# Every skill is **multi-harness**: the same logic is importable (`lib.py`), runnable from a shell or cron
# (`cli.py`), and self-verifying (`cli.py benchmark`). `--help` prints the SKILL.md description, so any agent
# can discover what a skill is for without reading code.

# %%
bench = []
for s in a.skill_catalog("self-evolution"):
    ok, first = a.run_benchmark("self-evolution", s["slug"])
    bench.append({"skill": s["slug"], "benchmark": "PASS" if ok else "FAIL", "first line": first[:60]})
a.show(bench)

cli = a.SKILLS_DIR / "self-evolution" / "intervention-selector" / "cli.py"
helptext = subprocess.run([sys.executable, str(cli), "--help"], capture_output=True, text=True, cwd=str(a.PROJECT_ROOT))
print(helptext.stdout[:260], "...")
scen = subprocess.run([sys.executable, str(a.SKILLS_DIR / "self-evolution" / "semantic-backprop-attributor" / "cli.py"),
                       "scenario", "devops-prediction"], capture_output=True, text=True, cwd=str(a.PROJECT_ROOT))
a.check("all 7 benchmarks pass", len(bench) == 7 and all(b["benchmark"] == "PASS" for b in bench))
a.check("--help prints the description and exits 0", helptext.returncode == 0 and "intervention" in helptext.stdout.lower())
a.check("CLI scenario blames CausalAttributionNode", scen.returncode == 0 and "causalattributionnode" in scen.stdout.lower())

# %% [markdown]
# ## 14. 🎯 Vibe-coding corner #1 — extend the Cypher
#
# **Vibe coding** = describe the goal and constraints to an AI assistant (Copilot / Claude Code) and *verify* the output
# yourself. A good prompt has: **context** (schema), **task**, **constraints**, **acceptance check**.
#
# Copy-paste prompt:
# ```
# Context: Neo4j graph with (:Execution)-[:HAS_STEP]->(:Step {label,type,latency_ms,cost}),
# (a:Step)-[:NEXT]->(b:Step), (:Diagnosis)-[:TARGETS]->(:Step), (:Diagnosis)-[:RESULTS_IN]->(:Intervention).
# Task: write ONE Cypher query returning, per Intervention type, the number of diagnoses, the collected target step
# labels, and the total latency of the steps that come AFTER each target (use a variable-length NEXT path).
# Constraints: parameterised, read-only, no APOC, use OPTIONAL MATCH so interventions with no downstream steps still appear.
# Acceptance: returns exactly one row for PROMPT_REFINEMENT with downstream_ms ~= 305.1.
# ```
# **Verify:** run it via `a.run(...)` and `assert` the acceptance line. Never accept a query you cannot explain clause by clause.
# Reference solution (try the prompt first!):

# %%
ref = a.run("""
MATCH (d:Diagnosis)-[:RESULTS_IN]->(i:Intervention)
MATCH (d)-[:TARGETS]->(t:Step)
OPTIONAL MATCH (t)-[:NEXT*]->(after:Step)
WITH i.type AS type, count(DISTINCT d) AS diagnoses, collect(DISTINCT t.label) AS targets, sum(after.latency_ms) AS downstream_ms
RETURN type, diagnoses, targets, downstream_ms
""")
a.show(ref)
a.check("vibe query #1 acceptance", len(ref) == 1 and ref[0]["type"] == "PROMPT_REFINEMENT" and abs(ref[0]["downstream_ms"] - 305.1) < 1e-6)

# %% [markdown]
# ## 15. 🎯 Vibe-coding corner #2 — a helper written "as if vibe-coded", plus its test
#
# Prompt that produced the helper below:
# ```
# Write a pure-Python function cheapest_safe_fix(candidates, max_risk) using the intervention-selector skill's
# risk_rank(). Return the lowest-risk candidate whose risk_rank <= max_risk; ties keep input order; return None if none
# qualify. No I/O, stdlib only, type hints, a one-line docstring. Then write 4 asserts incl. the None case.
# ```
# **What to check in AI output:** (1) does it use the *skill's* ranking instead of re-inventing one? (2) edge cases
# (empty list, nothing qualifies)? (3) is it pure (no hidden Neo4j calls) so it is testable?

# %%
def cheapest_safe_fix(candidates, max_risk):
    """Return the lowest-risk intervention type with risk_rank <= max_risk, else None (stable on ties)."""
    allowed = [c for c in candidates if selector.risk_rank(c) <= max_risk]
    return min(allowed, key=selector.risk_rank) if allowed else None

assert cheapest_safe_fix(["FINE_TUNE", "PROMPT_REFINEMENT", "CODE_MODIFICATION"], 3) == "PROMPT_REFINEMENT"
assert cheapest_safe_fix(["CODE_MODIFICATION", "FINE_TUNE"], 2) == "FINE_TUNE"
assert cheapest_safe_fix(["CODE_MODIFICATION"], 1) is None
assert cheapest_safe_fix([], 3) is None
a.check("vibe-coded helper passes its 4 asserts", True)

# %% [markdown]
# ## 16. 🎯 Vibe-coding corner #3 — a whole new skill
#
# ```
# Create a new skill skills/self-evolution/rollback-planner/ following the anatomy of skills/self-evolution/
# graduated-validation-protocol/: SKILL.md with the same 7 sections + Source Attribution (Ch7), a stdlib-only lib.py
# with plan_rollback(provenance_ledger, bad_signature) returning the ordered list of changes to revert, and a cli.py with
# argparse where --help prints the SKILL.md description and a `benchmark` subcommand runs self-checks and exits non-zero on failure.
# Constraints: no pip installs, mark production swaps with TODO(production). Acceptance: python cli.py benchmark exits 0.
# ```
# **Verify:** `a.run_benchmark("self-evolution", "rollback-planner")` returns `(True, ...)` and `cli.py --help` exits 0.
# (Don't ship it unreviewed — read lib.py like a pull request.)

# %% [markdown]
# ## 17. 🧪 Experiments — predict, then change
#
# Write your prediction **before** running each. Edit the value in the code cell and re-run just that cell.
#
# 1. **`ki_floor`.** In section 8 we used `0.95`. Predict the intervention at `ki_floor=0.85`, then `0.92`. (Hint: KI is 0.91.)
# 2. **Canary p-value.** In `cand1`, `target_pvalue` is 0.01. Predict `passed` at 0.049, 0.05 and 0.051. Is the boundary inclusive?
# 3. **GC ttl.** `garbage_collect(learnings, ttl_days=30)` removed the 45-day-old learning. Predict what stays at `ttl_days=60` and `ttl_days=1`.

# %%
for fl in (0.85, 0.92):
    print(f"ki_floor={fl}:", selector.select_intervention(report, ki_floor=fl).type)
for pv in (0.049, 0.05, 0.051):
    c = {**cand1, "metrics": {**cand1["metrics"], "target_pvalue": pv}}
    print(f"p={pv}: passed={gvp.graduated_validation(c).passed}")
probe = [gvp.Learning("fresh", 0.8, 1), gvp.Learning("mid", 0.6, 45), gvp.Learning("old", 0.6, 90)]
for ttl in (60, 1):
    k, _ = gvp.garbage_collect(probe, ttl_days=ttl)
    print(f"ttl={ttl}: kept={[l.id for l in k]}")
a.check("ki_floor=0.85 keeps PROMPT_REFINEMENT; 0.92 flips to FINE_TUNE",
        selector.select_intervention(report, ki_floor=0.85).type == "PROMPT_REFINEMENT"
        and selector.select_intervention(report, ki_floor=0.92).type == "FINE_TUNE")
a.check("p = 0.05 fails (strictly less than)", not gvp.graduated_validation({**cand1, "metrics": {**cand1["metrics"], "target_pvalue": 0.05}}).passed)

# %% [markdown]
# ## 18. Common pitfalls
#
# * **Hard-coding skill inputs.** Then Neo4j is decoration. Always derive inputs from Cypher and write outputs back.
# * **Blaming the node where the error surfaced.** Use semantic backprop with neighbor context; the culprit may be upstream.
# * **Jumping to fine-tuning.** It is slower, riskier and less reversible than a prompt refinement. Follow the intensity ladder.
# * **No gate for "small" changes.** Even prompts go through a canary; a p-value of 0.06 is *not* a pass.
# * **Memory without forgetting.** Without GC/decay, stale learnings drown the fresh ones (entropy collapse).
# * **Routing by description.** Similar-sounding skills with poor records win otherwise; route by demonstrated success.
# * **Cypher mistakes:** `MATCH` instead of `OPTIONAL MATCH` silently drops rows; unbounded `[:NEXT*]` on large graphs explodes
#   — always cap hops (`*1..5`); never string-format values into queries; Neo4j properties must be primitives (JSON-encode dicts).
# * **Mapping to the book:** the 4-layer cascade, intervention routing, semantic backprop, graduated validation and XSkill are
#   Chapter 7's building blocks; here they are composed into one closed loop.

# %% [markdown]
# ## 19. Verification battery
# One last read-only pass over the graph the whole notebook built.

# %%
closing = a.run("""
MATCH (e:Execution {id:'pred_7f3a9c'})-[:HAS_STEP]->(s:Step {label:'CausalAttributionNode'})
MATCH (d:Diagnosis)-[:TARGETS]->(s)
MATCH (d)-[:RESULTS_IN]->(i:Intervention)-[:MODIFIES]->(s)
MATCH (v:PromptVersion)-[:REFINES]->(s)
MATCH (f:Feedback)-[:ATTRIBUTED_TO]->(s)
MATCH (x:Experience)-[:DISTILLED_FROM]->(s)
RETURN s.label AS faulty_step, d.failure_type AS diagnosis, i.type AS fix, v.id AS new_prompt, v.passed AS passed
""")
a.show(closing)
a.check("full loop is one connected subgraph", len(closing) == 1 and closing[0]["fix"] == "PROMPT_REFINEMENT")
a.check("pool-exhaustion events recorded", a.run("MATCH (l:LogEvent) WHERE l.pool_exhausted RETURN count(l) AS c")[0]["c"] == 2)
a.check("GC removed 3 learnings", a.run("MATCH (l:Learning) RETURN count(l) AS c")[0]["c"] == 1)
a.check("agent trace persisted", a.run("MATCH (:Run)-[:STEP]->(c:ToolCall) RETURN count(c) AS c")[0]["c"] >= 4)
a.check("all 7 skills exercised", all(m is not None for m in (exec_graph, cascade, taxonomy, selector, backprop, gvp, xskill)))
a.check("report names the right intervention", report["recommended_intervention"] == "PROMPT_REFINEMENT")
print("\nALL CH7 ADVANCED CHECKS PASSED")

# %% [markdown]
# ## Recap
#
# * **Execution graph** turned the agent's reasoning into a queryable DAG in Neo4j.
# * **Cascade** localised the failure to a *reasoning* fault at steps 4-5; what-if probes showed it discriminates.
# * **Taxonomy + selector** picked the lightest safe lever (prompt refinement) deterministically and auditably.
# * **Semantic backprop** assigned blame with neighbor context; **graduated validation** gated the fix by risk tier.
# * **XSkill** kept the lesson, decayed old memories, and routed by track record.
# * A **MiniAgent** orchestrated skills + Cypher, and its own trace became queryable data.
#
# **Next:** Chapter 8 — optimization: making this loop cheap and fast enough for production.
