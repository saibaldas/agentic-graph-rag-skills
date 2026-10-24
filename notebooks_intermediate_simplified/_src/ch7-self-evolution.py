# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > [Beginner (tier 1/3)](../notebooks_simplified/ch7-self-evolution.ipynb)  →  **Intermediate (tier 2/3) ← you are here**  →  [Advanced (tier 3/3)](../notebooks_advanced_simplified/ch7-self-evolution.ipynb)

# %% [markdown]
# # Chapter 7 (Intermediate) — Self-Evolution: from a missed prediction to a validated fix
#
# **Tier 2 of 3.** In the beginner notebook you called each skill once with ready-made inputs. In the advanced notebook
# you wire everything into an agent loop. Here you learn **why each skill takes the inputs it does, how to read its output,
# and how to feed it from a real Neo4j graph** — and you start writing some of the code yourself.
#
# **Scenario** (fictional AWS account `123456789012`, no LLM calls, no API keys): a prediction agent said
# *"stripe-python 3.3.0 will break checkout-api via an API contract violation"*. The real cause was **connection-pool
# exhaustion** after a timeout change (30s -> 10s). The self-evolution loop turns that miss into a validated fix:
#
# `execution graph -> 4-layer diagnosis -> intervention -> semantic backprop -> graduated validation -> XSkill routing`
#
# **You will:** query the graph with Cypher, derive skill inputs from the results, write skill outputs back as nodes,
# chain skills in a small hand-written pipeline, do two guided exercises, and learn how to *trust* a skill (CLI + benchmark).

# %%
import int_helpers as h
import json, os, subprocess, sys
from datetime import date

h.load_devops_graph()   # wipes + reloads the shared DevOps graph: the expected start of every notebook

exec_graph = h.load_skill("self-evolution", "execution-graph")
cascade    = h.load_skill("self-evolution", "four-layer-eval-cascade")
taxonomy   = h.load_skill("self-evolution", "evolution-taxonomy-classifier")
selector   = h.load_skill("self-evolution", "intervention-selector")
backprop   = h.load_skill("self-evolution", "semantic-backprop-attributor")
gvp        = h.load_skill("self-evolution", "graduated-validation-protocol")
xskill     = h.load_skill("self-evolution", "xskill-self-improving-object")

catalog = h.skill_catalog("self-evolution")
h.show([{"skill": s["slug"], "what it does": s["description"][:60] + "..."} for s in catalog])
h.check("7 chapter skills catalogued", len(catalog) == 7)

# %% [markdown]
# ## 1. Put the evidence in the graph
#
# We add what the diagnosis needs on top of the DevOps graph: the **library** that changed (with changelog entries),
# **log events** from `checkout-api`, and four stored **learnings** (used later for garbage collection).
#
# **Why a graph, not a dict?** "Which change, on which dependency path, produced which symptom?" is a *path* question.
# In SQL it is several joins; in Cypher it is one pattern. The skills below are pure Python — the graph is where their
# inputs come from and where their verdicts live.

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
learnings = [
    {"id": "premature-closure-fix",         "conf": 0.9, "days": 2,  "promoted": True,  "contra": False},
    {"id": "stale-api-hypothesis",          "conf": 0.6, "days": 45, "promoted": False, "contra": False},
    {"id": "contradicted-guess",            "conf": 0.4, "days": 5,  "promoted": False, "contra": True},
    {"id": "fresh-pool-exhaustion-pattern", "conf": 0.8, "days": 1,  "promoted": False, "contra": False},
]
h.run("""
MATCH (c:Service {name:'checkout-api'})
CREATE (c)-[:USES {methods:['payments.charge','payments.retrieve']}]->(l:Library {name:'stripe-python', from_v:'3.2.1', to_v:'3.3.0'})
WITH l
UNWIND $changes AS ch
CREATE (l)-[:HAS_CHANGE]->(:Change {id:ch.id, kind:ch.kind, text:ch.text})
""", changes=changes)
h.run("""
MATCH (c:Service {name:'checkout-api'})
UNWIND $events AS e
CREATE (c)-[:EMITTED]->(:LogEvent {id:e.id, level:e.level, duration_ms:e.ms, pool_exhausted:e.pool, msg:e.msg})
""", events=events)
h.run("""
UNWIND $ls AS l
CREATE (:Learning {id:l.id, confidence:l.conf, last_accessed_days:l.days, resolved_promoted:l.promoted, contradicted:l.contra})
""", ls=learnings)
n = h.run("MATCH (n) RETURN count(n) AS nodes")[0]["nodes"]
h.check("graph has Library, Change, LogEvent, Learning nodes", n >= 20)

# %% [markdown]
# ## 2. Cypher #1 — blast radius (variable-length path)
#
# *You read tier 1, so only the new parts:*
# * `p = (d)-[:DEPENDS_ON*1..3]->(x)` binds a whole **path** `p` and follows 1–3 hops; `length(p)` is its hop count.
# * `DISTINCT` inside `collect(...)` de-duplicates when several paths reach the same node.
# * `min(length(p))` per dependency keeps the *shortest* route when there are several.
#
# **Why it matters:** this tells the agent who suffers (callers upstream) and what it leans on (dependencies downstream).

# %%
upstream = h.run("""
MATCH p = (d:Service)-[:DEPENDS_ON*1..3]->(:Service {name:'checkout-api'})
RETURN collect(DISTINCT d.name) AS callers_hurt, count(DISTINCT d) AS n
""")[0]
downstream = h.run("""
MATCH p = (:Service {name:'checkout-api'})-[:DEPENDS_ON*1..3]->(x)
RETURN x.name AS dependency, min(length(p)) AS hops
ORDER BY hops, dependency
""")
print("Callers hurt if checkout-api degrades:", upstream["callers_hurt"])
h.show(downstream)
h.check("cart-service is in the upstream blast radius", "cart-service" in upstream["callers_hurt"])
h.check("orders-db is reachable within 2 hops", any(r["dependency"] == "orders-db" and r["hops"] == 2 for r in downstream))

# Cypher #2 - evidence per service (aggregation + OPTIONAL MATCH)
# OPTIONAL MATCH keeps services with NO log events (like a LEFT JOIN); silence on payment-service is evidence too.
# WITH s, count(e) closes the aggregation (s is the grouping key); sum(CASE WHEN ... THEN 1 ELSE 0 END) is an inline if/else count.
evidence = h.run("""
MATCH (s:Service)
OPTIONAL MATCH (s)-[:EMITTED]->(e:LogEvent)
WITH s, count(e) AS events,
     sum(CASE WHEN e.pool_exhausted THEN 1 ELSE 0 END) AS pool_hits,
     max(e.duration_ms) AS worst_ms
RETURN s.name AS service, events, pool_hits, worst_ms
ORDER BY pool_hits DESC, service
""")
h.show(evidence)
h.check("checkout-api has 2 pool-exhaustion events", evidence[0]["service"] == "checkout-api" and evidence[0]["pool_hits"] == 2)
h.check("silent services are kept by OPTIONAL MATCH", any(r["events"] == 0 for r in evidence))

# %% [markdown]
# ## 4. Skill 1 — `execution-graph`: the agent's autobiography
#
# **Idea.** Every step the agent took is a node in a DAG with latency, tokens and cost. A **two-phase write** —
# `begin_node` records structure *before* the step runs (a crash still leaves a trace), `complete_node` fills in results.
#
# **Anatomy of the API**
# * `ExecutionGraph(execution_id=...)` — one graph per run.
# * `begin_node(node_type, input_payload, parent_id)` -> node id. `parent_id` wires the edge; `None` = root.
# * `complete_node(id, output_payload, latency_ms, token_count, cost_usd)` — closes the node.
# * `summary()` -> `node_count`, `failed_node_count`, totals. `causal_chain(id)` -> ancestors root-first.
#
# We store the 5 recorded steps in Neo4j, then **read them back with Cypher** to drive the skill. `OPTIONAL MATCH (p)-[:NEXT]->(s)`
# finds each step's parent (the root has none), so the database stays the source of truth.

# %%
steps = [
    ("KnowledgeGraphQuery",   "Retrieval", 98.7,   0,   0.0,   {"deps": "checkout-api uses payments.charge/retrieve; no batch_charge usage"}),
    ("ChangelogRetrieval",    "Retrieval", 120.4,  0,   0.0,   {"changelog": [c["text"] for c in changes]}),
    ("ChangeAnalyzer",        "LLM_Call",  610.2,  420, 0.006, {"classified": "API deprecations present, configuration changes present"}),
    ("CausalAttributionNode", "LLM_Call",  1400.0, 847, 0.013, {"failure_mode": "API contract violation", "confidence": 0.78}),
    ("PredictionSynthesis",   "LLM_Call",  305.1,  210, 0.004, {"prediction": "HIGH impact; review checkout-api for batch_charge usage"}),
]
h.run("CREATE (:Execution {id:'pred_7f3a9c', query:'Will stripe-python 3.2.1 -> 3.3.0 break checkout-api?', verdict:'partially correct'})")
h.run("""
UNWIND $steps AS s
MATCH (e:Execution {id:'pred_7f3a9c'})
CREATE (e)-[:HAS_STEP]->(:Step {execution:'pred_7f3a9c', seq:s.seq, label:s.label, type:s.type,
        latency_ms:s.lat, tokens:s.tok, cost:s.cost, output:s.out})
""", steps=[{"seq": i, "label": l, "type": t, "lat": lat, "tok": tok, "cost": c, "out": json.dumps(o)}
            for i, (l, t, lat, tok, c, o) in enumerate(steps)])
h.run("""
MATCH (a:Step {execution:'pred_7f3a9c'}), (b:Step {execution:'pred_7f3a9c'})
WHERE b.seq = a.seq + 1
CREATE (a)-[:NEXT]->(b)
""")

rows = h.run("""
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
h.check("execution graph has 5 nodes, 0 failed", summary["node_count"] == 5 and summary["failed_node_count"] == 0)
h.check("causal chain ends at the faulty node", chain[-1] == "CausalAttributionNode" and chain[0] == "KnowledgeGraphQuery")

# %% [markdown]
# ## 5. Cypher #3 — the same chain, asked of Neo4j directly
#
# `[:NEXT*]` (no bounds) walks any number of hops; `nodes(p)` returns the nodes on path `p` and `UNWIND` flattens them to
# rows. An aggregation then answers a different question: *where do time and money go?*
#
# **Trade-off:** the skill computes the chain in memory (fast, testable, no DB); Cypher computes it at scale where the
# data already lives. Check that both agree — that is how you know your graph model and the skill share one idea of "parent".

# %%
path_rows = h.run("""
MATCH p = (:Step {execution:'pred_7f3a9c', label:'KnowledgeGraphQuery'})-[:NEXT*]->(:Step {label:'CausalAttributionNode'})
UNWIND nodes(p) AS n
RETURN n.label AS step
""")
cost = h.run("""
MATCH (:Execution {id:'pred_7f3a9c'})-[:HAS_STEP]->(s:Step)
WITH s.type AS type, count(s) AS steps, sum(s.latency_ms) AS total_ms, sum(s.cost) AS usd, collect(s.label) AS labels
RETURN type, steps, round(total_ms) AS total_ms, usd, labels ORDER BY total_ms DESC
""")
h.show(cost)
h.check("Neo4j path equals the skill's causal chain", [r["step"] for r in path_rows] == chain)
h.check("LLM calls dominate cost", cost[0]["type"] == "LLM_Call" and abs(cost[0]["usd"] - 0.023) < 1e-9)

# %% [markdown]
# ## 6. Skill 2 — `four-layer-eval-cascade`: the cognitive autopsy
#
# It asks questions from cheapest/most general to most specific and **stops at the first failing layer**:
# **L0** hallucination gate (grounded in the premise?) -> **L1** context evaluator (did the agent have the facts?) ->
# **L2** cognitive fault isolator (KNOWLEDGE vs REASONING) -> **L3** TIR-judge (are numeric claims right?).
#
# **Input — `run_cascade(execution: dict)`**
# | key | meaning |
# |---|---|
# | `answer`, `query` | what the agent said, and to what |
# | `context_premise` | the retrieved text the answer must be grounded in (we build it from `Change` nodes) |
# | `required_claims` | facts that must appear in the context for L1 to pass |
# | `infogain_trace` | per-step gain in confidence; steps **below 0.05** are "low-information" (reasoning stalled) |
# | `knowledge_index` | 0-1 share of needed facts the agent held; **below 0.8** means a *knowledge* failure |
# | `fault_node` | where to point the fix |
#
# **Output fields:** `stopped_at_layer`, `layer_2_cognitive.{failure_type, low_infogain_steps, knowledge_index}`,
# `recommended_intervention`, `target_nodes`. We write the verdict back as a `Diagnosis` node that `TARGETS` the step.

# %%
changelog = [r["t"] for r in h.run("MATCH (:Library)-[:HAS_CHANGE]->(c:Change) RETURN c.text AS t ORDER BY c.id")]
h.run("MATCH (e:Execution {id:'pred_7f3a9c'}) SET e.infogain_trace=$t, e.knowledge_index=0.91",
      t=[0.34, 0.29, 0.22, 0.03, -0.01, 0.19])
ex = h.run("MATCH (e:Execution {id:'pred_7f3a9c'}) RETURN e.query AS q, e.infogain_trace AS t, e.knowledge_index AS ki")[0]
fault = h.run("""
MATCH (:Execution {id:'pred_7f3a9c'})-[:HAS_STEP]->(s:Step {type:'LLM_Call'})
RETURN s.label AS node_id, s.type AS node_type ORDER BY s.latency_ms DESC LIMIT 1
""")[0]
premise = ("Knowledge graph and changelog for stripe-python 3.3.0. Predicted analysis of the failure mode. "
           + ". ".join(changelog) + ". "
           "Default connection timeout reduced from 30 to 10 seconds. "
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
print("L2:", cog["failure_type"], "| low steps:", cog["low_infogain_steps"], "| KI:", cog["knowledge_index"])

h.run("""
MATCH (s:Step {execution:'pred_7f3a9c', label:$node})
CREATE (:Diagnosis {id:'dx-1', failure_type:$ft, layer:$layer, low_steps:$low, knowledge_index:$ki})-[:TARGETS]->(s)
""", node=report["target_nodes"][0], ft=cog["failure_type"], layer=report["stopped_at_layer"],
      low=cog["low_infogain_steps"], ki=cog["knowledge_index"])
back = h.run("MATCH (d:Diagnosis {id:'dx-1'})-[:TARGETS]->(s:Step) RETURN d.failure_type AS ft, d.low_steps AS low, s.label AS target")[0]
print("read back from Neo4j:", back)
h.check("cascade stops at layer 2 with REASONING", report["stopped_at_layer"] == 2 and cog["failure_type"] == "REASONING")
h.check("low-InfoGain steps are [4, 5]", cog["low_infogain_steps"] == [4, 5])
h.check("Diagnosis node written and read back", back["ft"] == "REASONING" and back["target"] == "CausalAttributionNode")

# %%
# What-if: is the cascade discriminating? A useful diagnostic routes DIFFERENT evidence to DIFFERENT fixes;
# if every failure ended in "fine-tune" the cascade would be an expensive way to say the same thing.
def stop_layer(**over):
    r = cascade.run_cascade({**execution, **over})
    return r["stopped_at_layer"], r["recommended_intervention"]

what_if = [
    {"scenario": "baseline", "result": stop_layer()},
    {"scenario": "timeout fact missing from context", "result": stop_layer(context_premise="Predicted analysis. nothing relevant here")},
    {"scenario": "low knowledge index 0.5", "result": stop_layer(knowledge_index=0.5)},
]
h.show([{"scenario": w["scenario"], "stopped_at_layer": w["result"][0], "intervention": w["result"][1]} for w in what_if])
h.check("different evidence -> different routes", len({w["result"] for w in what_if}) == 3)
h.check("KI 0.5 forces FINE_TUNE", what_if[2]["result"][1] == "FINE_TUNE")

# %% [markdown]
# ## 7. Skills 3 and 4 — place the fix (`taxonomy`) and route it (`selector`)
#
# **Taxonomy.** Before changing anything, answer four questions: **what** evolves (model/context/tool/architecture),
# **when** (`intra_test_time` = inside the request, `inter_test_time` = between requests), **how** (reward/imitation/population),
# **where** (general vs domain-specialised). `route_failure(failure_type, recurring=)` suggests an axis;
# `budget_check(when_fires, measured_op_ms, request_budget_ms)` asks whether an in-request evolution step fits your SLA.
#
# **Selector.** `select_intervention(report, ki_floor=0.8, low_step_max=2)` applies four ordered branches: insufficient context
# -> `RETRIEVAL_FIX`; format violation -> `STRUCTURAL_CONSTRAINT`; localized reasoning failure with intact knowledge ->
# `PROMPT_REFINEMENT`; else `FINE_TUNE`. Output fields: `.type`, `.target` (list of nodes), `.action`.
#
# **Trade-off:** deterministic routing is auditable (same report, same fix) but its thresholds are *policy* — tune them on history.
# We write `Diagnosis -[:RESULTS_IN]-> Intervention -[:MODIFIES]-> Step`.

# %%
classification = taxonomy.classify_from_signals(
    what="model", when="inter_test_time", how="reward_based", where="domain_specialized",
    notes="Prompt refinement for CausalAttributionNode after premature-closure diagnosis.")
routes = [{"failure": f, "recurring": rec, "mechanism": taxonomy.route_failure(f, recurring=rec)["mechanism"]}
          for f, rec in [("REASONING", False), ("REASONING", True), ("KNOWLEDGE", False)]]
h.show(routes)
slow = taxonomy.budget_check(when_fires="intra_test_time", measured_op_ms=2500, request_budget_ms=2000)
print("2.5s step inside a 2s request budget ->", slow["verdict"])

intervention = selector.select_intervention(report)
print(selector.explain(report))
h.run("""
MATCH (d:Diagnosis {id:'dx-1'})-[:TARGETS]->(s:Step {label:$node})
CREATE (d)-[:RESULTS_IN]->(:Intervention {type:$t, action:$act, risk_rank:$r})-[:MODIFIES]->(s)
""", node=intervention.target[0], t=intervention.type, act=intervention.action, r=selector.risk_rank(intervention.type))
fix_row = h.run("""
MATCH (:Diagnosis {id:'dx-1'})-[:RESULTS_IN]->(i:Intervention)-[:MODIFIES]->(s:Step)
RETURN i.type AS type, s.label AS target
""")[0]
print("graph now says:", fix_row)
h.check("taxonomy has all four axes", set(classification.graph_rationale) == {"what", "when", "how", "where"})
h.check("recurring REASONING escalates", routes[1]["mechanism"] != routes[0]["mechanism"])
h.check("slow in-path op is moved off the request path", slow["verdict"] == "MOVE_TO_INTER")
h.check("intervention is PROMPT_REFINEMENT on the faulty node", fix_row == {"type": "PROMPT_REFINEMENT", "target": "CausalAttributionNode"})
h.check("risk ordering prompt < fine-tune < code",
        selector.risk_rank("PROMPT_REFINEMENT") < selector.risk_rank("FINE_TUNE") < selector.risk_rank("CODE_MODIFICATION"))

# %% [markdown]
# ## 8. Skill 5 — `semantic-backprop-attributor`: who is really to blame?
#
# In neural nets, gradients flow backward as numbers. Here, **textual feedback** flows backward through the DAG. The node where
# an error *surfaced* may not be the node *responsible*.
#
# **Input — `backprop(edges, node_outputs, failure_node, predicted, actual)`**: `edges` is a list of `(parent, child)` tuples,
# `node_outputs` maps node -> output text. **Output (`Feedback`):** `.target_node` (blamed node), `.neighbor_context` (the
# sibling outputs used as evidence), `.feedback` (the grounded instruction).
#
# Demo 1 (classic): the `Validator` complains, but the `CurrencyConverter` upstream used the wrong rate — blame moves upstream.
# Demo 2 (our incident): neighbors were sound, so blame **stays** on `CausalAttributionNode`. We read edges and outputs from Neo4j.

# %%
edges_cur = [("Extractor", "CurrencyConverter"), ("CurrencyConverter", "Validator"), ("DateChecker", "Validator")]
outs_cur = {"Extractor": "Revenue: $10M", "CurrencyConverter": "EUR 9.5M", "DateChecker": "Date: 2022", "Validator": "conversion error flagged"}
fb1 = backprop.backprop(edges_cur, outs_cur, failure_node="Validator", predicted="EUR 9.5M",
                        actual="EUR 9.0M; 2022 rate was 0.9 not 0.95")
print("demo 1 -> blame:", fb1.target_node, "| neighbor evidence:", list(fb1.neighbor_context))

edges7 = [(r["a"], r["b"]) for r in h.run("MATCH (p:Step {execution:'pred_7f3a9c'})-[:NEXT]->(q) RETURN p.label AS a, q.label AS b")]
outs7 = {r["l"]: r["o"] for r in h.run("MATCH (s:Step {execution:'pred_7f3a9c'}) RETURN s.label AS l, s.output AS o")}
fb7 = backprop.backprop(edges7, outs7, failure_node="CausalAttributionNode",
                        predicted="API contract violation (confidence 0.78)",
                        actual="Connection pool exhaustion after timeout 30s->10s")
h.run("""
MATCH (s:Step {execution:'pred_7f3a9c', label:$n})
CREATE (:Feedback {id:'fb-1', text:$t})-[:ATTRIBUTED_TO]->(s)
""", n=fb7.target_node, t=fb7.feedback)
print("demo 2 -> blame:", fb7.target_node)
h.check("blame moves upstream to CurrencyConverter", fb1.target_node == "CurrencyConverter" and "DateChecker_output" in fb1.neighbor_context)
h.check("blame stays on CausalAttributionNode when inputs were sound", fb7.target_node == "CausalAttributionNode")

# %% [markdown]
# ## 9. Skill 6 — `graduated-validation-protocol` (+ the entropy-collapse guard)
#
# Self-modification needs gates proportional to risk. `graduated_validation(candidate: dict)` picks the tier from
# `candidate["intervention_type"]`:
# * **Tier 1 canary** (prompt/hyperparameter): needs `metrics.target_lift > 0`, `target_pvalue < 0.05`, no KPI regression.
# * **Tier 2 staging** (fine-tune): weighted utility of `scores` must beat `baseline`.
# * **Tier 3 airlock** (new tools/safety): needs `human_decision`; none = *pending*, not pass.
#
# Output (`.tier`, `.passed`, `.requires_human`, `.reasons`). `provenance_signature(candidate)` is a SHA-256 audit trail.
# **Entropy guard:** `garbage_collect(learnings)` drops promoted, contradicted or stale (>30d) learnings so memory doesn't fill with noise.
# We rebuild `Learning` objects from a Cypher query, and delete the removed ones in the graph.

# %%
cand1 = {"intervention_type": "prompt_refinement", "target_node": "CausalAttributionNode",
         "metrics": {"target_lift": 0.12, "target_pvalue": 0.01, "kpi_deltas": {"latency": 0.0, "accuracy_other": 0.01}}}
cand_weak = {**cand1, "metrics": {"target_lift": 0.12, "target_pvalue": 0.2, "kpi_deltas": {"latency": 0.0}}}
cand_tool = {"intervention_type": "tool_integration", "new_tool": True, "risk_reward": {"risk": "high", "reward": "medium"}}
results = {"canary": gvp.graduated_validation(cand1), "canary p=0.2": gvp.graduated_validation(cand_weak),
           "airlock (no human)": gvp.graduated_validation(cand_tool),
           "airlock (approved)": gvp.graduated_validation({**cand_tool, "human_decision": "approve"})}
h.show([{"case": k, "tier": v.tier, "passed": v.passed, "needs_human": v.requires_human} for k, v in results.items()])
sig = gvp.provenance_signature(cand1)
h.run("""
MATCH (s:Step {execution:'pred_7f3a9c', label:'CausalAttributionNode'})
CREATE (:PromptVersion {id:'prompt-v2', tier:$tier, passed:$ok, signature:$sig, status:'candidate'})-[:REFINES]->(s)
""", tier=results["canary"].tier, ok=results["canary"].passed, sig=sig)

rows = h.run("MATCH (l:Learning) RETURN l.id AS id, l.confidence AS c, l.last_accessed_days AS d, l.resolved_promoted AS p, l.contradicted AS x ORDER BY id")
objs = [gvp.Learning(r["id"], confidence=r["c"], last_accessed_days=r["d"], resolved_promoted=r["p"],
                     contradicted_by_higher_conf=r["x"]) for r in rows]
kept, removed = gvp.garbage_collect(objs)
h.run("MATCH (l:Learning) WHERE l.id IN $ids DETACH DELETE l", ids=[l.id for l in removed])
left = [r["id"] for r in h.run("MATCH (l:Learning) RETURN l.id AS id")]
print("kept:", [l.id for l in kept], "| removed:", [l.id for l in removed], "| in graph now:", left)
h.check("canary passes, weak p-value fails", results["canary"].passed and not results["canary p=0.2"].passed)
h.check("airlock waits for a human, then passes on approve",
        results["airlock (no human)"].requires_human and not results["airlock (no human)"].passed and results["airlock (approved)"].passed)
h.check("GC keeps only the fresh learning, in the graph too", [l.id for l in kept] == ["fresh-pool-exhaustion-pattern"] == left)
h.check("provenance signature is a SHA-256 hex", len(sig) == 64)

# %% [markdown]
# ## 10. Skill 7 — `xskill-self-improving-object`: remember by track record
#
# What was learned must not evaporate. `extract_experiences(nodes)` turns execution nodes into action-level
# *experiences* (only nodes that diverged or failed produce a lesson). `SkillGraph.learn(skill_id, ctx, outcome)` records
# track records and `route(task)` picks a skill by **demonstrated success**, not by how similar its description looks.
#
# We keep outcomes as `(:Skill)-[:RAN]->(:Outcome)` and compute the success rate with an aggregation (`count`, conditional `sum`).
# A vague-but-similar `deployment` skill (40% success) must lose to the specialised `canary-deployment` (88%).

# %%
nodes = [
    {"id": "CausalAttributionNode", "task_type": "deployment_prediction", "action": "select primary failure mode",
     "context": "changelog has API + config changes", "outcome": "diverged", "caused_task_failure": True, "neighbors": ["ChangeAnalyzer"]},
    {"id": "PredictionSynthesis", "task_type": "deployment_prediction", "action": "emit structured prediction",
     "context": "single-hypothesis input", "outcome": "success", "caused_task_failure": False},
]
experiences = xskill.extract_experiences(nodes)
print("experiences:", [(e.outcome, e.action_taken, e.lesson) for e in experiences])

h.run("""
CREATE (:Skill {id:'deployment', description:'general deployment skill for services'})
CREATE (:Skill {id:'canary-deployment', description:'canary progressive rollout with pool checks'})
""")
for sid, ok, bad in (("deployment", 2, 3), ("canary-deployment", 7, 1)):
    h.run("""
    MATCH (s:Skill {id:$sid})
    UNWIND $results AS r
    CREATE (s)-[:RAN {pattern:'canary'}]->(:Outcome {result:r})
    """, sid=sid, results=["success"] * ok + ["failure"] * bad)
rates = h.run("""
MATCH (s:Skill)-[:RAN {pattern:'canary'}]->(o:Outcome)
RETURN s.id AS skill, count(o) AS runs, sum(CASE o.result WHEN 'success' THEN 1 ELSE 0 END) AS wins
ORDER BY skill
""")
h.show(rates)

sg = xskill.SkillGraph()
sg.add("deployment", {"name": "deployment", "description": "general deployment skill for services", "steps": ["check changelog", "deploy"]})
sg.add("canary-deployment", {"name": "canary deployment", "description": "canary progressive rollout with pool checks",
                             "steps": ["canary 1 percent", "monitor kpis", "promote"]})
for r in h.run("MATCH (s:Skill)-[:RAN {pattern:'canary'}]->(o:Outcome) RETURN s.id AS sid, o.result AS res ORDER BY sid"):
    sg.learn(r["sid"], {"pattern": "canary", "description": "canary deployment"}, r["res"])
chosen = sg.route({"description": "canary deployment for checkout-api", "pattern": "canary"})
h.run("MATCH (s:Skill {id:$id}) SET s.routed_for_canary=true", id=chosen.skill_id)
print("route() ->", chosen.skill_id)
h.check("exactly one diverged experience (the faulty step)", len(experiences) == 1 and experiences[0].outcome == "failure")
h.check("route() picks by demonstrated success", chosen.skill_id == "canary-deployment")

# %% [markdown]
# ## 11. A hand-written pipeline: chain three skills on *checkout-api is slow*
#
# So far each skill ran in its own cell. A pipeline is just a plain Python function where **one skill's output is the next one's
# input**. No agent framework needed (tier 3 wraps this in a `MiniAgent` with a persisted trace).
#
# Evidence (Cypher) -> `run_cascade` -> `select_intervention` -> `graduated_validation` -> readable decision report.

# %%
def investigate(service: str, candidate: dict) -> dict:
    ev = h.run("""
        MATCH (:Service {name:$s})-[:EMITTED]->(e:LogEvent)
        RETURN count(e) AS events, sum(CASE WHEN e.pool_exhausted THEN 1 ELSE 0 END) AS pool_hits
        """, s=service)[0]
    if ev["pool_hits"] < 2:
        return {"service": service, "decision": "NO ACTION", "why": "no repeated pool exhaustion", "evidence": ev}
    rep = cascade.run_cascade(execution)
    fix = selector.select_intervention(rep)
    gate = gvp.graduated_validation(candidate)
    return {"service": service, "evidence": ev, "diagnosis": rep["layer_2_cognitive"]["failure_type"],
            "stopped_at_layer": rep["stopped_at_layer"], "fix": fix.type, "target": fix.target[0],
            "gate": gate.tier, "gate_passed": gate.passed,
            "decision": "SHIP to canary" if gate.passed else "HOLD"}

decision = investigate("checkout-api", cand1)
print("DECISION REPORT")
print("=" * 40)
for k, v in decision.items():
    print(f"{k:>16}: {v}")
h.check("pipeline decides to ship a validated prompt refinement",
        decision["decision"] == "SHIP to canary" and decision["fix"] == "PROMPT_REFINEMENT" and decision["evidence"]["pool_hits"] == 2)
h.check("a quiet service gets NO ACTION", investigate("payment-service", cand1)["decision"] == "NO ACTION")

# %% [markdown]
# ## 12. Your turn — two guided exercises
#
# **Exercise 1 (Cypher).** Return the number of pool-exhaustion `LogEvent`s emitted by `checkout-api` as a single integer.
# **Hint:** `MATCH (:Service {name:...})-[:EMITTED]->(e:LogEvent)`, filter on `e.pool_exhausted`, and `count(e)`.
# Try it in a new cell first, then compare with the reference below (it runs, and is checked against the expected value).

# %%
def ex1():
    return h.run("""
        MATCH (:Service {name:'checkout-api'})-[:EMITTED]->(e:LogEvent)
        WHERE e.pool_exhausted = true
        RETURN count(e) AS n
    """)[0]["n"]

h.your_turn("count pool-exhaustion events on checkout-api",
            "WHERE e.pool_exhausted = true, then RETURN count(e)", ex1, 2)

# %% [markdown]
# **Exercise 2 (skill input).** The same diagnostic `report` currently yields `PROMPT_REFINEMENT`. Which intervention does `select_intervention`
# return if you demand a **stricter knowledge floor** of `0.95` (our knowledge index is 0.91)?
# **Hint:** `select_intervention(report, ki_floor=...)`. Predict first: does 0.91 clear a 0.95 floor?

# %%
def ex2():
    return selector.select_intervention(report, ki_floor=0.95).type

h.your_turn("strict ki_floor changes the fix", "pass ki_floor=0.95 as a keyword argument", ex2, "FINE_TUNE")

# %% [markdown]
# ## 14. How to trust a skill: its CLI and its benchmark
#
# Each skill is importable (`lib.py`), runnable from a shell/cron (`cli.py`) and self-verifying (`cli.py benchmark`).
# `--help` prints the SKILL.md description so any agent can discover it. Below: one CLI run and **one** benchmark.
# Tier 3 benchmarks all seven. **Set `PYTHONIOENCODING=utf-8`** — skill output contains non-ASCII characters and Windows consoles may choke.

# %%
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
cli = h.SKILLS_DIR / "self-evolution" / "intervention-selector" / "cli.py"
helptext = subprocess.run([sys.executable, str(cli), "--help"], capture_output=True, text=True, encoding="utf-8",
                          cwd=str(h.PROJECT_ROOT), env=env)
scen = subprocess.run([sys.executable, str(cli), "scenario", "devops-prediction"], capture_output=True, text=True, encoding="utf-8",
                      cwd=str(h.PROJECT_ROOT), env=env)
print(helptext.stdout[:200], "...\n")
print(scen.stdout[:400])
ok, first_line = h.run_benchmark("self-evolution", "intervention-selector")
print("benchmark:", "PASS" if ok else "FAIL", "|", first_line[:80])
h.check("--help exits 0 and describes the skill", helptext.returncode == 0 and "intervention" in helptext.stdout.lower())
h.check("scenario subcommand exits 0 and names the fix", scen.returncode == 0 and "PROMPT_REFINEMENT" in scen.stdout)
h.check("benchmark passes", ok)

# %% [markdown]
# ## 15. 🎯 Vibe-coding tip — ask an AI assistant, then *verify*
#
# A good prompt has **context** (schema), **task**, **constraints** and an **acceptance check**. Copy-paste this:
#
# ```
# Context: Neo4j graph (:Execution)-[:HAS_STEP]->(:Step {label, latency_ms, cost}),
# (:Diagnosis {id})-[:TARGETS]->(:Step), (:Diagnosis)-[:RESULTS_IN]->(:Intervention {type}).
# Task: write ONE Cypher query returning, per Intervention type, how many diagnoses led to it and the
# labels of the steps it MODIFIES (as a list).
# Constraints: parameterised, read-only, no APOC, columns named type, diagnoses, targets.
# ```
#
# **Acceptance check — never trust, run it.** The expected row for our graph is one `PROMPT_REFINEMENT` row,
# 1 diagnosis, targets `['CausalAttributionNode']`. Paste the AI's query into `candidate_query` and run the cell
# (a reference answer is pre-filled so the notebook stays runnable).

# %%
candidate_query = """
MATCH (d:Diagnosis)-[:RESULTS_IN]->(i:Intervention)-[:MODIFIES]->(s:Step)
RETURN i.type AS type, count(DISTINCT d) AS diagnoses, collect(DISTINCT s.label) AS targets
"""
vibe_rows = h.run(candidate_query)
h.show(vibe_rows)
h.check("AI query returns the expected row",
        vibe_rows == [{"type": "PROMPT_REFINEMENT", "diagnoses": 1, "targets": ["CausalAttributionNode"]}])
h.check("query is read-only", not any(w in candidate_query.upper() for w in ("CREATE", "DELETE", "SET ", "MERGE")))

# %%
# Final verification battery: every stage left its artifact in the graph.
counts = {r["label"]: r["n"] for r in h.run("MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n")}
h.check("5 steps and 1 diagnosis recorded", counts.get("Step") == 5 and counts.get("Diagnosis") == 1)
h.check("intervention, feedback and prompt version recorded",
        all(counts.get(k) == 1 for k in ("Intervention", "Feedback", "PromptVersion")))
h.check("only the fresh learning survives GC", counts.get("Learning") == 1)
h.check("skill routed by track record is flagged in the graph",
        h.run("MATCH (s:Skill {routed_for_canary:true}) RETURN s.id AS id")[0]["id"] == "canary-deployment")
h.check("pipeline, cascade and selector agree", decision["fix"] == intervention.type == report["recommended_intervention"])
print("ALL CH7 INTERMEDIATE CHECKS PASSED")

# %% [markdown]
# ## 16. Why this matters, common mistakes, and recap
#
# **Trade-offs**
# * *Prompt refinement first.* It is the lightest, most reversible lever; fine-tuning/code changes cost more and are harder to undo.
# * *Deterministic routing vs LLM judgement.* Rules are auditable but thresholds are policy you must tune.
# * *Graph as truth, skills as analysers.* You pay a query round-trip but gain a queryable history of every diagnosis.
#
# **Common mistakes**
# 1. Hard-coding skill inputs instead of deriving them from the graph — the notebook then proves nothing about your data.
# 2. Blaming the node where the error *surfaced* instead of using neighbor context (semantic backprop exists for this).
# 3. Skipping the gate: shipping a change without `graduated_validation` and a provenance signature.
# 4. Running a Cypher write twice (`CREATE`) and duplicating nodes — use `MERGE` or a clean graph when re-running.
# 5. Forgetting `PYTHONIOENCODING=utf-8` when calling a CLI from a Windows subprocess.
#
# ## Recap
#
# * Skill inputs came from **Cypher** (variable-length path, `OPTIONAL MATCH` aggregation, `[:NEXT*]` path) and verdicts were **written back** (`Diagnosis`, `Intervention`, `Feedback`, `PromptVersion`) and read again.
# * Seven skills, each with clear inputs and output fields: `execution-graph`, `four-layer-eval-cascade`, `evolution-taxonomy-classifier`, `intervention-selector`, `semantic-backprop-attributor`, `graduated-validation-protocol`, `xskill-self-improving-object`.
# * A plain function chained evidence -> diagnosis -> fix -> gate into a decision report.
# * You trust a skill by running its CLI and its `benchmark`, and you trust AI-written Cypher by running an acceptance check.
#
# **Next:** [`../notebooks_advanced_simplified/ch7-self-evolution.ipynb`](../notebooks_advanced_simplified/ch7-self-evolution.ipynb) — the same loop as a `MiniAgent` with a persisted trace, all seven benchmarks, more vibe-coding corners and experiments.
