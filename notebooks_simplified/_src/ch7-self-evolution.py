# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > **Beginner (tier 1/3) ← you are here**  →  [Intermediate (tier 2/3)](../notebooks_intermediate_simplified/ch7-self-evolution.ipynb)  →  [Advanced (tier 3/3)](../notebooks_advanced_simplified/ch7-self-evolution.ipynb)

# %% [markdown]
# # Chapter 7 (Simplified) — Self-Evolution and Evaluation, with a real Neo4j graph
#
# **Who this is for:** you are new to GraphRAG and agents. Every step is explained in plain language.
#
# **What you will learn**
# 1. How an agent's *reasoning* can be saved as a graph (an **execution graph**) so it can be inspected later.
# 2. How a **four-layer evaluation cascade** does an "autopsy" on a partly wrong answer.
# 3. How to classify a fix, choose the lightest **intervention**, and send **neighbor-aware feedback** backwards along the graph.
# 4. How a fix is **validated** safely (canary), how stale learnings are cleaned up, and how skills are routed by **track record**.
#
# **The story:** a DevOps agent predicts what a dependency update will break. It is *partially* wrong:
# it blames a deprecated API, but the real cause is a connection-pool problem. Instead of a human rewriting
# the prompt by hand, a **self-evolution loop** finds the faulty reasoning step and ships a fix.
# (We keep the repo's checkout-api-is-slow world, and store everything in a **real Neo4j graph**.)
#
# > **Before running:** start Neo4j with `docker compose -f docker-compose/docker-compose.yml up -d`,
# > then run cells top to bottom. No LLM or API keys are needed.

# %% [markdown]
# ## 0. Setup: connect to Neo4j and load the incident graph
#
# A **graph database** stores **nodes** (things: a service, a database) and **relationships**
# (how they connect: `checkout-api -[:DEPENDS_ON]-> orders-service`). **Cypher** is its query language;
# `(a)-[:DEPENDS_ON]->(b)` reads "a depends on b". Helpers from `graph_helpers.py`:
# `g.run(cypher)` runs a query, `g.show(rows)` prints a table, `g.load_skill(chapter, slug)` imports a skill's `lib.py`.

# %%
import json
import subprocess
import sys

import graph_helpers as g

print("Loaded into Neo4j:", g.load_devops_graph())

# %% [markdown]
# ## 1. Load the skills
#
# Chapter 7 has seven skills (folder `self-evolution`). Each is a small Python file we import and call,
# so you see the real repo code running.

# %%
CH = "self-evolution"
exec_graph = g.load_skill(CH, "execution-graph")
cascade = g.load_skill(CH, "four-layer-eval-cascade")
taxonomy = g.load_skill(CH, "evolution-taxonomy-classifier")
selector = g.load_skill(CH, "intervention-selector")
backprop = g.load_skill(CH, "semantic-backprop-attributor")
gvp = g.load_skill(CH, "graduated-validation-protocol")
xskill = g.load_skill(CH, "xskill-self-improving-object")
print("Loaded 7 skills from skills/" + CH)

# %% [markdown]
# ## 2. The trigger: a dependency update and its blast radius
#
# The platform team queues an update of the `stripe-python` library from 3.2.1 to 3.3.0 for `checkout-api`.
# Before deploying, the agent asks the graph for the **blast radius**: who uses the library, what changed in
# it, and which services sit downstream. We add the library and its changelog to the graph, then *query* it.

# %%
g.run("""
MATCH (c:Service {name:'checkout-api'})
CREATE (lib:Library {name:'stripe-python', from_version:'3.2.1', to_version:'3.3.0'})
CREATE (c)-[:USES {methods:['payments.charge','payments.retrieve']}]->(lib)
CREATE (lib)-[:HAS_CHANGE]->(:Change {id:'c1', text:'deprecated synchronous batch_charge API', kind:'api'})
CREATE (lib)-[:HAS_CHANGE]->(:Change {id:'c2', text:'added async payment methods', kind:'api'})
CREATE (lib)-[:HAS_CHANGE]->(:Change {id:'c3', text:'default connection timeout reduced from 30 to 10 seconds', kind:'config'})
""")

row = g.run("""
    MATCH (s:Service {name:'checkout-api'})-[u:USES]->(lib:Library)
    RETURN s.name AS service, lib.name AS library, lib.from_version + ' -> ' + lib.to_version AS version_change,
           u.methods AS methods_used
""")[0]
changelog = [r["text"] for r in g.run("MATCH (:Library {name:'stripe-python'})-[:HAS_CHANGE]->(c) RETURN c.text AS text ORDER BY c.id")]
downstream = [r["name"] for r in g.run("MATCH (d:Service)-[:DEPENDS_ON*1..3]->(:Service {name:'checkout-api'}) RETURN DISTINCT d.name AS name")]
blast_radius = {**row, "downstream": downstream, "changelog": changelog}
for k, v in blast_radius.items():
    print(f"  {k}: {v}")

assert "deprecated synchronous batch_charge API" in blast_radius["changelog"]
assert "default connection timeout reduced from 30 to 10 seconds" in blast_radius["changelog"]
print("\nGATE: both the API-deprecation and the timeout-config signals are present in the input.")

# %% [markdown]
# **How to read this:** the agent holds *two* competing clues — a deprecated API (`batch_charge`) and a
# lowered timeout. checkout-api never calls `batch_charge`, yet the agent will over-weight the API clue.

# %% [markdown]
# ## 3. The actual outcome: telemetry stored in the graph
#
# Twenty minutes after the deploy, requests slow down and fail. We store the log events as `LogEvent`
# nodes attached to the service, then count the ones that show the *real* failure: database pool exhaustion.

# %%
g.run("""
MATCH (c:Service {name:'checkout-api'})
UNWIND [
  {id:'r001', level:'INFO',  ms:41,   pool_exhausted:false, msg:'ok'},
  {id:'d001', level:'INFO',  ms:0,    pool_exhausted:false, msg:'deploy stripe-python 3.2.1->3.3.0 timeout=10s'},
  {id:'r107', level:'WARN',  ms:2810, pool_exhausted:false, msg:'db_pool_wait_ms=2700'},
  {id:'r113', level:'ERROR', ms:5012, pool_exhausted:true,  msg:'status=504'},
  {id:'r119', level:'ERROR', ms:4998, pool_exhausted:true,  msg:'status=504'}
] AS e
CREATE (c)-[:EMITTED]->(:LogEvent {id:e.id, level:e.level, duration_ms:e.ms, pool_exhausted:e.pool_exhausted, msg:e.msg})
""")
g.show(g.run("""
    MATCH (:Service {name:'checkout-api'})-[:EMITTED]->(l:LogEvent)
    WHERE l.pool_exhausted OR l.duration_ms > 1000
    RETURN l.id AS event, l.level AS level, l.duration_ms AS duration_ms, l.pool_exhausted AS pool_exhausted
    ORDER BY l.id
"""))
exhausted = g.run("MATCH (l:LogEvent) WHERE l.pool_exhausted RETURN count(l) AS c")[0]["c"]
assert exhausted >= 2
print(f"\nGATE: {exhausted} pool-exhaustion events confirm the ACTUAL failure is not an API break.")
ACTUAL_FAILURE = "Connection pool exhaustion from timeout reduction (30s->10s)"

# %% [markdown]
# ## 4. The autobiography: build and persist the execution graph
#
# An **execution graph** records each reasoning step the agent took as a node, linked in order:
#
# `KnowledgeGraphQuery -> ChangelogRetrieval -> ChangeAnalyzer -> CausalAttributionNode -> PredictionSynthesis`
#
# The skill builds it in memory; we then **save it to Neo4j** as `Step` nodes with `NEXT` relationships,
# so the agent's own thinking becomes queryable data. `CausalAttributionNode` is where it committed to the wrong hypothesis.

# %%
eg = exec_graph.ExecutionGraph(execution_id="pred_7f3a9c")
label_to_id, steps = {}, []

def add(label, node_type, parent_label, output, latency_ms, tokens=0, cost=0.0):
    nid = eg.begin_node(node_type, input_payload={"node": label}, parent_id=label_to_id.get(parent_label))
    eg.complete_node(nid, output_payload=output, latency_ms=latency_ms, token_count=tokens, cost_usd=cost)
    label_to_id[label] = nid
    steps.append({"label": label, "type": node_type, "parent": parent_label, "latency_ms": latency_ms,
                  "output": json.dumps(output)})

add("KnowledgeGraphQuery", "Retrieval", None,
    {"deps": "checkout-api uses payments.charge/retrieve; no batch_charge usage"}, 98.7)
add("ChangelogRetrieval", "Retrieval", "KnowledgeGraphQuery", {"changelog": changelog}, 120.4)
add("ChangeAnalyzer", "LLM_Call", "ChangelogRetrieval",
    {"classified": "API deprecations present, configuration changes present"}, 610.2, 420, 0.006)
add("CausalAttributionNode", "LLM_Call", "ChangeAnalyzer",
    {"failure_mode": "API contract violation", "confidence": 0.78}, 1400.0, 847, 0.013)
add("PredictionSynthesis", "LLM_Call", "CausalAttributionNode",
    {"prediction": "HIGH impact; review checkout-api for batch_charge usage"}, 305.1, 210, 0.004)

# Persist: one Execution node, one Step per node, NEXT edges following the parent links.
g.run("""
CREATE (:Execution {id:'pred_7f3a9c', query:'Will stripe-python 3.2.1 -> 3.3.0 break checkout-api?', verdict:'partially correct'})
""")
g.run("""
UNWIND $steps AS s
MATCH (e:Execution {id:'pred_7f3a9c'})
CREATE (e)-[:HAS_STEP]->(:Step {execution:'pred_7f3a9c', label:s.label, type:s.type,
                                latency_ms:s.latency_ms, output:s.output})
""", steps=steps)
g.run("""
UNWIND $steps AS s
WITH s WHERE s.parent IS NOT NULL
MATCH (p:Step {execution:'pred_7f3a9c', label:s.parent}), (c:Step {execution:'pred_7f3a9c', label:s.label})
CREATE (p)-[:NEXT]->(c)
""", steps=steps)

summary = eg.summary()
print("Skill summary:", json.dumps(summary, default=str))
chain = eg.causal_chain(label_to_id["CausalAttributionNode"])
print("\nCausal chain (skill, in memory):")
for node in chain:
    print(f"  <- {node.input_payload['node']} ({node.type})")

# %% [markdown]
# **Same question, asked of the real graph.** Cypher walks `NEXT` arrows backwards from the faulty step
# (`*` = any number of hops). The path from the root to the bad step *is* the causal chain.

# %%
rows = g.run("""
    MATCH p = (root:Step {execution:'pred_7f3a9c', label:'KnowledgeGraphQuery'})-[:NEXT*]->(f:Step {label:'CausalAttributionNode'})
    UNWIND nodes(p) AS n
    RETURN n.label AS step, n.type AS type, n.latency_ms AS latency_ms
""")
g.show(rows)

assert summary["node_count"] == 5, summary
chain_labels = [n.input_payload["node"] for n in chain]
assert chain_labels == ["KnowledgeGraphQuery", "ChangelogRetrieval", "ChangeAnalyzer", "CausalAttributionNode"]
assert [r["step"] for r in rows] == chain_labels, "Neo4j path must match the skill's causal chain"
assert summary["failed_node_count"] == 0  # nothing crashed; the failure is cognitive
print("\nGATE: execution graph is complete; skill and Neo4j agree on the causal chain.")

# %% [markdown]
# ## 5. The cognitive autopsy: four-layer evaluation cascade
#
# The cascade checks from the most general failure cause to the most specific and **stops at the first failing layer**:
#
# * **Layer 0** – hallucination gate: is the answer grounded in the retrieved text?
# * **Layer 1** – context evaluator: did the agent *have* the needed information?
# * **Layer 2** – cognitive fault isolator: a *knowledge* failure or a *reasoning* failure?
# * **Layer 3** – TIR-judge: are the numeric claims correct?
#
# **InfoGain** is how much new information each reasoning step added. Steps 4 and 5 add almost nothing
# (`0.03`, `-0.01`): the agent had already made up its mind — **premature closure**. We store the trace on
# the `Execution` node and read it back from Neo4j to feed the cascade.

# %%
infogain = [0.34, 0.29, 0.22, 0.03, -0.01, 0.19]
g.run("MATCH (e:Execution {id:'pred_7f3a9c'}) SET e.infogain_trace=$t, e.knowledge_index=0.91", t=infogain)
ex = g.run("MATCH (e:Execution {id:'pred_7f3a9c'}) RETURN e.query AS q, e.infogain_trace AS t, e.knowledge_index AS ki")[0]
fault = g.run("""
    MATCH (:Execution {id:'pred_7f3a9c'})-[:HAS_STEP]->(s:Step {label:'CausalAttributionNode'})
    RETURN s.label AS node_id, s.type AS node_type
""")[0]

premise = (
    "Knowledge graph and changelog for stripe-python 3.3.0. Predicted analysis of the failure mode. "
    + ". ".join(changelog) + ". "
    "checkout-service uses payments charge and payments retrieve. "
    "API contract violation is one candidate. Connection pool exhaustion is another candidate."
)
execution = {
    "execution_id": "pred_7f3a9c",
    "query": ex["q"],
    "answer": "API contract violation from deprecated batch_charge",
    "context_premise": premise,
    "required_claims": ["deprecated batch_charge", "connection timeout reduced 30 to 10"],
    "infogain_trace": ex["t"],
    "knowledge_index": ex["ki"],
    "fault_node": {"node_id": fault["node_id"], "node_type": fault["node_type"], "description": fault["node_id"]},
    "diagnosis": ("Premature commitment to 'API contract violation' at step 4. The configuration "
                  "change (timeout 30s -> 10s) had equal evidentiary support but was not explored "
                  "after the early commitment to the API path."),
}
report = cascade.run_cascade(execution)
print(json.dumps(report, indent=2, default=str))

# %% [markdown]
# **Reading the report:** layers 0 and 1 pass (grounded, enough context, `knowledge_index` 0.91 = high), so it
# stops at layer 2 with `REASONING`: the agent knew the facts but reasoned badly. The lightest fix is a prompt refinement.

# %%
assert report["layer_1_context"]["sufficient"] is True
assert report["stopped_at_layer"] == 2
cog = report["layer_2_cognitive"]
assert cog["failure_type"] == "REASONING", cog
assert cog["low_infogain_steps"] == [4, 5], cog
assert abs(cog["knowledge_index"] - 0.91) < 1e-9
assert report["recommended_intervention"] == "PROMPT_REFINEMENT"
assert report["target_nodes"] == ["CausalAttributionNode"]

# Save the verdict on the graph: Diagnosis -[:TARGETS]-> faulty Step.
g.run("""
MATCH (s:Step {execution:'pred_7f3a9c', label:$node})
CREATE (d:Diagnosis {failure_type:$ft, stopped_at_layer:$layer, recommended:$rec})-[:TARGETS]->(s)
""", node=report["target_nodes"][0], ft=cog["failure_type"], layer=report["stopped_at_layer"],
      rec=report["recommended_intervention"])
print("GATE: reasoning failure localized to CausalAttributionNode, steps 4-5, prompt refinement recommended.")

# %% [markdown]
# ## 6. Classify the evolution across four axes
#
# Before changing anything we place the fix in a taxonomy of four questions: **what** evolves, **when**,
# **how**, and **where**. Choosing the wrong lever wastes compute or causes regressions.

# %%
classification = taxonomy.classify_from_signals(
    what="model", when="inter_test_time", how="reward_based", where="domain_specialized",
    notes="Prompt refinement for CausalAttributionNode after premature-closure diagnosis.",
)
print(json.dumps(classification.to_dict(), indent=2, default=str))
assert classification.what == "model" and classification.when == "inter_test_time"
assert classification.how == "reward_based" and classification.where == "domain_specialized"
assert set(classification.graph_rationale) == {"what", "when", "how", "where"}
print("\nGATE: evolution classified across all four axes with graph rationale attached.")

# %% [markdown]
# ## 7. Choose the intervention (deterministic routing)
#
# The diagnosis already names the failure type and target, so picking a fix is a fixed, auditable rule rather
# than each engineer's opinion. Interventions are ranked by risk: prompt refinement < fine-tune < code change.

# %%
intervention = selector.select_intervention(report)
print(json.dumps(intervention.to_dict(), indent=2, default=str))
print("\nAudit line:\n ", selector.explain(report))
for t in ("PROMPT_REFINEMENT", "FINE_TUNE", "CODE_MODIFICATION"):
    prof = selector.intervention_intensity(t)
    print(f"  {t:20s} risk={prof['risk']:8s} reversibility={prof['reversibility']}")

assert intervention.type == "PROMPT_REFINEMENT"
assert intervention.target == ["CausalAttributionNode"]
assert selector.risk_rank("PROMPT_REFINEMENT") < selector.risk_rank("FINE_TUNE") < selector.risk_rank("CODE_MODIFICATION")
g.run("""
MATCH (d:Diagnosis)-[:TARGETS]->(s:Step {label:$node})
CREATE (d)-[:RESULTS_IN]->(:Intervention {type:$t, risk_rank:$r})-[:MODIFIES]->(s)
""", node=intervention.target[0], t=intervention.type, r=selector.risk_rank(intervention.type))
print("\nGATE: intervention resolves to a prompt refinement scoped to the faulty node.")

# %% [markdown]
# ## 8. Semantic backpropagation: neighbor-aware feedback
#
# Neural nets pass error *numbers* backwards; here we pass *text feedback* backwards along the graph. The
# trap: the step that **shows** the error is not always the one that **caused** it. Feedback must include the
# outputs of neighboring steps so credit lands on the right node.
#
# **Teaching example:** `Extractor -> CurrencyConverter -> Validator`, plus `DateChecker -> Validator`.
# We store it in Neo4j (execution `currency_demo`), then read edges and outputs *from the graph*.

# %%
g.run("""
UNWIND [['Extractor','Revenue: $10M'],['CurrencyConverter','EUR 9.5M'],
        ['DateChecker','Date: 2022'],['Validator','conversion error flagged']] AS n
CREATE (:Step {execution:'currency_demo', label:n[0], output:n[1]})
""")
g.run("""
UNWIND [['Extractor','CurrencyConverter'],['CurrencyConverter','Validator'],['DateChecker','Validator']] AS e
MATCH (a:Step {execution:'currency_demo', label:e[0]}), (b:Step {execution:'currency_demo', label:e[1]})
CREATE (a)-[:NEXT]->(b)
""")
edges = [(r["a"], r["b"]) for r in g.run(
    "MATCH (a:Step {execution:'currency_demo'})-[:NEXT]->(b) RETURN a.label AS a, b.label AS b")]
node_outputs = {r["l"]: r["o"] for r in g.run(
    "MATCH (s:Step {execution:'currency_demo'}) RETURN s.label AS l, s.output AS o")}
print("Edges read from Neo4j:", edges)

fb = backprop.backprop(edges, node_outputs, failure_node="Validator",
                       predicted="EUR 9.5M", actual="EUR 9.0M; 2022 rate was 0.9 not 0.95")
print(json.dumps(fb.to_dict(), indent=2, default=str))

assert fb.target_node == "CurrencyConverter", "error must attribute to the converter, not the extractor"
assert "DateChecker_output" in fb.neighbor_context
assert "Extractor" not in fb.neighbor_context and fb.target_node != "Extractor"
print("\nGATE: neighbor-awareness attributes to CurrencyConverter and leaves the Extractor unchanged.")

# %% [markdown]
# **Now the DevOps case through the skill's command-line interface** (`cli.py scenario devops-prediction`),
# which shows the skill works from scripts/CI too. The feedback to `CausalAttributionNode` carries the neighbor
# evidence: the timeout change (30s -> 10s) *was* in its input.

# %%
cli = g.SKILLS_DIR / CH / "semantic-backprop-attributor" / "cli.py"
proc = subprocess.run([sys.executable, str(cli), "scenario", "devops-prediction"],
                      cwd=str(g.PROJECT_ROOT), capture_output=True, text=True)
print(proc.stdout)
assert proc.returncode == 0, proc.stderr
out = proc.stdout.lower()
assert "causalattributionnode" in out
assert "30" in out and "10" in out
print("GATE: CLI-driven semantic backprop emits neighbor-aware feedback for CausalAttributionNode.")

# %% [markdown]
# **Same skill on our own persisted trace.** We read the `pred_7f3a9c` edges and outputs from Neo4j and
# attribute the final prediction's failure. Which step gets the blame depends on the skill's rules; we just print it.

# %%
edges7 = [(r["a"], r["b"]) for r in g.run(
    "MATCH (a:Step {execution:'pred_7f3a9c'})-[:NEXT]->(b) RETURN a.label AS a, b.label AS b")]
outs7 = {r["l"]: r["o"] for r in g.run(
    "MATCH (s:Step {execution:'pred_7f3a9c'}) RETURN s.label AS l, s.output AS o")}
fb7 = backprop.backprop(edges7, outs7, failure_node="PredictionSynthesis",
                        predicted="API contract violation", actual=ACTUAL_FAILURE)
print(json.dumps(fb7.to_dict(), indent=2, default=str)[:1500])
g.run("MATCH (s:Step {execution:'pred_7f3a9c', label:$n}) SET s.feedback=$f",
      n=fb7.target_node, f=json.dumps(fb7.to_dict(), default=str))
assert fb7.target_node in {r[0] for r in edges7} | {r[1] for r in edges7}

# %% [markdown]
# ## 9. Graduated validation and the entropy-collapse guard
#
# A fix must not reach production unchecked. Low-risk changes like a prompt refinement enter at **Tier 1 (canary)**:
# serve ~1% of traffic, require a statistically significant lift (`p < 0.05`) and no regression in core
# KPIs, otherwise roll back. A **provenance signature** (a hash of the change) is recorded for audit.
#
# We save the candidate and its verdict in Neo4j as a `PromptVersion` linked to the faulty step.

# %%
candidate = {
    "intervention_type": "prompt_refinement",
    "target_node": "CausalAttributionNode",
    "metrics": {
        "target_lift": 0.12, "target_pvalue": 0.01,
        "kpi_deltas": {"latency": 0.0, "accuracy_other_predictions": 0.01},
    },
}
result = gvp.graduated_validation(candidate)
print(json.dumps(result.to_dict(), indent=2, default=str))
signature = gvp.provenance_signature(candidate)
print(f"\nProvenance signature: {signature[:16]}...")

assert result.tier == "TIER1_CANARY"
assert result.passed is True
g.run("""
MATCH (s:Step {execution:'pred_7f3a9c', label:'CausalAttributionNode'})
CREATE (v:PromptVersion {id:'prompt-v2', status:'canary', tier:$tier, passed:$passed,
                         lift:0.12, pvalue:0.01, signature:$sig})-[:REFINES]->(s)
""", tier=result.tier, passed=result.passed, sig=signature)
print("\nGATE: prompt refinement passes the Tier 1 canary with no KPI regression.")

# %% [markdown]
# **Entropy-collapse guard.** An agent that keeps writing "learnings" ends up with a pile of stale or
# contradicting notes. A daily **garbage collection** (GC) removes learnings that were already promoted to a
# permanent skill, were contradicted by a more confident one, or are older than 30 days unused.
# We store the learnings as nodes, read them back, run GC, and delete the removed ones from Neo4j.

# %%
g.run("""
UNWIND [
 {id:'premature-closure-fix', c:0.9, d:1, promoted:true,  contra:false},
 {id:'stale-api-hypothesis', c:0.4, d:45, promoted:false, contra:false},
 {id:'contradicted-guess', c:0.5, d:3, promoted:false, contra:true},
 {id:'fresh-pool-exhaustion-pattern', c:0.88, d:5, promoted:false, contra:false}
] AS l
CREATE (:Learning {id:l.id, confidence:l.c, last_accessed_days:l.d, resolved_promoted:l.promoted, contradicted:l.contra})
""")
rows = g.run("MATCH (l:Learning) RETURN l.id AS id, l.confidence AS c, l.last_accessed_days AS d, "
             "l.resolved_promoted AS p, l.contradicted AS x ORDER BY id")
Learning = gvp.Learning
learnings = [Learning(r["id"], confidence=r["c"], last_accessed_days=r["d"], resolved_promoted=r["p"],
                      contradicted_by_higher_conf=r["x"]) for r in rows]
kept, removed = gvp.garbage_collect(learnings)
print("kept:", [l.id for l in kept], "\nremoved:", [l.id for l in removed])
g.run("MATCH (l:Learning) WHERE l.id IN $ids DETACH DELETE l", ids=[l.id for l in removed])
rate = gvp.promotion_rate(total_learnings=len(learnings), promoted=1)
print(f"Promotion rate: {rate:.0%} -> {gvp.promotion_health(rate)}")

assert [l.id for l in kept] == ["fresh-pool-exhaustion-pattern"]
assert len(removed) == 3
left = [r["id"] for r in g.run("MATCH (l:Learning) RETURN l.id AS id")]
assert left == ["fresh-pool-exhaustion-pattern"], left
print("GATE: GC keeps only the fresh, uncontradicted learning (and Neo4j agrees).")

# %% [markdown]
# ## 10. Skills as self-improving graph objects (XSkill + Cognee)
#
# What we learned should not evaporate. **XSkill** distills an action-level *experience* ("in context X, doing Y
# led to outcome Z") from execution steps. **Cognee** stores skills as graph objects and routes tasks by
# **demonstrated success**, not by how similar a description sounds. A specialised `canary-deployment`
# skill with a strong record should beat a vague, similar-sounding `deployment` skill.

# %%
g.run("""
MATCH (:Execution {id:'pred_7f3a9c'})-[:HAS_STEP]->(s:Step)
WHERE s.label IN ['CausalAttributionNode','PredictionSynthesis']
SET s.task_type='deployment_prediction',
    s.action = CASE s.label WHEN 'CausalAttributionNode' THEN 'select primary failure mode' ELSE 'emit structured prediction' END,
    s.context = CASE s.label WHEN 'CausalAttributionNode' THEN 'changelog has API + config changes' ELSE 'single-hypothesis input' END,
    s.outcome = CASE s.label WHEN 'CausalAttributionNode' THEN 'diverged' ELSE 'success' END,
    s.caused_task_failure = (s.label = 'CausalAttributionNode')
""")
nodes = [{"id": r["id"], "task_type": r["tt"], "action": r["a"], "context": r["c"], "outcome": r["o"],
          "caused_task_failure": r["f"], "neighbors": r["nb"]} for r in g.run("""
    MATCH (s:Step {execution:'pred_7f3a9c'}) WHERE s.action IS NOT NULL
    OPTIONAL MATCH (s)-[:NEXT]-(n:Step)
    RETURN s.label AS id, s.task_type AS tt, s.action AS a, s.context AS c, s.outcome AS o,
           s.caused_task_failure AS f, [x IN collect(DISTINCT n.label) WHERE x = 'ChangeAnalyzer'] AS nb
    ORDER BY id""")]
experiences = xskill.extract_experiences(nodes)
for e in experiences:
    print(f"  [{e.outcome}] {e.action_taken}: {e.lesson}")
assert len(experiences) == 1 and experiences[0].outcome == "failure"
g.run("""
MATCH (s:Step {execution:'pred_7f3a9c', label:'CausalAttributionNode'})
CREATE (:Experience {outcome:$o, action:$a, lesson:$l})-[:DISTILLED_FROM]->(s)
""", o=experiences[0].outcome, a=experiences[0].action_taken, l=experiences[0].lesson)

# %% [markdown]
# **Cognee routing.** Each skill's track record on the `canary` pattern is stored as `Skill` nodes with
# `Outcome` nodes; we rebuild the skill graph from those, then ask `route()` to pick a skill.

# %%
g.run("""
CREATE (a:Skill {id:'deployment', description:'general deployment skill for services'})
CREATE (b:Skill {id:'canary-deployment', description:'canary progressive rollout with pool checks'})
WITH a, b
UNWIND range(1,5) AS i
  FOREACH (_ IN CASE WHEN i<=3 THEN [1] ELSE [] END | CREATE (a)-[:RAN {pattern:'canary'}]->(:Outcome {result:'success'}))
  FOREACH (_ IN CASE WHEN i>3  THEN [1] ELSE [] END | CREATE (a)-[:RAN {pattern:'canary'}]->(:Outcome {result:'failure'}))
WITH DISTINCT b
UNWIND range(1,8) AS j
  FOREACH (_ IN CASE WHEN j<=7 THEN [1] ELSE [] END | CREATE (b)-[:RAN {pattern:'canary'}]->(:Outcome {result:'success'}))
  FOREACH (_ IN CASE WHEN j>7  THEN [1] ELSE [] END | CREATE (b)-[:RAN {pattern:'canary'}]->(:Outcome {result:'failure'}))
""")
g.show(g.run("""
    MATCH (s:Skill)-[:RAN {pattern:'canary'}]->(o:Outcome)
    RETURN s.id AS skill, count(o) AS runs, sum(CASE o.result WHEN 'success' THEN 1 ELSE 0 END) AS successes,
           round(100.0*sum(CASE o.result WHEN 'success' THEN 1 ELSE 0 END)/count(o))/100 AS success_rate
    ORDER BY skill
"""))

# %%
graph = xskill.SkillGraph()
graph.add("deployment", {"name": "deployment", "description": "general deployment skill for services",
                         "steps": ["check changelog", "deploy"]})
graph.add("canary-deployment", {"name": "canary deployment",
                                "description": "canary progressive rollout with pool checks",
                                "steps": ["canary 1 percent", "monitor kpis", "promote"]})
for r in g.run("MATCH (s:Skill)-[:RAN {pattern:'canary'}]->(o:Outcome) RETURN s.id AS sid, o.result AS res ORDER BY sid"):
    graph.learn(r["sid"], {"pattern": "canary", "description": "canary deployment"}, r["res"])

task = {"description": "canary deployment for checkout-api", "pattern": "canary"}
chosen = graph.route(task)
print(f"  deployment        success@canary = {graph.skills['deployment'].success_rate_for('canary'):.3f}")
print(f"  canary-deployment success@canary = {graph.skills['canary-deployment'].success_rate_for('canary'):.3f}")
print(f"  -> route() picked: {chosen.skill_id}")
assert chosen.skill_id == "canary-deployment", "must route by demonstrated success, not description"
g.run("MATCH (s:Skill {id:$id}) SET s.routed_for_canary=true", id=chosen.skill_id)
print("GATE: routing selects the skill with the better track record, not the closer description.")

# %% [markdown]
# ## 11. Closing verdict: the whole loop is one connected graph
#
# `execution -> diagnosis -> classification -> intervention -> feedback -> validation -> deployment`.
# Because every stage was saved in Neo4j, we can read the full story in a single Cypher query:

# %%
g.show(g.run("""
    MATCH (e:Execution {id:'pred_7f3a9c'})-[:HAS_STEP]->(s:Step {label:'CausalAttributionNode'})
    MATCH (d:Diagnosis)-[:TARGETS]->(s)
    MATCH (d)-[:RESULTS_IN]->(i:Intervention)
    MATCH (v:PromptVersion)-[:REFINES]->(s)
    RETURN s.label AS faulty_step, d.failure_type AS diagnosis, i.type AS intervention,
           v.id AS new_prompt, v.tier AS validation_tier, v.passed AS passed
"""))

shipped_improvement = {
    "target_node": report["target_nodes"][0],
    "intervention": intervention.type,
    "prompt_change": ("When a changelog contains both API changes and configuration changes, "
                      "evaluate each as a separate hypothesis and assign independent confidence "
                      "scores before selecting a primary failure mode."),
    "actual_failure": ACTUAL_FAILURE,
}
g.run("MATCH (v:PromptVersion {id:'prompt-v2'}) SET v.prompt_change=$p, v.status='shipped'",
      p=shipped_improvement["prompt_change"])
print(json.dumps(shipped_improvement, indent=2))

assert report["recommended_intervention"] == "PROMPT_REFINEMENT"
assert intervention.type == "PROMPT_REFINEMENT"
assert classification.what == "model"
assert result.tier == "TIER1_CANARY" and result.passed
assert chosen.skill_id == "canary-deployment"
assert exhausted >= 2
assert g.run("MATCH (v:PromptVersion {status:'shipped'}) RETURN count(v) AS c")[0]["c"] == 1
print("\nALL CH7 SELF-EVOLUTION GATES PASSED - the loop closed end to end.")

# %% [markdown]
# ## Recap
#
# * An **execution graph** turns an agent's reasoning into nodes and edges you can query (we stored it in Neo4j).
# * The **four-layer cascade** told us *what kind* of failure it was (reasoning, not missing knowledge) and *where* (steps 4–5).
# * The **taxonomy** and **intervention selector** chose the lightest, safest fix: a prompt refinement.
# * **Semantic backpropagation** sent neighbor-aware feedback to the right node, not just the one that showed the error.
# * **Graduated validation** gated the fix through a canary; **garbage collection** kept learnings from piling up.
# * **XSkill/Cognee** keep the lesson and route future tasks by track record.
#
# **Next:** Chapter 8 — optimization (making all of this fast and cheap enough for production).
