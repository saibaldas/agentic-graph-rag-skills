# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > **Beginner (tier 1/3) ← you are here**  →  Intermediate (tier 2/3, no notebook for this topic)  →  Advanced (tier 3/3, no notebook for this topic)

# %% [markdown]
# # Spike C (Simplified) — The execution graph: an agent's autobiography in Neo4j
#
# **Who this is for:** beginners. Every term is explained when it first appears.
#
# **What is a "spike"?** A small experiment proving one idea works. The idea here: *if an AI agent records every
# step it takes as a graph, then it (or you) can later ask "what went wrong, and what led up to it?"*.
#
# **What you will learn**
# 1. What an **execution graph** is: each thing an agent does (an LLM call, a retrieval, a tool call, a decision) is
#    a **node**, and a **parent → child** link records what caused what.
# 2. How to record a DevOps investigation, *including a step that fails*.
# 3. How to **find the failed step** and rebuild its **causal chain** back to the root.
# 4. How to save the trace in **Neo4j** as `Step` nodes and ask Cypher questions: slow LLM calls, cost and token totals.
#
# **Jargon**
# * **Node** – a thing in a graph. **Relationship** – a link between two nodes. **Cypher** – Neo4j's query language,
#   e.g. `MATCH (s:Step) WHERE s.latency_ms > 1000 RETURN s`.
# * **Causal chain** – the path from the very first step ("root") to a given step: *what happened before it, and why*.
# * **Token** – a chunk of text an LLM reads or writes; LLM cost is billed per token.
# * **Latency** – how long a step took, in milliseconds (ms).
#
# **Story:** the agent is asked *"Why is checkout latency spiking?"* (our `checkout-api` is slow). It plans, retrieves
# recent deploys, calls a metrics tool (works), calls an instance-lookup tool (fails!), then asks an LLM for a hypothesis.
#
# > **Before running:** Neo4j must be running (`docker compose up -d`).

# %% [markdown]
# ## 1. Load the skill and put the "world" the agent inspects in Neo4j
#
# The original spike used a fake AWS to make a tool succeed and another fail. We use Neo4j as the environment instead:
# a `:Metric` node holds the p99 latency and a few `:Instance` nodes exist. Asking for an instance that
# is *not* in the database will be our realistic failure.

# %%
import json
import graph_helpers as g

eg = g.load_skill("self-evolution", "execution-graph")
ExecutionGraph = eg.ExecutionGraph

g.clear_graph()
g.run("""
    CREATE (:Metric {name:'TargetResponseTime', service:'checkout-api', p99_ms: 8200, datapoints: 12}),
           (:Instance {id:'i-0a1b2c3d', service:'checkout-api', type:'m5.xlarge'}),
           (:Instance {id:'i-0e4f5a6b', service:'payments', type:'m5.large'})
""")
g.show(g.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS nodes"))

# %% [markdown]
# ## 2. Trace the agent's investigation
#
# `ExecutionGraph` has three key calls:
# * `begin_node(type, input_payload, parent_id)` — a step starts (and is linked to its parent);
# * `complete_node(...)` — it finished, with latency, tokens and cost;
# * `fail_node(id, error=...)` — it failed.
#
# Node types: `Decision_Point`, `Retrieval`, `Tool_Invocation`, `LLM_Call`. The structure we build:
#
# ```
# Decision_Point (root: plan)
# ├── Retrieval (recent deploys)
# │   ├── Tool_Invocation (metrics)       ✔ ok
# │   └── Tool_Invocation (instance lookup) ✘ fails
# └── LLM_Call (hypothesis)
# ```
# The numbers (latencies, tokens, costs) are the same illustrative values as the original spike.

# %%
xg = ExecutionGraph(execution_id="investigate-checkout-latency-2026-03-15")

# ROOT: the agent decides on a plan
root = xg.begin_node("Decision_Point", input_payload={"query": "Why is checkout latency spiking?"})
xg.complete_node(root, output_payload={"plan": "check metrics, check deploys, hypothesize"},
                 latency_ms=12.3, token_count=350, cost_usd=0.001)

# Retrieval of recent deployments
retr1 = xg.begin_node("Retrieval", input_payload={"q": "recent deploys checkout-api"}, parent_id=root)
xg.complete_node(retr1, output_payload={"deploys": ["v3.5.0", "v3.4.1"]},
                 latency_ms=14.5, token_count=0, cost_usd=0.0)

# Tool 1: read the metric from Neo4j (succeeds)
tool1 = xg.begin_node("Tool_Invocation", input_payload={"tool": "metrics.get", "name": "TargetResponseTime"},
                      parent_id=retr1)
metric = g.run("MATCH (m:Metric {name:'TargetResponseTime'}) RETURN m.p99_ms AS p99_ms, m.datapoints AS n")[0]
xg.complete_node(tool1, output_payload={"p99_ms": metric["p99_ms"], "datapoints_count": metric["n"]},
                 latency_ms=145.0, token_count=0, cost_usd=0.0)

# Tool 2: look up an instance that does not exist (fails)
tool2 = xg.begin_node("Tool_Invocation",
                      input_payload={"tool": "instances.describe", "instance_id": "i-INVALID"}, parent_id=retr1)
found = g.run("MATCH (i:Instance {id:$id}) RETURN i.id AS id", id="i-INVALID")
if found:
    xg.complete_node(tool2, output_payload={"instance": found[0]["id"]}, latency_ms=50.0)
else:
    xg.fail_node(tool2, error="NotFound: InvalidInstanceID.NotFound (i-INVALID)")

# LLM call forming a hypothesis
llm1 = xg.begin_node("LLM_Call", input_payload={"prompt": "p99=8200ms, instance lookup failed — hypothesize"},
                     parent_id=root)
xg.complete_node(llm1, output_payload={"hypothesis": "payments backpressure post-v3.5.0"},
                 latency_ms=2340.0, token_count=1200, cost_usd=0.018)

print(json.dumps(xg.summary(), indent=2))

# %% [markdown]
# **How to read the summary:** 5 nodes in total, counted per type, with latency/tokens/cost added up and
# `failed_node_count` = 1. Everything the agent did is now one inspectable object.

# %% [markdown]
# ## 3. Save the trace in Neo4j
#
# We write each execution node as a `(:Step)` and each parent link as a `[:CAUSED]` relationship
# (`(parent)-[:CAUSED]->(child)`). Failed steps get an extra `:Failed` label so they stand out in the graph view.

# %%
rows = [{"id": n.id, "type": n.type, "parent": n.parent_id, "latency_ms": n.latency_ms,
         "tokens": n.token_count, "cost": n.cost_usd, "error": n.error,
         "input": json.dumps(n.input_payload), "output": json.dumps(n.output_payload)}
        for n in xg.nodes.values()]

g.run("""
    MERGE (e:Execution {id:$eid})
    WITH e
    UNWIND $rows AS r
    CREATE (s:Step {id:r.id, type:r.type, latency_ms:r.latency_ms, tokens:r.tokens,
                    cost_usd:r.cost, error:r.error, input:r.input, output:r.output})
    MERGE (e)-[:HAS_STEP]->(s)
""", eid=xg.execution_id, rows=rows)
g.run("""
    UNWIND $rows AS r
    WITH r WHERE r.parent IS NOT NULL
    MATCH (p:Step {id:r.parent}), (c:Step {id:r.id})
    CREATE (p)-[:CAUSED]->(c)
""", rows=rows)
g.run("MATCH (s:Step) WHERE s.error IS NOT NULL SET s:Failed")

g.show(g.run("""
    MATCH (p:Step)-[:CAUSED]->(c:Step)
    RETURN p.type AS parent_step, c.type AS child_step, c.error AS error
"""))

# %% [markdown]
# **How to read it:** four `CAUSED` links = the tree drawn above. The `error` column is filled only for the failed child.

# %% [markdown]
# ## 4. Find the failed step and its causal chain
#
# First with the skill (`failed()` and `causal_chain()`), then with Cypher. In Cypher, `*` means "follow the
# relationship any number of hops": `(root)-[:CAUSED*]->(failed)` is the whole path.

# %%
failed_nodes = xg.failed()
print(f"failed nodes: {len(failed_nodes)}")
assert len(failed_nodes) == 1, f"expected 1 failed node, got {len(failed_nodes)}"
fn = failed_nodes[0]
print(f"  - {fn.type} | error: {fn.error}")

chain = xg.causal_chain(fn.id)
print(f"\ncausal chain from root ({len(chain)} nodes):")
for c in chain:
    print(f"  - {c.type} | id={c.id[:8]}")
assert chain[0].parent_id is None, "chain root must have no parent"
assert chain[-1].id == fn.id, "chain must end at failed node"
print("\n[PASS] Failed node located AND causal chain reconstructed from root.")

# %%
cypher_chain = g.run("""
    MATCH path = (root:Step)-[:CAUSED*0..]->(f:Failed)
    WHERE NOT ()-[:CAUSED]->(root)
    RETURN [s IN nodes(path) | s.type] AS chain, f.error AS error
""")
print(cypher_chain)
assert cypher_chain[0]["chain"] == [c.type for c in chain]
print("[PASS] Neo4j returns the same causal chain as the skill.")

# %% [markdown]
# ## 5. Query the trace: find slow LLM calls
#
# The skill's `query()` takes a Python predicate (Cypher-*like*). The same question in real Cypher is a
# one-line `WHERE`.

# %%
slow_llms = xg.query(lambda n: n.type == "LLM_Call" and n.latency_ms is not None and n.latency_ms > 1000)
print(f"slow LLM calls (> 1000ms): {len(slow_llms)}")
for n in slow_llms:
    print(f"  - latency={n.latency_ms:.1f}ms tokens={n.token_count} cost=${n.cost_usd}")
assert len(slow_llms) == 1, f"expected 1 slow LLM call, got {len(slow_llms)}"
print("\n[PASS] Predicate query returned the slow LLM call.")

neo_slow = g.run("MATCH (s:Step {type:'LLM_Call'}) WHERE s.latency_ms > 1000 "
                 "RETURN s.type AS type, s.latency_ms AS latency_ms, s.tokens AS tokens, s.cost_usd AS cost_usd")
g.show(neo_slow)
assert len(neo_slow) == 1

# %% [markdown]
# ## 6. Cost and token totals
#
# Rolling up totals across the whole run is the basis of later "is my agent efficient?" analyses.
# We compute them with the skill and with a Cypher aggregation (`sum`, grouped by step type).

# %%
s = xg.summary()
print(json.dumps(s, indent=2))
assert s["failed_node_count"] == 1, "summary should report 1 failed node"
assert s["total_cost_usd"] > 0, "total cost should be > 0 after an LLM call"
assert s["total_token_count"] > 0, "total tokens should be > 0"

g.show(g.run("""
    MATCH (s:Step)
    RETURN s.type AS step_type, count(*) AS steps, sum(s.tokens) AS tokens,
           round(sum(s.cost_usd) * 1000) / 1000.0 AS cost_usd, sum(s.latency_ms) AS latency_ms
    ORDER BY cost_usd DESC
"""))
tot = g.run("MATCH (s:Step) RETURN sum(s.tokens) AS t, sum(s.cost_usd) AS c")[0]
assert tot["t"] == s["total_token_count"] and abs(tot["c"] - s["total_cost_usd"]) < 1e-9
print("\n[PASS] Cost + token totals roll up identically in the skill and in Neo4j.")

# %% [markdown]
# **How to read it:** the single LLM call accounts for most tokens, nearly all cost and most of the time —
# exactly the kind of insight a self-improving agent needs.

# %% [markdown]
# ## Recap
# * An **execution graph** records each agent step (`Decision_Point`, `Retrieval`, `Tool_Invocation`, `LLM_Call`) with its parent.
# * A **failed step** is easy to locate, and its **causal chain** shows how the agent got there.
# * In Neo4j this is `(:Step)-[:CAUSED]->(:Step)` plus a `:Failed` label; Cypher answers "slow?", "failed?", "how costly?".
# * Cost, token and latency totals roll up the same way in the skill and in the database.
# * Without this trace, every later self-evolution technique is guesswork; with it, the agent's history is a queryable asset.
