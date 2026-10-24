# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > **Beginner (tier 1/3) ← you are here**  →  [Intermediate (tier 2/3)](../notebooks_intermediate_simplified/ch2-architecture-foundations.ipynb)  →  [Advanced (tier 3/3)](../notebooks_advanced_simplified/ch2-architecture-foundations.ipynb)

# %% [markdown]
# # Chapter 2 (Simplified) — Architecture Foundations, with a real Neo4j graph
#
# **Who this is for:** you are new to GraphRAG and agents. Every step is explained in plain language.
#
# **What you will learn**
# 1. The big idea of the book: **two graphs + one harness**
#    (a *vertical* graph of facts, a *horizontal* graph of work steps, and a harness that splits work into safe steps).
# 2. How "strings become things": answering *"what does checkout-api depend on?"* by walking relationships.
# 3. How an agent **routes** a request: facts only (vertical), process only (horizontal), or **both**.
# 4. How to split a workflow into **nodes by tool surface** (not by prompt), and give each node a least-privilege scope.
# 5. How a workflow is a **DAG** (a directed graph with no loops) and how to check that with Cypher.
# 6. The **seam** where the two graphs meet: facts from the graph decide which live metrics to fetch.
# 7. The **eight pillars** roadmap: what to build next.
#
# **The story (same as Chapter 1):** the checkout API is slow. This time we store *everything* —
# the facts, the workflow steps, and the metrics — in a **real Neo4j database** and query it with Cypher.
#
# **Skills used** (small Python files in `skills/architecture/`, loaded with `g.load_skill`):
# `dual-graph-router`, `harness-node-splitter`, `eight-pillar-readiness-map`.
#
# > **Before running:** start Neo4j with `docker compose -f docker-compose/docker-compose.yml up -d`,
# > then run cells top to bottom.

# %% [markdown]
# ## 0. The repo at a glance (the map)
#
# This repo is the *practice* companion to the book *Agentic GraphRAG*: **51 skills across 8 chapters**.
# Chapter 2 gives the shape that all the others plug into:
#
# * **Vertical knowledge graph** = *what is true* (services, dependencies, owners, incidents). Chapter 3 fills it, Chapter 4 makes it durable over time.
# * **Horizontal workflow graph** = *what the agent does* (a chain/DAG of small steps). Chapter 5 plans over it, Chapter 6 runs it with tools.
# * **The harness** = the boundary that splits work into nodes, each with a limited set of tools and memory. Chapter 7 lets it improve itself, Chapter 8 makes it cheap.
#
# | # | Chapter | Folder | Skills | Adds to the foundation |
# |---|---|---|---|---|
# | 1 | Defining Agentic AI | `skills/crisis/` | 5 | the failure modes that motivate everything |
# | **2** | **Architecture Foundations** | `skills/architecture/` | **3** | **dual-graph router, harness splitter, readiness map (this notebook)** |
# | 3 | Knowledge Representation | `skills/knowledge-representation/` | 8 | fills the vertical graph |
# | 4 | Memory | `skills/memory/` | 8 | durable, time-aware graph |
# | 5 | Reasoning & Planning | `skills/reasoning-planning/` | 6 | plans over the horizontal graph |
# | 6 | Tool Orchestration | `skills/tool-orchestration/` | 9 | runs the workflow as tools |
# | 7 | Self-Evolution & Evaluation | `skills/self-evolution/` | 7 | closes the improvement loop |
# | 8 | Optimization | `skills/optimization/` | 5 | makes it affordable |
#
# Each skill folder has `SKILL.md` (when/how to use it), `lib.py` (pure-Python code) and `cli.py` (command line).
# Below we build the Chapter 2 row.

# %% [markdown]
# ## 1. Setup: connect to Neo4j and load the incident graph
#
# **Quick vocabulary.** A graph database stores **nodes** (things, with a *label* like `Service`)
# and **relationships** (typed arrows like `DEPENDS_ON`). Both can carry **properties** (key/value data).
# **Cypher** is the query language: `(a)-[:DEPENDS_ON]->(b)` reads "a depends on b".
#
# `graph_helpers` gives us `g.run(cypher)` (returns rows), `g.show(rows)` (prints a table) and `g.load_skill(...)`.

# %%
import graph_helpers as g

counts = g.load_devops_graph()
print("Loaded into Neo4j:", counts)

# %% [markdown]
# ## 2. Strings to things — the vertical knowledge graph
#
# A vector store turns "checkout-api" into a list of numbers (an **embedding**). It can say two paragraphs are
# *similar*, but it cannot tell you that checkout-api **DEPENDS_ON** orders-db. A knowledge graph stores that as an
# explicit, queryable fact: a *thing* with *connections*.
#
# Our loaded graph already has services and databases. Let us enrich it a little, like the book's Example 2-2:
# a library node and a `since` date on each dependency (temporal metadata on the relationship itself).

# %%
g.run("""
    MATCH (c:Service {name:'checkout-api'})-[r:DEPENDS_ON]->(d)
    SET r.since = CASE d.name
        WHEN 'payment-service' THEN '2024-01-15'
        WHEN 'orders-service'  THEN '2024-03-01'
        ELSE '2024-05-10' END
""")
g.run("""
    MATCH (c:Service {name:'checkout-api'})
    CREATE (c)-[:USES_LIBRARY {version:'5.4.0'}]->(:Library {name:'stripe-python', version:'5.4.0'})
""")

# %% [markdown]
# **Question a vector store cannot answer:** *what is checkout-api connected to?*
# We answer it by **traversing relationships** (following arrows), not by ranking text.
# Read each row as: `checkout-api -[relationship]-> target`.

# %%
deps = g.run("""
    MATCH (:Service {name:'checkout-api'})-[r]->(t)
    RETURN type(r) AS relationship, labels(t)[0] AS target_type, t.name AS target,
           r.since AS since, r.version AS version
    ORDER BY relationship, target
""")
g.show(deps)

# These are explicit facts, not statistical guesses.
assert any(r["relationship"] == "DEPENDS_ON" and r["target"] == "orders-service" for r in deps)
assert any(r["target"] == "stripe-python" for r in deps)
print("\nPASS: the dependency chain is a traversal, not a text-similarity guess.")

# %% [markdown]
# ## 3. Route the request — `dual-graph-router`
#
# Before doing any work, the harness must decide **what kind** of request it received:
#
# * a pure **lookup** ("which services depend on orders-db?") is one graph query → **vertical**;
# * a pure **process** ("classify, plan, decide on rollback") is a workflow → **horizontal**;
# * a **diagnosis** ("why is checkout slow — what does it depend on and what changed?") needs *both*:
#   a workflow whose retrieval step queries the vertical graph. This is *where the two graphs meet*.
#
# The router scores keywords for each side and returns a decision with the scores so you can see *why*.

# %%
dg = g.load_skill("architecture", "dual-graph-router")

requests = [
    "Which services depend on the orders-db and were deployed in the last 24 hours?",
    "Classify the alert severity, plan the remediation, and decide whether to roll back.",
    "Diagnose why the checkout API is slow: which services does it depend on and what changed recently?",
]

decisions = dg.route_batch(requests)
for d in decisions:
    print(f"[{d.target.upper():<10}] {d.request}")
    print(f"             vertical={d.vertical_score} horizontal={d.horizontal_score} "
          f"(V:{d.matched_vertical}  H:{d.matched_horizontal})")

lookup, process, diagnosis = decisions
assert lookup.target == "vertical"
assert process.target == "horizontal"
assert diagnosis.target == "both"

meet = dg.explain_meeting_point(diagnosis)
print("\nWhere the two graphs meet (for the diagnosis request):")
for role, text in meet.items():
    print(f"  {role}: {text}")
assert meet["forward_flow"] and meet["backward_flow"]
print("\nPASS: lookup->vertical, process->horizontal, diagnosis->both with a meeting point.")

# %% [markdown]
# **How to read it:** the lookup only matched *vertical* words, the process only *horizontal* words, and the diagnosis
# matched both, so it is routed to **both**. `forward_flow` = the workflow asks the knowledge graph for facts;
# `backward_flow` = what the workflow learns is written back to the graph.
#
# **Try the "vertical" route for real.** The first request is a pure lookup, so one Cypher query answers it:
# which services depend (directly or through other services) on `orders-db`, and what was deployed to them?

# %%
lookup_answer = g.run("""
    MATCH (s:Service)-[:DEPENDS_ON*1..3]->(:Database {name:'orders-db'})
    OPTIONAL MATCH (d:Deployment)-[:DEPLOYED_TO|CHANGED]->(x)
    WHERE x = s OR (s)-[:DEPENDS_ON]->(x)
    RETURN DISTINCT s.name AS service, s.tier AS tier, collect(DISTINCT d.id) AS related_deployments
    ORDER BY service
""")
g.show(lookup_answer)
assert {r["service"] for r in lookup_answer} >= {"checkout-api", "orders-service", "payment-service"}

# %% [markdown]
# ## 4. Split the workflow into harness nodes — `harness-node-splitter`
#
# The diagnosis routed to a **horizontal workflow**. Now we split it into **nodes** (small steps).
# The rule of the chapter: **nodes differ by tool surface, not by prompt.**
# * Two operations that call the *same tools* are one node with prompt variations (merge them).
# * Two operations that call *different tools* are different roles and stay separate
#   (the book's "scanner vs validator" lesson: a code scanner only needs the filesystem; a validator needs a browser/network).
#
# This is also the **horizontal graph**, so let us store it in Neo4j: each candidate operation is an `Operation` node
# connected to the `Tool`s it uses. Then we read the operations *back from the graph* and feed them to the skill.

# %%
import json

sample = g.SKILLS_DIR / "architecture" / "harness-node-splitter" / "sample-workflow.json"
rows = json.loads(sample.read_text(encoding="utf-8"))["operations"]

g.run("""
    UNWIND $rows AS r
    CREATE (o:Operation) SET o = r
    WITH o
    UNWIND o.tools AS tool
    MERGE (t:Tool {name: tool})
    MERGE (o)-[:USES_TOOL]->(t)
""", rows=rows)

g.show(g.run("""
    MATCH (o:Operation)
    OPTIONAL MATCH (o)-[:USES_TOOL]->(t:Tool)
    RETURN o.id AS operation, o.node_type AS type, collect(t.name) AS tools
    ORDER BY operation
"""), max_rows=20)

# %% [markdown]
# **Reading it:** `get_metrics` and `get_cpu` both use the same tool (so they are really one job with two prompts),
# while `scan_code` and `validate_finding` use totally different tools. Now we ask Neo4j for the operations and let the skill decide.

# %%
splitter = g.load_skill("architecture", "harness-node-splitter")

op_rows = g.run("MATCH (o:Operation) RETURN properties(o) AS p")
ops = splitter.operations_from_dicts([r["p"] for r in op_rows])
result = splitter.split_nodes(ops)

print(f"{len(ops)} candidate operations -> {len(result.nodes)} nodes "
      f"(tool-overlap threshold {result.threshold})\n")
for n in result.nodes:
    tag = "  [MERGED]" if n.is_merged else ""
    print(f"  {n.id:<18}{n.node_type:<11}tools={n.tools}{tag}")
    if n.is_merged:
        print(f"      prompt variants: {n.tasks}")

merged = [n for n in result.nodes if n.is_merged]
assert len(merged) == 1 and set(merged[0].merged_from) == {"get_metrics", "get_cpu"}

assert splitter.tool_overlap({"filesystem"},
                             {"browser_driver", "ios_simulator", "network_stack", "scripting_runtime"}) == 0.0
ids = {n.id for n in result.nodes}
assert {"scan_code", "validate_finding"} <= ids
print("\nPASS: same tool surface merged; distinct tool surfaces stayed split.")

# %% [markdown]
# **How to read it:** the *overlap* is the share of tools two operations have in common. Above the threshold (80%)
# they are merged into one node (`[MERGED]`, listing both prompts). With zero overlap they stay split.
#
# ### The per-node constrained context scope
# Each node gets a **scope**: which tools it may call, which memory slices it reads/writes, and its input/output contract.
# A node sees only its own tools and memory — **least privilege by construction** (a security idea: give each part only what it needs).

# %%
retrieval_node = next(n for n in result.nodes if n.id == "query_kg")
scope = splitter.node_scope(retrieval_node)
print("Scope of the retrieval node that traverses the vertical graph:")
for k, v in scope.items():
    print(f"  {k}: {v}")

assert scope["tool_surface"] == ["graph_read"]
assert "vertical_kg" in scope["memory_reads"]
print("\nPASS: the retrieval node is scoped to graph_read + the vertical_kg memory slice.")

# %% [markdown]
# ## 5. The horizontal workflow graph as a DAG (Example 2-3)
#
# A **DAG** (directed acyclic graph) is a graph of arrows with **no loops**. Edges here mean "must run before".
# `query_kg` feeds *both* `get_metrics` and `analyze`; `analyze` waits for the graph context *and* the live metrics.
# Independent branches can run in parallel; dependent ones wait.
#
# We create the `PRECEDES` relationships between the `Operation` nodes already stored in Neo4j.

# %%
workflow_edges = [
    ("classify", "query_kg"),
    ("query_kg", "get_metrics"),
    ("query_kg", "analyze"),
    ("get_metrics", "analyze"),
    ("analyze", "report"),
]
g.run("""
    UNWIND $edges AS e
    MATCH (a:Operation {id: e[0]}), (b:Operation {id: e[1]})
    CREATE (a)-[:PRECEDES]->(b)
""", edges=[list(e) for e in workflow_edges])

stored = g.run("MATCH (a)-[:PRECEDES]->(b) RETURN a.id AS from_step, b.id AS to_step ORDER BY from_step, to_step")
g.show(stored)
assert len(stored) == len(workflow_edges)

# %% [markdown]
# **Two checks, both in Cypher:**
# 1. *Is there a cycle?* Look for any path that leads back to its own start. A DAG must return **0**.
# 2. *How many inputs does each step wait on?* This is the **in-degree** (number of incoming arrows).

# %%
cycles = g.run("MATCH p=(n:Operation)-[:PRECEDES*1..10]->(n) RETURN count(p) AS cycles")[0]["cycles"]
print("cycles found:", cycles)

indeg_rows = g.run("""
    MATCH (n:Operation) WHERE n.id IN ['classify','query_kg','get_metrics','analyze','report']
    OPTIONAL MATCH (m)-[:PRECEDES]->(n)
    RETURN n.id AS step, count(m) AS waits_on_inputs
    ORDER BY waits_on_inputs, step
""")
g.show(indeg_rows)
indeg = {r["step"]: r["waits_on_inputs"] for r in indeg_rows}

assert cycles == 0, "the workflow graph must be acyclic"
assert indeg["analyze"] == 2, "analyze waits on both query_kg and get_metrics"
print("\nPASS: valid DAG; analyze waits on two inputs (graph context + live metrics).")

# %% [markdown]
# ## 6. The seam — facts from the graph decide which metrics to fetch
#
# This is *where the two graphs meet*: the workflow step `get_metrics` needs to know **which services to check**.
# It does not guess — the vertical graph already told us (checkout-api and everything it depends on).
#
# In production this step would call a monitoring API. Here, to stay self-contained, we store a few latency readings
# as `MetricPoint` nodes in Neo4j (as if a monitoring tool had written them), then fetch them with Cypher.
# The spike starts right after deploy-481 changed the database connection pool at 08:30.

# %%
g.run("""
    MATCH (c:Service {name:'checkout-api'})
    UNWIND [['08:20',42.0],['08:25',45.0],['08:31',3100.0],['08:36',3300.0],['08:41',2400.0]] AS p
    CREATE (c)-[:HAS_METRIC]->(:MetricPoint {metric:'p99_latency_ms', at:p[0], value:p[1]})
""")

affected_rows = g.run("""
    MATCH (c:Service {name:'checkout-api'})
    OPTIONAL MATCH (c)-[:DEPENDS_ON]->(d)
    WITH c, collect(d.name) AS deps
    RETURN [c.name] + deps AS affected
""")
affected = affected_rows[0]["affected"]
print("Vertical graph says the affected surface is:", affected)

# %% [markdown]
# Now the execution node queries metrics *only for those names* (services with no readings simply return nothing,
# which is itself useful information for the `analyze` step).

# %%
metric_rows = g.run("""
    UNWIND $names AS name
    MATCH (s {name: name})
    OPTIONAL MATCH (s)-[:HAS_METRIC]->(m:MetricPoint)
    RETURN name AS service, count(m) AS datapoints, max(m.value) AS peak_latency_ms
    ORDER BY datapoints DESC, service
""", names=affected)
g.show(metric_rows)

metrics = next(r for r in metric_rows if r["service"] == "checkout-api")
assert metrics["datapoints"] > 0, "no datapoints returned"
assert metrics["peak_latency_ms"] and metrics["peak_latency_ms"] > 1000, "the spike should be visible"

# %% [markdown]
# **Reading it:** checkout-api has 5 readings and a peak of 3300 ms (normal is ~40 ms). The graph also lets us
# line the spike up with the cause — metric readings *after* the deployment time versus *before*:

# %%
g.show(g.run("""
    MATCH (d:Deployment {id:'deploy-481'})
    MATCH (:Service {name:'checkout-api'})-[:HAS_METRIC]->(m:MetricPoint)
    RETURN m.at AS time, m.value AS p99_ms,
           CASE WHEN m.at >= '08:30' THEN 'after ' + d.id ELSE 'before ' + d.id END AS phase
    ORDER BY time
"""))
print("\nPASS: the execution node used graph-derived targets and got shaped metric data.")

# %% [markdown]
# ## 7. Map the agent across the eight pillars — `eight-pillar-readiness-map`
#
# The dual-graph architecture is the framework; the **eight pillars** are the build roadmap, and they are *layered*:
# knowledge representation comes first because everything else depends on the graph.
# The DevOps agent's starting point is a *representation problem, not a technology problem*: the data exists in logs
# and configs (knowledge representation only **partial**), and every other pillar is missing.
# The map reports readiness, which Chapter-1 flaws are still open, and what to build next.

# %%
pillars = g.load_skill("architecture", "eight-pillar-readiness-map")

report = pillars.assess(pillars.initial_state())
print(f"Readiness: {report.readiness_pct}%   next pillar to build: {report.next_pillar}\n")
for p in report.per_pillar:
    mark = {"present": "[x]", "partial": "[~]", "missing": "[ ]"}[p["status"]]
    print(f"  {mark} {p['order']}. {p['name']:<30}{p['chapter']:<8}solves: {', '.join(p['solves_flaws'])}")

print("\nUnresolved Chapter-1 flaws:")
for f in report.unresolved_flaws:
    print(f"  - {f['flaw']} ({f['state']}) — needs {', '.join(f['solving_pillars'])}")

print(f"\nRoadmap: {' -> '.join(report.roadmap)}")

assert report.next_pillar == "knowledge_representation"
assert {f["flaw"] for f in report.unresolved_flaws} == set(pillars.FLAW_TO_PILLARS)
assert not report.dependency_violations
print("\nPASS: initial state maps to a representation problem; roadmap starts at Chapter 3.")

# %% [markdown]
# **How to read it:** `[x]` built, `[~]` partly built, `[ ]` missing. Because the pillars are layered, the roadmap
# starts at knowledge representation (Chapter 3). For perspective, here is how much structure we actually hold in Neo4j now:

# %%
g.show(g.run("""
    MATCH (n) RETURN labels(n)[0] AS node_type, count(*) AS nodes ORDER BY nodes DESC
"""))

# %% [markdown]
# ## 8. Verdict — all gates in one cell

# %%
checks = {
    "vertical traversal": any(r["target"] == "orders-service" for r in deps),
    "router: diagnosis -> both": dg.route(requests[2]).target == "both",
    "splitter: 9 ops -> 8 nodes": len(splitter.split_nodes(ops).nodes) == 8,
    "workflow is a DAG (no cycles in Neo4j)": cycles == 0,
    "metrics fetched via graph-derived targets": metrics["datapoints"] > 0,
    "pillars: next = knowledge_representation": pillars.assess(pillars.initial_state()).next_pillar == "knowledge_representation",
}
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
assert all(checks.values()), "one or more Chapter-2 gates failed"
print("\nCh2 architecture foundations: all gates passed.")

# %% [markdown]
# ## Recap
# * **Two graphs + a harness:** a vertical graph of facts, a horizontal graph of workflow steps, and a harness that splits work into scoped nodes.
# * **Strings to things:** relationships are explicit facts you traverse with Cypher.
# * **`dual-graph-router`:** lookup → vertical, process → horizontal, diagnosis → both.
# * **`harness-node-splitter`:** split by tool surface, merge when tool overlap is high; each node has a least-privilege scope.
# * **DAG:** no cycles; a step waits on all its inputs. Verified with a Cypher cycle check and in-degree count.
# * **The seam:** graph facts pick which metrics to fetch.
# * **`eight-pillar-readiness-map`:** build knowledge representation first.
#
# **Next:** `ch3-knowledge-representation.ipynb` — building the vertical knowledge graph properly.
