# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > [Beginner (tier 1/3)](../notebooks_simplified/ch2-architecture-foundations.ipynb)  →  [Intermediate (tier 2/3)](../notebooks_intermediate_simplified/ch2-architecture-foundations.ipynb)  →  **Advanced (tier 3/3) ← you are here**

# %% [markdown]
# # Chapter 2 (Advanced, explained) — Architecture Foundations on a real Neo4j graph
#
# **Tier 3 of 3 (advanced).** You have done the beginner notebook (`notebooks_simplified/`). Here we go deeper:
# every skill of the chapter is used, *every skill input comes from a Cypher query*, every skill output is written
# **back** into the graph, and a tiny agent drives it all and leaves an auditable trace.
#
# **What you will build**
# 1. A graph model of the chapter: services, a workflow DAG, harness nodes, tools, pillars, flaws.
# 2. **`dual-graph-router`** — decide *vertical* (facts), *horizontal* (process), or *both*; store the decision as edges.
# 3. **`harness-node-splitter`** — split operations into nodes **by tool surface**; store scopes and prove least privilege in Cypher.
# 4. **`eight-pillar-readiness-map`** — a layered roadmap, with system status kept in the graph.
# 5. A **plan → act → observe** agent (`a.MiniAgent`) whose tools are skills + Cypher, with its trace stored as a graph.
# 6. Each skill's **CLI and self-benchmark**, then **vibe-coding prompts**, **experiments** and **pitfalls**.
#
# **Maps to the book:** Ch2 *"The Dual-Graph Architecture"*, *"Defining the harness"*, *"Splitting a workflow into nodes"*,
# *"The Eight Pillars"*. Read the book for the theory; this notebook is the practice.
#
# > Needs only Neo4j (`bolt://localhost:7687`) — no LLM, no API keys. Run top to bottom; the first cell **resets the graph**.

# %% [markdown]
# ## 1. Setup and the skill catalog
#
# `adv_helpers` (imported as `a`) wraps the connection (`a.run` returns rows, `a.show` prints a table),
# loads skills by path (`a.load_skill`) and provides `a.check` (prints PASS/FAIL and raises on FAIL).
#
# **Cypher so far:** `CREATE (n:Label {prop: value})` makes a node; `(a)-[:TYPE]->(b)` is an arrow. The loader
# creates the book's running DevOps scenario: services that `DEPENDS_ON` each other, a deployment that changed a database, an alarm and incidents.

# %%
import adv_helpers as a

counts = a.load_devops_graph()
print("Loaded:", counts)

catalog = a.skill_catalog("architecture")
for row in catalog:
    print(f"- {row['slug']}: {row['description'][:90]}...")
a.check("chapter folder exposes exactly 3 skills", len(catalog) == 3)

dg = a.load_skill("architecture", "dual-graph-router")
splitter = a.load_skill("architecture", "harness-node-splitter")
pillars = a.load_skill("architecture", "eight-pillar-readiness-map")

# %% [markdown]
# ## 2. The vertical graph: facts with *time* on the edge
#
# A vector store can say two paragraphs are *similar*; it cannot say `checkout-api` **DEPENDS_ON** `orders-db`.
# A knowledge graph stores that as an explicit edge. We enrich the edges with a `since` date (when the dependency began).
#
# * `MATCH` finds a pattern; `SET` writes properties; `CASE ... WHEN` is an if/else inside a query.
# * `MERGE` = "create it if it does not exist, otherwise reuse it" (safe to re-run), unlike `CREATE`, which always makes a new one.
#
# **Bi-temporal** (defined): a fact has two times — *valid time* (when it was true in the world, our `since`) and
# *record time* (when the database learned it, our `recorded_at`). Keeping both lets you ask "what did we believe on date X?" (Chapter 4 goes deep).

# %%
a.run("""
    MATCH (c:Service {name:'checkout-api'})-[r:DEPENDS_ON]->(d)
    SET r.since = CASE d.name
            WHEN 'payment-service' THEN '2024-01-15'
            WHEN 'orders-service'  THEN '2024-03-01'
            ELSE '2024-05-10' END,
        r.recorded_at = '2026-10-24T08:00:00Z'
""")
a.run("""
    MATCH (c:Service {name:'checkout-api'})
    MERGE (c)-[:USES_LIBRARY {version:'5.4.0'}]->(:Library {name:'stripe-python', version:'5.4.0'})
""")

# Point-in-time query: which dependencies had already started by 2024-02-01?
asof = a.run("""
    MATCH (:Service {name:'checkout-api'})-[r:DEPENDS_ON]->(d)
    WHERE r.since <= $as_of
    RETURN d.name AS dependency, r.since AS since ORDER BY since
""", as_of="2024-02-01")
a.show(asof)
a.check("as of 2024-02-01 only payment-service was a dependency", [r["dependency"] for r in asof] == ["payment-service"])

# %% [markdown]
# ### Query 1 — multi-hop *blast radius*
#
# **Blast radius** (defined): the set of things that break or slow down if one component fails — i.e. everything that
# depends on it, directly or through other things.
#
# * `-[:DEPENDS_ON*1..3]->` is a **variable-length path**: follow 1 to 3 hops of that edge type.
# * `p = (...)` names the whole path so we can call `length(p)` (number of hops).
# * `min(length(p))` with `RETURN s.name, ...` is an **aggregation**: rows are grouped by the non-aggregated columns.
#
# Why it matters: the alarm fired on `checkout-api`, but the changed component is `orders-db`, two hops away. Only a traversal sees that.

# %%
blast = a.run("""
    MATCH p = (s:Service)-[:DEPENDS_ON*1..3]->(:Database {name:'orders-db'})
    RETURN s.name AS service, s.tier AS tier, min(length(p)) AS hops_to_db
    ORDER BY hops_to_db, service
""")
a.show(blast)
hops = {r["service"]: r["hops_to_db"] for r in blast}
a.check("checkout-api is 2 hops from orders-db", hops["checkout-api"] == 2)
a.check("cart-service is in the blast radius (3 hops)", hops["cart-service"] == 3)

# %% [markdown]
# ## 3. `dual-graph-router` — vertical, horizontal, or both
#
# The harness first decides **what kind of request** it got:
# * a *lookup* ("which services depend on …?") is one traversal → **vertical** graph (the *map*);
# * a *process* ("classify, plan, decide on rollback") is a workflow → **horizontal** graph (the *route*);
# * a *diagnosis* needs a workflow whose steps query the map → **both** (*where the two graphs meet*).
#
# Design choice: the skill is a transparent keyword scorer, so you can see *why* it routed. The **contract**
# (`vertical/horizontal/both/unroutable` + rationale) is the stable seam where production swaps in an LLM classifier.
#
# **Derive input from Cypher:** the requests live in the graph as `:Request` nodes (with the label we *expect*), we read
# them with Cypher, route them, and write each decision back as a `-[:ROUTED_TO]->(:Route)` edge.

# %%
a.run("""
    UNWIND $rows AS r
    CREATE (:Request {id:r[0], text:r[1], expected:r[2]})
""", rows=[
    ["R1", "Which services depend on the orders-db and were deployed in the last 24 hours?", "vertical"],
    ["R2", "Classify the alert severity, plan the remediation, and decide whether to roll back.", "horizontal"],
    ["R3", "Diagnose why the checkout API is slow: which services does it depend on and what changed recently?", "both"],
    ["R4", "Hello there, nice weather", "unroutable"],
])

req_rows = a.run("MATCH (r:Request) RETURN r.id AS id, r.text AS text ORDER BY id")
for r in req_rows:
    d = dg.route(r["text"])
    a.run("""
        MATCH (r:Request {id:$id})
        MERGE (k:Route {kind:$kind})
        MERGE (r)-[x:ROUTED_TO]->(k)
        SET x.vertical_score=$v, x.horizontal_score=$h
    """, id=r["id"], kind=d.target, v=d.vertical_score, h=d.horizontal_score)

# %% [markdown]
# ### Query 2 — aggregation with `collect` and `count`, plus an audit
#
# * `count(r)` counts rows per group; `collect(r.id)` gathers values into a list.
# * `WHERE r.expected <> k.kind` keeps only *mismatches* — a graph-native regression test: zero rows means the router agrees with our labels.

# %%
a.show(a.run("""
    MATCH (r:Request)-[x:ROUTED_TO]->(k:Route)
    RETURN k.kind AS route, count(r) AS requests, collect(r.id) AS ids,
           max(x.vertical_score) AS max_v, max(x.horizontal_score) AS max_h
    ORDER BY route
"""))
mismatch = a.run("MATCH (r:Request)-[:ROUTED_TO]->(k:Route) WHERE r.expected <> k.kind RETURN r.id AS id")
a.check("router agrees with every labelled request", mismatch == [])

meet = dg.explain_meeting_point(dg.route(req_rows[2]["text"]))
print("\nWhere the graphs meet:")
for k, v in meet.items():
    print(f"  {k}: {v[:95]}...")
a.check("only a 'both' request has a meeting point", dg.explain_meeting_point(dg.route(req_rows[0]["text"])) == {})

# %% [markdown]
# ## 4. `harness-node-splitter` — nodes differ by *tool surface*, not by prompt
#
# The harness executes the workflow. The chapter's rule: **before adding a node, list the tools it calls.** If the tool list
# overlaps ≥ 80 % with an existing node (same type), it is just a *prompt variation* — merge. Otherwise it is a new role — split.
#
# **Jaccard overlap** (defined): `|A ∩ B| / |A ∪ B|`. Identical tool sets → 1.0, disjoint → 0.0.
# Trade-off: fewer nodes = simpler, but one node holding many tools has a bigger permission surface (and a bigger place for mistakes).
# Splitting by tools keeps each node **least-privilege**: it can only do what its role needs.
#
# We store candidate operations as `:Operation` nodes linked to `:Tool` nodes (`UNWIND` turns a list into one row per element).

# %%
import json

sample = a.SKILLS_DIR / "architecture" / "harness-node-splitter" / "sample-workflow.json"
op_dicts = json.loads(sample.read_text(encoding="utf-8"))["operations"]

a.run("""
    UNWIND $rows AS r
    CREATE (o:Operation) SET o = r
    WITH o
    UNWIND o.tools AS tool
    MERGE (t:Tool {name: tool})
    MERGE (o)-[:USES_TOOL]->(t)
""", rows=op_dicts)

# %% [markdown]
# ### Query 3 — `OPTIONAL MATCH` and tool sharing
#
# * `OPTIONAL MATCH` is like a left join: operations with **no** tool still appear (with `null`), instead of vanishing.
# * `WITH t, collect(o.id) AS users WHERE size(users) > 1` filters *after* aggregating — "tools used by more than one operation".
#
# Tool-less operations (`classify`, `analyze`, `report`) are pure reasoning/generation; their role lives in the prompt and in their *position* in the DAG.

# %%
a.show(a.run("""
    MATCH (o:Operation)
    OPTIONAL MATCH (o)-[:USES_TOOL]->(t:Tool)
    RETURN o.id AS operation, o.node_type AS type, count(t) AS tool_count, collect(t.name) AS tools
    ORDER BY type, operation
"""))
shared = a.run("""
    MATCH (t:Tool)<-[:USES_TOOL]-(o:Operation)
    WITH t, collect(o.id) AS users WHERE size(users) > 1
    RETURN t.name AS tool, users
""")
print("\nTools shared by several operations (merge candidates):")
a.show(shared)
a.check("exactly one shared tool: cloudwatch_get_metric_data", [r["tool"] for r in shared] == ["cloudwatch_get_metric_data"])

# %% [markdown]
# Now the skill. Its **input is read from Neo4j** (`properties(o)` returns a node's properties as a map), and its
# **output — the harness nodes and their scopes — is written back** as `:HarnessNode` nodes that `MERGES` the operations
# they absorbed, `CAN_CALL` tools and `MAY_READ` memory slices.

# %%
op_rows = a.run("MATCH (o:Operation) RETURN properties(o) AS p")
ops = splitter.operations_from_dicts([r["p"] for r in op_rows])
result = splitter.split_nodes(ops)
print(f"{len(ops)} operations -> {len(result.nodes)} harness nodes (threshold {result.threshold})")

for n in result.nodes:
    sc = splitter.node_scope(n)
    a.run("""
        MERGE (h:HarnessNode {id:$id})
        SET h.node_type=$type, h.merged=$merged, h.output_schema=$out
        WITH h
        UNWIND $merged_from AS op_id
          MATCH (o:Operation {id:op_id}) MERGE (h)-[:MERGES]->(o)
    """, id=sc["id"], type=sc["node_type"], merged=sc["is_merged"], out=sc["output_schema"],
          merged_from=n.merged_from)
    a.run("""
        MATCH (h:HarnessNode {id:$id})
        UNWIND $tools AS tl MERGE (t:Tool {name:tl}) MERGE (h)-[:CAN_CALL]->(t)
    """, id=sc["id"], tools=sc["tool_surface"])
    a.run("""
        MATCH (h:HarnessNode {id:$id})
        UNWIND $slices AS s MERGE (m:MemorySlice {name:s}) MERGE (h)-[:MAY_READ]->(m)
    """, id=sc["id"], slices=sc["memory_reads"])

merged_nodes = [n for n in result.nodes if n.is_merged]
a.check("9 operations -> 8 nodes", len(ops) == 9 and len(result.nodes) == 8)
a.check("only get_metrics+get_cpu merged", len(merged_nodes) == 1 and set(merged_nodes[0].merged_from) == {"get_metrics", "get_cpu"})
a.check("RedAI scanner vs validator overlap is 0", splitter.tool_overlap({"filesystem"}, {"browser_driver", "ios_simulator", "network_stack", "scripting_runtime"}) == 0.0)

# %% [markdown]
# **Prove least privilege with the graph.** If two *different* harness nodes could call the same tool, their roles overlap.
# The query below looks for exactly that (`a.id < b.id` avoids listing each pair twice). Zero rows = clean separation.
# Then we print each node's scope: how many tools and memory slices it may touch.

# %%
overlap_pairs = a.run("""
    MATCH (x:HarnessNode)-[:CAN_CALL]->(t:Tool)<-[:CAN_CALL]-(y:HarnessNode)
    WHERE x.id < y.id RETURN x.id AS a, y.id AS b, t.name AS tool
""")
print("harness nodes sharing a tool:", overlap_pairs)
a.check("no two harness nodes share a tool", overlap_pairs == [])

a.show(a.run("""
    MATCH (h:HarnessNode)
    OPTIONAL MATCH (h)-[:CAN_CALL]->(t:Tool)
    WITH h, count(t) AS tools
    OPTIONAL MATCH (h)-[:MAY_READ]->(m:MemorySlice)
    RETURN h.id AS node, h.node_type AS type, h.merged AS merged, tools, count(m) AS memory_slices
    ORDER BY tools DESC, node
"""))

# %% [markdown]
# ## 5. The horizontal graph as a **DAG**
#
# **DAG** (defined): *directed acyclic graph* — arrows ("must run before") with **no loops**. A step runs when all its
# inputs are ready; independent branches can run in parallel. If a cycle sneaked in, the workflow could never finish.
#
# We add `PRECEDES` edges between the operations, then ask Cypher three things:
# cycles (must be 0), **in-degree** (how many arrows enter a step = inputs it waits for) and **topological level**
# (the longest path from a start step = which "wave" it can run in; same level ⇒ can run in parallel).

# %%
edges = [("classify", "query_kg"), ("query_kg", "get_metrics"), ("query_kg", "analyze"),
         ("get_metrics", "analyze"), ("analyze", "report")]
a.run("""
    UNWIND $edges AS e
    MATCH (x:Operation {id:e[0]}), (y:Operation {id:e[1]})
    MERGE (x)-[:PRECEDES]->(y)
""", edges=[list(e) for e in edges])

cycles = a.run("MATCH p=(n:Operation)-[:PRECEDES*1..10]->(n) RETURN count(p) AS cycles")[0]["cycles"]
levels = a.run("""
    MATCH (root:Operation) WHERE NOT ()-[:PRECEDES]->(root) AND (root)-[:PRECEDES]->()
    MATCH p = (root)-[:PRECEDES*0..]->(n:Operation)
    OPTIONAL MATCH (m)-[:PRECEDES]->(n)
    RETURN n.id AS step, max(length(p)) AS level, count(DISTINCT m) AS waits_on_inputs
    ORDER BY level, step
""")
a.show(levels)
lv = {r["step"]: r for r in levels}
a.check("workflow has no cycles", cycles == 0)
a.check("analyze waits on 2 inputs (graph context + live metrics)", lv["analyze"]["waits_on_inputs"] == 2)
a.check("report is the last wave (level 4)", lv["report"]["level"] == 4)

# %% [markdown]
# **Why the DAG check matters:** `analyze` has in-degree 2, so it is a *join*: the harness must wait for both the
# graph context and the metrics. To see the cycle check actually *work*, we add a deliberately bad edge that we delete right after.

# %%
a.run("MATCH (x:Operation {id:'report'}), (y:Operation {id:'classify'}) CREATE (x)-[:PRECEDES {tmp:true}]->(y)")
bad = a.run("MATCH p=(n:Operation)-[:PRECEDES*1..10]->(n) RETURN count(p) AS cycles")[0]["cycles"]
a.run("MATCH ()-[r:PRECEDES {tmp:true}]->() DELETE r")
after = a.run("MATCH p=(n:Operation)-[:PRECEDES*1..10]->(n) RETURN count(p) AS cycles")[0]["cycles"]
print(f"with bad edge: {bad} cyclic paths; after removal: {after}")
a.check("cycle detector fires on a bad edge and clears after removal", bad > 0 and after == 0)

# %% [markdown]
# ## 6. The seam — graph facts choose which live metrics to fetch
#
# This is *where the two graphs meet*: the `get_metrics` node does not guess which services to check — the vertical graph
# says so (the service and everything it depends on, found with a variable-length path).
# Here the "monitoring tool" is simulated by writing `MetricPoint` nodes; the spike starts after `deploy-481` (08:30).

# %%
a.run("""
    MATCH (c:Service {name:'checkout-api'})
    UNWIND [['08:20',42.0],['08:25',45.0],['08:31',3100.0],['08:36',3300.0],['08:41',2400.0]] AS p
    CREATE (c)-[:HAS_METRIC]->(:MetricPoint {metric:'p99_latency_ms', at:p[0], value:p[1]})
""")
affected = a.run("""
    MATCH (c:Service {name:'checkout-api'})
    OPTIONAL MATCH (c)-[:DEPENDS_ON*1..2]->(d)
    WITH c, collect(DISTINCT d.name) AS deps
    RETURN [c.name] + deps AS affected
""")[0]["affected"]
print("graph-derived targets:", affected)

ba = a.run("""
    MATCH (d:Deployment {id:'deploy-481'})
    WITH substring(d.at, 11, 5) AS hhmm
    MATCH (:Service {name:'checkout-api'})-[:HAS_METRIC]->(m:MetricPoint)
    RETURN round(avg(CASE WHEN m.at <  hhmm THEN m.value END)) AS before_avg_ms,
           round(avg(CASE WHEN m.at >= hhmm THEN m.value END)) AS after_avg_ms,
           max(m.value) AS peak_ms
""")[0]
print(ba)
a.check("targets include the database two hops away", "orders-db" in affected and "checkout-api" in affected)
a.check("latency jumped >10x after the deployment", ba["after_avg_ms"] / ba["before_avg_ms"] > 10)

# %% [markdown]
# ## 7. `eight-pillar-readiness-map` — a layered roadmap kept in the graph
#
# The dual graph is the *structure*; the **eight pillars** are the *build order*, and they are **layered**: a higher
# pillar cannot be sound if a lower one it depends on is missing (memory needs a knowledge graph to remember *into*).
# The Ch1 flaws map to pillars (Table 2-1), e.g. `relationship_blindness → knowledge_representation`.
#
# We model **pillars as nodes with `REQUIRES` edges**, **flaws as nodes with `SOLVED_BY` edges**, and a **system's status**
# as `HAS_STATUS {status}` edges. The skill's input (the capabilities dict) is then read back with Cypher.

# %%
a.run("""
    UNWIND $ps AS p
    MERGE (x:Pillar {key:p.key}) SET x.name=p.name, x.chapter=p.chapter, x.order=p.order, x.production_only=p.production_only
""", ps=[dict(key=p.key, name=p.name, chapter=p.chapter, order=p.order, production_only=p.production_only)
         for p in pillars.PILLARS.values()])
a.run("""
    UNWIND $deps AS d
    MATCH (x:Pillar {key:d[0]}), (y:Pillar {key:d[1]}) MERGE (x)-[:REQUIRES]->(y)
""", deps=[[p.key, dep] for p in pillars.PILLARS.values() for dep in p.depends_on])
a.run("""
    UNWIND $fs AS f
    MERGE (fl:Flaw {name:f[0]})
    WITH fl, f UNWIND f[1] AS pk
    MATCH (p:Pillar {key:pk}) MERGE (fl)-[:SOLVED_BY]->(p)
""", fs=[[flaw, list(ps)] for flaw, ps in pillars.FLAW_TO_PILLARS.items()])


def set_status(system: str, statuses: dict) -> None:
    a.run("MERGE (:System {name:$n})", n=system)
    a.run("""
        UNWIND $items AS it
        MATCH (s:System {name:$n}), (p:Pillar {key:it[0]})
        MERGE (s)-[h:HAS_STATUS]->(p) SET h.status = it[1]
    """, n=system, items=[[k, v] for k, v in statuses.items()])


def caps_from_graph(system: str) -> dict:
    rows = a.run("MATCH (:System {name:$n})-[h:HAS_STATUS]->(p:Pillar) RETURN p.key AS k, h.status AS s", n=system)
    return {r["k"]: r["s"] for r in rows}


set_status("devops-agent", pillars.initial_state())
set_status("legacy-bot", {"memory": "present", "reasoning": "partial"})   # built top-down: a bad idea

# %% [markdown]
# ### Query 4 — what is the true *foundation*? (aggregation over `REQUIRES`)
#
# Counting incoming `REQUIRES` edges per pillar tells us how many other pillars lean on it.

# %%
foundation = a.run("""
    MATCH (p:Pillar)
    OPTIONAL MATCH (q:Pillar)-[:REQUIRES]->(p)
    RETURN p.key AS pillar, p.chapter AS chapter, count(q) AS dependents
    ORDER BY dependents DESC, pillar LIMIT 4
""")
a.show(foundation)
a.check("knowledge_representation is the most depended-on pillar", foundation[0]["pillar"] == "knowledge_representation")

# %% [markdown]
# Now run the skill on **graph-derived** statuses and write its verdict back: a `NEXT_TO_BUILD` edge for the roadmap,
# and `VIOLATES` edges for any layering mistake (a pillar claimed while its prerequisite is missing).
# Finally, cross-check the skill's "unresolved flaws" against a pure-Cypher computation.

# %%
def assess_and_store(system: str):
    rep = pillars.assess(caps_from_graph(system))
    if rep.next_pillar:
        a.run("""MATCH (s:System {name:$n}), (p:Pillar {key:$k})
                 MERGE (s)-[:NEXT_TO_BUILD]->(p)""", n=system, k=rep.next_pillar)
    for v in rep.dependency_violations:
        for dep in v["missing_dependencies"]:
            a.run("""MATCH (p:Pillar {key:$p}), (d:Pillar {key:$d})
                     MERGE (p)-[x:VIOLATES]->(d) SET x.system=$n""", p=v["pillar"], d=dep, n=system)
    return rep


rep = assess_and_store("devops-agent")
print(f"devops-agent: readiness {rep.readiness_pct}%, next = {rep.next_pillar}")
print("roadmap:", " -> ".join(rep.roadmap))

cy_unresolved = a.run("""
    MATCH (f:Flaw)-[:SOLVED_BY]->(p:Pillar)
    OPTIONAL MATCH (:System {name:'devops-agent'})-[h:HAS_STATUS]->(p)
    WITH f, collect(coalesce(h.status, 'missing')) AS statuses
    WHERE NOT 'present' IN statuses
    RETURN f.name AS flaw, statuses
""")
a.show(cy_unresolved)
a.check("Cypher and skill agree on unresolved flaws", {r["flaw"] for r in cy_unresolved} == {f["flaw"] for f in rep.unresolved_flaws})

bad = assess_and_store("legacy-bot")
viol = a.run("""
    MATCH (p:Pillar)-[x:VIOLATES {system:'legacy-bot'}]->(d:Pillar)
    RETURN p.key AS claimed_pillar, collect(d.key) AS missing_prerequisites ORDER BY claimed_pillar
""")
a.show(viol)
a.check("legacy-bot is flagged: memory/reasoning without knowledge_representation",
        {v["pillar"] for v in bad.dependency_violations} == {"memory", "reasoning"})

# %% [markdown]
# ## 8. The agent — plan → act → observe over skills + Cypher
#
# An agent in this book is **a loop**: *plan* the steps, *act* by calling a tool, *observe* the result, repeat.
# `a.MiniAgent` is that loop in ~30 lines; its "tools" are plain Python functions. We register **skills and Cypher queries as tools**.
# The planner here is deterministic — it is the **router's decision** (`both` ⇒ run the investigation workflow). In production an LLM
# plans; the *tools and the trace* stay the same. Every call is logged by `AgentTrace` into Neo4j as `(:Run)-[:STEP]->(:ToolCall)-[:NEXT]->(:ToolCall)`.

# %%
GOAL = "Diagnose why the checkout API is slow: which services does it depend on and what changed recently?"
agent = a.MiniAgent(GOAL)


@agent.tool("route")
def t_route(request):
    d = dg.route(request)
    return {"target": d.target, "v": d.vertical_score, "h": d.horizontal_score}


@agent.tool("kg_dependencies")
def t_deps(service):
    rows = a.run("MATCH (:Service {name:$s})-[:DEPENDS_ON*1..2]->(d) RETURN DISTINCT d.name AS n ORDER BY n", s=service)
    return [r["n"] for r in rows]


@agent.tool("recent_changes")
def t_changes(service):
    return a.run("""
        MATCH (:Service {name:$s})-[:DEPENDS_ON*0..3]->(x)<-[:CHANGED]-(d:Deployment)
        RETURN d.id AS deployment, d.change AS change, x.name AS changed_component ORDER BY d.at DESC
    """, s=service)


@agent.tool("metrics_peak")
def t_peak(service):
    return a.run("MATCH (:Service {name:$s})-[:HAS_METRIC]->(m:MetricPoint) RETURN max(m.value) AS peak", s=service)[0]["peak"]


@agent.tool("similar_past_incident")
def t_similar(component):
    return a.run("""
        MATCH (c {name:$c})
        OPTIONAL MATCH (old:Incident {status:'resolved'})-[:ROOT_CAUSE]->(c)
        RETURN old.id AS incident, old.fix AS fix
    """, c=component)[0]


@agent.tool("record_root_cause")
def t_record(incident, deployment):
    # backward flow: the workflow writes its conclusion into the vertical graph
    a.run("""MATCH (i:Incident {id:$i}), (d:Deployment {id:$d})
             MERGE (i)-[x:CAUSED_BY]->(d) SET x.by_run=$run""", i=incident, d=deployment, run=agent.trace.run_id)
    return f"{incident} -CAUSED_BY-> {deployment}"


@agent.tool("assess_pillars")
def t_pillars(system):
    return pillars.assess(caps_from_graph(system)).next_pillar


# PLAN: the route decides the workflow shape
decision = agent.act("route", request=GOAL)
plan = ["kg_dependencies", "recent_changes", "metrics_peak", "similar_past_incident", "record_root_cause"] if decision["target"] == "both" else []
print("router said:", decision, "\nplan:", plan)

# ACT + OBSERVE: each step reads what earlier steps observed (agent.memory)
for step in plan:
    if step == "kg_dependencies":
        out = agent.act(step, service="checkout-api")
    elif step == "recent_changes":
        out = agent.act(step, service="checkout-api")
    elif step == "metrics_peak":
        out = agent.act(step, service="checkout-api")
    elif step == "similar_past_incident":
        out = agent.act(step, component=agent.memory["recent_changes"][0]["changed_component"])
    else:
        out = agent.act(step, incident="INC-1001", deployment=agent.memory["recent_changes"][0]["deployment"])
    print(f"  {step:<22} -> {str(out)[:95]}")

a.check("agent found deploy-481 as the culprit", agent.memory["recent_changes"][0]["deployment"] == "deploy-481")
a.check("agent saw the latency spike", agent.memory["metrics_peak"] > 1000)

# %% [markdown]
# ### Failures belong in the trace too
# An agent will sometimes call a tool it does not have (the harness's *typed tool registry* should refuse it). `MiniAgent.act` logs
# the failure with `ok=False` and re-raises; we catch it so the notebook continues. One more successful step then closes the run.

# %%
try:
    agent.act("restart_service", service="orders-service")   # not registered: least privilege in action
except KeyError as exc:
    print("refused unregistered tool:", exc)
agent.act("assess_pillars", system="devops-agent")

# %% [markdown]
# ## 9. Query the trace back from Neo4j
#
# Because the trace is a graph, auditing is just Cypher: per-run summary (aggregation), the ordered chain (a variable-length
# `NEXT*` path ending at a node with no outgoing `NEXT`), and the **write-back** the agent made to the knowledge graph.

# %%
rid = agent.trace.run_id
a.show(a.run("""
    MATCH (r:Run {id:$rid})-[:STEP]->(c:ToolCall)
    WITH r, c ORDER BY c.seq
    RETURN r.goal AS goal, count(c) AS steps, sum(CASE WHEN c.ok THEN 1 ELSE 0 END) AS ok_steps,
           sum(CASE WHEN c.ok THEN 0 ELSE 1 END) AS failed_steps
""", rid=rid))

chain = a.run("""
    MATCH (:Run {id:$rid})-[:STEP]->(f:ToolCall {seq:1})
    MATCH p = (f)-[:NEXT*]->(l:ToolCall) WHERE NOT (l)-[:NEXT]->()
    RETURN [n IN nodes(p) | n.tool] AS chain
""", rid=rid)[0]["chain"]
print("\ncall chain:", " -> ".join(chain))

writeback = a.run("""
    MATCH (i:Incident {id:'INC-1001'})-[:CAUSED_BY]->(d:Deployment)-[:CHANGED]->(x)
    OPTIONAL MATCH (old:Incident)-[:ROOT_CAUSE]->(x)
    RETURN i.id AS incident, d.id AS caused_by, d.change AS change, x.name AS component, old.id AS past_incident, old.fix AS past_fix
""")
print()
a.show(writeback)
a.check("trace has 8 steps: 7 ok + 1 refused", len(chain) == 8 and a.run("MATCH (c:ToolCall {ok:false}) WHERE c.id STARTS WITH $p RETURN count(c) AS n", p=rid)[0]["n"] == 1)
a.check("graph now links INC-1001 to deploy-481 and to the 2026-06 precedent INC-0877",
        writeback and writeback[0]["caused_by"] == "deploy-481" and writeback[0]["past_incident"] == "INC-0877")

# %% [markdown]
# **Design notes (maps to Ch2):** the *route* chose the horizontal shape; the retrieval steps were Cypher over the vertical
# graph (forward flow); `record_root_cause` wrote a `CAUSED_BY` edge back (backward flow); and the run is itself a graph.
# That is the claim of the chapter: *each graph makes the other more useful.*

# %% [markdown]
# ## 10. Skill CLI & benchmark
#
# Every skill is also a command-line tool (`cli.py`) for cron, CI or any agent harness with zero pip installs.
# `--help` prints the SKILL.md description; `benchmark` runs the skill's self-checks and exits non-zero on failure.
# `a.run_benchmark(chapter, slug)` runs that subcommand and returns `(passed, first_output_line)`.

# %%
bench = {}
for row in catalog:
    ok, line = a.run_benchmark("architecture", row["slug"])
    bench[row["slug"]] = ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {row['slug']:<28} {line[:70]}")
a.check("all 3 chapter benchmarks pass", all(bench.values()) and len(bench) == 3)

import subprocess, sys
cli = a.SKILLS_DIR / "architecture" / "dual-graph-router" / "cli.py"
helptext = subprocess.run([sys.executable, str(cli), "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace")
print("\n$ python cli.py --help   (exit code", helptext.returncode, ")\n" + helptext.stdout[:500])
routed = subprocess.run([sys.executable, str(cli), "route", GOAL, "--json"], capture_output=True, text=True, encoding="utf-8", errors="replace")
cli_decision = json.loads(routed.stdout)
a.check("--help exits 0 and prints text", helptext.returncode == 0 and len(helptext.stdout) > 50)
a.check("CLI route agrees with the library (both)", cli_decision.get("target") == "both")

# %% [markdown]
# ## 11. 🎯 Vibe-coding corner
#
# *Vibe coding* = describing what you want to an AI coding assistant (Copilot, Claude Code) and steering by running checks.
# It works best when your prompt has **context** (what exists), **constraints** (what must not change), and an **acceptance
# check** (how you will know it is right). Then you **verify yourself** — never trust unrun code.
#
# **Corner A — a new Cypher query.** Copy-paste:
# ```text
# Context: Neo4j has (:Service)-[:DEPENDS_ON {since}]->(x), (:Deployment)-[:CHANGED]->(:Database).
# Task: write ONE read-only Cypher query returning, for every Service, how many distinct databases it reaches within 3 DEPENDS_ON hops
#       (use OPTIONAL MATCH so services with none show 0), ordered by count desc.
# Constraints: parameterise values, no writes, no APOC. Explain each clause in a comment.
# Acceptance: checkout-api must return 1 (orders-db), cart-service must return 1, session-cache must not appear as a Service.
# ```
# *What makes it good:* names the labels/edges, forbids writes, states expected rows. *Verify:* run it with `a.run` and compare to the acceptance line.
#
# **Corner B — a worked example (below).** We "ask" for a helper `blast_radius(component, max_hops)` and write the test that guards it.
# Notice the AI-risk: Cypher cannot take a *parameter* for the `*1..N` bound, so a naive prompt tempts string-building — we validate `int()` to keep it injection-safe.

# %%
def blast_radius(component: str, max_hops: int = 3) -> dict:
    """Return {service: min_hops} for everything that transitively depends on `component`."""
    n = int(max_hops)                      # bound cannot be a Cypher parameter; force an int
    if not 1 <= n <= 6:
        raise ValueError("max_hops must be between 1 and 6")
    rows = a.run("""
        MATCH p = (s:Service)-[:DEPENDS_ON*1.."""+ str(n) + """]->(x {name:$c})
        RETURN s.name AS service, min(length(p)) AS hops
    """, c=component)
    return {r["service"]: r["hops"] for r in rows}


# The test you (not the AI) own: expected values come from the graph you can see in section 2.
assert blast_radius("orders-db", 3) == {"orders-service": 1, "payment-service": 1, "checkout-api": 2, "cart-service": 3}
assert blast_radius("orders-db", 1) == {"orders-service": 1, "payment-service": 1}
assert blast_radius("session-cache", 3) == {"checkout-api": 1, "cart-service": 2}
try:
    blast_radius("orders-db", 99)
    raise SystemExit("guard missing")
except ValueError:
    pass
print("blast_radius: 4 tests passed (values, hop limit, other component, input guard)")

# %% [markdown]
# **Corner C — a whole new skill.** Copy-paste:
# ```text
# Create skills/architecture/workflow-cycle-guard/ following the repo's skill anatomy (look at skills/architecture/dual-graph-router/):
#  - SKILL.md with the 7 sections + Source Attribution to Ch2 "The Horizontal Workflow Graph"
#  - lib.py: stdlib only; detect_cycles(edges: list[tuple[str,str]]) -> list[list[str]] and topo_levels(edges) -> dict[str,int]
#  - cli.py: argparse; --help prints the SKILL.md description; `benchmark` subcommand with >=4 self-checks, exit code 1 on failure
# Constraints: no pip installs, no network, don't touch other skills. Mark any production swap with TODO.
# Acceptance: `python skills/architecture/workflow-cycle-guard/cli.py --help` exits 0 and `... benchmark` passes;
#             topo_levels must reproduce the levels we computed in Cypher in the notebook.
# ```
# *Verify:* run the benchmark, then compare `topo_levels` against the Cypher result — two independent implementations that agree is strong evidence.

# %% [markdown]
# ## 12. Experiments — change a parameter, predict, then run
#
# **Experiment 1 — the merge threshold.** With the default 0.8 we got 8 nodes. Predict the count for 0.2, and for **0.0**.
# (Hint: tool-less nodes score overlap 0.0, and the rule is `overlap >= threshold`.)

# %%
for th in (0.8, 0.2, 0.0):
    print(f"threshold {th}: {len(splitter.split_nodes(ops, threshold=th).nodes)} nodes")
a.check("0.2 still gives 8 (no partial overlaps exist), 0.0 collapses same-type nodes to 4",
        len(splitter.split_nodes(ops, 0.2).nodes) == 8 and len(splitter.split_nodes(ops, 0.0).nodes) == 4)

# %% [markdown]
# *Lesson:* the threshold only matters when there are **partial** overlaps; at 0.0 everything of the same type merges, destroying role separation.
#
# **Experiment 2 — promote a pillar.** Predict `next_pillar` and readiness after setting `knowledge_representation` to *present* and
# `memory` to *partial*. Then predict what happens if you set `reasoning` to *present* while `memory` is *missing*.

# %%
exp = dict(pillars.initial_state(), knowledge_representation="present", memory="partial")
r1 = pillars.assess(exp)
print("after promotion:", r1.readiness_pct, "% next =", r1.next_pillar)
exp2 = dict(exp, memory="missing", reasoning="present")
print("reasoning without memory -> violations:", [v["pillar"] for v in pillars.assess(exp2).dependency_violations])
a.check("promotion moves roadmap to memory; skipping a layer is flagged",
        r1.next_pillar == "memory" and [v["pillar"] for v in pillars.assess(exp2).dependency_violations] == ["reasoning"])

# %% [markdown]
# **Experiment 3 — hop limit.** Predict the size of `blast_radius("orders-db", n)` for n = 1, 2, 3 before running.

# %%
sizes = {n: len(blast_radius("orders-db", n)) for n in (1, 2, 3)}
print(sizes)
a.check("blast radius grows 2 -> 3 -> 4 with hops", sizes == {1: 2, 2: 3, 3: 4})

# %% [markdown]
# ## 13. Common pitfalls
#
# * **Merging by prompt instead of tools.** "These two nodes sound similar" is not the rule — compare *tool lists*. And two tool-less nodes score 0.0 on purpose; do not merge them by accident (threshold 0.0).
# * **Unbounded variable-length paths** (`*` with no upper bound) can explode on dense or cyclic graphs. Always cap (`*1..3`) and keep a DAG check.
# * **`CREATE` vs `MERGE`.** Re-running a cell with `CREATE` duplicates nodes; use `MERGE` for idempotent writes (we used `CREATE` only for the throw-away `:Request` seed).
# * **Building higher pillars first.** A "smart" reasoning layer over no knowledge graph is a *dependency violation*, however well it demos.
# * **Aggregation grouping.** In `RETURN a.x, count(b)`, rows are grouped by `a.x`; forgetting `OPTIONAL MATCH` silently drops groups with zero matches.
# * **Trusting AI-written Cypher/skills unrun.** Always execute the acceptance check (see the vibe-coding corner).
# * **Shared database.** This notebook wipes the graph at the top; never point it at a graph you care about.

# %% [markdown]
# ## 14. Verification battery

# %%
a.check("graph has all chapter-2 structures",
        all(a.run(q)[0]["n"] > 0 for q in [
            "MATCH (:Request)-[:ROUTED_TO]->(:Route) RETURN count(*) AS n",
            "MATCH (:HarnessNode)-[:MERGES]->(:Operation) RETURN count(*) AS n",
            "MATCH (:Operation)-[:PRECEDES]->(:Operation) RETURN count(*) AS n",
            "MATCH (:Pillar)-[:REQUIRES]->(:Pillar) RETURN count(*) AS n",
            "MATCH (:Run)-[:STEP]->(:ToolCall) RETURN count(*) AS n"]))
a.check("router: lookup->vertical, process->horizontal, diagnosis->both",
        [dg.route(r["text"]).target for r in req_rows[:3]] == ["vertical", "horizontal", "both"])
a.check("splitter: 9 ops -> 8 nodes, no shared tools", len(result.nodes) == 8 and overlap_pairs == [])
a.check("workflow is a DAG", cycles == 0 and after == 0)
a.check("pillars: devops-agent next = knowledge_representation", rep.next_pillar == "knowledge_representation")
a.check("agent wrote CAUSED_BY back to the graph", len(writeback) == 1)
a.check("vibe-coded blast_radius passes its tests", blast_radius("orders-db") == {"orders-service": 1, "payment-service": 1, "checkout-api": 2, "cart-service": 3})
a.check("all 3 skills covered (router, splitter, readiness map) and benchmarks green", all(bench.values()))
print("\nCh2 advanced: all gates passed.")

# %% [markdown]
# ## Recap
# * **Two graphs + a harness.** Vertical = what is true (with time on edges); horizontal = what to do (a DAG); the harness scopes each node.
# * **`dual-graph-router`** decided vertical/horizontal/both; decisions are stored as `ROUTED_TO` edges and audited with a Cypher mismatch query.
# * **`harness-node-splitter`** merged by tool overlap (Jaccard ≥ 0.8); the scopes live in the graph, and Cypher proves no two nodes share a tool.
# * **DAG hygiene:** cycle detection, in-degree (joins) and topological level (parallel waves) — all in Cypher.
# * **`eight-pillar-readiness-map`** read status from the graph, wrote `NEXT_TO_BUILD` / `VIOLATES` back; knowledge representation is the foundation (most dependents).
# * **Agent loop:** plan (from the router) → act (skills + Cypher tools) → observe → trace stored as a graph, including a refused call, and a write-back that links the incident to its cause and to a past precedent.
# * You also met CLIs/benchmarks, three vibe-coding prompts with a verified helper, three predict-then-run experiments and the main pitfalls.
#
# **Next:** `ch3-knowledge-representation.ipynb` (advanced tier) — building the vertical knowledge graph properly.
