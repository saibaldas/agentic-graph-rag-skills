# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > [Beginner (tier 1/3)](../notebooks_simplified/ch2-architecture-foundations.ipynb)  →  **Intermediate (tier 2/3) ← you are here**  →  [Advanced (tier 3/3)](../notebooks_advanced_simplified/ch2-architecture-foundations.ipynb)

# %% [markdown]
# # Chapter 2 — Architecture Foundations (Intermediate)
#
# We will work between the beginner tour and the advanced agent lab: understand the design trade-offs, then compose a few primitives yourself. The book supplies the theory; these three skills make parts of it executable.
#
# **Today:** use the vertical graph for facts, the horizontal graph for work, and a harness boundary for constrained nodes. Inputs come from Cypher, one verdict is persisted, and a plain Python pipeline handles the `checkout-api-is-slow` incident. No LLM or API key is involved.
#
# **Chapter skills:** `dual-graph-router`, `harness-node-splitter`, and `eight-pillar-readiness-map`.

# %% [markdown]
# ## 1. Start with the running incident
#
# `h.load_devops_graph()` resets and loads the shared local Neo4j DevOps fixture, including checkout-api, its dependencies, `deploy-481`, and the open incident. The helpers wrap the configured Bolt connection and keep this notebook focused on Cypher and skill contracts.

# %%
import int_helpers as h

counts = h.load_devops_graph()
print("Loaded shared Neo4j fixture:", counts)

# %% [markdown]
# ## 2. Vertical graph: facts are traversable
#
# A vector similarity score can retrieve related text; an explicit `DEPENDS_ON` edge answers a structural question. Cypher patterns are readable contracts: `MATCH` finds a pattern, `WHERE` filters it, `RETURN` shapes rows, and `ORDER BY` makes results stable.

# %%
blast = h.run("""
    MATCH p=(s:Service)-[:DEPENDS_ON*1..3]->(:Database {name:'orders-db'})
    RETURN s.name AS service, min(length(p)) AS hops
    ORDER BY hops, service
""")
h.show(blast)
h.check("checkout-api is two hops from orders-db", {r["service"]: r["hops"] for r in blast}["checkout-api"] == 2)
h.check("cart-service is three hops from orders-db", {r["service"]: r["hops"] for r in blast}["cart-service"] == 3)

# %% [markdown]
# **Query anatomy:** `p=` binds each path; `*1..3` permits one to three dependency edges; `length(p)` measures each path. `min(...)` groups paths by service and keeps the nearest route to the database. This is useful for blast-radius questions, but a depth bound is intentional: unbounded paths can be expensive or return more than an incident workflow needs.

# %%
fanout = h.run("""
    MATCH (s:Service)-[:DEPENDS_ON]->(target)
    RETURN target.name AS dependency, count(s) AS direct_consumers,
           collect(s.name) AS services
    ORDER BY direct_consumers DESC, dependency
""")
h.show(fanout)
h.check("orders-db has two direct service consumers", next(r["direct_consumers"] for r in fanout if r["dependency"] == "orders-db") == 2)

# %% [markdown]
# **Aggregation anatomy:** `count(s)` counts incoming service rows and `collect(s.name)` preserves which services contributed. Grouping happens by `dependency`; this is a compact summary, not a replacement for the underlying edges.

# %% [markdown]
# ## 3. `dual-graph-router`: select the structure before acting
#
# A single lookup belongs on the **vertical** knowledge graph; a multi-step process belongs on the **horizontal** workflow graph. A diagnosis that needs both is the meeting point: workflow steps query facts, then feed decisions back. The current implementation is an explainable keyword heuristic, not a universal intent model.
#
# API: `route(request: str) -> RouteDecision`. `target` is `vertical`, `horizontal`, `both`, or `unroutable`; the score fields count matched signal words; matched-token lists and `rationale` expose why; `node_hint` suggests the handling structure.

# %%
dg = h.load_skill("architecture", "dual-graph-router")
incident_row = h.run("""
    MATCH (i:Incident {id:'INC-1001'})-[:AFFECTS]->(s:Service)
    RETURN i.id AS incident, i.title AS title, s.name AS service
""")[0]
request_text = f"Diagnose why {incident_row['service']} is slow; determine what it depends on and plan an investigation."
decision = dg.route(request_text)
print("Request:", request_text)
print("Route:", decision.target, "| scores V/H:", decision.vertical_score, decision.horizontal_score)
print("Matched:", decision.matched_vertical, decision.matched_horizontal)
print("Rationale:", decision.rationale)
h.check("incident diagnosis needs both graphs", decision.target == "both")

# %% [markdown]
# **Why this matters:** routing a diagnosis as only a lookup loses the process; routing it as only a workflow loses grounding. The explicit `both` contract lets the harness retain each role. For ambiguous requests, `unroutable` is safer than inventing intent.

# %% [markdown]
# ## 4. Harness boundary: split by tool surface, not prompt wording
#
# A node is a constrained job with a tool list and input/output contract. Same-role operations with highly overlapping tool sets can merge; different surfaces stay separate. Here, operation records are stored then read back from Neo4j, so the skill consumes graph-derived inputs.
#
# API: `operations_from_dicts(rows) -> list[Operation]`; `split_nodes(operations, threshold=0.8) -> SplitResult`. Each `Node` exposes `id`, `node_type`, `tools`, prompt `tasks`, memory `reads`/`writes`, contracts, and `merged_from`; `SplitResult` contains `nodes`, pairwise `decisions`, and the threshold. `node_scope(node)` summarizes the enforced boundary.

# %%
splitter = h.load_skill("architecture", "harness-node-splitter")
ops_data = [
    {"id":"query_facts", "task":"Traverse service dependencies", "node_type":"retrieval", "tools":["graph_read"], "reads":["vertical_kg"], "writes":[], "input_contract":"incident id", "output_schema":"service list"},
    {"id":"check_metrics", "task":"Read latency metrics", "node_type":"execution", "tools":["monitor_read"], "reads":["service list"], "writes":[], "input_contract":"service names", "output_schema":"latency rows"},
    {"id":"compare_metrics", "task":"Compare latency windows", "node_type":"execution", "tools":["monitor_read"], "reads":["latency rows"], "writes":["evidence"], "input_contract":"latency rows", "output_schema":"comparison"},
    {"id":"write_report", "task":"Summarize evidence", "node_type":"generation", "tools":["report_write"], "reads":["evidence"], "writes":["incident_report"], "input_contract":"evidence", "output_schema":"incident summary"},
]
h.run("""
    UNWIND $rows AS row
    MERGE (o:Operation {id:row.id})
    SET o = row
""", rows=ops_data)
op_rows = h.run("MATCH (o:Operation) RETURN properties(o) AS operation ORDER BY o.id")
operations = splitter.operations_from_dicts([r["operation"] for r in op_rows])
split = splitter.split_nodes(operations)
print(f"{len(operations)} operations -> {len(split.nodes)} scoped nodes; threshold={split.threshold}")
for node in split.nodes:
    print(node.id, "tools=", node.tools, "merged_from=", node.merged_from)
print("Scope of retrieval node:", splitter.node_scope(next(n for n in split.nodes if n.id == "query_facts")))
h.check("same execution surface merges prompt variants", any(set(n.merged_from) == {"check_metrics", "compare_metrics"} for n in split.nodes))
h.check("retrieval keeps graph-read scope", splitter.node_scope(next(n for n in split.nodes if n.id == "query_facts"))["tool_surface"] == ["graph_read"])

# %% [markdown]
# `node_scope` output anatomy: `tool_surface` is the allowed tool set; `memory_reads`/`memory_writes` narrow context; `input_contract` and `output_schema` define the handoff; `prompt_variants` preserve merged tasks. `is_merged` makes the merge decision inspectable. The abstraction is only useful if a runtime actually enforces these fields.

# %% [markdown]
# ## 5. Readiness: make the build order explicit
#
# The eight-pillar model is layered: knowledge representation comes before memory and reasoning. We copy the skill's initial status map into graph records, read those statuses with Cypher, and assess the retrieved mapping.
#
# API: `assess(capabilities: dict[str, str]) -> ReadinessReport`. `readiness_pct` is a weighted completeness score; `dependency_violations` flags claimed pillars whose prerequisites are missing; `unresolved_flaws` describes remaining gaps; `next_pillar` is the earliest incomplete layer; `roadmap` lists incomplete pillars in order.

# %%
pillars = h.load_skill("architecture", "eight-pillar-readiness-map")
initial = pillars.initial_state()
h.run("""
    UNWIND $items AS item
    MERGE (c:Capability {system:'checkout-agent', pillar:item[0]})
    SET c.status = item[1]
""", items=[[key, value] for key, value in initial.items()])
cap_rows = h.run("MATCH (c:Capability {system:'checkout-agent'}) RETURN c.pillar AS pillar, c.status AS status")
capabilities = {r["pillar"]: r["status"] for r in cap_rows}
readiness = pillars.assess(capabilities)
print(f"Readiness: {readiness.readiness_pct}% | next: {readiness.next_pillar}")
print("Unresolved flaws:", [f["flaw"] for f in readiness.unresolved_flaws])
h.check("the first incomplete layer is knowledge representation", readiness.next_pillar == "knowledge_representation")

# %% [markdown]
# **Persist and verify one skill result:** store the recommended next pillar as a graph edge, then read it back. `MERGE` makes this write repeatable instead of adding duplicate recommendations on every execution.

# %%
h.run("""
    MATCH (c:Capability {system:'checkout-agent'})
    WITH c LIMIT 1
    MERGE (s:System {name:'checkout-agent'})
    MERGE (p:Pillar {key:$key})
    MERGE (s)-[:NEXT_TO_BUILD]->(p)
""", key=readiness.next_pillar)
stored_next = h.run("MATCH (:System {name:'checkout-agent'})-[:NEXT_TO_BUILD]->(p:Pillar) RETURN p.key AS next")
h.show(stored_next)
h.check("read-back matches readiness skill verdict", stored_next[0]["next"] == readiness.next_pillar)

# %% [markdown]
# ## 6. A small hand-written pipeline — no agent loop
#
# Tier 3 adds `MiniAgent`; here we write the orchestration ourselves. The function chains three skills: route the incident, split graph-derived work into nodes, then assess the graph-derived capability map. It returns a readable report so a human can inspect the decision rather than trusting an opaque chain.

# %%
def investigate_checkout(incident_id: str) -> str:
    incident = h.run("""
        MATCH (i:Incident {id:$id})-[:AFFECTS]->(s:Service)
        RETURN i.title AS title, s.name AS service
    """, id=incident_id)[0]
    routed = dg.route(f"Diagnose why {incident['service']} is slow; determine dependencies and plan an investigation.")
    graph_ops = h.run("MATCH (o:Operation) RETURN properties(o) AS operation ORDER BY o.id")
    nodes = splitter.split_nodes(splitter.operations_from_dicts([r["operation"] for r in graph_ops]))
    caps = h.run("MATCH (c:Capability {system:'checkout-agent'}) RETURN c.pillar AS pillar, c.status AS status")
    report = pillars.assess({r["pillar"]: r["status"] for r in caps})
    return (f"Incident: {incident['title']} ({incident_id})\n"
            f"Route: {routed.target} — use workflow steps grounded in the knowledge graph\n"
            f"Harness: {len(nodes.nodes)} nodes from {len(graph_ops)} operations\n"
            f"Readiness: {report.readiness_pct}% — next build: {report.next_pillar}")

print(investigate_checkout("INC-1001"))

# %% [markdown]
# ### Your turn 1 — query direct consumers
#
# First try writing a Cypher aggregation that counts services with a direct `DEPENDS_ON` edge to `orders-db`. Then compare with the runnable reference below. `h.your_turn` prints the task and hint, runs the reference, and checks the expected value.

# %%
h.your_turn(
    "Count direct service consumers of orders-db",
    "MATCH service-to-database edges, then use count(service).",
    lambda: h.run("MATCH (s:Service)-[:DEPENDS_ON]->(:Database {name:'orders-db'}) RETURN count(s) AS n")[0]["n"],
    2,
)

# %% [markdown]
# ### Your turn 2 — change the request, not the router
#
# Try another lookup phrasing and see whether the contract remains `vertical`. This is a heuristic skill, so use the matched tokens and scores to understand its limits.

# %%
h.your_turn(
    "Route a direct dependency lookup",
    "Pass a question containing a dependency / lookup signal, without a workflow action.",
    lambda: dg.route("Which services depend on orders-db?").target,
    "vertical",
)

# %% [markdown]
# ## 7. Trust the skill with its CLI and self-check
#
# The CLI is the same library contract in a process boundary; the benchmark subcommand runs the skill's built-in self-checks. Here we run one CLI explicitly, then one separate helper benchmark. Tier 3 runs benchmarks for every chapter skill.

# %%
import subprocess
import sys

cli_path = h.SKILLS_DIR / "architecture" / "dual-graph-router" / "cli.py"
cli_run = subprocess.run([sys.executable, str(cli_path), "benchmark"], capture_output=True, text=True, timeout=120)
print((cli_run.stdout or cli_run.stderr).strip())
h.check("router CLI benchmark exits successfully", cli_run.returncode == 0)
bench_ok, bench_summary = h.run_benchmark("architecture", "harness-node-splitter")
print("Splitter benchmark:", bench_summary)
h.check("splitter helper benchmark passes", bench_ok)

# %% [markdown]
# ## 8. 🎯 Vibe-coding tip
#
# **Copy-paste prompt:** “Using the existing Neo4j helper and `checkout-api` fixture, write a parameterized Cypher query that returns every service within three `DEPENDS_ON` hops of `orders-db`, with the shortest hop count. Do not modify the graph. Keep the query in one function and explain its grouping.”
#
# **Acceptance check:** run a bounded variable-length traversal; it should include checkout-api at 2 hops and cart-service at 3, with no graph writes.

# %%
cypher_acceptance = h.run("""
    MATCH p=(s:Service)-[:DEPENDS_ON*1..3]->(:Database {name:'orders-db'})
    RETURN s.name AS service, min(length(p)) AS hops
    ORDER BY hops, service
""")
accepted = {r["service"]: r["hops"] for r in cypher_acceptance}
h.check("vibe prompt acceptance: expected shortest paths", accepted.get("checkout-api") == 2 and accepted.get("cart-service") == 3)

# %% [markdown]
# ## Why this matters — and common mistakes
#
# **Trade-off:** explicit edges and bounded traversal give explainable answers, but only for modeled relationships. A harness can constrain tools and context, but a data structure alone does not enforce policy; the runtime must honor it. Readiness scores summarize declared status, not independently verified production capability.
#
# **Common mistakes:**
# - Treating text similarity as proof of dependency; traverse typed edges for topology.
# - Using unbounded variable-length paths on a large graph; cap depth to the question.
# - Merging jobs because prompts look alike while their tool surfaces differ.
# - Treating the router's keyword scores as certainty; inspect the rationale and allow clarification.
# - Assuming a stored scope or readiness claim enforces itself; validate at the execution boundary.

# %% [markdown]
# ## 9. Check battery
#
# These assertions cover the key contracts and the persisted round trip; they are deliberately small, deterministic checks rather than an integration-wide benchmark suite.

# %%
h.check("three chapter skills loaded", all(x is not None for x in (dg, splitter, pillars)))
h.check("route includes vertical and horizontal signals", decision.target == "both" and decision.vertical_score > 0 and decision.horizontal_score > 0)
h.check("pipeline produces a decision report", "Route: both" in investigate_checkout("INC-1001"))
h.check("stored readiness result remains readable", stored_next[0]["next"] == "knowledge_representation")

# %% [markdown]
# ## Recap
#
# You queried a multi-hop dependency graph and an aggregation, inspected the three skill contracts, derived their inputs from Cypher, persisted and re-read a readiness verdict, and hand-wrote a three-skill pipeline. The main architectural lesson is that the horizontal workflow should use the vertical graph for grounding, while harness scopes keep each step focused.
#
# **Next:** continue to the advanced lab: [Chapter 2 — Architecture Foundations](../notebooks_advanced_simplified/ch2-architecture-foundations.ipynb). It adds all-skill coverage, an auditable `MiniAgent`, and broader experiments.
