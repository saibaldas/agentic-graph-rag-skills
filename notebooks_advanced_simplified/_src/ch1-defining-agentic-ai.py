# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > [Beginner (tier 1/3)](../notebooks_simplified/ch1-defining-agentic-ai.ipynb)  →  [Intermediate (tier 2/3)](../notebooks_intermediate_simplified/ch1-defining-agentic-ai.ipynb)  →  **Advanced (tier 3/3) ← you are here**

# %% [markdown]
# # Chapter 1 (Advanced, explained) — Defining Agentic AI on a real Neo4j graph
#
# **Tier 3 of 3 (advanced).** You already met the five Chapter 1 skills in `notebooks_simplified/`.
# Here we go deeper: the **graph becomes the single source of truth**. Skill inputs are read *from* Neo4j
# with Cypher, skill outputs are written *back* as nodes and edges, and a tiny **agent** uses the skills
# and Cypher as tools while its every move is stored (and queried) as a graph.
#
# **What you will do**
# 1. Model the Chapter 1 world (services, incidents, symptoms, agent systems, tools) as a graph.
# 2. Use **all five** skills in `skills/crisis/` — failure classifier, spectrum classifier,
#    constraint-triangle scorer, vector-vs-graph selector, enterprise-readiness scorer.
# 3. Run non-trivial Cypher: variable-length paths, aggregation, `OPTIONAL MATCH`, cycle detection, trace queries.
# 4. Run a **plan → act → observe** agent on *"checkout-api is slow"* and read its trace back from Neo4j.
# 5. Call skills through their **CLI + benchmark**, then try **vibe-coding** prompts and **experiments**.
#
# **Why this matters (book mapping):** Chapter 1 argues vector-only RAG fails agents because it sees
# *strings*, not *connected things*. Every section below turns one of that chapter's claims into something
# you can run and assert.
#
# > Prereq: Neo4j running (`docker compose -f docker-compose/docker-compose.yml up -d`). No API keys needed.

# %% [markdown]
# ## 0. Setup — connect, load the incident graph, see the skill catalog
#
# `adv_helpers` (imported as `a`) gives us `a.run(cypher, **params)` (returns list of dicts), `a.show(rows)`,
# `a.load_skill(chapter_folder, skill)`, `a.check(name, condition)` (prints PASS/FAIL), `a.MiniAgent`,
# `a.AgentTrace` and `a.run_benchmark`. The "catalog" below reads each skill's `SKILL.md` front‑matter —
# the same discovery step an AI coding assistant performs before it picks a skill.

# %%
import adv_helpers as a

counts = a.load_devops_graph()          # wipes the DB, then loads the shared DevOps story
print("Loaded:", counts)

catalog = a.skill_catalog("crisis")
for s in catalog:
    print(f"- {s['slug']:<34} {s['description'][:90]}...")
a.check("chapter 1 folder has 5 skills", len(catalog) == 5)

# %% [markdown]
# ## 1. The crisis, as Cypher
#
# Vector RAG finds *similar text*. A graph answers *connected* questions. We prove it with three queries.
# Cypher clauses explained on first use:
#
# * `MATCH` — find a pattern. `(a)-[:DEPENDS_ON]->(b)` = "a depends on b".
# * `*1..3` — **variable-length path**: follow the relationship 1 to 3 hops (this is *multi-hop* reasoning).
# * `p = ...` — name the whole path so we can ask `length(p)` (number of hops).
# * `min(...)` — an **aggregation**; non-aggregated columns in `RETURN` become the grouping key.
#
# **Blast radius** = the set of components hurt when one component fails. Here: if `orders-db` is in trouble,
# who is affected and how many hops away?

# %%
blast = a.run("""
    MATCH p = (affected:Service)-[:DEPENDS_ON*1..3]->(:Database {name:'orders-db'})
    RETURN affected.name AS service, affected.tier AS tier, min(length(p)) AS hops
    ORDER BY hops, service
""")
a.show(blast)
hops = {r["service"]: r["hops"] for r in blast}
a.check("cart-service is 3 hops from orders-db", hops["cart-service"] == 3)
a.check("blast radius is 4 services", len(blast) == 4)

# %% [markdown]
# Why `min(length(p))`? `checkout-api` reaches `orders-db` by *two* routes (via orders and via payment).
# We want the **shortest** distance, so we group by service and take the minimum.
#
# **Second query — aggregation with `collect` / `count`.** `WITH DISTINCT` de-duplicates rows before the next
# stage; `count()` counts per group; `collect()` gathers values into a list. Business question: *which
# criticality tiers are exposed?*

# %%
by_tier = a.run("""
    MATCH (s:Service)-[:DEPENDS_ON*1..3]->(:Database {name:'orders-db'})
    WITH DISTINCT s
    RETURN s.tier AS tier, count(s) AS n_services, collect(s.name) AS services
    ORDER BY n_services DESC, tier
""")
a.show(by_tier)
a.check("critical tier has 2 exposed services",
        next(r for r in by_tier if r["tier"] == "critical")["n_services"] == 2)

# %% [markdown]
# **Third query — `OPTIONAL MATCH` (context amnesia fix).** A plain `MATCH` drops a row if the pattern is
# missing. `OPTIONAL MATCH` keeps the row and fills `null` — essential when only *some* incidents have a
# recorded root cause. The currently open incident has none yet; the June one does.

# %%
history = a.run("""
    MATCH (i:Incident)-[:AFFECTS]->(:Service {name:'checkout-api'})
    OPTIONAL MATCH (i)-[:ROOT_CAUSE]->(cause)
    RETURN i.id AS incident, i.status AS status, cause.name AS root_cause, i.fix AS known_fix
    ORDER BY i.opened_at
""")
a.show(history)
a.check("open incident has no root cause yet (null kept by OPTIONAL MATCH)",
        any(r["incident"] == "INC-1001" and r["root_cause"] is None for r in history))

# %% [markdown]
# **Fourth query — is the dependency graph a DAG?** A **DAG** (directed acyclic graph) has no loops: nobody
# transitively depends on themselves. Many agent plans assume this (otherwise "restart dependencies first"
# never terminates). We *ask the graph*: any path that starts and ends at the same service?

# %%
cycles = a.run("MATCH p = (s:Service)-[:DEPENDS_ON*1..6]->(s) RETURN count(p) AS cycles")[0]["cycles"]
print("dependency cycles found:", cycles)
a.check("DEPENDS_ON is a DAG (0 cycles)", cycles == 0)

# %% [markdown]
# ## 2. Context failure classifier → graph write-back
#
# Chapter 1 names **five fatal flaws** of naive vector RAG (amnesia, relationship blindness, temporal
# ignorance, reasoning paralysis, tool chaos). They compound into agent-level failure modes — Chapter 1's
# "cascade of agent incompetence".
#
# Pipeline: (1) *store* post-mortem symptoms of INC‑1001 as `Symptom` nodes; (2) *read them back* with Cypher;
# (3) classify with the skill; (4) *write* `FailureMode → Flaw → Capability` edges.
# `MERGE` = "create if missing, otherwise reuse" (idempotent, prevents duplicate nodes).

# %%
failure = a.load_skill("crisis", "context-failure-classifier")

postmortem = [
    ("S1", "The agent restarted orders-service unaware it would cascade to checkout-api"),
    ("S2", "Each new session it starts from scratch and forgets the June pool incident"),
    ("S3", "It trusted the old config and reported a stale pool size after the rollback"),
    ("S4", "Could not connect the dots between the alarm and the deploy; no multi-step plan"),
    ("S5", "It called the wrong tool and had to guess which API to use"),
]
a.run("""
    UNWIND $rows AS row
    MATCH (inc:Incident {id:'INC-1001'})
    CREATE (inc)-[:HAS_SYMPTOM]->(:Symptom {id:row[0], text:row[1]})
""", rows=[list(r) for r in postmortem])

# derive classifier input FROM the graph, not from the python list
symptoms = a.run("""
    MATCH (:Incident {id:'INC-1001'})-[:HAS_SYMPTOM]->(s:Symptom)
    RETURN s.id AS id, s.text AS text ORDER BY s.id
""")
batch = failure.classify_batch([s["text"] for s in symptoms])

for s, res in zip(symptoms, batch["results"]):
    p = res["primary"]
    a.run("""
        MATCH (s:Symptom {id:$sid})
        MERGE (fm:FailureMode {name:$mode})
        MERGE (fl:Flaw {name:$flaw})
        MERGE (cap:Capability {name:$cure})
        MERGE (s)-[:CLASSIFIED_AS]->(fm)
        MERGE (fm)-[:ROOT_FLAW]->(fl)
        MERGE (fl)-[:CURED_BY]->(cap)
    """, sid=s["id"], mode=p["failure_mode"], flaw=p["root_flaw"], cure=p["cure"])
    print(f"{s['id']} -> {p['failure_mode']:<22} flaw={p['root_flaw']:<22} cure={p['cure']}")

expected = {"S1": "action_blindness", "S2": "memory_fragmentation", "S3": "context_drift",
            "S4": "planning_paralysis", "S5": "tool_chaos"}
got = {s["id"]: r["primary"]["failure_mode"] for s, r in zip(symptoms, batch["results"])}
a.check("all five symptoms land in the expected failure mode", got == expected)

# %% [markdown]
# **Aggregation over the written-back graph.** A multi-hop path `Symptom → FailureMode → Flaw → Capability`
# lets us ask a *prioritisation* question the skill alone cannot answer in context: which graph capability
# cures the most symptoms? (Here each is 1 — a perfectly spread incident, which itself is a finding:
# INC‑1001 shows *all five* flaws, i.e. the agent needs the whole graph toolkit.)

# %%
cures = a.run("""
    MATCH (:Incident {id:'INC-1001'})-[:HAS_SYMPTOM]->(s)-[:CLASSIFIED_AS]->(fm)-[:ROOT_FLAW]->(fl)-[:CURED_BY]->(cap)
    RETURN cap.name AS graph_capability, count(DISTINCT s) AS symptoms_cured, collect(DISTINCT fl.name) AS flaws
    ORDER BY symptoms_cured DESC, graph_capability
""")
a.show(cures)
a.check("5 distinct capabilities needed", len(cures) == 5)

# %% [markdown]
# **Design note.** The classifier is regex/keyword based (stdlib-only, deterministic, free). In production you
# would swap `classify()` for an LLM classifier with the *same return shape* — the `TODO` seam in `lib.py`.
# Trade-off: rules are explainable and repeatable but brittle on unusual wording.

# %% [markdown]
# ## 3. Workflow ↔ agent spectrum (three dials)
#
# Agency has three dials from 0 to 1: **autonomy** (decides alone), **action** (can change the world — a
# *gate*: no action means advisor, not agent) and **authority** (what it is *allowed* to do).
# We store four systems as `AgentSystem` nodes, read them back, classify, and link each to a `Band` node.
# `UNWIND $rows AS row` turns a list parameter into one row per item; `SET s = row` copies all keys as properties.

# %%
spectrum = a.load_skill("crisis", "workflow-agent-spectrum-classifier")

systems_seed = [
    dict(name="scripted-latency-runbook", autonomy=0.1, action=0.2, authority=0.2,
         path_determinism=0.95, memory=False, tool_use=True, contextual=False),
    dict(name="human-in-loop-remediation", autonomy=0.5, action=0.6, authority=0.4,
         path_determinism=0.55, memory=True, tool_use=True, contextual=True),
    dict(name="autonomous-devops-agent", autonomy=0.9, action=0.8, authority=0.5,
         path_determinism=0.15, memory=True, tool_use=True, contextual=True),
    dict(name="latency-advisor (read-only)", autonomy=0.85, action=0.05, authority=0.05,
         path_determinism=0.2, memory=True, tool_use=True, contextual=True),
]
a.run("UNWIND $rows AS row CREATE (s:AgentSystem) SET s = row", rows=systems_seed)

placed = {}
for rec in a.run("MATCH (s:AgentSystem) RETURN s{.*} AS p ORDER BY s.name"):
    r = spectrum.classify(**rec["p"])
    placed[r["name"]] = r
    a.run("""
        MATCH (s:AgentSystem {name:$n})
        MERGE (b:Band {name:$band})
        MERGE (s)-[:PLACED_IN {position:$pos, is_agent:$agent}]->(b)
    """, n=r["name"], band=r["band"], pos=r["spectrum_position"], agent=r["is_agent_by_action_test"])

a.show(a.run("""
    MATCH (s:AgentSystem)-[p:PLACED_IN]->(b:Band)
    RETURN b.name AS band, count(s) AS systems, collect(s.name) AS names, round(avg(p.position), 2) AS avg_position
    ORDER BY avg_position
"""))
a.check("runbook is a WORKFLOW", placed["scripted-latency-runbook"]["band"] == "WORKFLOW")
a.check("human-in-loop is BLENDED", placed["human-in-loop-remediation"]["band"] == "BLENDED")
a.check("autonomous agent is AGENT", placed["autonomous-devops-agent"]["band"] == "AGENT")
a.check("read-only advisor fails the action test", placed["latency-advisor (read-only)"]["is_agent_by_action_test"] is False)

# %% [markdown]
# **Read the trap:** the advisor lands in the `AGENT` *band* (high autonomy, dynamic path) yet fails the
# *action gate* (`action 0.05 < 0.15`). Band and gate answer different questions — never collapse them into one
# number. Also see `notes` below: high autonomy with low authority is a *calibration* concern, not a bug.

# %%
for note in placed["latency-advisor (read-only)"]["notes"]:
    print("-", note)

# %% [markdown]
# ## 4. The agent constraint triangle — scored from live graph data
#
# Three pressures feed each other: **complexity** (more steps → compounding errors), **tool orchestration**
# (more/ambiguous tools → wrong calls) and **context utilisation** (fuller window → *context rot*, falling recall).
# Chapter 1's principle: the smallest set of high-signal tokens that still succeeds.
#
# Advanced twist: our *own* mini agent is a config too. We model its tools as `Tool` nodes with `FEEDS`
# edges (output of one tool is input to the next), then **derive `tool_count` with Cypher** (`count()`),
# not by typing a number.

# %%
triangle = a.load_skill("crisis", "agent-constraint-triangle-scorer")

tool_names = ["find_suspects", "blast_radius", "classify_symptoms", "select_retrieval",
              "score_triangle", "assess_readiness", "record_conclusion"]
a.run("UNWIND $n AS name CREATE (:Tool {name:name})", n=tool_names)
feeds = list(zip(tool_names, tool_names[1:]))
a.run("""
    UNWIND $edges AS e
    MATCH (x:Tool {name:e[0]}), (y:Tool {name:e[1]})
    CREATE (x)-[:FEEDS]->(y)
""", edges=[list(e) for e in feeds])

a.run("""
    UNWIND $rows AS row CREATE (c:AgentConfig) SET c = row
""", rows=[
    dict(name="balanced-graph-agent", avg_task_steps=5, tool_count=12, tools_disambiguable=True,
         context_window_tokens=200_000, avg_context_tokens_used=60_000),
    dict(name="bloated-mcp-agent", avg_task_steps=8, tool_count=36, tools_disambiguable=False,
         context_window_tokens=200_000, avg_context_tokens_used=178_000),
])
a.run("""
    CREATE (c:AgentConfig {name:'graph-devops-agent', avg_task_steps:7, tools_disambiguable:true,
                           context_window_tokens:200000, avg_context_tokens_used:40000})
    WITH c
    MATCH (t:Tool)
    WITH c, collect(t) AS tools
    SET c.tool_count = size(tools)
    FOREACH (t IN tools | CREATE (c)-[:USES]->(t))
""")

for rec in a.run("MATCH (c:AgentConfig) RETURN c{.*} AS cfg ORDER BY c.name"):
    cfg = rec["cfg"]
    r = triangle.score(cfg)
    a.run("MATCH (c:AgentConfig {name:$n}) SET c.overall = $o, c.dominant = $d",
          n=cfg["name"], o=r["overall_band"], d=r["dominant_constraint"])
    print(f"{cfg['name']:<22} tools={cfg['tool_count']:<3} -> {r['overall_band']:<16} worst corner: {r['dominant_constraint']}")

bands = {r["name"]: r["overall"] for r in a.run("MATCH (c:AgentConfig) RETURN c.name AS name, c.overall AS overall")}
a.check("balanced agent is BALANCED", bands["balanced-graph-agent"] == "BALANCED")
a.check("bloated agent is OVERCONSTRAINED", bands["bloated-mcp-agent"] == "OVERCONSTRAINED")
a.check("our 7-tool agent derived tool_count from the graph",
        a.run("MATCH (c:AgentConfig {name:'graph-devops-agent'}) RETURN c.tool_count AS n")[0]["n"] == 7)

# %% [markdown]
# **How the skill thinks (trade-offs):** the curves are *heuristics* that encode the chapter's qualitative
# claims (saturating exponential for steps, ambiguity penalty for tools, fill-ratio^0.7 for context rot) —
# not measured benchmarks. They are a conversation starter: `recommendations` tells you which corner to
# shrink first. Inspect the bloated agent's active pressure cycles:

# %%
bloated = triangle.score(a.run("MATCH (c:AgentConfig {name:'bloated-mcp-agent'}) RETURN c{.*} AS c")[0]["c"])
for cyc in bloated["active_pressure_cycles"]:
    print(f"* {cyc['edge']}: {cyc['cascade'][:100]}...")
print("\nFirst fix:", bloated["recommendations"][0]["action"][:110], "...")

# %% [markdown]
# ## 5. Vector, graph or hybrid? — selector + RRF
#
# `vector-vs-graph-retrieval-selector` maps a workload to **VECTOR**, **GRAPH** or **HYBRID** using the
# Microsoft BenchmarkQED quadrants (local/global × data/activity). Workloads live in the graph as
# `Workload` nodes; each recommendation becomes a `RECOMMENDED` edge to a `Strategy` node.

# %%
retrieval = a.load_skill("crisis", "vector-vs-graph-retrieval-selector")

a.run("UNWIND $rows AS row CREATE (w:Workload) SET w = row", rows=[
    dict(name="specific-5xx-lookup", query_scope="local", query_type="data", multi_hop=False,
         temporal=False, structured_domain=False, latency_critical=True),
    dict(name="cascading-migration-impact", query_scope="global", query_type="activity",
         multi_hop=True, temporal=True, structured_domain=True),
    dict(name="autonomous-devops-agent", query_scope="mixed", query_type="activity", multi_hop=True,
         temporal=True, structured_domain=True, agentic=True, dataset_scale_pages=120000),
    dict(name="unstructured-global-sensemaking", query_scope="global", query_type="activity",
         multi_hop=True, structured_domain=False),
])
for rec in a.run("MATCH (w:Workload) RETURN w{.*} AS w ORDER BY w.name"):
    r = retrieval.recommend(**rec["w"])
    a.run("""
        MATCH (w:Workload {name:$n})
        MERGE (s:Strategy {name:$s})
        MERGE (w)-[:RECOMMENDED]->(s)
    """, n=rec["w"]["name"], s=r["recommendation"])

a.show(a.run("""
    MATCH (w:Workload)-[:RECOMMENDED]->(s:Strategy)
    RETURN w.name AS workload, s.name AS strategy ORDER BY strategy, workload
"""))
rec_map = {r["w"]: r["s"] for r in a.run(
    "MATCH (w:Workload)-[:RECOMMENDED]->(s:Strategy) RETURN w.name AS w, s.name AS s")}
a.check("5xx lookup -> VECTOR", rec_map["specific-5xx-lookup"] == "VECTOR")
a.check("migration impact -> GRAPH", rec_map["cascading-migration-impact"] == "GRAPH")
a.check("agent workload -> HYBRID", rec_map["autonomous-devops-agent"] == "HYBRID")
a.check("unstructured global -> HYBRID (graph costly)", rec_map["unstructured-global-sensemaking"] == "HYBRID")

# %% [markdown]
# **The "bigger context window" objection** — the skill ships the book's rebuttal, and a scale note when the
# corpus reaches 100k pages:

# %%
r = retrieval.recommend(query_scope="global", query_type="activity", multi_hop=True,
                        larger_context_window=True, dataset_scale_pages=120000)
print(r["larger_context_window_rebuttal"][:260], "...\n")
print(r["scale_note"])
print("GraphRAG costs:", *r.get("graphrag_costs", []), sep="\n  - ")

# %% [markdown]
# **How does HYBRID actually merge two result lists?** A common method is **RRF — Reciprocal Rank Fusion**:
# each list gives an item the score `1 / (k + rank)` (k≈60 dampens the top ranks); scores are summed across lists.
# Items that *both* retrievers like float to the top, with no need to compare incompatible raw scores
# (cosine similarity vs graph hops). The vector list below is a stand-in for an embedding search
# (production seam: your vector index); the graph list comes from Cypher.

# %%
def rrf(*rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])

vector_hits = ["orders-service", "session-cache", "checkout-api", "payment-service"]  # stand-in ANN result
graph_hits = [r["service"] for r in blast]                                           # Cypher result from section 1
fused = rrf(vector_hits, graph_hits)
for item, sc in fused:
    print(f"{item:<16} {sc:.4f}  vector={item in vector_hits}  graph={item in graph_hits}")
a.check("RRF puts the item both retrievers agree on first", fused[0][0] == "orders-service")
a.check("graph-only cart-service survives fusion", "cart-service" in dict(fused))

# %% [markdown]
# ## 6. Enterprise readiness — a graph-probed score, before and after
#
# `enterprise-readiness-scorer` normally trusts a *self-reported* profile. Its own `TODO` says production
# should **probe the substrate instead**. We do exactly that: five Cypher probes check whether the graph
# really provides each curing capability. It also applies **Marple's decision-trace test**: can the system say
# not only *what* happened but which *alternatives were considered and rejected*? (that is what separates a
# context graph from a relabelled search index.)
#
# Each probe is `RETURN count(...) > 0 AS ok` — an aggregate always returns one row, even with no matches.

# %%
readiness = a.load_skill("crisis", "enterprise-readiness-scorer")

PROBES = {
    "entity_relationships": "MATCH ()-[r]->() RETURN count(DISTINCT type(r)) >= 4 AS ok",
    "temporal_evolution": "MATCH (d:Deployment) WHERE d.at IS NOT NULL RETURN count(d) > 0 AS ok",
    "evolving_memory": "MATCH (i:Incident {status:'resolved'}) WHERE i.root_cause IS NOT NULL RETURN count(i) > 0 AS ok",
    "multi_hop_reasoning": "MATCH p = (:Service)-[:DEPENDS_ON*2..]->() RETURN count(p) > 0 AS ok",
    "tool_orchestration": "MATCH (:Tool)-[:FEEDS]->(:Tool) RETURN count(*) > 0 AS ok",
}

def probe_graph() -> dict:
    """Build an enterprise-readiness profile from what the graph actually contains."""
    ok = lambda q: bool(a.run(q)[0]["ok"])
    caps = {name: ok(q) for name, q in PROBES.items()}
    return {
        "graph_capabilities": caps,
        "agency": {"autonomy": 0.6, "action": 0.5, "authority": 0.3},   # deliberately calibrated, self-reported
        "capabilities": {
            "autonomous_decision_making": True,
            "contextual_understanding": caps["entity_relationships"],
            "strategic_tool_utilization": caps["tool_orchestration"],
            "memory_persistence": ok("MATCH (r:Run) RETURN count(r) > 0 AS ok"),   # persisted agent runs
        },
        "captures_rejected_alternatives": ok("MATCH (:ToolCall)-[:REJECTED]->(:Alternative) RETURN count(*) > 0 AS ok"),
    }

def rate(name: str, profile: dict) -> dict:
    """Assess a profile and (re)write its rating into the graph."""
    r = readiness.assess(profile)
    a.run("""
        MERGE (p:Profile {name:$n})
        WITH p
        OPTIONAL MATCH (p)-[old:RATED]->()
        DELETE old
        WITH DISTINCT p
        SET p.score = $score
        MERGE (b:Readiness {band:$band})
        MERGE (p)-[:RATED]->(b)
    """, n=name, score=r["score"], band=r["band"])
    return r

naive_profile = {
    "graph_capabilities": {k: False for k in PROBES},
    "agency": {"autonomy": 0.6, "action": 0.5, "authority": 0.3},
    "capabilities": {"autonomous_decision_making": True, "contextual_understanding": False,
                     "strategic_tool_utilization": False, "memory_persistence": False},
    "captures_rejected_alternatives": False,
}
naive = rate("naive-vector-bot", naive_profile)
before = rate("graph-devops-agent", probe_graph())
print(f"naive-vector-bot   : {naive['score']:>5} {naive['band']}")
print(f"graph-devops-agent : {before['score']:>5} {before['band']}   (before running the agent)")
for rec in before["recommendations"]:
    print("  gap ->", rec[:100])
a.check("naive bot is NAIVE-VECTOR", naive["band"] == "NAIVE-VECTOR")
a.check("graph agent is PILOT-READY before any run", before["band"] == "PILOT-READY")

# %% [markdown]
# The remaining gaps are **memory persistence** (no stored runs yet) and **decision traces** (no rejected
# alternatives recorded). Both are things an *agent run* can create — which is the next section.

# %% [markdown]
# ## 7. A mini agent: plan → act → observe on *"checkout-api is slow"*
#
# An agent is a loop: **plan** the next action, **act** (call a tool), **observe** the result, repeat.
# `a.MiniAgent` is that loop with deterministic tools — our five skills plus Cypher. A real agent would replace
# `next_step` with an LLM; everything else (tools, trace, graph) stays identical. That is the book's point:
# the graph + tools are the substrate, the planner is swappable.
#
# Every tool call is persisted by `AgentTrace` as `(:Run)-[:STEP]->(:ToolCall)-[:NEXT]->(:ToolCall)`. We also
# record, for each step, one **rejected alternative** (`REJECTED` edge) — the decision-trace test from section 6.

# %%
agent = a.MiniAgent("Find the likely cause of INC-1001 (checkout-api is slow) and what to do")

@agent.tool("find_suspects")
def find_suspects(incident_id):
    # affected service -> things it depends on (1-2 hops) -> deployments that changed them -> similar past incidents
    return a.run("""
        MATCH (inc:Incident {id:$iid})-[:AFFECTS]->(svc:Service)
        MATCH (svc)-[:DEPENDS_ON*1..2]->(dep)
        MATCH (d:Deployment)-[:CHANGED]->(dep)
        OPTIONAL MATCH (past:Incident {status:'resolved'})-[:ROOT_CAUSE]->(dep)
        RETURN DISTINCT dep.name AS suspect, d.id AS deploy, d.change AS change,
               past.id AS past_incident, past.fix AS past_fix
    """, iid=incident_id)

@agent.tool("blast_radius")
def blast_radius(component):
    return a.run("""
        MATCH p = (s:Service)-[:DEPENDS_ON*1..3]->({name:$c})
        RETURN s.name AS service, min(length(p)) AS hops ORDER BY hops, service
    """, c=component)

@agent.tool("classify_symptoms")
def classify_symptoms(incident_id):
    rows = a.run("MATCH (:Incident {id:$i})-[:HAS_SYMPTOM]->(s) RETURN s.text AS t ORDER BY s.id", i=incident_id)
    return failure.classify_batch([r["t"] for r in rows])["prioritized_cures"]

@agent.tool("select_retrieval")
def select_retrieval(workload):
    w = a.run("MATCH (w:Workload {name:$n}) RETURN w{.*} AS w", n=workload)[0]["w"]
    return retrieval.recommend(**w)["recommendation"]

@agent.tool("score_triangle")
def score_triangle(config):
    c = a.run("MATCH (c:AgentConfig {name:$n}) RETURN c{.*} AS c", n=config)[0]["c"]
    return triangle.score(c)["overall_band"]

@agent.tool("assess_readiness")
def assess_readiness():
    r = rate("graph-devops-agent", probe_graph())
    return f"{r['score']} {r['band']}"

@agent.tool("record_conclusion")
def record_conclusion(component, deploy, past_incident):
    a.run("""
        MATCH (i:Incident {id:'INC-1001'}), (c {name:$c}), (d:Deployment {id:$d})
        MERGE (i)-[l:LIKELY_CAUSE]->(c)
        SET l.deploy = $d, l.run = $run, l.change = d.change
        WITH i
        OPTIONAL MATCH (p:Incident {id:$p})
        FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | MERGE (i)-[:SIMILAR_TO]->(p))
    """, c=component, d=deploy, p=past_incident, run=agent.trace.run_id)
    return f"INC-1001 -[LIKELY_CAUSE]-> {component} (via {deploy})"

a.check("agent tools == Tool nodes in the graph",
        set(agent.tools) == {r["n"] for r in a.run("MATCH (t:Tool) RETURN t.name AS n")})

# %% [markdown]
# **The planner.** A tiny rule-based policy: pick the first tool not yet used, fill its arguments from what
# earlier tools *observed* (e.g. the suspect found by `find_suspects` feeds `blast_radius`), and stop early if
# no suspect exists. It also names one alternative it chose *not* to run now.

# %%
ORDER = ["find_suspects", "blast_radius", "classify_symptoms", "select_retrieval",
         "score_triangle", "assess_readiness", "record_conclusion"]

def next_step(mem):
    if mem.get("find_suspects") == []:
        return None                                    # no deploy suspect -> nothing more to conclude
    remaining = [t for t in ORDER if t not in mem]
    if not remaining:
        return None
    tool = remaining[0]
    top = (mem.get("find_suspects") or [{}])[0]
    args = {
        "find_suspects": {"incident_id": "INC-1001"},
        "blast_radius": {"component": top.get("suspect")},
        "classify_symptoms": {"incident_id": "INC-1001"},
        "select_retrieval": {"workload": "autonomous-devops-agent"},
        "score_triangle": {"config": "graph-devops-agent"},
        "assess_readiness": {},
        "record_conclusion": {"component": top.get("suspect"), "deploy": top.get("deploy"),
                              "past_incident": top.get("past_incident")},
    }[tool]
    rejected = (remaining[1], f"needs the output of {tool} first (FEEDS order)") if len(remaining) > 1 else None
    return tool, args, rejected

step = 0
while (nxt := next_step(agent.memory)) is not None and step < 10:
    tool, args, rejected = nxt
    step += 1
    print(f"PLAN    {step}: call {tool}({args})")
    obs = agent.act(tool, **args)                      # ACT (and the trace logs it)
    if rejected:
        a.run("MATCH (c:ToolCall {id:$cid}) CREATE (c)-[:REJECTED {reason:$why}]->(:Alternative {tool:$t})",
              cid=agent.trace.prev, why=rejected[1], t=rejected[0])
    print(f"OBSERVE {step}: {str(obs)[:120]}\n")      # OBSERVE

a.check("agent finished all 7 steps", step == 7)

# %% [markdown]
# **Reading the answer.** The agent found `orders-db` as the suspect (deploy‑481 shrank its pool 100→20),
# measured the blast radius, matched the June incident `INC-0877`, and wrote its conclusion *into the graph*.
# Now query the **trace itself** — the run is data, not a log file. Three clauses are new here:
# `NOT ()-[:NEXT]->(x)` is a *negative pattern* (x has no incoming NEXT = first step), `[:NEXT*]` follows the
# chain, and `[n IN nodes(p) | n.tool]` is a **list comprehension** projecting each node on the path.

# %%
tid = agent.trace.run_id
a.show(agent.trace.timeline())

seq = a.run("""
    MATCH (:Run {id:$id})-[:STEP]->(first:ToolCall) WHERE NOT ()-[:NEXT]->(first)
    MATCH p = (first)-[:NEXT*]->(last:ToolCall) WHERE NOT (last)-[:NEXT]->()
    RETURN length(p) + 1 AS steps, [n IN nodes(p) | n.tool] AS tool_sequence
""", id=tid)[0]
print("\nPath through the trace:", seq["steps"], "steps ->", " > ".join(seq["tool_sequence"]))

alts = a.run("""
    MATCH (:Run {id:$id})-[:STEP]->(c:ToolCall)
    OPTIONAL MATCH (c)-[rj:REJECTED]->(alt:Alternative)
    RETURN c.seq AS seq, c.tool AS chosen, alt.tool AS rejected, rj.reason AS why ORDER BY c.seq
""", id=tid)
a.show(alts)
a.check("trace has 7 steps in one NEXT chain", seq["steps"] == 7)
a.check("6 rejected alternatives recorded (last step has none)", sum(1 for r in alts if r["rejected"]) == 6)
a.check("every tool call succeeded", all(r["ok"] for r in agent.trace.timeline()))

# %% [markdown]
# **Cross-incident memory query.** The conclusion edges let *future* runs ask: "what did we conclude before,
# and from which run?" — closing the context-amnesia loop from section 1.

# %%
verdict = a.run("""
    MATCH (i:Incident {id:'INC-1001'})-[l:LIKELY_CAUSE]->(c)
    OPTIONAL MATCH (i)-[:SIMILAR_TO]->(p:Incident)
    RETURN i.id AS incident, c.name AS likely_cause, l.change AS change, p.id AS similar_to, p.fix AS suggested_fix, l.run AS run
""")
a.show(verdict)
a.check("conclusion: orders-db via the pool change, similar to INC-0877",
        verdict[0]["likely_cause"] == "orders-db" and verdict[0]["similar_to"] == "INC-0877")

after = rate("graph-devops-agent", probe_graph())
print(f"\nreadiness before={before['score']} {before['band']}  ->  after the run={after['score']} {after['band']}")
a.check("running the agent (persisted trace + rejected alternatives) raises readiness to PRODUCTION-READY",
        after["band"] == "PRODUCTION-READY" and after["score"] > before["score"])

# %% [markdown]
# ## 8. Skill CLI & benchmark
#
# Each skill is *also* a command-line tool, so it runs from cron, CI or a shell with zero installs.
# Convention: `cli.py --help` prints the SKILL.md description; `cli.py benchmark` runs the skill's own
# verification gate (exit code 0 = pass). `a.run_benchmark(chapter, slug)` wraps that in a subprocess.

# %%
bench = {}
for s in catalog:
    ok, first_line = a.run_benchmark("crisis", s["slug"])
    bench[s["slug"]] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] {s['slug']:<34} {first_line[:70]}")
a.check("all 5 chapter-1 benchmarks pass", all(bench.values()) and len(bench) == 5)

# %% [markdown]
# **One skill through its CLI**, using `subprocess` exactly as a shell script or CI job would.

# %%
import os, subprocess, sys

cli = a.SKILLS_DIR / "crisis" / "context-failure-classifier" / "cli.py"
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}     # Windows consoles default to cp1252
kw = dict(capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
helped = subprocess.run([sys.executable, str(cli), "--help"], **kw)
print(helped.stdout[:420], "...\n")
called = subprocess.run([sys.executable, str(cli), "classify", "--symptom", postmortem[0][1]], **kw)
print("\n".join(called.stdout.splitlines()[:8]))
a.check("cli --help exits 0 and prints a description", helped.returncode == 0 and len(helped.stdout) > 100)
a.check("cli classify exits 0 and mentions action_blindness",
        called.returncode == 0 and "action_blindness" in called.stdout)

# %% [markdown]
# ## 9. 🎯 Vibe-coding corner
#
# "Vibe coding" = describe the outcome to an AI coding assistant (Copilot, Claude Code) and steer by
# *verifying*, not by typing every line. The quality of the result tracks the quality of your prompt.
#
# **A good prompt has four parts:** (1) **context** — files/anatomy to imitate, (2) **task** — one concrete
# change, (3) **constraints** — stdlib-only, no hard-coded data, parameterised Cypher, (4) **acceptance check**
# — a command or assert that proves it works. Then *you* run that check; never trust "it should work".
#
# ### Prompt A — a new Cypher query
# ```text
# Context: Neo4j holds (:Service)-[:DEPENDS_ON]->(:Service|:Database), (:Deployment)-[:CHANGED]->(x) and
# (:Incident)-[:AFFECTS]->(:Service). Task: write a Cypher query returning, for each Service, how many
# distinct deployments changed anything within 2 DEPENDS_ON hops of it, plus collect() of those deploy ids.
# Use OPTIONAL MATCH so services with zero are included. Constraints: read-only, no property is hard-coded
# except labels, return columns service, n_deploys, deploys. Acceptance: checkout-api must show n_deploys=1
# (deploy-481) and cart-service must be present. Explain each clause in a comment.
# ```
# **Verify:** run it through `a.run(...)` and `a.check(...)` the two acceptance facts.
#
# ### Prompt B — a new skill, repo-conformant
# ```text
# Context: follow skills/crisis/context-failure-classifier/ exactly: SKILL.md with the 7 sections + Source
# Attribution, lib.py (stdlib-only, TODO markers at production swap seams), cli.py (argparse; --help prints
# the SKILL.md description; a `benchmark` subcommand running self-checks, exit 0 on pass).
# Task: create skills/crisis/blast-radius-ranker/ that ranks services by (tier weight x 1/hops) given rows
# [{service, tier, hops}]. Constraints: no pip packages, deterministic, no network. Acceptance:
# `python skills/crisis/blast-radius-ranker/cli.py --help` exits 0 and `... benchmark` exits 0.
# ```
# **Verify:** `a.run_benchmark("crisis", "blast-radius-ranker")` returns `(True, ...)`; also run the repo's CI checks.
#
# ### Prompt C — extend the agent
# ```text
# Context: notebook cell defining next_step(mem) for a.MiniAgent. Task: add a branch so that if blast_radius
# returns any tier='critical' service, the agent inserts an extra tool `page_oncall` BEFORE record_conclusion
# and logs it in the trace. Constraints: keep the REJECTED-alternative logging; add the Tool node + FEEDS edge
# in Cypher. Acceptance: the trace query shows 8 steps and readiness stays PRODUCTION-READY.
# ```
#
# ### Worked example — a helper "as if vibe-coded", plus its test
# Prompt used: *"Write `affected_by(component, max_hops=3)` returning a dict {service: hops} of services that
# transitively DEPEND_ON the component, via Cypher. Validate inputs. Acceptance tests: orders-db with 3 hops
# gives 4 services, with 1 hop gives orders-service and payment-service; bad hops raise ValueError."*
# Note the one subtle design call a reviewer should check: Cypher **cannot parameterise a path length**
# (`*1..$n` is invalid), so the number must be spliced into the string — which is only safe after validating
# it is an `int` in a small range. Reviewing for exactly this kind of injection risk is the human's job.

# %%
def affected_by(component: str, max_hops: int = 3) -> dict:
    """Return {service_name: shortest_hops} of services that transitively depend on `component`."""
    if not isinstance(max_hops, int) or isinstance(max_hops, bool) or not 1 <= max_hops <= 6:
        raise ValueError("max_hops must be an int between 1 and 6")
    rows = a.run(
        f"""
        MATCH p = (s:Service)-[:DEPENDS_ON*1..{max_hops}]->(c {{name:$name}})
        RETURN s.name AS service, min(length(p)) AS hops
        """,
        name=component,
    )
    return {r["service"]: r["hops"] for r in rows}

# the test that checks it (this is what you run after the assistant hands code back)
assert set(affected_by("orders-db", 3)) == {"orders-service", "payment-service", "checkout-api", "cart-service"}
assert set(affected_by("orders-db", 1)) == {"orders-service", "payment-service"}
assert affected_by("orders-db")["cart-service"] == 3
assert affected_by("no-such-component") == {}
for bad in (0, 7, "3", 2.5, True, "3]->() DETACH DELETE s //"):
    try:
        affected_by("orders-db", bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted bad max_hops={bad!r}")
a.check("vibe-coded affected_by passes its tests (incl. injection attempts rejected)", True)

# %% [markdown]
# ## 10. Experiments — change a parameter, predict, then run
#
# **How to do these:** write your prediction in a comment *before* running; then edit the value and re-run.
#
# **Experiment 1 — tool bloat.** Predict the tool-orchestration pressure at 5, 12, 24, 36 and 60 tools when the
# tools are *ambiguous* (`tools_disambiguable=False`). Is it linear? Where does it saturate? Then flip the flag
# to `True` and predict again.

# %%
DISAMBIGUABLE = False        # <- change me
sweep = {n: triangle.score_tool_orchestration(n, DISAMBIGUABLE) for n in (5, 12, 24, 36, 60)}
print(sweep)
vals = list(sweep.values())
a.check("pressure never decreases as tools are added", vals == sorted(vals))

# %% [markdown]
# **Experiment 2 — hop budget.** How many services does each hop limit reveal around `orders-db`?
# Predict the counts for 1, 2, 3 hops. (Hint: think about what `cart-service` needs.)

# %%
MAX_HOPS = 3                 # <- change me (1..6)
growth = {h: len(affected_by("orders-db", h)) for h in range(1, MAX_HOPS + 1)}
print(growth)
a.check("blast radius grows 2 -> 3 -> 4 with the hop budget", [growth[h] for h in (1, 2, 3) if h in growth] == [2, 3, 4][:len(growth)])

# %% [markdown]
# **Experiment 3 — the action gate.** Hold autonomy at 0.85 and sweep `action` across the 0.15 threshold.
# Predict which values make a system an *agent* and whether the **band** changes at all.

# %%
ACTIONS = (0.05, 0.14, 0.15, 0.5)    # <- change me
for act in ACTIONS:
    r = spectrum.classify(autonomy=0.85, action=act, authority=0.4, path_determinism=0.2)
    print(f"action={act:<5} band={r['band']:<6} is_agent={r['is_agent_by_action_test']}")
gate = {act: spectrum.classify(autonomy=0.85, action=act, authority=0.4, path_determinism=0.2)["is_agent_by_action_test"]
        for act in (0.14, 0.15)}
a.check("gate flips exactly at action=0.15, band unchanged", gate == {0.14: False, 0.15: True})

# %% [markdown]
# ## 11. Common pitfalls
#
# * **Hard-coding what the graph already knows.** If a skill input is typed by hand, the notebook can drift from
#   the data. Derive inputs with Cypher (we did `tool_count`, systems, workloads, symptoms).
# * **`MATCH` vs `OPTIONAL MATCH`.** A plain `MATCH` silently *removes* rows with no match — your "open incident"
#   disappears from the report. Use `OPTIONAL MATCH` when absence is meaningful.
# * **Unbounded variable-length paths** (`*` with no upper bound) can explode on dense graphs. Cap them (`*1..3`)
#   and use `DISTINCT` / `min(length(p))` so duplicate routes don't inflate counts.
# * **Cypher cannot parameterise path length or labels** — validate then format (see `affected_by`), never
#   concatenate raw user text.
# * **`CREATE` instead of `MERGE`** duplicates nodes on re-runs. Use `MERGE` for shared entities (`FailureMode`, `Band`).
# * **Band ≠ gate.** An advisor can sit in the AGENT band and still fail the action test.
# * **Heuristic scores are not benchmarks.** The triangle/readiness numbers encode the book's reasoning; cite
#   them as a rubric, not as measured accuracy. Only the BenchmarkQED figures quoted by the retrieval skill
#   come from the book.
# * **Rule-based classifiers are brittle** — if `classify()` returns `classified: False`, rephrase the symptom
#   as *behaviour* ("it restarted X unaware...") not infrastructure.

# %% [markdown]
# ## 12. Verification battery
#
# One last pass that re-reads the graph (read-only) and asserts every chapter claim end-to-end.

# %%
def one(q, **p):
    return a.run(q, **p)[0]["n"]

a.check("5 skills discovered, all benchmarks green", len(catalog) == 5 and all(bench.values()))
a.check("graph has Service, Incident, Symptom, Tool, Run node types",
        all(one(f"MATCH (x:{l}) RETURN count(x) AS n") > 0 for l in ("Service", "Incident", "Symptom", "Tool", "Run")))
a.check("dependency graph is acyclic", one("MATCH p=(s:Service)-[:DEPENDS_ON*1..6]->(s) RETURN count(p) AS n") == 0)
a.check("5 symptoms classified into 5 distinct failure modes",
        one("MATCH (:Symptom)-[:CLASSIFIED_AS]->(f) RETURN count(DISTINCT f) AS n") == 5)
a.check("4 systems placed on the spectrum", one("MATCH (:AgentSystem)-[:PLACED_IN]->() RETURN count(*) AS n") == 4)
a.check("3 agent configs scored", one("MATCH (c:AgentConfig) WHERE c.overall IS NOT NULL RETURN count(c) AS n") == 3)
a.check("4 workloads have a recommended strategy", one("MATCH (:Workload)-[:RECOMMENDED]->() RETURN count(*) AS n") == 4)
a.check("exactly one agent run persisted with 7 tool calls",
        one("MATCH (:Run)-[:STEP]->(c) RETURN count(c) AS n") == 7)
a.check("agent concluded orders-db is the likely cause",
        one("MATCH (:Incident {id:'INC-1001'})-[:LIKELY_CAUSE]->(c {name:'orders-db'}) RETURN count(c) AS n") == 1)
a.check("final readiness is PRODUCTION-READY",
        a.run("MATCH (:Profile {name:'graph-devops-agent'})-[:RATED]->(b) RETURN b.band AS n")[0]["n"] == "PRODUCTION-READY")
a.check("vector-vs-graph RRF fusion ranks the shared hit first", fused[0][0] == "orders-service")
print("\nAll verification checks passed.")

# %% [markdown]
# ## Recap
# * A graph answers *connected* questions (multi-hop, aggregation, history) that vector search cannot —
#   we used variable-length paths, `OPTIONAL MATCH`, `collect/count`, cycle detection and trace queries.
# * All **5 Chapter 1 skills** were driven by data read from Neo4j and their outputs written back as edges.
# * Agency is three dials plus an **action gate**; the constraint triangle explains why agents overload;
#   VECTOR / GRAPH / HYBRID (merged by **RRF**) is a decision, not a religion.
# * An agent is a plan→act→observe loop whose tools are skills + Cypher; its **trace is itself a graph**, and
#   recording rejected alternatives is what turns a search index into a context graph.
# * Skills are CLIs with benchmarks; vibe-coding works when you supply context, constraints and an acceptance check.
#
# **Next:** `ch2-architecture-foundations.ipynb` in this folder — the architecture that makes this graph-plus-agent pattern production-grade.
