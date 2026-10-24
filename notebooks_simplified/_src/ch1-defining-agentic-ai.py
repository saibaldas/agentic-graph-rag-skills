# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > **Beginner (tier 1/3) ← you are here**  →  [Intermediate (tier 2/3)](../notebooks_intermediate_simplified/ch1-defining-agentic-ai.ipynb)  →  [Advanced (tier 3/3)](../notebooks_advanced_simplified/ch1-defining-agentic-ai.ipynb)

# %% [markdown]
# # Chapter 1 (Simplified) — Defining Agentic AI, with a real Neo4j graph
#
# **Who this is for:** you are new to GraphRAG and agents. Every step is explained in plain language.
#
# **What you will learn**
# 1. Why a plain "vector search + LLM" agent breaks down in real companies.
# 2. What an *agent* actually is (three dials: autonomy, action, authority).
# 3. The **constraint triangle** — why agents are hard to build.
# 4. When to use **vector search**, a **graph**, or **both**.
#
# **The story used in the whole repo:** it is 08:42 and your pager goes off —
# *"the checkout API is slow"*. We will use a small **real graph in Neo4j** (a graph database)
# to see why a graph helps an agent solve this.
#
# **How the repo is organised:** each idea from the book is a small "skill" under `skills/`.
# A skill is just a Python file (`lib.py`) with one function. We *import and call* those
# functions here, so you see the real repo code running.
#
# > **Before running:** start Neo4j once with
# > `docker compose -f docker-compose/docker-compose.yml up -d`
# > then open this notebook and run cells top to bottom (Shift+Enter).

# %% [markdown]
# ## 0. Setup: connect to Neo4j and load the incident graph
#
# **What is a graph database?** Instead of tables, data is stored as
# **nodes** (things: a service, a database, an incident) and **relationships**
# (how things connect: `checkout-api -[:DEPENDS_ON]-> orders-service`).
# You ask questions with **Cypher**, a query language that looks like ASCII art:
# `(a)-[:DEPENDS_ON]->(b)` means "a depends on b".
#
# `graph_helpers.py` (next to this notebook) hides the boring plumbing:
# * `connect()` – opens the connection (default `bolt://localhost:7687`, user `neo4j`, password `password`)
# * `run("cypher...")` – runs a Cypher query and returns rows
# * `show(rows)` – prints rows as a table
# * `load_devops_graph()` – creates our incident story (services, a database, a deploy, an alarm, two incidents)

# %%
import graph_helpers as g

counts = g.load_devops_graph()          # wipes the DB, then loads the story
print("Loaded into Neo4j:", counts)

# %% [markdown]
# **Look at what we just stored.** Each row below is one relationship from the graph.
# Read it as a sentence: "checkout-api DEPENDS_ON orders-service".

# %%
g.show(g.run("""
    MATCH (a)-[r]->(b)
    RETURN labels(a)[0] AS from_type, coalesce(a.name, a.id) AS from_node,
           type(r) AS relationship,
           labels(b)[0] AS to_type,   coalesce(b.name, b.id) AS to_node
    ORDER BY relationship, from_node
"""), max_rows=40)

# %% [markdown]
# 💡 **Tip:** open <http://localhost:7474> (login `neo4j` / `password`) and run
# `MATCH (n)-[r]->(m) RETURN n,r,m` to *see* the graph as a picture.

# %% [markdown]
# ## 1. The crisis: why plain vector-RAG agents stall
#
# **Vector RAG** = split documents into chunks → turn each into numbers (embeddings) →
# find chunks *similar in wording* to the question → give them to an LLM.
# It is great at "find the paragraph that mentions X". The book lists five flaws that
# appear when you build an *agent* on it:
#
# | Flaw | What it means for our DevOps agent |
# |---|---|
# | Context amnesia | forgets the similar outage from June |
# | Relationship blindness | doesn't know checkout depends on the database |
# | Temporal ignorance | can't tell the *new* config from the *old* one |
# | Reasoning paralysis | finds similar text, not logical connections |
# | Tool chaos | guesses which API to call |
#
# They share one root cause: vector search sees **strings**, not **things and how they connect**.
# A knowledge graph stores the things and connections explicitly.
#
# **Demo — relationship blindness fixed by a graph.** Question: *"If orders-db is in trouble,
# which services are affected?"* A vector search would need a document that happens to say so.
# A graph just walks the `DEPENDS_ON` arrows backwards (`*1..3` = follow 1 to 3 hops):

# %%
rows = g.run("""
    MATCH (affected:Service)-[:DEPENDS_ON*1..3]->(:Database {name:'orders-db'})
    RETURN DISTINCT affected.name AS affected_service, affected.tier AS tier
    ORDER BY tier
""")
g.show(rows)
assert {r["affected_service"] for r in rows} >= {"checkout-api", "orders-service", "cart-service"}
print("\nNotice cart-service: it never mentions orders-db directly, but the graph finds it in 3 hops.")

# %% [markdown]
# **Demo — context amnesia fixed by a graph.** "Has this happened before?"
# The graph remembers past incidents attached to the same service:

# %%
g.show(g.run("""
    MATCH (i:Incident)-[:AFFECTS]->(:Service {name:'checkout-api'})
    OPTIONAL MATCH (i)-[:ROOT_CAUSE]->(cause)
    RETURN i.id AS incident, i.status AS status, i.title AS title,
           cause.name AS root_cause_component, i.fix AS known_fix
    ORDER BY i.opened_at
"""))

# %% [markdown]
# And **temporal ignorance** — what changed right before the alarm? The deployment node
# records *what* changed and *when*:

# %%
g.show(g.run("""
    MATCH (d:Deployment)-[:CHANGED]->(x)
    RETURN d.id AS deployment, d.at AS at, d.change AS change, x.name AS changed_component
"""))
print("\nA connected story: deploy-481 shrank orders-db's pool (100 -> 20) at 08:30; alarm fired 08:41.")
print("The June incident INC-0877 had the same cause. The graph connected all of it.")

# %% [markdown]
# ## 2. What is an *agent*? Three dials
#
# Ch.1 avoids "agent or not?" and instead defines **agency** with three dials from 0 to 1:
#
# * **Autonomy** – how much it decides on its own.
# * **Action** – can it *change* the world (restart a host, roll back a deploy)? This is a **gate**:
#   if it cannot act, it is an *advisor*, not an agent.
# * **Authority** – how much it is *allowed* to do.
#
# The skill `workflow-agent-spectrum-classifier` places a system on a line from
# **WORKFLOW** (fixed script) to **AGENT** (decides its own steps).
# `load_skill(chapter_folder, skill_name)` imports that skill's `lib.py`.

# %%
spectrum = g.load_skill("crisis", "workflow-agent-spectrum-classifier")

systems = [
    dict(name="scripted-latency-runbook", autonomy=0.1, action=0.2, authority=0.2,
         path_determinism=0.95, memory=False, tool_use=True, contextual=False),
    dict(name="human-in-loop-remediation", autonomy=0.5, action=0.6, authority=0.4,
         path_determinism=0.55, memory=True, tool_use=True, contextual=True),
    dict(name="autonomous-devops-agent", autonomy=0.9, action=0.8, authority=0.5,
         path_determinism=0.15, memory=True, tool_use=True, contextual=True),
    dict(name="latency-advisor (read-only)", autonomy=0.85, action=0.05, authority=0.05,
         path_determinism=0.2, memory=True, tool_use=True, contextual=True),
]
placed = {}
for s in systems:
    r = spectrum.classify(**s)
    placed[s["name"]] = r
    verdict = "AGENT" if r["is_agent_by_action_test"] else "advisor (cannot act)"
    print(f"{s['name']:<30} position={r['spectrum_position']:<5} band={r['band']:<9} action-test: {verdict}")

# %% [markdown]
# **How to read this:** the runbook is a WORKFLOW (low autonomy, deterministic path).
# The autonomous agent is at the AGENT end. The read-only advisor *thinks a lot* but fails
# the action test (`action=0.05`) — so it is an advisor. The `assert`s below turn that
# expectation into an automatic check:

# %%
assert placed["scripted-latency-runbook"]["band"] == "WORKFLOW"
assert placed["autonomous-devops-agent"]["band"] == "AGENT"
assert placed["human-in-loop-remediation"]["band"] == "BLENDED"
assert placed["latency-advisor (read-only)"]["is_agent_by_action_test"] is False
print("PASS: spectrum classifier behaves as Chapter 1 describes.")

# %% [markdown]
# ## 3. The agent constraint triangle
#
# Building an agent means balancing three pressures that feed each other:
#
# 1. **Tool orchestration** – too many or confusing tools → wrong tool picked.
# 2. **Context window** – the more you stuff into the prompt, the worse the model reasons ("context rot").
# 3. **Task steps** – long multi-step tasks multiply the above.
#
# The skill `agent-constraint-triangle-scorer` turns a configuration into pressure scores
# and an overall band (`BALANCED` … `OVERCONSTRAINED`). Compare a lean agent with a bloated one:

# %%
triangle = g.load_skill("crisis", "agent-constraint-triangle-scorer")

balanced = dict(name="balanced-graph-agent", avg_task_steps=5, tool_count=12,
                tools_disambiguable=True, context_window_tokens=200_000,
                avg_context_tokens_used=60_000)
bloated = dict(name="bloated-mcp-agent", avg_task_steps=8, tool_count=36,
               tools_disambiguable=False, context_window_tokens=200_000,
               avg_context_tokens_used=178_000)

for cfg in (balanced, bloated):
    r = triangle.score(cfg)
    print(f"{cfg['name']}: {r['overall_band']}  (worst corner: {r['dominant_constraint']})")
    for name, p in r["pressures"].items():
        print(f"    {name:<22} pressure={p['pressure']:<6} band={p['band']}")
    print()

assert triangle.score(bloated)["overall_band"] == "OVERCONSTRAINED"
assert triangle.score(balanced)["overall_band"] == "BALANCED"
print("Suggested fix for the bloated agent:", triangle.score(bloated)["recommendations"][0]["action"][:90], "...")

# %% [markdown]
# **Takeaway:** a graph helps *every* corner — it gives the agent a small, precise slice of
# context (not a 178k-token dump) and a clear map of what depends on what.

# %% [markdown]
# ## 4. Vector, graph, or hybrid?
#
# Not every question needs a graph. The skill `vector-vs-graph-retrieval-selector` encodes the
# book's rule of thumb. You describe the question and it recommends **VECTOR**, **GRAPH** or **HYBRID**.
#
# * *Local + fact lookup* ("show the 5xx errors in this log") → **VECTOR** (fast, cheap).
# * *Global + multi-hop + time* ("what is impacted by this migration?") → **GRAPH**.
# * *An autonomous agent doing both* → **HYBRID**.

# %%
retrieval = g.load_skill("crisis", "vector-vs-graph-retrieval-selector")

workloads = [
    dict(name="specific-5xx-lookup", query_scope="local", query_type="data",
         multi_hop=False, temporal=False, structured_domain=False, latency_critical=True),
    dict(name="cascading-migration-impact", query_scope="global", query_type="activity",
         multi_hop=True, temporal=True, structured_domain=True),
    dict(name="autonomous-devops-agent", query_scope="mixed", query_type="activity",
         multi_hop=True, temporal=True, structured_domain=True, agentic=True,
         dataset_scale_pages=120_000),
]
recs = {}
for w in workloads:
    r = retrieval.recommend(**w)
    recs[w["name"]] = r["recommendation"]
    print(f"{w['name']:<28} -> {r['recommendation']:<7} because: {r['reasons'][0]}")

assert recs == {"specific-5xx-lookup": "VECTOR", "cascading-migration-impact": "GRAPH",
                "autonomous-devops-agent": "HYBRID"}

# %% [markdown]
# **"Why not just use a huge context window?"** — a very common objection. The skill carries
# the book's rebuttal:

# %%
r = retrieval.recommend(query_scope="global", query_type="activity", multi_hop=True,
                        larger_context_window=True)
print(r["larger_context_window_rebuttal"])

# %% [markdown]
# ## 5. Putting it together on the live incident
#
# The agent's job: *find the likely cause of INC-1001 and what to do.* With the graph this
# is **one query** that joins the alarm, the affected service, recent deployments to anything it
# depends on, and past incidents with the same root cause:

# %%
g.show(g.run("""
    MATCH (inc:Incident {id:'INC-1001'})-[:AFFECTS]->(svc:Service)
    MATCH (svc)-[:DEPENDS_ON*1..2]->(dep)
    MATCH (d:Deployment)-[:CHANGED]->(dep)
    OPTIONAL MATCH (past:Incident {status:'resolved'})-[:ROOT_CAUSE]->(dep)
    RETURN inc.id AS incident, svc.name AS affected, dep.name AS suspect_component,
           d.id AS recent_deploy, d.change AS what_changed,
           past.id AS similar_past_incident, past.fix AS past_fix
"""))

# %% [markdown]
# **Reading the answer:** checkout-api → depends on `orders-service` → depends on `orders-db`;
# deploy-481 changed `orders-db`'s pool size; incident INC-0877 had the same root cause and was
# fixed by raising the pool. A vector search cannot assemble this chain; a graph does in one hop-walk.
#
# ## Recap
# * Vector RAG finds similar *text*; a graph answers *connected* questions.
# * Agency = autonomy + action (gate) + authority.
# * The constraint triangle explains why agents get overloaded.
# * Use VECTOR for local lookups, GRAPH for multi-hop/temporal, HYBRID for agents.
#
# **Next:** `ch2-architecture-foundations.ipynb`.
