# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > [Beginner (tier 1/3)](../notebooks_simplified/ch1-defining-agentic-ai.ipynb)  →  **Intermediate (tier 2/3) ← you are here**  →  [Advanced (tier 3/3)](../notebooks_advanced_simplified/ch1-defining-agentic-ai.ipynb)

# %% [markdown]
# # Chapter 1 (Intermediate) — Defining Agentic AI: the why and the how
#
# **Tier 2 of 3.** In the beginner notebook you ran the five Chapter 1 skills and saw their answers.
# Here we open the hood a little and start driving yourself:
#
# * skill **inputs come from Cypher** on a real Neo4j graph, skill **outputs are written back** as nodes/edges;
# * for each skill you see **the function signature** and an **anatomy of its output** (what each field means);
# * you chain skills in a **hand-written pipeline** on *"checkout-api is slow"*;
# * you solve **two guided exercises**, run a skill from its **CLI**, and learn to **trust a skill via its benchmark**.
#
# **Why this matters:** Chapter 1 argues vector-only RAG fails agents because it sees *strings*, not
# *connected things*. Each section turns one claim of that chapter into something you can run and assert.
#
# > Prereq: Neo4j at `bolt://localhost:7687` (neo4j/password). No API keys needed.
# > Not in this notebook (see tier 3): the `MiniAgent` loop, benchmarks of every skill, vibe-coding corners, experiments.

# %% [markdown]
# ## 0. Setup — connect, load the incident graph, discover the skills
#
# `h.run(cypher, **params)` returns a list of dicts, `h.show(rows)` prints a table, `h.check(name, cond)` prints PASS/FAIL,
# `h.load_skill(chapter_folder, slug)` imports a skill's `lib.py`. The catalog reads each skill's `SKILL.md`
# front-matter — the same discovery step an AI assistant does before choosing a skill.

# %%
import int_helpers as h

counts = h.load_devops_graph()          # wipes the DB, then loads the shared DevOps story
print("Loaded:", counts)

catalog = h.skill_catalog("crisis")
for s in catalog:
    print(f"- {s['slug']:<34} {s['description'][:80]}...")
h.check("chapter 1 folder has 5 skills", len(catalog) == 5)

# %% [markdown]
# ## 1. The crisis, as Cypher
#
# Vector RAG finds *similar text*; a graph answers *connected* questions. Two queries (you know the basics
# from tier 1, so only the new clauses are explained).
#
# **Multi-hop, variable length.** `[:DEPENDS_ON*1..3]` follows 1 to 3 hops; `p = ...` names the path so
# `length(p)` gives the hop count. `checkout-api` reaches `orders-db` by *two* routes, so we group by
# service and take `min(length(p))` — the shortest distance (non-aggregated `RETURN` columns form the grouping key).

# %%
blast = h.run("""
    MATCH p = (affected:Service)-[:DEPENDS_ON*1..3]->(:Database {name:'orders-db'})
    RETURN affected.name AS service, affected.tier AS tier, min(length(p)) AS hops
    ORDER BY hops, service
""")
h.show(blast)
hops = {r["service"]: r["hops"] for r in blast}
h.check("cart-service is 3 hops from orders-db", hops["cart-service"] == 3)
h.check("blast radius is 4 services", len(blast) == 4)

# %% [markdown]
# **Aggregation.** `WITH DISTINCT s` de-duplicates before grouping (otherwise a service reached by two paths
# is counted twice); `count()` counts per group and `collect()` gathers a list. Question: *which criticality tiers are exposed?*

# %%
by_tier = h.run("""
    MATCH (s:Service)-[:DEPENDS_ON*1..3]->(:Database {name:'orders-db'})
    WITH DISTINCT s
    RETURN s.tier AS tier, count(s) AS n_services, collect(s.name) AS services
    ORDER BY n_services DESC, tier
""")
h.show(by_tier)
h.check("critical tier has 2 exposed services",
        next(r for r in by_tier if r["tier"] == "critical")["n_services"] == 2)

# %% [markdown]
# **Why this matters (trade-off):** a variable-length path is cheap at `*1..3` but grows fast on dense graphs —
# always bound it. An unbounded `*` on a big graph is the classic way to hang a query.
#
# ### 🧑‍💻 Your turn 1 — write a Cypher query
# **Task:** which services depend on `orders-db` *directly* (exactly 1 hop)? Return a sorted list of names.
# Try writing it yourself first, then compare with the reference solution below.

# %%
def direct_dependents():
    rows = h.run("""
        MATCH (s:Service)-[:DEPENDS_ON]->(:Database {name:'orders-db'})
        RETURN s.name AS name ORDER BY name
    """)
    return [r["name"] for r in rows]

h.your_turn("direct dependents of orders-db",
            "No `*` needed: a single relationship hop. Return `s.name` and `ORDER BY` it.",
            direct_dependents, ["orders-service", "payment-service"])

# %% [markdown]
# ## 2. Skill 1 — `context-failure-classifier` (with graph write-back)
#
# Chapter 1 names **five fatal flaws** of naive vector RAG that compound into five agent failure modes
# (action blindness, memory fragmentation, context drift, planning paralysis, tool chaos).
#
# **Signature:** `classify(symptom: str)` and `classify_batch(symptoms: list[str])`.
#
# Flow: store INC-1001's post-mortem as `Symptom` nodes → **read them back with Cypher** → classify →
# **write** `FailureMode → Flaw → Capability` edges. `MERGE` = create-if-missing (idempotent, no duplicate nodes).

# %%
failure = h.load_skill("crisis", "context-failure-classifier")

postmortem = [
    ("S1", "The agent restarted orders-service unaware it would cascade to checkout-api"),
    ("S2", "Each new session it starts from scratch and forgets the June pool incident"),
    ("S3", "It trusted the old config and reported a stale pool size after the rollback"),
    ("S4", "Could not connect the dots between the alarm and the deploy; no multi-step plan"),
    ("S5", "It called the wrong tool and had to guess which API to use"),
]
h.run("""
    UNWIND $rows AS row
    MATCH (inc:Incident {id:'INC-1001'})
    CREATE (inc)-[:HAS_SYMPTOM]->(:Symptom {id:row[0], text:row[1]})
""", rows=[list(r) for r in postmortem])      # UNWIND: one row per list item

symptoms = h.run("""
    MATCH (:Incident {id:'INC-1001'})-[:HAS_SYMPTOM]->(s:Symptom)
    RETURN s.id AS id, s.text AS text ORDER BY s.id
""")
batch = failure.classify_batch([s["text"] for s in symptoms])

# %% [markdown]
# **Anatomy of the output.** `classify_batch` returns a dict; the parts we use:
#
# | field | meaning |
# |---|---|
# | `results[i]["primary"]` | best match for symptom *i*: `failure_mode`, `root_flaw` (architectural cause), `cure` (graph capability) |
# | `results[i]["classified"]` | `False` if no keyword matched (rephrase the symptom) |
# | `prioritized_cures` | the union of cures ranked across the whole post-mortem |

# %%
print("top-level keys:", sorted(batch))
print("one result keys:", sorted(batch["results"][0]))
for s, res in zip(symptoms, batch["results"]):
    p = res["primary"]
    h.run("""
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
h.check("all five symptoms land in the expected failure mode", got == expected)

# %% [markdown]
# **Read it back.** A 4-hop path `Symptom → FailureMode → Flaw → Capability` answers a prioritisation question:
# which capability cures the most symptoms? `count(DISTINCT s)` avoids double counting.

# %%
cures = h.run("""
    MATCH (:Incident {id:'INC-1001'})-[:HAS_SYMPTOM]->(s)-[:CLASSIFIED_AS]->(fm)-[:ROOT_FLAW]->(fl)-[:CURED_BY]->(cap)
    RETURN cap.name AS graph_capability, count(DISTINCT s) AS symptoms_cured, collect(DISTINCT fl.name) AS flaws
    ORDER BY symptoms_cured DESC, graph_capability
""")
h.show(cures)
h.check("5 distinct capabilities needed", len(cures) == 5)

# %% [markdown]
# **Why this matters (trade-off):** the classifier is keyword/regex based — deterministic, free and explainable,
# but brittle on unusual wording. Production swaps `classify()` for an LLM with the *same return shape*
# (the `TODO` seam in `lib.py`).
#
# ## 3. Skill 2 — `workflow-agent-spectrum-classifier`
#
# Agency has three dials from 0 to 1: **autonomy** (decides alone), **action** (can change the world — a *gate*:
# no action means advisor, not agent) and **authority** (what it is *allowed* to do).
#
# **Signature:** `classify(autonomy, action, authority, path_determinism, memory=False, tool_use=False, ...)`.

# %%
spectrum = h.load_skill("crisis", "workflow-agent-spectrum-classifier")

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
h.run("UNWIND $rows AS row CREATE (s:AgentSystem) SET s = row", rows=systems_seed)   # SET s = row copies all keys

placed = {}
for rec in h.run("MATCH (s:AgentSystem) RETURN s{.*} AS p ORDER BY s.name"):   # map projection -> dict of properties
    r = spectrum.classify(**rec["p"])
    placed[rec["p"]["name"]] = r
    h.run("""
        MATCH (s:AgentSystem {name:$n})
        MERGE (b:Band {name:$band})
        MERGE (s)-[:PLACED_IN {position:$pos, is_agent:$agent}]->(b)
    """, n=rec["p"]["name"], band=r["band"], pos=r["spectrum_position"], agent=r["is_agent_by_action_test"])

# Anatomy of the output: band (WORKFLOW/BLENDED/AGENT), spectrum_position (0-1 behind the band),
# is_agent_by_action_test (the gate, independent of the band), notes (calibration warnings).
h.show(h.run("""
    MATCH (s:AgentSystem)-[p:PLACED_IN]->(b:Band)
    RETURN b.name AS band, count(s) AS systems, collect(s.name) AS names, round(avg(p.position), 2) AS avg_position
    ORDER BY avg_position
"""))
h.check("runbook is a WORKFLOW", placed["scripted-latency-runbook"]["band"] == "WORKFLOW")
h.check("human-in-loop is BLENDED", placed["human-in-loop-remediation"]["band"] == "BLENDED")
h.check("autonomous agent is AGENT", placed["autonomous-devops-agent"]["band"] == "AGENT")
adv = placed["latency-advisor (read-only)"]
h.check("read-only advisor fails the action test", adv["is_agent_by_action_test"] is False)
print("advisor band:", adv["band"], "| notes:", *adv["notes"], sep="\n - ")

# %% [markdown]
# **Read the trap:** the advisor lands in the `AGENT` *band* yet fails the *action gate*. Band and gate answer
# different questions — never collapse them into one number.
#
# ## 4. Skill 3 — `agent-constraint-triangle-scorer`
#
# Three pressures feed each other: **complexity** (more steps → compounding errors), **tool orchestration**
# (more/ambiguous tools → wrong calls) and **context utilisation** (fuller window → *context rot*).
#
# **Signature:** `score(config: dict)` with keys `avg_task_steps`, `tool_count`, `tools_disambiguable`,
# `context_window_tokens`, `avg_context_tokens_used`.
#
# Intermediate move: derive `tool_count` **from the graph** (`count()` over `Tool` nodes) instead of typing it.

# %%
triangle = h.load_skill("crisis", "agent-constraint-triangle-scorer")

tool_names = ["find_suspects", "blast_radius", "classify_symptoms", "select_retrieval",
              "score_triangle", "assess_readiness", "record_conclusion"]
h.run("UNWIND $n AS name CREATE (:Tool {name:name})", n=tool_names)
h.run("""
    UNWIND $edges AS e
    MATCH (x:Tool {name:e[0]}), (y:Tool {name:e[1]})
    CREATE (x)-[:FEEDS]->(y)
""", edges=[list(e) for e in zip(tool_names, tool_names[1:])])

h.run("""
    CREATE (c:AgentConfig {name:'graph-devops-agent', avg_task_steps:5, tools_disambiguable:true,
                           context_window_tokens:200000, avg_context_tokens_used:40000})
    WITH c
    MATCH (t:Tool)
    WITH c, collect(t) AS tools
    SET c.tool_count = size(tools)
    FOREACH (t IN tools | CREATE (c)-[:USES]->(t))
""")
h.run("""
    CREATE (:AgentConfig {name:'bloated-mcp-agent', avg_task_steps:8, tool_count:36, tools_disambiguable:false,
                          context_window_tokens:200000, avg_context_tokens_used:178000})
""")

for rec in h.run("MATCH (c:AgentConfig) RETURN c{.*} AS cfg ORDER BY c.name"):
    cfg = rec["cfg"]
    r = triangle.score(cfg)
    h.run("MATCH (c:AgentConfig {name:$n}) SET c.overall = $o, c.dominant = $d",
          n=cfg["name"], o=r["overall_band"], d=r["dominant_constraint"])
    print(f"{cfg['name']:<22} tools={cfg['tool_count']:<3} -> {r['overall_band']:<16} worst corner: {r['dominant_constraint']}")

# %% [markdown]
# **Anatomy of the output:** `overall_band` (e.g. BALANCED … OVERCONSTRAINED), `dominant_constraint` (the worst
# corner), `active_pressure_cycles` (which pressures currently reinforce each other, with a `cascade` explanation)
# and `recommendations` (which corner to shrink first). The curves are **heuristics** encoding the chapter's
# qualitative claims — a conversation starter, not a measurement.

# %%
bands = {r["name"]: r["overall"] for r in h.run("MATCH (c:AgentConfig) RETURN c.name AS name, c.overall AS overall")}
h.check("our graph-derived agent is BALANCED", bands["graph-devops-agent"] == "BALANCED")
h.check("bloated agent is OVERCONSTRAINED", bands["bloated-mcp-agent"] == "OVERCONSTRAINED")
h.check("tool_count derived from the graph == 7",
        h.run("MATCH (c:AgentConfig {name:'graph-devops-agent'}) RETURN c.tool_count AS n")[0]["n"] == 7)
bloated = triangle.score(h.run("MATCH (c:AgentConfig {name:'bloated-mcp-agent'}) RETURN c{.*} AS c")[0]["c"])
print("active cycles:", [c["edge"] for c in bloated["active_pressure_cycles"]])
print("first fix:", bloated["recommendations"][0]["action"][:110], "...")

# %% [markdown]
# ### 🧑‍💻 Your turn 2 — change the inputs
# **Task:** take the bloated agent and fix the *tool* corner (8 tools, now disambiguable) **and** the *context*
# corner (only 60,000 of 200,000 tokens used). The 8 task steps stay. Which `overall_band` do you get?

# %%
def fix_tools_and_context():
    cfg = h.run("MATCH (c:AgentConfig {name:'bloated-mcp-agent'}) RETURN c{.*} AS c")[0]["c"]
    cfg.update(tool_count=8, tools_disambiguable=True, avg_context_tokens_used=60000)
    return triangle.score(cfg)["overall_band"]

h.your_turn("shrink the tool and context corners",
            "Copy the config from the graph, `cfg.update(...)` the three keys, call `triangle.score`. Long task chains still hurt.",
            fix_tools_and_context, "STRESSED")

# %% [markdown]
# ## 5. Skill 4 — `vector-vs-graph-retrieval-selector`
#
# Maps a workload to **VECTOR**, **GRAPH** or **HYBRID** (Microsoft BenchmarkQED quadrants: local/global × data/activity).
#
# **Signature:** `recommend(query_scope, query_type, multi_hop=False, temporal=False, structured_domain=True, dataset_scale_pages=0, ...)`.
# Anatomy: `recommendation` (the verdict) plus supporting text such as `scale_note` and `larger_context_window_rebuttal`.

# %%
retrieval = h.load_skill("crisis", "vector-vs-graph-retrieval-selector")

h.run("UNWIND $rows AS row CREATE (w:Workload) SET w = row", rows=[
    dict(name="specific-5xx-lookup", query_scope="local", query_type="data", multi_hop=False,
         temporal=False, structured_domain=False, latency_critical=True),
    dict(name="cascading-migration-impact", query_scope="global", query_type="activity",
         multi_hop=True, temporal=True, structured_domain=True),
    dict(name="autonomous-devops-agent", query_scope="mixed", query_type="activity", multi_hop=True,
         temporal=True, structured_domain=True, agentic=True, dataset_scale_pages=120000),
])
for rec in h.run("MATCH (w:Workload) RETURN w{.*} AS w ORDER BY w.name"):
    r = retrieval.recommend(**rec["w"])
    h.run("""
        MATCH (w:Workload {name:$n})
        MERGE (s:Strategy {name:$s})
        MERGE (w)-[:RECOMMENDED]->(s)
    """, n=rec["w"]["name"], s=r["recommendation"])

h.show(h.run("MATCH (w:Workload)-[:RECOMMENDED]->(s:Strategy) RETURN w.name AS workload, s.name AS strategy ORDER BY strategy"))
rec_map = {r["w"]: r["s"] for r in h.run("MATCH (w:Workload)-[:RECOMMENDED]->(s:Strategy) RETURN w.name AS w, s.name AS s")}
h.check("5xx lookup -> VECTOR", rec_map["specific-5xx-lookup"] == "VECTOR")
h.check("migration impact -> GRAPH", rec_map["cascading-migration-impact"] == "GRAPH")
h.check("agent workload -> HYBRID", rec_map["autonomous-devops-agent"] == "HYBRID")

# %% [markdown]
# **Why this matters (trade-off):** graph retrieval costs indexing time and schema design; vector retrieval is cheap
# but blind to relationships. The skill keeps you honest: simple lookups stay VECTOR, only multi-hop/temporal
# workloads justify the graph, and mixed agent workloads end up HYBRID.

#
# ## 6. Skill 5 — `enterprise-readiness-scorer`
#
# **Signature:** `assess(profile: dict)` where `profile` has `graph_capabilities` (5 booleans),
# `agency` (autonomy/action/authority), `capabilities`, and `captures_rejected_alternatives` (Marple's decision-trace test).
# Anatomy of the output: `score`, `band` (NAIVE-VECTOR … PILOT-READY … PRODUCTION-READY) and `recommendations` (the gaps).
#
# Instead of trusting a self-reported profile, we **probe the graph**: each probe is `RETURN count(...) > 0 AS ok`
# (an aggregate always returns one row, even with no matches).

# %%
readiness = h.load_skill("crisis", "enterprise-readiness-scorer")

PROBES = {
    "entity_relationships": "MATCH ()-[r]->() RETURN count(DISTINCT type(r)) >= 4 AS ok",
    "temporal_evolution": "MATCH (d:Deployment) WHERE d.at IS NOT NULL RETURN count(d) > 0 AS ok",
    "evolving_memory": "MATCH (i:Incident {status:'resolved'}) WHERE i.root_cause IS NOT NULL RETURN count(i) > 0 AS ok",
    "multi_hop_reasoning": "MATCH p = (:Service)-[:DEPENDS_ON*2..]->() RETURN count(p) > 0 AS ok",
    "tool_orchestration": "MATCH (:Tool)-[:FEEDS]->(:Tool) RETURN count(*) > 0 AS ok",
}
caps = {k: bool(h.run(q)[0]["ok"]) for k, q in PROBES.items()}
print("graph probes:", caps)

graph_profile = {
    "graph_capabilities": caps,
    "agency": {"autonomy": 0.6, "action": 0.5, "authority": 0.3},
    "capabilities": {"autonomous_decision_making": True,
                     "contextual_understanding": caps["entity_relationships"],
                     "strategic_tool_utilization": caps["tool_orchestration"],
                     "memory_persistence": False},
    "captures_rejected_alternatives": False,
}
naive_profile = {
    "graph_capabilities": {k: False for k in PROBES},
    "agency": graph_profile["agency"],
    "capabilities": {"autonomous_decision_making": True, "contextual_understanding": False,
                     "strategic_tool_utilization": False, "memory_persistence": False},
    "captures_rejected_alternatives": False,
}
naive = readiness.assess(naive_profile)
graphy = readiness.assess(graph_profile)
for name, r in [("naive-vector-bot", naive), ("graph-devops-agent", graphy)]:
    h.run("""
        MERGE (p:Profile {name:$n}) SET p.score = $score
        MERGE (b:Readiness {band:$band})
        MERGE (p)-[:RATED]->(b)
    """, n=name, score=r["score"], band=r["band"])
    print(f"{name:<20} {r['score']:>5} {r['band']}")
for gap in graphy["recommendations"]:
    print("  gap ->", gap[:100])
h.check("naive bot is NAIVE-VECTOR", naive["band"] == "NAIVE-VECTOR")
h.check("graph agent is PILOT-READY (no stored runs / decision traces yet)", graphy["band"] == "PILOT-READY")

# %% [markdown]
# The gaps are **memory persistence** and **decision traces** — both things an agent run can create (tier 3 does exactly that).

#
# ## 7. A plain-Python pipeline: *"checkout-api is slow"*
#
# Chaining skills does not need an agent framework: a normal function that reads the graph, calls two or three
# skills and prints a decision report is enough. (A planner loop with a recorded trace — `MiniAgent` — is tier 3.)

# %%
def investigate(incident_id: str) -> dict:
    # 1) suspects: service -> deps (1-2 hops) -> deployments that changed them, plus similar past incidents
    suspects = h.run("""
        MATCH (inc:Incident {id:$iid})-[:AFFECTS]->(svc:Service)
        MATCH (svc)-[:DEPENDS_ON*1..2]->(dep)
        MATCH (d:Deployment)-[:CHANGED]->(dep)
        OPTIONAL MATCH (past:Incident {status:'resolved'})-[:ROOT_CAUSE]->(dep)
        RETURN DISTINCT dep.name AS suspect, d.id AS deploy, d.change AS change,
               past.id AS past_incident, past.fix AS past_fix
    """, iid=incident_id)
    top = suspects[0]
    # 2) blast radius of the top suspect
    blast_r = h.run("""
        MATCH p = (s:Service)-[:DEPENDS_ON*1..3]->({name:$c})
        RETURN s.name AS service, min(length(p)) AS hops ORDER BY hops, service
    """, c=top["suspect"])
    # 3) skill: why would an agent mishandle this? (symptoms read from the graph)
    texts = [r["t"] for r in h.run(
        "MATCH (:Incident {id:$i})-[:HAS_SYMPTOM]->(s) RETURN s.text AS t ORDER BY s.id", i=incident_id)]
    cures = failure.classify_batch(texts)["prioritized_cures"]
    # 4) skill: which retrieval strategy does the agent workload need?
    w = h.run("MATCH (w:Workload {name:'autonomous-devops-agent'}) RETURN w{.*} AS w")[0]["w"]
    strategy = retrieval.recommend(**w)["recommendation"]
    # write the conclusion back
    h.run("""
        MATCH (i:Incident {id:$i}), (c {name:$c}), (d:Deployment {id:$d})
        MERGE (i)-[l:LIKELY_CAUSE]->(c) SET l.deploy = $d, l.change = d.change
    """, i=incident_id, c=top["suspect"], d=top["deploy"])
    return dict(top=top, blast=blast_r, cures=cures, strategy=strategy)

rep = investigate("INC-1001")
top = rep["top"]
print(f"INC-1001 LIKELY CAUSE : {top['suspect']} (deploy {top['deploy']}: {top['change']})")
print(f"  similar past incident: {top['past_incident']} -> fix: {top['past_fix']}")
print(f"  blast radius          : {[(b['service'], b['hops']) for b in rep['blast']]}")
print(f"  agent cures needed    : {len(rep['cures'])} capabilities")
print(f"  retrieval strategy    : {rep['strategy']}")
h.check("pipeline blames a dependency that has a deployment", top["deploy"] is not None)
h.check("conclusion persisted as LIKELY_CAUSE edge",
        h.run("MATCH (:Incident {id:'INC-1001'})-[l:LIKELY_CAUSE]->() RETURN count(l) AS n")[0]["n"] == 1)
h.check("pipeline picks HYBRID retrieval", rep["strategy"] == "HYBRID")

# %% [markdown]
# **Why this matters:** every step is a *tool with an explicit input and output shape*, so you can unit-test each
# one, and later hand the same functions to an LLM planner without rewriting them.

#
# ## 8. Trusting a skill: CLI and benchmark
#
# Every skill is also a command-line tool (stdlib only), so it runs from cron, CI or a shell. `cli.py --help`
# prints the SKILL.md description; `cli.py benchmark` runs its own verification gate (exit code 0 = pass).

# %%
import os, subprocess, sys
cli = h.SKILLS_DIR / "crisis" / "context-failure-classifier" / "cli.py"
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
kw = dict(capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
called = subprocess.run([sys.executable, str(cli), "classify", "--symptom", postmortem[0][1]], **kw)
print(called.stdout[:400])
h.check("cli classify exits 0 and mentions action_blindness",
        called.returncode == 0 and "action_blindness" in called.stdout)

ok, first_line = h.run_benchmark("crisis", "workflow-agent-spectrum-classifier")
print(f"[{'PASS' if ok else 'FAIL'}] workflow-agent-spectrum-classifier: {first_line[:80]}")
h.check("spectrum classifier benchmark passes", ok)

# %% [markdown]
# `run_benchmark` is the habit to adopt: **never rely on a skill you haven't benchmarked**. Tier 3 benchmarks all five.
#
# ## 🎯 Vibe-coding tip
# Paste this into your AI assistant (Copilot, Claude Code, Cursor…) from the repo root:
#
# > *Read `skills/crisis/agent-constraint-triangle-scorer/SKILL.md` and `lib.py`. Write a script that scores three
# > agent configs (5, 20 and 40 tools, tools_disambiguable false for the last) and prints a table of
# > `overall_band` and `dominant_constraint`. Use only the skill's `score()`, no new dependencies.*
#
# **Acceptance check (run it yourself, don't trust the assistant):**
# `python skills\crisis\agent-constraint-triangle-scorer\cli.py benchmark` exits 0, and the 40-tool row is
# not `BALANCED`. If the assistant invented config keys, compare them with the signature in section 4.

#
# ## Common mistakes
# * Typing skill inputs by hand instead of reading them from the graph — the whole point is a single source of truth.
# * Using `CREATE` where you meant `MERGE`, producing duplicate `Flaw`/`Band` nodes on every rerun.
# * Unbounded variable-length paths (`*`) on large graphs; always set an upper bound.
# * Treating `band` and `is_agent_by_action_test` as the same thing.
# * Ignoring `classified: False` from the classifier instead of rephrasing the symptom.
# * Reading the triangle heuristics as measured benchmarks.

#
# ## Final check battery

# %%
checks = {
    "graph has services": h.run("MATCH (s:Service) RETURN count(s) AS n")[0]["n"] > 0,
    "5 failure modes written back": h.run("MATCH (f:FailureMode) RETURN count(f) AS n")[0]["n"] == 5,
    "4 systems placed on the spectrum": h.run("MATCH (:AgentSystem)-[:PLACED_IN]->() RETURN count(*) AS n")[0]["n"] == 4,
    "3 workloads have a strategy": h.run("MATCH (:Workload)-[:RECOMMENDED]->() RETURN count(*) AS n")[0]["n"] == 3,
    "2 profiles rated": h.run("MATCH (:Profile)-[:RATED]->() RETURN count(*) AS n")[0]["n"] == 2,
    "DEPENDS_ON is a DAG": h.run("MATCH p = (s:Service)-[:DEPENDS_ON*1..6]->(s) RETURN count(p) AS n")[0]["n"] == 0,
}
for name, cond in checks.items():
    h.check(name, cond)
assert all(checks.values())

# %% [markdown]
# ## Recap
# * You used **all five** Chapter 1 skills, each fed by Cypher and (where useful) written back to Neo4j.
# * You can read a skill's **signature and output anatomy**, and chain skills in a **plain function**.
# * You practised Cypher (multi-hop, aggregation, `MERGE`, `UNWIND`) and verified skills via **CLI + benchmark**.
#
# **Next:** the same chapter at tier 3 — `MiniAgent` plan→act→observe, stored traces, all benchmarks, vibe corners and
# experiments: [`../notebooks_advanced_simplified/ch1-defining-agentic-ai.ipynb`](../notebooks_advanced_simplified/ch1-defining-agentic-ai.ipynb).
