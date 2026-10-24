# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > **Beginner (tier 1/3) ← you are here**  →  Intermediate (tier 2/3, no notebook for this topic)  →  Advanced (tier 3/3, no notebook for this topic)

# %% [markdown]
# # Spike B (Simplified) — Diagnosing agent-memory failures (the "Letta 8 failure modes" reviewer)
#
# **Who this is for:** beginners. Every term is defined when it first appears.
#
# **What is a "spike"?** A small experiment checking that one idea works end to end.
#
# **What does this skill do?** When people say *"my agent keeps forgetting things"*, that is a vibe, not a
# diagnosis. The `letta-failure-modes` skill is a **reviewer**: you hand it a **snapshot** (a bundle of
# numbers describing your memory design: how many facts in core, whether edges have timestamps,
# how retrieval works...) and it checks **8 known failure modes** from the Letta Leaderboard, each reported as
# `ok`, `warning`, `present` or `unknown`, with *evidence* and a *fix*.
#
# **What you will learn**
# 1. How to build a `MemorySnapshot` by hand (a deliberately **broken** design).
# 2. How to build a snapshot of a **clean** design from real objects (`hierarchical-memory` + `graphiti-incremental-update`).
# 3. How to read the diagnostic report, and its total *severity* score (lower = healthier).
# 4. How to store the verdicts in **Neo4j** and query them with Cypher.
#
# **Jargon:** a **node** is a thing in the graph; a **relationship** links two nodes (here:
# `Architecture -[:HAS_MODE]-> FailureMode`); **Cypher** is Neo4j's query language; **bi-temporal edges** record
# both *when a fact was true* and *when we learned it*; **disconnected components** are islands of nodes with no links
# to each other.
#
# > **Before running:** Neo4j must be running (`docker compose up -d`).

# %% [markdown]
# ## 1. Load the three skills we need

# %%
import graph_helpers as g

diag = g.load_skill("memory", "letta-failure-modes")
hier = g.load_skill("memory", "hierarchical-memory")
graph_mod = g.load_skill("memory", "graphiti-incremental-update")

g.clear_graph()
print("loaded:", diag.__name__, "|", hier.__name__, "|", graph_mod.__name__)

# %% [markdown]
# ## 2. A broken architecture (a hand-built anti-pattern)
#
# We invent numbers describing a bad memory: core stuffed with short-lived clutter (8 of 10),
# no interaction history (`recall_size=0`) while archival holds 200 entries, 15 important facts buried in archival,
# a fragmented graph (30 islands), no timestamps, a linear scan over everything (`O(n)`, meaning it gets slower
# the bigger memory is), no query log and no fact extractor.

# %%
broken_snap = diag.MemorySnapshot(
    core_size=10, core_limit=10, core_durable_count=2, core_short_lived_count=8,
    recall_size=0, archival_size=200, archival_durable_count=15,
    node_count=50, edge_count=10, disconnected_components=30, avg_degree=0.4,
    uses_bi_temporal_edges=False, facts_without_created_at=100,
    facts_without_invalidation_reason_when_invalidated=20,
    retrieval_method="linear-scan", retrieval_complexity_class="O(n)",
    has_query_log=False, extract_fn_present=False, size_stress_test_passed=False,
)
broken_report = diag.diagnose(broken_snap)
print(diag.format_text(broken_report))

# %% [markdown]
# **How to read it:** each mode has a status and a severity from 0 (fine) to 3 (critical). Statuses marked `present`
# are real problems; the report prints the *evidence* (the numbers that triggered it) and a concrete *fix*.

# %%
broken_score = diag.total_score(broken_report)
broken_present = sum(1 for m in broken_report.modes if m.status == "present")
assert broken_present >= 6, f"expected >= 6 failure modes present, got {broken_present}"
print(f"[PASS] broken snapshot triggers {broken_present}/8 failure modes; severity={broken_score}/24")

# %% [markdown]
# ## 3. A clean architecture, composed from real primitives
#
# Now the opposite: we *build* a small healthy memory and read its stats from the other skills.
# * `HierarchicalMemory` gives core/recall/archival sizes (core holds durable production facts; we also log 3 interactions).
# * `graphiti-incremental-update` builds a small knowledge graph from 4 DevOps episodes
#   (sentences like `person:Sarah opened incident:INC-1 ...`) so we get real node and edge counts.

# %%
mem = hier.HierarchicalMemory(core_limit=10)
for c, d in [
    ("Production AWS account is 123456789012", hier.DURABILITY_DURABLE),
    ("Primary region is us-east-1", hier.DURABILITY_DURABLE),
    ("On-call rotation Sarah primary", hier.DURABILITY_DURABLE),
    ("checkout-api uses synchronous http to payments", hier.DURABILITY_DURABLE),
    ("Running aws ec2 describe-instances", hier.DURABILITY_SHORT_LIVED),
    ("Tail 08:00 504 gateway", hier.DURABILITY_SHORT_LIVED),
]:
    mem.add_fact(c, d)
mem.process_interaction("what region is production", "us-east-1")
mem.process_interaction("who's on call", "Sarah")
mem.process_interaction("checkout deps", "payments, inventory")

kg = graph_mod.Graph()
for ep in [
    "person:Sarah opened incident:INC-1 affecting service:checkout-api in region:us-east-1",
    "service:checkout-api depends on service:payments",
    "deployment:v3.5.0 of service:checkout-api went live",
    "person:Sarah closed incident:INC-1",
]:
    graph_mod.add_episode(ep, kg)

mem_diag = mem.diagnostics()
print("hierarchical-memory diagnostics:", {k: v for k, v in mem_diag.items() if k != "warnings"})
print("knowledge graph:", len(kg.nodes), "nodes,", len(kg.edges), "edges")

# %% [markdown]
# ## 4. Fill a snapshot from those numbers and diagnose it
#
# Numbers we measured come straight from the objects. The rest (`bi-temporal`, `hybrid` retrieval, a query log...)
# describe design choices of the clean architecture, so we state them explicitly.

# %%
clean_snap = diag.MemorySnapshot(
    core_size=mem_diag["core_size"], core_limit=mem_diag["core_limit"],
    core_durable_count=mem_diag["core_durable_count"],
    core_short_lived_count=mem_diag["core_short_lived_count"],
    recall_size=mem_diag["recall_size"],
    archival_size=mem_diag["archival_size"],
    archival_durable_count=mem_diag["archival_durable_count"],
    node_count=len(kg.nodes),
    edge_count=len(kg.edges),
    disconnected_components=1,
    avg_degree=2 * len(kg.edges) / max(1, len(kg.nodes)),
    uses_bi_temporal_edges=True,
    facts_without_created_at=0,
    facts_without_invalidation_reason_when_invalidated=0,
    retrieval_method="hybrid",
    retrieval_complexity_class="O(log n)",
    has_query_log=True,
    extract_fn_present=True,
    size_stress_test_passed=True,
)
clean_report = diag.diagnose(clean_snap)
print(diag.format_text(clean_report))

clean_score = diag.total_score(clean_report)
clean_present = sum(1 for m in clean_report.modes if m.status == "present")
assert clean_present == 0, f"expected 0 failure modes in clean architecture, got {clean_present}"
print(f"[PASS] clean snapshot triggers {clean_present}/8 failure modes; severity={clean_score}/24 (production-ready)")

# %% [markdown]
# ## 5. Store both verdicts in Neo4j
#
# The original spike published the scores to a mocked CloudWatch. Here we write them to the graph instead:
# one `(:Architecture)` node per design, one `(:FailureMode)` node per mode, and a `[:HAS_MODE]` relationship that
# carries the `status` and `severity` as properties. Once it is a graph, *"which modes are broken, and where?"* is a one-line query.

# %%
for name, snap, report in [("broken", broken_snap, broken_report), ("clean", clean_snap, clean_report)]:
    g.run("MERGE (a:Architecture {name:$n}) SET a.total_severity = $s",
          n=name, s=diag.total_score(report))
    g.run("""
        MATCH (a:Architecture {name:$n})
        UNWIND $modes AS m
        MERGE (f:FailureMode {name: m.name})
        MERGE (a)-[r:HAS_MODE]->(f)
        SET r.status = m.status, r.severity = m.severity, r.evidence = m.evidence, r.fix = m.fix
    """, n=name, modes=[m.to_dict() for m in report.modes])

g.show(g.run("""
    MATCH (a:Architecture)-[r:HAS_MODE]->(f:FailureMode)
    RETURN f.name AS failure_mode,
           max(CASE WHEN a.name='broken' THEN r.status END) AS broken,
           max(CASE WHEN a.name='clean'  THEN r.status END) AS clean
    ORDER BY failure_mode
"""))

# %% [markdown]
# **How to read it:** one row per failure mode, two columns side by side. The `clean` column should be all `ok`.

# %% [markdown]
# ## 6. Ask the graph questions
#
# *Which modes are `present` in the broken design, worst first, and what is the recommended fix?*

# %%
worst = g.run("""
    MATCH (:Architecture {name:'broken'})-[r:HAS_MODE {status:'present'}]->(f:FailureMode)
    RETURN f.name AS failure_mode, r.severity AS severity, r.fix AS fix
    ORDER BY severity DESC, failure_mode
""")
g.show(worst)

# %% [markdown]
# And the headline comparison — the total severity out of 24 (8 modes × max 3):

# %%
scores = g.run("MATCH (a:Architecture) RETURN a.name AS architecture, a.total_severity AS severity ORDER BY severity DESC")
g.show(scores)
by_name = {r["architecture"]: r["severity"] for r in scores}
assert by_name["broken"] > by_name["clean"], "broken score should exceed clean score"
present_in_db = g.run("MATCH (:Architecture {name:'broken'})-[r:HAS_MODE {status:'present'}]->() RETURN count(r) AS n")[0]["n"]
assert present_in_db == broken_present
cp = g.run("MATCH (:Architecture {name:'clean'})-[r:HAS_MODE {status:'present'}]->() RETURN count(r) AS n")[0]["n"]
assert cp == 0
print(f"[PASS] Neo4j agrees: broken={by_name['broken']} > clean={by_name['clean']}; "
      f"{present_in_db} modes present vs {cp}.")

# %% [markdown]
# ## Recap
# * The skill turns *"my agent forgets things"* into a precise list of **8 orthogonal failure modes**, each with evidence and a fix.
# * A hand-built **broken** snapshot triggered ≥ 6 modes; a **clean** one composed from `hierarchical-memory` and
#   `graphiti-incremental-update` triggered 0.
# * We saved the verdicts in Neo4j as `Architecture -[:HAS_MODE {status, severity}]-> FailureMode`, so reviews can be compared,
#   tracked over time and queried with Cypher.
# * Lower total severity = healthier memory; the same numbers could be turned into an alert in a real monitoring system.
