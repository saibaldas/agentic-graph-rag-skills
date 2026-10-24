# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > **Beginner (tier 1/3) ← you are here**  →  Intermediate (tier 2/3, no notebook for this topic)  →  Advanced (tier 3/3, no notebook for this topic)

# %% [markdown]
# # Spike B (Simplified) — Three-tier hierarchical memory, stored in Neo4j
#
# **Who this is for:** beginners. Nothing is assumed.
#
# **What is a "spike"?** A small experiment that checks one risky idea works. Here the idea:
# *an AI agent has a tiny "working memory", so what should it forget when that memory is full?*
#
# **What you will learn**
# 1. The **three memory tiers** (borrowed from the Letta / MemGPT design):
#    * **core** – small and fast, like a desk. Always visible to the agent. Has a hard limit.
#    * **recall** – the raw conversation history, like a diary.
#    * **archival** – overflow storage that is searchable, like a filing cabinet.
# 2. **Eviction** – when core is full, one fact is moved (not deleted!) to archival.
# 3. **Durability-aware eviction** – facts marked `durable` ("production region is us-east-1")
#    are protected, while `short-lived` ones ("Running: kubectl get pods") rotate out first.
# 4. How to represent the tiers as **nodes in Neo4j** and ask Cypher where each fact lives.
#
# **Story:** a 5-day investigation of why `checkout-api` is slow. Day 1 the agent learns durable
# facts. Days 2–5 its memory floods with shell output. Can it still answer "what region is
# production?" (durable, still in core) and "what was the circuit-breaker line?" (evicted, still
# findable in archival)?
#
# **Jargon:** a **node** is a thing in the graph, a **label** (like `:CoreFact`) is its category,
# a **relationship** links two nodes, **Cypher** is Neo4j's query language.
#
# > **Before running:** Neo4j must be running (`docker compose up -d`).

# %% [markdown]
# ## 1. Load the skill and a clean database

# %%
import json
import graph_helpers as g

hm = g.load_skill("memory", "hierarchical-memory")
HierarchicalMemory = hm.HierarchicalMemory
DURABLE, SHORT = hm.DURABILITY_DURABLE, hm.DURABILITY_SHORT_LIVED

g.clear_graph()
print("durable protection factor:", hm.DURABILITY_PROTECTION_FACTOR,
      "(durable facts' eviction score is multiplied by 1/this = 10x harder to evict)")

# %% [markdown]
# ## 2. Helper: mirror the memory into Neo4j
#
# The skill keeps memory in Python objects. This helper writes the current state to Neo4j:
# one `(:Memory)` hub node, and every fact as a node labelled **`:CoreFact`** or **`:ArchivalFact`**,
# connected with `[:IN_CORE]` / `[:IN_ARCHIVAL]` relationships. Calling it again rewrites the picture,
# so Neo4j always shows "who lives where right now".

# %%
def sync_to_neo4j(mem):
    g.run("MATCH (f:Fact) DETACH DELETE f")
    g.run("MERGE (m:Memory {name:'agent-memory'}) SET m.core_limit = $lim", lim=mem.core_limit)
    def row(f, tier):
        return {"content": f.content, "durability": f.durability, "access_count": f.access_count,
                "reason": f.eviction_reason, "tier": tier}
    rows = [row(f, "core") for f in mem.core.values()] + [row(f, "archival") for f in mem.archival]
    g.run("""
        UNWIND $rows AS r
        MATCH (m:Memory {name:'agent-memory'})
        CREATE (f:Fact {content: r.content, durability: r.durability,
                        access_count: r.access_count, eviction_reason: r.reason, tier: r.tier})
        FOREACH (_ IN CASE WHEN r.tier = 'core' THEN [1] ELSE [] END |
                 SET f:CoreFact CREATE (m)-[:IN_CORE]->(f))
        FOREACH (_ IN CASE WHEN r.tier = 'archival' THEN [1] ELSE [] END |
                 SET f:ArchivalFact CREATE (m)-[:IN_ARCHIVAL]->(f))
    """, rows=rows)

def tier_summary():
    return g.run("""
        MATCH (f:Fact) RETURN f.tier AS tier, f.durability AS durability, count(*) AS facts
        ORDER BY tier, durability
    """)

# %% [markdown]
# ## 3. Day 1 — the agent learns five durable facts
#
# We create a memory whose core can hold only **10** facts (a tiny limit, on purpose, so that
# eviction happens quickly). Then we add five production-environment facts marked `durable`.

# %%
mem = HierarchicalMemory(core_limit=10)

for content, dur in [
    ("Production AWS account is 123456789012", DURABLE),
    ("Primary region is us-east-1", DURABLE),
    ("On-call rotation: Sarah (primary), Marcus (secondary)", DURABLE),
    ("Checkout API uses synchronous HTTP to payments service", DURABLE),
    ("Service-checkout-api runs EC2 m5.xlarge", DURABLE),
]:
    mem.add_fact(content, dur)

print(f"After Day 1: core={len(mem.core)}, recall={len(mem.recall)}, archival={len(mem.archival)}")
sync_to_neo4j(mem)
g.show(g.run("MATCH (:Memory)-[:IN_CORE]->(f:CoreFact) RETURN f.durability AS durability, f.content AS fact"))

# %% [markdown]
# **How to read it:** core has 5 of 10 slots used, all durable. Nothing has been evicted yet.

# %% [markdown]
# ## 4. Days 2–5 — a flood of short-lived shell output
#
# The agent now runs many commands. Each command and log line is a `short-lived` fact.
# That is 14 new facts for only 5 free slots, so eviction must happen 9 times.
# The question: **which facts get thrown out?**

# %%
short_lived_pool = [
    "Running: aws ec2 describe-instances", "Tail: 08:00 504 gateway",
    "Running: kubectl get pods", "Tail: 08:01 503",
    "Running: gh pr view 1234", "Tail: 08:02 retry-storm",
    "Running: terraform plan", "Tail: 08:03 circuit-breaker tripped",
    "Running: psql -c 'SELECT count(*)'", "Tail: 08:04 db-pool full",
    "Running: redis-cli ping", "Tail: 08:05 p99 8200ms",
    "Running: aws sts get-caller-identity", "Tail: 08:06 page-fired",
]
for s in short_lived_pool:
    mem.add_fact(s, SHORT)

print(f"After flood: core={len(mem.core)}, recall={len(mem.recall)}, archival={len(mem.archival)}")
sync_to_neo4j(mem)
g.show(tier_summary())

# %% [markdown]
# **How to read it:** core is still exactly 10 (its hard limit). The 9 evicted facts are in archival.
# Look at the `durability` column: *all 5 durable facts are still in core*, and every archival
# fact is short-lived. The score used to pick victims is `(access_count+1)/age`, multiplied by 10 for
# durable facts — so durable facts are always the last to go.

# %%
g.show(g.run("""
    MATCH (:Memory)-[:IN_ARCHIVAL]->(f:ArchivalFact)
    RETURN f.content AS evicted_fact, f.durability AS durability, f.eviction_reason AS why
"""), max_rows=12)

# %% [markdown]
# ## 5. Question 1: "what region is production?" (durable → still in core)
#
# `mem.query` searches all three tiers (a simple substring match here; production would use
# vector or BM25 search). Then we ask Neo4j the same thing with Cypher.

# %%
result_region = mem.query("region")
print(json.dumps(result_region["core"], indent=2))
assert any("us-east-1" in r["content"] for r in result_region["core"]), "durable fact should still be in core"

neo = g.run("""
    MATCH (:Memory)-[:IN_CORE]->(f:CoreFact) WHERE toLower(f.content) CONTAINS 'region'
    RETURN f.content AS fact, f.tier AS tier
""")
g.show(neo)
assert any("us-east-1" in r["fact"] for r in neo)
print("[PASS] durable 'us-east-1' fact survived eviction (skill AND Neo4j agree)")

# %% [markdown]
# ## 6. Question 2: "what was the circuit-breaker line?" (short-lived → archival)
#
# That line was evicted from core, but **evicting is not deleting**: archival is still searchable.

# %%
result_cb = mem.query("circuit-breaker")
print(json.dumps(result_cb["archival"], indent=2))
assert any("circuit-breaker" in r["content"] for r in result_cb["archival"])
assert not result_cb["core"], "it should no longer be in core"

neo = g.run("""
    MATCH (:Memory)-[:IN_ARCHIVAL]->(f:ArchivalFact) WHERE f.content CONTAINS 'circuit-breaker'
    RETURN f.content AS fact, f.tier AS tier
""")
g.show(neo)
assert len(neo) == 1
print("[PASS] evicted 'circuit-breaker' fact is still queryable from archival")

# %% [markdown]
# ## 7. Cross-check in Neo4j: no durable fact was ever lost
#
# The original spike checked a mocked AWS EC2 region here. We instead check a cleaner invariant
# directly in the database: *zero durable facts live in archival.*

# %%
rows = g.run("MATCH (f:ArchivalFact {durability:'durable'}) RETURN count(f) AS durable_in_archival")
print(rows)
assert rows[0]["durable_in_archival"] == 0
total = g.run("MATCH (f:Fact) RETURN count(f) AS n")[0]["n"]
assert total == 5 + len(short_lived_pool), "no fact should have been deleted"
print(f"[PASS] 0 durable facts evicted, and all {total} facts are still stored (forgetting != deleting).")

# %% [markdown]
# ## 8. Diagnostics — a health check
#
# `mem.diagnostics()` flags unhealthy patterns, e.g. a core that is mostly short-lived clutter,
# or durable facts landing in archival.

# %%
diag = mem.diagnostics()
print(json.dumps(diag, indent=2))
assert diag["core_durable_count"] == 5 and diag["archival_durable_count"] == 0

# %% [markdown]
# **How to read it:** core has 5 durable + 5 short-lived. `core_short_lived_count / core_size` is
# exactly 50%, which is not *more than* 50%, so there are no warnings. Had the core been
# dominated by clutter, `warnings` would explain it.

# %% [markdown]
# ## Recap
# * **Three tiers**: core (small, bounded), recall (history), archival (searchable overflow).
# * **Eviction** moves the lowest-scoring fact to archival; **durable** facts get a 10× protection factor.
# * After a 14-item flood, all 5 durable facts remained in core, 9 short-lived ones went to archival.
# * We modelled the tiers as `:CoreFact` / `:ArchivalFact` nodes in Neo4j and queried them with Cypher.
# * Forgetting became an **explicit design choice**, not an accident.
