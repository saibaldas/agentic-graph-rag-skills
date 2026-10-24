# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > **Beginner (tier 1/3) ← you are here**  →  Intermediate (tier 2/3, no notebook for this topic)  →  Advanced (tier 3/3, no notebook for this topic)

# %% [markdown]
# # Spike B (Simplified) — Incremental graph updates (the Graphiti pattern) in Neo4j
#
# **Who this is for:** you are new to GraphRAG and agents. Every step is explained in plain language.
#
# **What is a "spike"?** A small experiment that tests one risky idea. Here the risk is:
# *when new information arrives, can we update only the small part of the graph it affects,
# instead of rebuilding the whole graph?*
#
# **What you will learn**
# 1. What an **episode** is — one new piece of information (a sentence about an incident).
# 2. The three steps of an incremental update: **extract → resolve → update**.
# 3. **Entity resolution** — recognising that "Sarah" and "Sarah(s.chen)" are the same person.
# 4. The **locality invariant**: each update touches only a handful of nodes.
# 5. How to mirror each update into a real **Neo4j** graph and check the result with Cypher.
#
# **Story:** during the checkout-API incident, a stream of 20 short log-like sentences arrives
# (who opened which incident, which deploy broke what...). We turn them into a graph as they arrive.
#
# **Jargon, defined once**
# * **Node** – a thing (a person, a service, an incident). **Relationship** – a link between two
#   nodes. **Cypher** – Neo4j's query language: `MATCH (a)-[:CO_OCCURS_IN]->(b) RETURN a, b`.
# * **Entity extraction** – pulling "things" out of text. The skill uses a simple pattern
#   (`type:Name`); in production an LLM would do this.
# * **Alias** – another name for the same thing (`s.chen` for Sarah).
# * **Locality** – a change stays *local*: only nearby nodes are touched.
#
# > **Before running:** start Neo4j with `docker compose -f docker-compose/docker-compose.yml up -d`.

# %% [markdown]
# ## 1. Load the skill and prepare an empty Neo4j graph
#
# The skill keeps its own small in-memory `Graph` (a Python object) to decide *what changed*.
# We copy each change into Neo4j. The skill file marks Neo4j as exactly the "production swap" seam.

# %%
import graph_helpers as g

gi = g.load_skill("memory", "graphiti-incremental-update")
Graph, add_episode, verify_locality = gi.Graph, gi.add_episode, gi.verify_locality

g.clear_graph()
g.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
print("Neo4j is empty and ready:", g.run("MATCH (n) RETURN count(n) AS nodes"))

# %% [markdown]
# ## 2. One episode, step by step
#
# Take a single sentence and watch the pipeline work. Entities are written `type:Name`.
# * **extract_entities** finds them.
# * **add_episode** resolves each one against the existing graph (`canonical` exact name,
#   `alias`, `fuzzy`, or `new`) and updates only what is needed.

# %%
demo = Graph()
sentence = "person:Sarah(s.chen) opened incident:INC-1042 affecting service:checkout-api in region:us-east-1"
for ent in gi.extract_entities(sentence):
    print(f"extracted: type={ent.type:9s} name={ent.name:12s} aliases={ent.aliases}")

log = add_episode(sentence, demo, episode_id="demo-1")
print("\nresolution:", log["resolution_method_counts"], "| touched nodes:", log["touched_nodes"])

# %% [markdown]
# **How to read it:** 4 entities were extracted; all were `new` (the graph was empty), so 4 nodes
# were created and 6 `co_occurs_in` edges (every pair mentioned together) were added.

# %% [markdown]
# ## 3. The 20-episode stream, mirrored into Neo4j
#
# For each episode we:
# 1. take a snapshot, call `add_episode`, take another snapshot → `verify_locality` says what % of nodes changed;
# 2. copy the **touched nodes** and **new edges** into Neo4j (`MERGE` = "create if missing, else reuse").
#
# We also tag each touched node with `last_episode`, so Neo4j itself can tell us what each episode touched.

# %%
episodes = [
    "person:Sarah(s.chen) opened incident:INC-1042 affecting service:checkout-api in region:us-east-1",
    "person:Marcus joined incident:INC-1042 as secondary on-call",
    "service:checkout-api depends on service:payments and service:inventory",
    "deployment:v3.5.0 of service:checkout-api went live at 22:30 UTC",
    "deployment:v3.5.0 introduced 504 timeouts on service:payments calls",
    "person:Sarah triggered rollback to deployment:v3.4.1 of service:checkout-api",
    "incident:INC-1042 closed by person:Sarah after rollback succeeded",
    "person:Sarah opened incident:INC-1043 for service:inventory in region:us-east-1",
    "incident:INC-1043 traced to deployment:v2.1.0 of service:inventory",
    "team:platform owns service:checkout-api and service:payments",
    "team:fulfillment owns service:inventory",
    "person:Marcus opened incident:INC-1044 for service:payments in region:us-west-2",
    "incident:INC-1044 was a region-specific failover test, not a real outage",
    "person:Sarah(s.chen) approved deployment:v3.6.0 of service:checkout-api",
    "deployment:v3.6.0 fixes the payments-timeout regression from deployment:v3.5.0",
    "service:checkout-api now has region:us-east-1 and region:us-west-2 deployed",
    "team:platform added service:fraud-detection as a dependency for service:checkout-api",
    "incident:INC-1045 opened for service:fraud-detection in region:us-east-1",
    "person:Sarah(s.chen) closed incident:INC-1045",
    "team:platform completed Q1 reliability review covering service:checkout-api and service:payments",
]

graph = Graph()
max_pct_changed, max_touched = 0.0, 0
per_episode = []

for i, ep in enumerate(episodes, 1):
    ep_id = f"ep-{i:03d}"
    edges_before = len(graph.edges)
    before = graph.snapshot()
    log = add_episode(ep, graph, episode_id=ep_id)
    loc = verify_locality(before, graph.snapshot())

    touched = [graph.nodes[nid] for nid in log["touched_ids"]]
    new_edges = graph.edges[edges_before:]
    g.run("""
        UNWIND $nodes AS n
        MERGE (e:Entity {id: n.id})
        ON CREATE SET e.first_episode = $ep
        SET e.name = n.name, e.type = n.type, e.aliases = n.aliases, e.last_episode = $ep
    """, nodes=[{"id": n.id, "name": n.name, "type": n.type, "aliases": n.aliases} for n in touched], ep=ep_id)
    g.run("""
        UNWIND $edges AS x
        MATCH (a:Entity {id: x.s}), (b:Entity {id: x.t})
        CREATE (a)-[:CO_OCCURS_IN {episode: x.ep}]->(b)
    """, edges=[{"s": e.source_id, "t": e.target_id, "ep": ep_id} for e in new_edges])

    max_pct_changed = max(max_pct_changed, loc["percent_changed"])
    max_touched = max(max_touched, log["touched_nodes"])
    per_episode.append({"episode": ep_id, "touched": log["touched_nodes"],
                        "new_nodes": loc["nodes_added"], "graph_size": loc["after_count"],
                        "pct_changed": round(loc["percent_changed"], 1)})

g.show(per_episode, max_rows=20)
print(f"\nFinal (Python): {len(graph.nodes)} nodes, {len(graph.edges)} edges")
print(f"Max % of graph changed by a single episode: {max_pct_changed:.1f}%")
print(f"Max nodes touched by a single episode:      {max_touched}")

# %% [markdown]
# **How to read the table:** `touched` is how many nodes an episode changed; `graph_size` keeps
# growing, so `pct_changed` shrinks. That is **locality**: work per update stays small even as the
# graph grows — the reason this pattern scales to millions of nodes.

# %% [markdown]
# ## 4. Did Neo4j end up with the same graph?
#
# Count what Neo4j holds and compare with the Python graph.

# %%
counts = g.run("""
    MATCH (n:Entity) WITH count(n) AS nodes
    MATCH ()-[r:CO_OCCURS_IN]->() RETURN nodes, count(r) AS relationships
""")[0]
print(counts)
assert counts["nodes"] == len(graph.nodes) and counts["relationships"] == len(graph.edges)
print("[PASS] Neo4j matches the incrementally-built graph.")

g.show(g.run("MATCH (n:Entity) RETURN n.type AS type, count(*) AS how_many ORDER BY how_many DESC, type"))

# %% [markdown]
# ## 5. Entity resolution: one canonical `checkout-api`
#
# `checkout-api` was mentioned in 10+ different sentences. If resolution failed we'd have many
# duplicate nodes. In Cypher: group by (type, name) and look for any group bigger than one.

# %%
g.show(g.run("""
    MATCH (n:Entity {type:'service', name:'checkout-api'})
    OPTIONAL MATCH (n)-[r:CO_OCCURS_IN]-()
    RETURN n.name AS service, count(DISTINCT n) AS canonical_nodes, count(r) AS mentions_with_others
"""))
dupes = g.run("""
    MATCH (n:Entity) WITH n.type AS type, toLower(n.name) AS name, count(*) AS c
    WHERE c > 1 RETURN type, name, c
""")
checkout = g.run("MATCH (n:Entity {type:'service'}) WHERE toLower(n.name) = 'checkout-api' RETURN n.id AS id")
assert len(checkout) == 1 and dupes == [], f"entity resolution failed: {dupes}"
print("[PASS] service:checkout-api is a single canonical node; no duplicate entities anywhere.")

# %% [markdown]
# ## 6. Alias resolution: "Sarah" and "Sarah(s.chen)" are one person
#
# Episode 1 introduced `Sarah` with alias `s.chen`; later sentences used just `Sarah` or the
# alias form again. The skill must keep **one** node and remember the alias.

# %%
sarah = g.run("""
    MATCH (n:Entity {type:'person'})
    WHERE toLower(n.name) = 'sarah' OR 's.chen' IN [a IN n.aliases | toLower(a)]
    RETURN n.name AS name, n.aliases AS aliases, n.first_episode AS first_seen, n.last_episode AS last_seen
""")
g.show(sarah)
assert len(sarah) == 1 and "s.chen" in sarah[0]["aliases"]
print("[PASS] 'Sarah(s.chen)' resolved to ONE canonical node WITH the s.chen alias captured.")

# %% [markdown]
# A payoff of having a graph: **who was involved with checkout-api incidents?**
# (Follow `CO_OCCURS_IN` from a person to an incident that also touches `checkout-api`.)

# %%
g.show(g.run("""
    MATCH (p:Entity {type:'person'})-[:CO_OCCURS_IN]-(i:Entity {type:'incident'})-[:CO_OCCURS_IN]-(:Entity {name:'checkout-api'})
    RETURN DISTINCT p.name AS person, i.name AS incident ORDER BY incident, person
"""))

# %% [markdown]
# ## 7. The locality invariant, measured in Neo4j
#
# Ask Neo4j directly: how many nodes were last touched by each episode?
# No episode should have touched more than a few nodes, and certainly never the whole graph.

# %%
rows = g.run("""
    MATCH (n:Entity) RETURN n.last_episode AS episode, count(n) AS nodes_last_touched
    ORDER BY episode
""")
g.show(rows, max_rows=20)

total = len(graph.nodes)
print(f"\nFinal graph: {total} nodes | max touched in one episode: {max_touched} "
      f"({max_touched / total * 100:.1f}% of the graph)")
assert max_touched < total, "locality invariant broken"
print("[PASS] Locality invariant held — every episode touched far less than the whole graph.")

# %% [markdown]
# ## Recap
# * An **episode** is one new fact; the pipeline is *extract → resolve → incrementally update*.
# * Only **touched** nodes were written to Neo4j (`MERGE` + `SET`), never the whole graph.
# * **Entity resolution** kept `checkout-api` and `Sarah(s.chen)` as single nodes (checked with Cypher).
# * **Locality** held: the biggest single update touched only a few nodes out of ~20+.
# * Production swaps documented in the skill: an LLM extractor and embedding-based fuzzy matching.
