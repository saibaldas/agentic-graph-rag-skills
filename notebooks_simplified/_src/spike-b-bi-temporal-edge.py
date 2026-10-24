# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > **Beginner (tier 1/3) ← you are here**  →  Intermediate (tier 2/3, no notebook for this topic)  →  Advanced (tier 3/3, no notebook for this topic)

# %% [markdown]
# # Spike B (Simplified) — Bi-temporal edges: "what was true at the time of the outage?"
#
# **Who this is for:** you are new to GraphRAG and agents. Every step is explained in plain language.
#
# **What is a "spike"?** A small experiment that tests one risky idea end-to-end. Here:
# *can a graph store facts that change over time, so an agent can ask about the past?*
#
# **What you will learn**
# 1. What **bi-temporal** means: every fact has *two* clocks.
# 2. How the `bi-temporal-edge` skill models a fact as an edge with `valid_from` / `valid_until`.
# 3. How to store those edges in **Neo4j** and ask "what was the config **as of** a date?" in Cypher.
# 4. **Ingestion lag** — facts the agent learned about *late*.
# 5. How to *change* a fact without erasing history (invalidate / supersede).
#
# **Story:** the checkout API had an outage at `2026-03-15 08:00 UTC`. By the time of the
# post-mortem, the config store says the servers are `m5.xlarge`. But the question that matters is
# *what were they at the time of the outage?*
#
# **Jargon, defined once**
# * **Bi-temporal** – two time axes per fact: **valid time** (when it was true in the real world)
#   and **ingestion time** (when our system *learned* it).
# * **Node / relationship / property** – a graph stores *nodes* (things), *relationships*
#   (links), each with *properties* (key/value details such as `valid_from`).
# * **Cypher** – Neo4j's query language: `MATCH (a)-[r]->(b) WHERE ... RETURN ...`.
# * **Point-in-time (“as of”) query** – ask the graph about its state at a past moment.
#
# > **Before running:** start Neo4j with `docker compose -f docker-compose/docker-compose.yml up -d`.

# %% [markdown]
# ## 1. Load the skill and its sample timeline
#
# The skill ships a sample file of 10 timeline facts for `service-checkout-api` (instance type
# changes, deployments, dependencies...). `load_edges` returns Python objects (`TemporalEdge`).

# %%
import os
from datetime import datetime, timedelta, timezone

import graph_helpers as g

bt = g.load_skill("memory", "bi-temporal-edge")
SKILL_DIR = g.SKILLS_DIR / "memory" / "bi-temporal-edge"

edges = bt.load_edges(str(SKILL_DIR / "sample-config-timeline.json"))
print(f"loaded {len(edges)} edges")
for e in bt.history("service-checkout-api", edges):
    vu = e.valid_until.isoformat() if e.valid_until else "now"
    print(f"  [{e.valid_from.date()} -> {vu[:10]}] {e.relationship} = {e.value}")

# %% [markdown]
# **How to read it:** `instance_type` was `t3.medium`, then `t3.large`, then `m5.xlarge`.
# Each row is valid for a *window* of time; `now` means "still current".

# %% [markdown]
# ## 2. Store the timeline in Neo4j
#
# We make one node per entity (`service-checkout-api`, `aws-ec2`, ...) and one **`FACT` relationship**
# per edge. The two clocks become properties:
#
# | property | meaning |
# |---|---|
# | `valid_from` / `valid_until` | the window when the fact was **true in the world** (`valid_until` empty = still true) |
# | `ingested_at` | when **our system** found out |
#
# We pass the dates as ISO text and let Neo4j convert them with `datetime(...)`.

# %%
g.clear_graph()
rows = [{
    "id": e.id, "source": e.source, "target": e.target, "rel": e.relationship,
    "value": e.value, "vf": e.valid_from.isoformat(),
    "vu": e.valid_until.isoformat() if e.valid_until else None,
    "ing": e.ingested_at.isoformat(), "reason": e.invalidation_reason,
    "link_type": e.link_type, "weight": e.weight, "note": e.metadata.get("note"),
} for e in edges]

g.run("""
    UNWIND $rows AS r
    MERGE (s:Entity {name: r.source})
    MERGE (t:Entity {name: r.target})
    CREATE (s)-[:FACT {id: r.id, relationship: r.rel, value: r.value,
                       valid_from: datetime(r.vf),
                       valid_until: CASE WHEN r.vu IS NULL THEN null ELSE datetime(r.vu) END,
                       ingested_at: datetime(r.ing), invalidation_reason: r.reason,
                       link_type: r.link_type, weight: r.weight, note: r.note}]->(t)
""", rows=rows)

g.show(g.run("""
    MATCH (s)-[f:FACT]->(t)
    RETURN f.relationship AS fact, toString(f.value) AS value,
           toString(date(f.valid_from)) AS valid_from,
           toString(date(f.valid_until)) AS valid_until
    ORDER BY fact, valid_from
"""), max_rows=15)

# %% [markdown]
# ## 3. Point-in-time queries at the outage moment
#
# "True at time T" means: `valid_from <= T` **and** (`valid_until` is empty **or** `T < valid_until`).
# That is the whole trick. First the skill's Python answer, then the same question in Cypher.

# %%
outage_ts = datetime.fromisoformat("2026-03-15T08:00:00+00:00")
print("OUTAGE TIME:", outage_ts.isoformat(), "\n")

for rel in ("instance_type", "deployment_version", "region"):
    e = bt.as_of("service-checkout-api", rel, outage_ts, edges)
    print(f"{rel:20s} = {e.value if e else 'unknown':10s} (valid_from={e.valid_from.date() if e else '-'})")

# %% [markdown]
# Now the **same question as Cypher**. `$at` is a parameter holding the timestamp.

# %%
AS_OF = """
    MATCH (:Entity {name: $svc})-[f:FACT {relationship: $rel}]->()
    WHERE f.valid_from <= datetime($at)
      AND (f.valid_until IS NULL OR datetime($at) < f.valid_until)
    RETURN f.value AS value, toString(f.valid_from) AS valid_from
"""
for rel in ("instance_type", "deployment_version", "region"):
    out = g.run(AS_OF, svc="service-checkout-api", rel=rel, at=outage_ts.isoformat())
    print(rel, "->", out)

# %% [markdown]
# **Verification gate:** Neo4j must agree with the skill's pure-Python answer
# (this replaces the old "compare with a mocked AWS" check — the graph is now our source of truth).

# %%
recovered = bt.as_of("service-checkout-api", "instance_type", outage_ts, edges)
neo = g.run(AS_OF, svc="service-checkout-api", rel="instance_type", at=outage_ts.isoformat())
assert recovered.value == "m5.xlarge"
assert len(neo) == 1 and neo[0]["value"] == recovered.value, "Neo4j and skill disagree!"
print("[PASS] Instance type at outage =", recovered.value, "(Neo4j == skill)")

# %% [markdown]
# ## 4. Time travel: what was true *before* the upgrade?
#
# Same query, different `$at`. The graph answers for any date, including "now".

# %%
for label, at in [("before migration", "2026-03-08T12:00:00+00:00"),
                  ("early January",    "2026-01-10T00:00:00+00:00"),
                  ("right now",        datetime.now(timezone.utc).isoformat())]:
    out = g.run(AS_OF, svc="service-checkout-api", rel="instance_type", at=at)
    py = bt.as_of("service-checkout-api", "instance_type", datetime.fromisoformat(at), edges)
    print(f"{label:17s} -> Neo4j: {out[0]['value']:10s} skill: {py.value}")
    assert out[0]["value"] == py.value

# %% [markdown]
# ## 5. Ingestion lag — facts learned late
#
# Ingestion lag = `ingested_at − valid_from`. A big lag means the fact was true for a long time
# before the agent knew it, so reasoning that happened earlier may have used stale data.
# In Cypher, `duration.between(a, b)` computes the gap.

# %%
lagged = g.run("""
    MATCH (:Entity)-[f:FACT]->()
    WITH f, duration.inSeconds(f.valid_from, f.ingested_at).seconds / 3600.0 AS lag_hours
    WHERE lag_hours > 1
    RETURN f.relationship AS fact, toString(f.value) AS value,
           round(lag_hours / 24.0, 1) AS lag_days, f.note AS note
    ORDER BY lag_days DESC
""")
g.show(lagged)

py_lagged = [e for e in edges if e.ingestion_lag() > timedelta(hours=1)]
assert len(lagged) == len(py_lagged) == 1
print("\nSkill agrees:", py_lagged[0].relationship, "lag =", py_lagged[0].ingestion_lag())

# %% [markdown]
# ## 6. Weighted neighbours: causal links beat weak "similar" links
#
# The skill gives each link a type (`entity`, `causal`, `semantic`, `temporal`) and a weight, so
# causal clues (a dependency, a recent deploy) outrank vague similarity. This is the HINDSIGHT idea.
# `score` = `weight × type multiplier` (causal ×1.5, semantic ×0.9...).

# %%
neighbors = bt.weighted_traverse("service-checkout-api", edges, depth=1, at_time=outage_ts)
for n in neighbors:
    print(f"score={n['score']:.2f}  {n['link_type']:9s} -[{n['relationship']}]-> {n['target']}")
assert neighbors[0]["link_type"] == "causal"

# %% [markdown]
# The same facts, but from Neo4j — only what was valid at the outage, strongest link type first:

# %%
g.show(g.run("""
    MATCH (:Entity {name:'service-checkout-api'})-[f:FACT]->(t)
    WHERE f.valid_from <= datetime($at) AND (f.valid_until IS NULL OR datetime($at) < f.valid_until)
    RETURN t.name AS target, f.relationship AS relationship, f.link_type AS link_type, f.weight AS weight
    ORDER BY CASE f.link_type WHEN 'causal' THEN 0 WHEN 'entity' THEN 1 ELSE 2 END, weight DESC
""", at=outage_ts.isoformat()))

# %% [markdown]
# ## 7. Changing a fact without erasing history
#
# Real systems never *delete* the old value. `supersede` closes the old edge (sets `valid_until` and
# a mandatory *reason*) and opens a new one. We do it in Python with a fixed clock, then mirror
# it in Neo4j: update the old relationship, create the new one.

# %%
change_time = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
current = bt.as_of("service-checkout-api", "instance_type", change_time, edges)
new_edge = bt.supersede(current, "m5.2xlarge", reason="scale up after post-mortem",
                        edges=edges, now_fn=lambda: change_time)
print("old edge closed:", current.value, "valid_until =", current.valid_until.date(), "|", current.invalidation_reason)
print("new edge opened:", new_edge.value, "valid_from =", new_edge.valid_from.date())

g.run("""
    MATCH (:Entity {name:'service-checkout-api'})-[old:FACT {id:$old_id}]->(t)
    SET old.valid_until = datetime($when), old.invalidation_reason = $reason
    WITH t
    MATCH (s:Entity {name:'service-checkout-api'})
    CREATE (s)-[:FACT {id:$new_id, relationship:'instance_type', value:$new_value,
                       valid_from: datetime($when), valid_until: null, ingested_at: datetime($when),
                       link_type:'entity', weight:1.0}]->(t)
""", old_id=current.id, when=change_time.isoformat(), reason="scale up after post-mortem",
      new_id=new_edge.id, new_value="m5.2xlarge")

# %% [markdown]
# **Audit trail:** the whole `instance_type` history, including *why* each value ended.
# The outage-time answer must still be `m5.xlarge`, even though "now" says `m5.2xlarge`.

# %%
g.show(g.run("""
    MATCH ()-[f:FACT {relationship:'instance_type'}]->()
    RETURN f.value AS value, toString(date(f.valid_from)) AS from_date,
           toString(date(f.valid_until)) AS until_date, f.invalidation_reason AS why_it_ended
    ORDER BY f.valid_from
"""))

at_outage = g.run(AS_OF, svc="service-checkout-api", rel="instance_type", at=outage_ts.isoformat())
at_now = g.run(AS_OF, svc="service-checkout-api", rel="instance_type", at="2026-05-01T00:00:00+00:00")
assert at_outage[0]["value"] == "m5.xlarge" and at_now[0]["value"] == "m5.2xlarge"
print("PASS: history preserved — outage:", at_outage[0]["value"], "| after change:", at_now[0]["value"])

# %% [markdown]
# ## Recap
# * A **bi-temporal edge** carries *valid time* (when true) and *ingestion time* (when learned).
# * In Neo4j that is just properties on a relationship; "as of T" is one `WHERE` clause.
# * We recovered `m5.xlarge` for the outage moment, matching the skill's pure-Python result.
# * One fact (`ebs_volume_size_gb`) had a 56-day ingestion lag — worth flagging to the agent.
# * Causal links outrank weak semantic ones; superseding keeps a full audit trail.
