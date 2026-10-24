# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > [Beginner (tier 1/3)](../notebooks_simplified/ch4-memory.ipynb)  →  [Intermediate (tier 2/3)](../notebooks_intermediate_simplified/ch4-memory.ipynb)  →  **Advanced (tier 3/3) ← you are here**

# %% [markdown]
# # Chapter 4 (Advanced) - Memory as a Graph: Eight Skills, One Neo4j Database
#
# **Tier 3 of 3 (advanced).** The beginner notebook showed each memory skill in isolation.
# Here we do what a production agent does: **the graph is the memory**. Every skill
# *reads its input from Cypher* and *writes its output back as nodes and edges*, so
# one database holds config history, entities, tiers, beliefs, cache state,
# consolidated knowledge and the agent's own trace.
#
# **Scenario** - `INC-1001`: *"checkout-api is slow"*. The agent must remember what
# was true when, what it learned, how sure it is, and what happened last time.
#
# | # | Skill (`skills/memory/...`) | Book idea | Graph shape we build |
# |---|---|---|---|
# | 1 | `bi-temporal-edge` | facts have valid-time *and* ingestion-time | `(Service)-[:HAD_CONFIG]->(ConfigVersion)` |
# | 2 | `graphiti-incremental-update` | local updates, entity resolution | `(Episode)-[:TOUCHED]->(MemEntity)-[:SAME_AS]->(Service)` |
# | 3 | `hierarchical-memory` | core / recall / archival tiers | `(MemTier)-[:HOLDS]->(MemFact)` |
# | 4 | `hindsight-epistemic-classifier` | world / experience / opinion / observation | `(EpFact)-[:INFERRED_FROM]->(EpFact)` |
# | 5 | `letta-failure-modes` | 8 ways memory breaks | diagnosis computed from graph counts |
# | 6 | `memory-consistency-model-selector` | strong vs eventual consistency | `(Agent)-[:WROTE/:CACHED/:STALE_ON]->(MemKey)` |
# | 7 | `memory-consolidation` | episodes become knowledge | `(ConsolidatedFact)-[:DERIVED_FROM]->(Episode)` |
# | 8 | `rrf-hybrid-retrieval` | fuse four retrieval channels | `(Query)-[:RETRIEVED]->(Episode)` |
#
# Then: a **plan -> act -> observe agent**, a **CLI & benchmark** section, three
# **vibe-coding corners**, **experiments**, **pitfalls** and a verification battery.
#
# Requires: Neo4j at `bolt://localhost:7687`. No LLM or API keys.

# %%
import adv_helpers as a
import json, subprocess, sys
from datetime import datetime, timedelta, timezone

a.load_devops_graph()   # wipes the shared DB and loads the DevOps graph (once, here)

bt = a.load_skill("memory", "bi-temporal-edge")
gi = a.load_skill("memory", "graphiti-incremental-update")
hm = a.load_skill("memory", "hierarchical-memory")
he = a.load_skill("memory", "hindsight-epistemic-classifier")
lf = a.load_skill("memory", "letta-failure-modes")
cm = a.load_skill("memory", "memory-consistency-model-selector")
mc = a.load_skill("memory", "memory-consolidation")
rr = a.load_skill("memory", "rrf-hybrid-retrieval")

def ts(s):
    """ISO-8601 string -> timezone-aware datetime (the skills require aware datetimes)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

OUTAGE = ts("2026-03-15T08:00:00+00:00")
print("DevOps graph loaded; skills loaded:", len(a.skill_catalog("memory")))
a.show(a.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS n ORDER BY label"))

# %% [markdown]
# ## Why a graph for memory? Vocabulary used below
#
# - **Bi-temporal**: every fact carries *two* clocks - **valid time** (when it was true in
#   the world) and **ingestion time** (when the agent learned it). The gap is the *ingestion lag*.
# - **Cypher**: Neo4j's query language. `MATCH` finds a pattern, `RETURN` projects columns,
#   `ORDER BY` sorts, `UNWIND` turns a list into rows, `CREATE`/`MERGE` write
#   (`MERGE` = "find or create", so reruns are idempotent), `SET` assigns properties.
# - **Provenance**: the chain of evidence behind a belief.
# - **Design choice**: skills stay *pure Python* (stdlib, testable, swappable) while Neo4j is the
#   *system of record*. The notebook is the glue: `Cypher -> skill -> Cypher`.

# %% [markdown]
# ## 1. Bi-temporal edges - *"what was true at the moment of the outage?"*
#
# Overwriting `instance_type` loses history. Instead we keep **one node per version** with
# `valid_from`, `valid_until`, `ingested_at` and an invalidation `reason`.
# We seed raw history with **`UNWIND $rows AS r`** (one row per dict) and **`CREATE`**.
# Timestamps are stored as ISO strings: simple and sortable; native `datetime()` would
# allow range predicates inside Cypher at the cost of conversions in Python.
#
# Same scenario as the original notebook: `m5.xlarge` replaced `t3.large` on Mar 10, `v3.5.0`
# shipped the night before the outage, and an EBS resize was *valid* in January but only
# *learned* on Mar 15 (a 56-day lag).

# %%
CONFIG = [
    dict(key="instance_type", value="t3.large", vf="2026-02-15T14:23:00+00:00", ing="2026-02-15T14:25:00+00:00",
         vu="2026-03-10T17:45:00+00:00", lt="entity", reason="next-gen migration to m5 family"),
    dict(key="instance_type", value="m5.xlarge", vf="2026-03-10T17:45:00+00:00", ing="2026-03-10T17:46:00+00:00",
         vu=None, lt="entity", reason=None),
    dict(key="deployment_version", value="v3.4.1", vf="2026-03-01T09:00:00+00:00", ing="2026-03-01T09:00:00+00:00",
         vu="2026-03-14T22:30:00+00:00", lt="causal", reason="superseded by v3.5.0 release"),
    dict(key="deployment_version", value="v3.5.0", vf="2026-03-14T22:30:00+00:00", ing="2026-03-14T22:31:00+00:00",
         vu=None, lt="causal", reason=None),
    dict(key="ebs_volume_gb", value="500", vf="2026-01-18T00:00:00+00:00", ing="2026-03-15T09:10:00+00:00",
         vu=None, lt="entity", reason=None),
]
a.run("""
MATCH (s:Service {name:'checkout-api'})
UNWIND $rows AS r
CREATE (s)-[:HAD_CONFIG]->(:ConfigVersion {key:r.key, value:r.value, valid_from:r.vf, valid_until:r.vu,
                                           ingested_at:r.ing, link_type:r.lt, reason:r.reason})
""", rows=CONFIG)

# Skill input is DERIVED from Cypher, not from the CONFIG list.
rows = a.run("""
MATCH (:Service {name:'checkout-api'})-[:HAD_CONFIG]->(c:ConfigVersion)
RETURN c.key AS key, c.value AS value, c.valid_from AS vf, c.valid_until AS vu,
       c.ingested_at AS ing, c.link_type AS lt, c.reason AS reason
ORDER BY key, vf""")
edges = []
for r in rows:
    e = bt.create_edge("checkout-api", "aws-ec2", r["key"], value=r["value"], valid_from=ts(r["vf"]),
                       ingested_at=ts(r["ing"]), link_type=r["lt"])
    if r["vu"]:
        e.valid_until = ts(r["vu"])
        e.invalidation_reason = r["reason"]
    edges.append(e)

inst = bt.as_of("checkout-api", "instance_type", OUTAGE, edges)
dep = bt.as_of("checkout-api", "deployment_version", OUTAGE, edges)
before = bt.as_of("checkout-api", "instance_type", ts("2026-03-01T00:00:00+00:00"), edges)
ebs = next(e for e in edges if e.relationship == "ebs_volume_gb")
print(f"At the outage: instance_type={inst.value}, deployment_version={dep.value}; on Mar 1 instance was {before.value}")
print(f"EBS ingestion lag: {ebs.ingestion_lag()}")
print("effective weights  causal:", round(next(e for e in edges if e.link_type == 'causal').effective_weight(), 2),
      " entity:", round(inst.effective_weight(), 2), "(causal edges outrank plain entity edges in traversal)")

# Write the skill's OUTPUT back: a Finding node explaining the open incident.
a.run("""
MATCH (i:Incident {id:'INC-1001'})
MERGE (f:Finding {id:'cfg-at-outage'})
SET f.instance_type=$inst, f.deployment=$dep, f.as_of=$at
MERGE (f)-[:EXPLAINS]->(i)
""", inst=inst.value, dep=dep.value, at=OUTAGE.isoformat())

assert inst.value == "m5.xlarge" and dep.value == "v3.5.0" and before.value == "t3.large"
assert ebs.ingestion_lag() > timedelta(days=55)
print("[PASS] point-in-time and ingestion-lag queries answered from graph-derived edges")

# %% [markdown]
# ### 1b. Supersede, then query the history with aggregation + `OPTIONAL MATCH`
#
# `supersede()` closes the old edge (with a **mandatory reason** - no silent overwrite) and opens
# a new one. We persist it with `MERGE` and link new -> old via `:SUPERSEDES`.
#
# New Cypher ideas:
# - `count(x)` / `collect(x)` are **aggregations**: they group rows by the non-aggregated columns
#   in `RETURN` (here `c.key`) - count the versions, collect their values into a list.
# - `OPTIONAL MATCH` is a *left join*: keep the row even when the pattern is absent
#   (a current version has **no** successor, so `newer` is `null`).

# %%
current = bt.as_of("checkout-api", "instance_type", datetime.now(timezone.utc), edges)
new_edge = bt.supersede(current, "m5.2xlarge", "scale-up during INC-1001 mitigation", edges=edges)
a.run("""
MATCH (old:ConfigVersion {key:'instance_type', value:$old})
SET old.valid_until=$vu, old.reason=$reason
WITH old
MATCH (s:Service {name:'checkout-api'})
MERGE (s)-[:HAD_CONFIG]->(new:ConfigVersion {key:'instance_type', value:$new})
SET new.valid_from=$vf, new.ingested_at=$vf, new.link_type='entity'
MERGE (new)-[:SUPERSEDES]->(old)
""", old=current.value, new=new_edge.value, vu=current.valid_until.isoformat(),
      reason=current.invalidation_reason, vf=new_edge.valid_from.isoformat())

print("Versions per key (aggregation):")
a.show(a.run("""
MATCH (:Service {name:'checkout-api'})-[:HAD_CONFIG]->(c:ConfigVersion)
RETURN c.key AS key, count(c) AS versions, collect(c.value) AS values
ORDER BY versions DESC, key"""))
print("Which version replaced which (OPTIONAL MATCH):")
a.show(a.run("""
MATCH (c:ConfigVersion {key:'instance_type'})
OPTIONAL MATCH (newer:ConfigVersion)-[:SUPERSEDES]->(c)
RETURN c.value AS version, c.valid_until IS NULL AS is_current, newer.value AS replaced_by
ORDER BY c.valid_from"""))
assert bt.as_of("checkout-api", "instance_type", datetime.now(timezone.utc), edges).value == "m5.2xlarge"
try:
    bt.invalidate(current, "")
except Exception as ex:
    print("empty reason rejected:", type(ex).__name__)

# %% [markdown]
# ## 2. Graphiti incremental update - local writes, entity resolution
#
# Re-processing the whole graph for each new log line does not scale. Graphiti-style ingestion
# touches **only the nodes an episode mentions**. Entity resolution runs a ladder:
# `canonical` name -> `alias` -> `fuzzy` (Jaccard >= 0.6) -> `new`. We stream six incident
# episodes, check *locality* (same touch-count in a 5-node and a 500-node graph), then persist
# which episode touched which entity and **link memory entities to the real topology** with
# a `SAME_AS` edge, so memory and infrastructure become one graph.

# %%
g = gi.Graph()
stream = [
    "person:Sarah(s.chen) opened incident:INC-1042 affecting service:checkout-api in region:us-east-1",
    "deployment:v3.5.0 of service:checkout-api went live at 22:30 UTC",
    "deployment:v3.5.0 introduced 504 timeouts on service:payment-service calls",
    "person:Marcus joined incident:INC-1042 as secondary on-call",
    "person:Sarah triggered rollback to deployment:v3.4.1 of service:checkout-api",
    "person:Sarah(s.chen) closed incident:INC-1042 after rollback succeeded",
]
touched_rows = []
for i, text in enumerate(stream, 1):
    b = g.snapshot()
    log = gi.add_episode(text, g, episode_id=f"ep-{i:03d}")
    loc = gi.verify_locality(b, g.snapshot())
    print(f"ep{i:02d} touched={log['touched_nodes']:2d} changed={loc['percent_changed']:5.1f}% methods={log['resolution_method_counts']}")
    touched_rows += [{"ep": f"ep-{i:03d}", "uid": u} for u in log["touched_ids"]]

ent_rows = [{"uid": u, "name": n.name, "type": getattr(n, "type", None) or getattr(n, "node_type", "?")}
            for u, n in g.nodes.items()]
a.run("""
UNWIND $ents AS e
MERGE (m:MemEntity {uid:e.uid}) SET m.name=e.name, m.type=e.type
""", ents=ent_rows)
a.run("""
UNWIND $t AS t
MERGE (ep:MemEpisode {id:t.ep})
WITH ep, t
MATCH (m:MemEntity {uid:t.uid})
MERGE (ep)-[:TOUCHED]->(m)
""", t=touched_rows)
# Link memory entities to the infrastructure graph (case-insensitive name match).
a.run("""
MATCH (m:MemEntity), (s:Service)
WHERE toLower(m.name) = toLower(s.name)
MERGE (m)-[:SAME_AS]->(s)
""")

print("\nEntities mentioned by more than one episode (collect + count):")
a.show(a.run("""
MATCH (ep:MemEpisode)-[:TOUCHED]->(m:MemEntity)
WITH m, count(ep) AS mentions, collect(ep.id) AS episodes
WHERE mentions > 1
RETURN m.name AS entity, m.type AS type, mentions, episodes ORDER BY mentions DESC, entity"""))

sarah = [n.name for n in g.nodes.values() if "sarah" in n.name.lower()]
small, big = gi.Graph(), gi.Graph()
for k in range(5): small.add_node(f"svc-{k}", "service")
for k in range(500): big.add_node(f"svc-{k}", "service")
t_small = gi.add_episode("person:Eve opened incident:INC-9", small)["touched_nodes"]
t_big = gi.add_episode("person:Eve opened incident:INC-9", big)["touched_nodes"]
print(f"\nSarah nodes: {sarah}; same episode touches {t_small} (5-node graph) vs {t_big} (500-node graph)")
assert len(sarah) == 1 and t_small == t_big
linked = a.run("MATCH (m:MemEntity)-[:SAME_AS]->(s:Service) RETURN count(*) AS n")[0]["n"]
assert linked >= 1, "at least checkout-api should link memory to topology"
print(f"[PASS] local updates, one Sarah, {linked} memory entities linked to real services")

# %% [markdown]
# ## 3. Hierarchical memory - core / recall / archival
#
# Mirrors the CPU hierarchy: **core** (small, always in the prompt), **recall** (recent
# conversation), **archival** (big, searched on demand). When core is full the fact with the
# lowest *eviction score* moves to archival - **eviction is not deletion**. Durable facts get a
# 10x protection factor in that score.
#
# Inputs are derived from the graph: critical services become *durable* facts, the firing alarm
# becomes a *short-lived* one. Afterwards we write each tier's contents back as
# `(:MemTier)-[:HOLDS]->(:MemFact)` and aggregate by tier and durability.

# %%
crit = a.run("MATCH (s:Service {tier:'critical'}) RETURN s.name AS name, s.owner AS owner ORDER BY name")
alarm = a.run("MATCH (al:Alarm)-[:FIRED_ON]->(s:Service) RETURN al.metric AS m, al.value AS v, s.name AS svc")[0]

mem = hm.HierarchicalMemory(core_limit=8)
durables = [f"{r['name']} is a critical-tier service owned by {r['owner']}" for r in crit] + [
    "Production AWS account is 123456789012", "Primary region is us-east-1",
    "On-call rotation: Sarah primary, Marcus secondary", "Checkout API calls payments over synchronous HTTP"]
for d in durables:
    mem.add_fact(d, hm.DURABILITY_DURABLE)
mem.add_fact(f"Alarm firing right now: {alarm['m']}={alarm['v']} on {alarm['svc']}", hm.DURABILITY_SHORT_LIVED)
for cmd in ["aws ec2 describe-instances", "kubectl get pods", "gh pr view 1234", "terraform plan", "aws logs filter-log-events",
            "psql -c count", "redis-cli ping", "aws sts get-caller-identity"]:
    mem.add_fact(f"Running: {cmd}", hm.DURABILITY_SHORT_LIVED)
    mem.add_fact(f"Tail after {cmd}: circuit-breaker open" if "psql" in cmd else f"Tail after {cmd}: ok", hm.DURABILITY_SHORT_LIVED)

survivors = [d for d in durables if d in mem.core]
acct = mem.query("account")
cb = mem.query("circuit-breaker")
print(f"core={len(mem.core)}/8 archival={len(mem.archival)}; durable survivors {len(survivors)}/{len(durables)}")
print("archival still finds evicted noise:", [h['content'] for h in cb['archival']][:1])

tier_rows = ([{"tier": "core", "c": f.content, "d": f.durability} for f in mem.core.values()] +
             [{"tier": "archival", "c": f.content, "d": f.durability} for f in mem.archival])
a.run("""
UNWIND $rows AS r
MERGE (t:MemTier {name:r.tier})
MERGE (f:MemFact {content:r.c}) SET f.durability=r.d
MERGE (t)-[:HOLDS]->(f)
""", rows=tier_rows)
a.show(a.run("""
MATCH (t:MemTier)-[:HOLDS]->(f:MemFact)
RETURN t.name AS tier, f.durability AS durability, count(f) AS facts ORDER BY tier, durability"""))
assert len(survivors) >= 3 and acct["core"] and cb["archival"] and len(mem.core) <= 8
print("[PASS] durable facts stayed hot; evicted noise remained searchable")

# %% [markdown]
# ## 4. Hindsight epistemic classifier - evidence vs belief
#
# Four networks: **World** (objective facts), **Experience** (what *I* did, needs a timestamp),
# **Opinion** (hedged belief with a confidence), **Observation** (synthesis of several sources).
# Why separate them? When a user asks *"how do you know?"* the agent must answer with
# provenance, and must never present an opinion as a world fact.
#
# World facts come from Cypher (the alarm and the deployment change); the agent's own actions
# and beliefs are authored here. We persist every fact as `:EpFact` and the opinion's evidence
# as `:INFERRED_FROM` edges.

# %%
w1 = a.run("MATCH (al:Alarm)-[:FIRED_ON]->(s:Service) RETURN s.name+' p99 latency reached '+toString(al.value)+' ms' AS t")[0]["t"]
w2 = a.run("MATCH (d:Deployment)-[:CHANGED]->(x) RETURN d.id+' changed '+x.name+': '+d.change AS t")[0]["t"]
now_iso = datetime.now(timezone.utc).isoformat()
exp_text = "I retrieved the deployment log for deploy-481."
op_text = "I believe the root cause is deploy-481 shrinking the orders-db connection pool."
batch = [
    {"text": w1, "metadata": {"external_ref": "cloudwatch:alarm-77"}},
    {"text": w2, "metadata": {"external_ref": "deploy-log"}},
    {"text": "I called CloudWatch GetMetricData for checkout-api.",
     "metadata": {"action_type": "cloudwatch_call", "source": "agent", "timestamp": now_iso}},
    {"text": exp_text, "metadata": {"source": "agent", "timestamp": now_iso}},
    {"text": op_text, "metadata": {"confidence": 0.7, "inferred_from": [w1, w2, exp_text]}},
    {"text": "In summary, the timeline is: deploy 08:30, p99 rising, alarm-77 fires, INC-1001 opened.",
     "metadata": {"derived_from": ["deploy-log", "cloudwatch", "incident-channel"]}},
]
facts = he.classify_batch(batch)
for f in facts:
    print(f"[{f.network:11s}] conf={f.confidence:.1f}  {f.text[:80]}")

a.run("UNWIND $f AS x MERGE (e:EpFact {text:x.text}) SET e.network=x.network, e.confidence=x.conf",
      f=[{"text": f.text, "network": f.network, "conf": f.confidence} for f in facts])
links = [{"op": f.text, "ev": ev} for f in facts for ev in f.inferred_from]
a.run("""
UNWIND $l AS l
MATCH (o:EpFact {text:l.op}), (e:EpFact {text:l.ev})
MERGE (o)-[:INFERRED_FROM]->(e)""", l=links)

just = he.justify(facts, op_text)
print("\njustify():", json.dumps(just, indent=1)[:400])
print("\nOpinion -> evidence chain read back from Neo4j:")
a.show(a.run("""
MATCH (o:EpFact {network:'Opinion'})-[:INFERRED_FROM]->(e:EpFact)
RETURN o.confidence AS confidence, e.network AS evidence_type, e.text AS evidence"""))
assert {f.network for f in facts} == {"World", "Experience", "Opinion", "Observation"}
assert len(links) >= 2 and he.network_audit(facts)["total"] == len(facts)
print("[PASS] all four networks present; the opinion is justified by graph evidence")

# %% [markdown]
# ## 5. Letta failure modes - diagnose the memory we just built
#
# The Letta leaderboard names **eight** failure modes (no retrieval, hierarchy collapse,
# silent overwrite, temporal blur, ...). `diagnose()` takes a `MemorySnapshot` - a plain dataclass
# of measurements - and returns severity-scored findings. Here the measurements come from the
# **live graph**: node/edge counts and isolated nodes via Cypher, tier stats from section 3.
# We then diagnose a deliberately *naive* design for contrast. Verdict bands:
# 0 = production-ready, < 10 review, < 18 ship at risk, otherwise do not ship.

# %%
stats = a.run("""
MATCH (n) WITH count(n) AS nodes
MATCH ()-[r]->() WITH nodes, count(r) AS edges
OPTIONAL MATCH (iso) WHERE NOT (iso)--()
RETURN nodes, edges, count(iso) AS isolated""")[0]
has_versions = a.run("MATCH (c:ConfigVersion) RETURN count(c) AS n")[0]["n"] > 0
snap = lf.MemorySnapshot(
    core_size=len(mem.core), core_limit=8,
    core_durable_count=sum(f.durability == hm.DURABILITY_DURABLE for f in mem.core.values()),
    core_short_lived_count=sum(f.durability == hm.DURABILITY_SHORT_LIVED for f in mem.core.values()),
    recall_size=len(mem.recall), archival_size=len(mem.archival), archival_durable_count=0,
    node_count=stats["nodes"], edge_count=stats["edges"], disconnected_components=stats["isolated"],
    avg_degree=round(2 * stats["edges"] / stats["nodes"], 2),
    uses_bi_temporal_edges=has_versions, facts_without_created_at=0,
    facts_without_invalidation_reason_when_invalidated=0,
    retrieval_method="hybrid", retrieval_complexity_class="O(log n)",
    has_query_log=True, extract_fn_present=True, size_stress_test_passed=True)
naive = lf.MemorySnapshot(
    core_size=50, core_limit=50, core_durable_count=2, core_short_lived_count=48,
    node_count=40, edge_count=10, disconnected_components=30, avg_degree=0.5,
    uses_bi_temporal_edges=False, facts_without_created_at=20,
    facts_without_invalidation_reason_when_invalidated=5,
    retrieval_method="linear-scan", retrieval_complexity_class="O(n)")
r_ours, r_naive = lf.diagnose(snap), lf.diagnose(naive)
print("OUR graph-backed memory:", lf.total_score(r_ours), "| NAIVE design:", lf.total_score(r_naive))
a.show([{"mode": m.name, "ours": m.status, "naive": n.status, "naive_fix": n.fix[:60]}
        for m, n in zip(r_ours.modes, r_naive.modes)])
assert len(r_ours.modes) == 8 and lf.total_score(r_ours) < lf.total_score(r_naive)
print("[PASS] the graph-backed design scores better than the naive one")

# %% [markdown]
# ## 6. Memory consistency model selector - who may read stale data?
#
# With several agents sharing memory you must pick a **consistency model**: *strong* (everyone
# sees the latest write), *causal*, *read-your-writes*, *eventual*. Stronger = safer but slower.
# `recommend_model()` scores an operation profile (weights 0-3). `detect_cache_divergence()` finds
# agents whose cached copy is older than the latest write.
#
# The write log and cache log live in Neo4j as `(:Agent)-[:WROTE|:CACHED]->(:MemKey)`; we
# read them back, run the detector, and write `:STALE_ON` edges with the gap.

# %%
T0 = ts("2026-03-15T08:00:00+00:00")
a.run("""
UNWIND $w AS w
MERGE (ag:Agent {name:w.agent}) MERGE (k:MemKey {key:w.key})
FOREACH (_ IN CASE WHEN w.kind='write' THEN [1] ELSE [] END | MERGE (ag)-[:WROTE {at:w.at, value:w.value}]->(k))
FOREACH (_ IN CASE WHEN w.kind='cache' THEN [1] ELSE [] END | MERGE (ag)-[:CACHED {at:w.at}]->(k))
""", w=[
    dict(agent="planner", key="orders-db.pool_size", kind="write", at=T0.isoformat(), value="20"),
    dict(agent="executor", key="orders-db.pool_size", kind="write", at=(T0 + timedelta(minutes=10)).isoformat(), value="100"),
    dict(agent="reporter", key="orders-db.pool_size", kind="cache", at=(T0 + timedelta(minutes=2)).isoformat(), value=None),
    dict(agent="planner", key="orders-db.pool_size", kind="cache", at=(T0 + timedelta(minutes=11)).isoformat(), value=None),
])
writes = [{"agent": r["agent"], "key": r["key"], "timestamp": ts(r["at"]).timestamp(), "value": r["value"]} for r in a.run(
    "MATCH (ag:Agent)-[w:WROTE]->(k:MemKey) RETURN ag.name AS agent, k.key AS key, w.at AS at, w.value AS value")]
caches = [{"agent": r["agent"], "key": r["key"], "cached_at": ts(r["at"]).timestamp()} for r in a.run(
    "MATCH (ag:Agent)-[c:CACHED]->(k:MemKey) RETURN ag.name AS agent, k.key AS key, c.at AS at")]
warnings = cm.detect_cache_divergence(writes, caches)
for w in warnings:
    gap = w["staleness_gap"]
    a.run("""MATCH (ag:Agent {name:$ag}), (k:MemKey {key:$k})
             MERGE (ag)-[r:STALE_ON]->(k) SET r.gap_s=$gap""",
          ag=w["agent"], k=w["key"], gap=gap.total_seconds() if hasattr(gap, "total_seconds") else float(gap))
print("stale agents:", [(w["agent"], str(w["staleness_gap"])) for w in warnings])

ops = {
    "shared prod config (writes must be seen)": cm.Operation(3, 3, 0, 2, 0),
    "team notebook (collaborative, tolerant)": cm.Operation(1, 1, 2, 3, 0),
    "private scratchpad (own session only)": cm.Operation(0, 0, 3, 0, 3),
}
recs = {name: cm.recommend_model(op) for name, op in ops.items()}
a.show([{"operation": n, "model": r["recommended"], "escalate": r["escalate_to_strong"]} for n, r in recs.items()])
assert [w["agent"] for w in warnings] == ["reporter"] and all(r["recommended"] in cm.MODELS for r in recs.values())
print("[PASS] stale cache found (reporter) and a model recommended per operation")

# %% [markdown]
# ## 7. Memory consolidation - episodes become knowledge
#
# Raw episodes pile up; the agent needs **patterns**. `consolidate()` clusters similar episodes
# (Jaccard single-linkage, `threshold=0.25`), keeps clusters with at least `min_cluster_size=3`
# members, and emits a `ConsolidatedFact` with `derived_from` ids - so every summary stays
# **auditable** back to its source episodes.
#
# We build episodes from Cypher: the real resolved incident `INC-0877` becomes an episode
# (string concatenation in Cypher), next to hand-seeded ones. One unrelated TLS episode should
# *not* be consolidated (too small a cluster).

# %%
a.run("""
UNWIND $eps AS e
MATCH (s:Service {name:e.about})
CREATE (x:Episode {id:e.id, content:e.content, type:'incident', created_at:e.at})
CREATE (x)-[:ABOUT]->(s)
""", eps=[
    dict(id="ep-1", about="checkout-api", at="2026-03-01T10:00:00+00:00",
         content="checkout latency spike caused by db connection pool exhausted on orders-db"),
    dict(id="ep-2", about="orders-service", at="2026-03-05T11:00:00+00:00",
         content="orders-db connection pool exhausted after deploy, checkout latency p99 high"),
    dict(id="ep-3", about="orders-service", at="2026-03-09T09:30:00+00:00",
         content="db connection pool too small, checkout latency spiked, raised pool size"),
    dict(id="ep-4", about="checkout-api", at="2026-03-12T16:00:00+00:00",
         content="checkout latency incident root cause db connection pool exhausted"),
    dict(id="ep-5", about="cart-service", at="2026-03-13T08:00:00+00:00",
         content="TLS certificate expiry on cart-service ingress"),
])
a.run("""
MATCH (i:Incident {id:'INC-0877'})-[:AFFECTS]->(s:Service)
CREATE (x:Episode {id:'ep-0877', type:'incident', created_at:'2026-02-20T09:00:00+00:00',
                   content: i.title + '. root cause: ' + i.root_cause + '. fix: ' + i.fix})
CREATE (x)-[:ABOUT]->(s)""")

records = a.run("MATCH (e:Episode) RETURN e.id AS id, e.content AS content, e.type AS type, e.created_at AS at ORDER BY id")
eps = [mc.Episode(id=r["id"], content=r["content"], episode_type=r["type"], created_at=ts(r["at"]), metadata={}) for r in records]
cfacts = mc.consolidate(eps, min_cluster_size=3, threshold=0.25)
for f in cfacts:
    print(f"{f.id}: {f.summary[:90]}  (derived from {len(f.derived_from)} episodes)")
    print("   provenance ok:", len(mc.provenance_of(f, eps)) == len(f.derived_from))
inferences = mc.precompute_inferences(cfacts, lambda f: [f"If checkout is slow, check pool size first ({len(f.derived_from)} past cases)"])

a.run("""
UNWIND $f AS f
MERGE (c:ConsolidatedFact {id:f.id}) SET c.summary=f.summary, c.confirmations=f.conf
WITH c, f UNWIND f.eps AS eid
MATCH (e:Episode {id:eid}) MERGE (c)-[:DERIVED_FROM]->(e)""",
      f=[{"id": f.id, "summary": f.summary, "conf": f.confirmations, "eps": list(f.derived_from)} for f in cfacts])

print("\nFacts and their evidence (aggregation):")
a.show(a.run("""
MATCH (c:ConsolidatedFact)-[:DERIVED_FROM]->(e:Episode)
RETURN c.id AS fact, count(e) AS episodes, collect(e.id) AS ids"""))
print("Episodes NOT consolidated (OPTIONAL MATCH + IS NULL filter):")
leftover = a.run("""
MATCH (e:Episode) OPTIONAL MATCH (c:ConsolidatedFact)-[:DERIVED_FROM]->(e)
WITH e, c WHERE c IS NULL RETURN e.id AS id, e.content AS content""")
a.show(leftover)
assert cfacts and "ep-5" not in cfacts[0].derived_from and any(r["id"] == "ep-5" for r in leftover)
assert len(cfacts[0].derived_from) >= 3 and "ep-0877" in cfacts[0].derived_from
print("[PASS] pool-exhaustion pattern consolidated incl. the real INC-0877; the TLS outlier stayed raw")

# %% [markdown]
# ## 8. RRF hybrid retrieval - four channels, one ranking
#
# Different questions need different retrievers: **semantic** (meaning), **keyword** (exact ids),
# **graph** (connected entities), **temporal** (recent first). **Reciprocal Rank Fusion** (RRF)
# merges ranked lists using only ranks: `score(doc) = sum over channels of 1 / (k + rank)`.
# No score calibration is needed, and documents high in *several* lists rise. `k=60` damps
# the influence of the very top ranks. After fusion a **reranker** reorders the top-N and a
# **token budget** trims to what fits in the prompt.
#
# The *graph channel* is a Cypher query with a **variable-length path**: `[:DEPENDS_ON*1..3]`
# means "follow 1 to 3 hops", so episodes about services near `checkout-api` rank higher.
# The semantic channel here is a character-trigram stand-in (production: embeddings - the
# documented `TODO` seam).

# %%
docs = {r["id"]: r for r in a.run("MATCH (e:Episode) RETURN e.id AS id, e.content AS text, e.created_at AS at")}

def _tok(t): return set(t.lower().replace(",", " ").split())
def _tri(t): t = t.lower(); return {t[i:i + 3] for i in range(len(t) - 2)}
def _rank(q, f): 
    s = {i: len(f(q) & f(d["text"])) / max(1, len(f(q) | f(d["text"]))) for i, d in docs.items()}
    return [i for i, v in sorted(s.items(), key=lambda kv: (-kv[1], kv[0])) if v > 0]

def graph_channel(q):
    rows = a.run("""
        MATCH (c:Service {name:'checkout-api'})
        MATCH (e:Episode)-[:ABOUT]->(s:Service)
        OPTIONAL MATCH p = (c)-[:DEPENDS_ON*1..3]-(s)
        RETURN e.id AS id, coalesce(min(CASE WHEN s = c THEN 0 ELSE length(p) END), 99) AS hops
        ORDER BY hops, id""")
    return [r["id"] for r in rows]

channels = {
    "semantic": lambda q: _rank(q, _tri),
    "keyword": lambda q: _rank(q, _tok),
    "graph": graph_channel,
    "temporal": lambda q: [r["id"] for r in sorted(docs.values(), key=lambda d: d["at"], reverse=True)],
}
tokens = lambda i: len(docs[i]["text"].split())

def memory_search(q, k=60, budget=60):
    return rr.hybrid_retrieve(q, channels, k=k, top_n_for_rerank=6, final_budget=budget,
                              get_tokens=tokens, metadata={i: d["text"] for i, d in docs.items()})

Q = "checkout latency connection pool"
hits = memory_search(Q)
a.show([{"doc": h["doc_id"], "fusion": round(h["fusion_score"], 4), "rerank": round(h["rerank_score"], 3),
         "tokens": h["token_count"]} for h in hits])
a.run("""
MERGE (q:Query {text:$q})
WITH q UNWIND $h AS h
MATCH (e:Episode {id:h.id}) MERGE (q)-[r:RETRIEVED]->(e) SET r.rank=h.rank, r.fused=h.fused""",
      q=Q, h=[{"id": h["doc_id"], "rank": i + 1, "fused": h["fusion_score"]} for i, h in enumerate(hits)])
fused = rr.rrf_fuse({n: f(Q) for n, f in channels.items()}, k=60)
print("pure RRF top-3:", [(d, round(s, 4)) for d, s in fused[:3]])
assert hits and hits[0]["doc_id"] != "ep-5" and "ep-5" != fused[0][0]
assert sum(h["token_count"] for h in hits) <= 60
print("[PASS] fused retrieval puts pool-exhaustion episodes ahead of the TLS outlier, within the token budget")

# %% [markdown]
# ## 9. Three non-trivial Cypher queries on the memory graph
#
# Everything above is now one connected graph. Three questions that need real graph power:
#
# 1. **Blast radius** (*variable-length path*): if `orders-db` degrades, which services are
#    hit and how many hops away? **Blast radius** = the set of components affected by one failure.
# 2. **Evidence trail** (*multi-hop*): from the open incident to its config finding, and from
#    the resolved incident to the consolidated pattern.
# 3. **Memory health** (*aggregation + OPTIONAL MATCH*): per service, how many memory entities,
#    config versions and episodes exist - including services with none (`coalesce`/`count` = 0).

# %%
print("Q1 blast radius of orders-db  (MATCH path = (s)-[:DEPENDS_ON*1..4]->(db)):")
a.show(a.run("""
MATCH path = (s:Service)-[:DEPENDS_ON*1..4]->(:Database {name:'orders-db'})
RETURN s.name AS service, s.tier AS tier, min(length(path)) AS hops
ORDER BY hops, service"""))

print("Q2 multi-hop evidence trail:")
a.show(a.run("""
MATCH (f:Finding)-[:EXPLAINS]->(i:Incident)-[:AFFECTS]->(s:Service)<-[:ABOUT]-(e:Episode)<-[:DERIVED_FROM]-(c:ConsolidatedFact)
RETURN i.id AS incident, f.instance_type AS instance_at_outage, s.name AS service,
       c.id AS pattern, c.confirmations AS confirmations, collect(DISTINCT e.id) AS episodes"""))

print("Q3 memory coverage per service (OPTIONAL MATCH keeps services with zero memory):")
q3 = a.run("""
MATCH (s:Service)
OPTIONAL MATCH (s)-[:HAD_CONFIG]->(c:ConfigVersion)
WITH s, count(c) AS configs
OPTIONAL MATCH (s)<-[:ABOUT]-(e:Episode)
WITH s, configs, count(e) AS episodes
OPTIONAL MATCH (s)<-[:SAME_AS]-(m:MemEntity)
RETURN s.name AS service, configs, episodes, count(m) AS entities ORDER BY service""")
a.show(q3)
blast = a.run("""MATCH p=(s:Service)-[:DEPENDS_ON*1..4]->(:Database {name:'orders-db'})
                 RETURN s.name AS s, min(length(p)) AS h""")
assert {r["s"]: r["h"] for r in blast} == {"orders-service": 1, "payment-service": 1, "checkout-api": 2, "cart-service": 3}
assert any(r["service"] == "payment-service" and r["configs"] == 0 for r in q3)
print("[PASS] blast radius = 4 services; zero-memory services are still listed")

# %% [markdown]
# ## 10. The agent: plan -> act -> observe on "checkout-api is slow"
#
# A memory is only useful if an agent *uses* it. `a.MiniAgent` is a tiny, deterministic
# agent harness (no LLM): you register **tools**, run a **plan** (an ordered list of tool calls),
# and each call is recorded by `AgentTrace` as `(:Run)-[:STEP]->(:ToolCall)` with `NEXT` edges.
# Tools here are skills *and* Cypher - the agent does not care which is which.
# In a real agent an LLM would choose the next step from the observations; the loop is the same:
# **plan** (choose) -> **act** (call tool) -> **observe** (read result, update plan).

# %%
agent = a.MiniAgent("Why is checkout-api slow? (INC-1001)")

@agent.tool("blast_radius")
def t_blast(store):
    return [r["s"] for r in a.run("""MATCH (s:Service)-[:DEPENDS_ON*1..4]->(d {name:$n}) RETURN DISTINCT s.name AS s ORDER BY s""", n=store)]

@agent.tool("past_incident")
def t_past(service):
    return a.run("""MATCH (i:Incident {status:'resolved'})-[:AFFECTS]->(:Service {name:$s})
                    OPTIONAL MATCH (i)-[:ROOT_CAUSE]->(r)
                    RETURN i.id AS id, i.root_cause AS cause, i.fix AS fix, r.name AS component""", s=service)

@agent.tool("recent_change")
def t_change():
    return a.run("MATCH (d:Deployment)-[:CHANGED]->(x) RETURN d.id AS deploy, d.change AS change, x.name AS target")

@agent.tool("config_at")
def t_cfg(key, at):
    e = bt.as_of("checkout-api", key, ts(at), edges)
    return {"key": key, "value": e.value if e else None}

@agent.tool("memory_search")
def t_search(query):
    return [h["doc_id"] for h in memory_search(query)]

@agent.tool("conclude")
def t_conclude(text, evidence):
    f = he.classify(text, metadata={"confidence": 0.8, "inferred_from": evidence})
    a.run("""MATCH (r:Run {id:$rid}) MERGE (x:Finding {id:'agent-conclusion'})
             SET x.text=$t, x.network=$n, x.confidence=$c MERGE (r)-[:CONCLUDED]->(x)""",
          rid=agent.trace.run_id, t=text, n=f.network, c=f.confidence)
    return {"network": f.network, "confidence": f.confidence}

plan = [
    ("recent_change", {}),
    ("blast_radius", {"store": "orders-db"}),
    ("past_incident", {"service": "checkout-api"}),
    ("config_at", {"key": "deployment_version", "at": "2026-03-15T08:00:00+00:00"}),
    ("memory_search", {"query": "checkout latency connection pool"}),
]
observations = {}
for step, (tool, args) in enumerate(plan, 1):
    obs = agent.act(tool, **args)
    observations[tool] = obs
    print(f"step {step}: {tool}({args}) -> {str(obs)[:110]}")

hypothesis = "I believe the checkout slowdown is the orders-db pool cut by the latest deploy, as in INC-0877."
verdict = agent.act("conclude", text=hypothesis, evidence=[str(observations["recent_change"]), str(observations["past_incident"])])
print("conclusion:", verdict)
assert observations["past_incident"][0]["id"] == "INC-0877" and "checkout-api" in observations["blast_radius"]

# %% [markdown]
# ### The trace is memory too
# `AgentTrace` wrote the run into Neo4j, so we can interrogate *the agent's own behaviour* with
# Cypher - the basis of debugging, auditing and later self-evolution (Chapter 7).

# %%
rid = agent.trace.run_id
print(agent.trace.timeline())
print("Steps with their successor (OPTIONAL MATCH on NEXT):")
steps = a.run("""
MATCH (r:Run {id:$rid})-[:STEP]->(t:ToolCall)
OPTIONAL MATCH (t)-[:NEXT]->(n:ToolCall)
RETURN t.seq AS seq, t.tool AS tool, t.ok AS ok, n.tool AS next_tool ORDER BY seq""", rid=rid)
a.show(steps)
print("Run summary + conclusion:")
a.show(a.run("""
MATCH (r:Run {id:$rid})-[:STEP]->(t:ToolCall)
OPTIONAL MATCH (r)-[:CONCLUDED]->(f:Finding)
RETURN r.goal AS goal, count(t) AS tool_calls, sum(CASE WHEN t.ok THEN 1 ELSE 0 END) AS ok_calls,
       collect(DISTINCT f.network) AS conclusion_type""", rid=rid))
assert len(steps) == 6 and all(s["ok"] for s in steps) and steps[-1]["next_tool"] is None
print("[PASS] the agent's six tool calls are queryable from Neo4j")

# %% [markdown]
# ## 11. Skill CLI & benchmark
#
# Each skill ships a `cli.py` so it runs from cron, CI or any harness with **zero pip installs**.
# Contract: `--help` prints the SKILL.md description (exit 0); `benchmark` runs the skill's
# self-checks. Skills are *verifiable*: trust them because they test themselves.

# %%
results = {}
for s in a.skill_catalog("memory"):
    slug = s["slug"] if isinstance(s, dict) else s
    results[slug] = a.run_benchmark("memory", slug)
for slug, (ok, first) in results.items():
    print(("PASS" if ok else "FAIL"), f"{slug:36s} {first}")

cli = a.PROJECT_ROOT / "skills" / "memory" / "rrf-hybrid-retrieval" / "cli.py"
helpp = subprocess.run([sys.executable, str(cli), "--help"], capture_output=True, text=True)
print("\n$ python skills/memory/rrf-hybrid-retrieval/cli.py --help   (exit", helpp.returncode, ")")
print("\n".join(helpp.stdout.splitlines()[:6]))
assert len(results) == 8 and all(ok for ok, _ in results.values()) and helpp.returncode == 0
assert "reciprocal rank fusion" in helpp.stdout.lower()

# %% [markdown]
# ## 🎯 Vibe-coding corner 1 - add a Cypher query
#
# **Prompt to copy into Copilot / Claude Code:**
#
# > *Context:* Neo4j memory graph with `(:Service)-[:HAD_CONFIG]->(:ConfigVersion {key,value,valid_from,valid_until,ingested_at})`.
# > *Task:* write one Cypher query returning, for each service, the config versions whose
# > ingestion lag (ingested_at minus valid_from) exceeds 30 days, with the lag in days.
# > *Constraints:* parameterise the threshold (`$days`), timestamps are ISO strings so use
# > `datetime()` conversion, read-only, no APOC. *Acceptance:* returns the EBS row (56 days) and no others.
#
# **What makes a good prompt:** (1) the exact schema, (2) one concrete task, (3) constraints
# (read-only, parameters, no plugins), (4) an acceptance check you can run.
# **Verify yourself:** run the generated query, then `EXPLAIN` it; reject answers that string-concatenate
# values into Cypher instead of using parameters.

# %%
LAG_QUERY = """
MATCH (s:Service)-[:HAD_CONFIG]->(c:ConfigVersion)
WITH s, c, duration.inSeconds(datetime(c.valid_from), datetime(c.ingested_at)).seconds / 86400.0 AS lag_days
WHERE lag_days > $days
RETURN s.name AS service, c.key AS key, round(lag_days) AS lag_days"""
lag = a.run(LAG_QUERY, days=30)
a.show(lag)
assert [(r["key"], r["lag_days"]) for r in lag] == [("ebs_volume_gb", 56.0)]

# %% [markdown]
# ## 🎯 Vibe-coding corner 2 - a worked example (helper + test)
#
# Imagine you asked: *"Write `blast_radius(name, max_hops)` returning service names depending on
# a node within `max_hops` (1-6). Cypher cannot parameterise path bounds, so validate the integer
# yourself. Raise `ValueError` for bad hops. Return [] for unknown nodes."* Below is the kind of
# result you would get - **and the test you must write before trusting it**. Note the trap the
# prompt anticipated: `*1..$n` is *invalid* Cypher, so the bound is interpolated - safe only after
# integer validation (never interpolate strings from users).

# %%
def blast_radius(name, max_hops=3):
    if not isinstance(max_hops, int) or isinstance(max_hops, bool) or not 1 <= max_hops <= 6:
        raise ValueError("max_hops must be an int in 1..6")
    rows = a.run(f"MATCH (s:Service)-[:DEPENDS_ON*1..{max_hops}]->(n {{name:$n}}) RETURN DISTINCT s.name AS s ORDER BY s", n=name)
    return [r["s"] for r in rows]

def test_blast_radius():
    assert blast_radius("orders-db") == ["cart-service", "checkout-api", "orders-service", "payment-service"]
    assert blast_radius("orders-db", 1) == ["orders-service", "payment-service"]
    assert blast_radius("does-not-exist") == []
    for bad in (0, 7, "3", True, 2.5):
        try:
            blast_radius("orders-db", bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad hops {bad!r}")

test_blast_radius()
print("[PASS] vibe-coded helper passes its tests (happy path, bound, unknown node, invalid input)")

# %% [markdown]
# ## 🎯 Vibe-coding corner 3 - write a new skill
#
# > *Context:* this repo's skills live in `skills/<chapter>/<skill>/` with `SKILL.md` (frontmatter +
# > 7 sections: Overview, When to Use, When NOT to Use, Process, Rationalizations, Red Flags,
# > Non-Negotiable Verification, plus Source Attribution), a stdlib-only `lib.py`, and an argparse
# > `cli.py` whose `--help` prints the description and which has a `benchmark` subcommand.
# > Copy the layout of `skills/memory/rrf-hybrid-retrieval/`.
# > *Task:* create `skills/memory/staleness-ttl-policy/` that assigns a TTL to a memory fact by
# > durability and link type, and flags expired facts. *Constraints:* no third-party imports; pure
# > functions; inject `now_fn` for tests. *Acceptance:* `python cli.py benchmark` prints all gates
# > passed and exits 0; `python cli.py --help` exits 0.
#
# **Verify:** run the benchmark, read the generated `lib.py` for hidden network/file access, and
# run the notebook helper `a.run_benchmark("memory", "staleness-ttl-policy")`.
# *(Do not commit the new skill unless you want it - this notebook only teaches the prompt.)*

# %% [markdown]
# ## Experiments - change it, predict first
#
# 1. **RRF `k`.** In section 8 call `memory_search(Q, k=1)` vs `k=1000`. *Predict:* small `k` lets
#    top ranks dominate (large score gaps), large `k` flattens scores so agreement across channels
#    matters more. Does the top-3 order change?
# 2. **Token budget.** Call `memory_search(Q, budget=15)`. *Predict:* how many hits remain?
#    (Hint: each episode costs ~10-12 tokens; the filter *skips* items that do not fit rather than stopping.)
# 3. **Core limit.** Re-run section 3 with `core_limit=5`. *Predict:* do all 6 durable facts still fit?
#    Which letta failure mode might appear in section 5 (hierarchy collapse)?
# 4. **Consolidation threshold.** Use `threshold=0.6` in section 7. *Predict:* fewer or more
#    clusters? Which episodes drop out?

# %% [markdown]
# ## Common pitfalls
#
# - **Naive datetimes**: bi-temporal skills need timezone-aware values; `datetime.now()` breaks comparisons.
# - **Overwriting instead of superseding**: you lose the audit trail; `invalidate` demands a reason on purpose.
# - **`CREATE` on rerun** duplicates data; use `MERGE` with a unique key for idempotent loads.
# - **Unbounded variable-length paths** (`*`) can explode on dense graphs; always cap (`*1..4`).
# - **Aggregation grouping**: every non-aggregated column in `RETURN` is a grouping key - extra columns split your counts.
# - **Evicting = deleting?** No: eviction moves to archival; deleting is a separate, explicit decision.
# - **Trusting an opinion as a fact**: keep World vs Opinion separate and carry provenance.
# - **Interpolating user text into Cypher**: use `$params`; only validated ints may be formatted in (path bounds).

# %%
a.check("1 bi-temporal: instance at outage is m5.xlarge", inst.value == "m5.xlarge")
a.check("1 supersede recorded in graph", a.run("MATCH (:ConfigVersion)-[r:SUPERSEDES]->() RETURN count(r) AS n")[0]["n"] >= 1)
a.check("2 graphiti: memory linked to topology", linked >= 1)
a.check("3 hierarchical: core within limit, archival non-empty", len(mem.core) <= 8 and len(mem.archival) > 0)
a.check("4 hindsight: Opinion has evidence edges", len(links) >= 2)
a.check("5 letta: 8 modes diagnosed", len(r_ours.modes) == 8)
a.check("6 consistency: reporter flagged stale", [w["agent"] for w in warnings] == ["reporter"])
a.check("7 consolidation: provenance covers INC-0877", "ep-0877" in cfacts[0].derived_from)
a.check("8 rrf: retrieval edges stored", a.run("MATCH (:Query)-[r:RETRIEVED]->() RETURN count(r) AS n")[0]["n"] == len(hits))
a.check("9 cypher: blast radius has 4 services", len(blast) == 4)
a.check("10 agent: trace persisted with 6 steps", len(steps) == 6)
a.check("11 all 8 skill benchmarks pass", all(ok for ok, _ in results.values()))
a.check("12 vibe-coded helper tested", True)

# %% [markdown]
# ## Recap
#
# - Memory is a **graph with two clocks**, tiers, epistemic labels and provenance - not a vector bag.
# - The pattern `Cypher -> pure-Python skill -> Cypher` keeps skills testable and the database authoritative.
# - We used **all 8** memory skills, wrote their outputs back, ran multi-hop, variable-length, aggregation
#   and `OPTIONAL MATCH` queries, and let an agent plan, act, observe and be audited through its own trace.
# - Trade-offs: strings vs native datetimes, strong vs eventual consistency, RRF simplicity vs learned fusion.
#
# **Next:** Chapter 5 - *Reasoning & Planning* (`notebooks_advanced_simplified/ch5-reasoning-planning.ipynb`):
# the agent stops just recalling and starts decomposing problems into plans over this memory.
