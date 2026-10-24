# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > [Beginner (tier 1/3)](../notebooks_simplified/ch4-memory.ipynb)  →  **Intermediate (tier 2/3) ← you are here**  →  [Advanced (tier 3/3)](../notebooks_advanced_simplified/ch4-memory.ipynb)

# %% [markdown]
# # Chapter 4 (Intermediate) - Memory as a Graph
#
# **Tier 2 of 3.** You have seen each memory skill in isolation (beginner). Here you learn
# *how a skill is wired to a real database*: **Cypher -> skill -> Cypher**, and then you start
# writing some of that glue yourself. The full eight-skill / agent / benchmark-everything
# treatment is in tier 3.
#
# **Scenario** - `INC-1001`: *"checkout-api is slow"*. The agent must remember what was true
# *when*, what it learned, how sure it is, and what happened last time.
#
# | # | Skill (`skills/memory/...`) | Question it answers | Graph shape we write |
# |---|---|---|---|
# | 1 | `bi-temporal-edge` | what was true *as of* a date? | `(Service)-[:HAD_CONFIG]->(ConfigVersion)` |
# | 2 | `hierarchical-memory` | what stays hot, what is archived? | `(MemTier)-[:HOLDS]->(MemFact)` |
# | 3 | `hindsight-epistemic-classifier` | fact, action or belief? | `(EpFact)-[:INFERRED_FROM]->(EpFact)` |
# | 4 | `memory-consolidation` | which past episodes form a pattern? | `(ConsolidatedFact)-[:DERIVED_FROM]->(Episode)` |
# | 5 | `rrf-hybrid-retrieval` | which memories answer this query? | `(Query)-[:RETRIEVED]->(Episode)` |
#
# Requires Neo4j at `bolt://localhost:7687`. No LLM or API keys.

# %%
import int_helpers as h
import json, subprocess, sys, os
from datetime import datetime, timedelta, timezone

h.load_devops_graph()   # wipes the shared DB and loads the DevOps graph (once, here)

bt = h.load_skill("memory", "bi-temporal-edge")
hm = h.load_skill("memory", "hierarchical-memory")
he = h.load_skill("memory", "hindsight-epistemic-classifier")
mc = h.load_skill("memory", "memory-consolidation")
rr = h.load_skill("memory", "rrf-hybrid-retrieval")

def ts(s):
    """ISO-8601 string -> timezone-aware datetime (the skills require aware datetimes)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

OUTAGE = ts("2026-03-15T08:00:00+00:00")
h.show(h.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS n ORDER BY label"))

# %% [markdown]
# ## 1. Bi-temporal edges - *"what was true at the moment of the outage?"*
#
# Overwriting `instance_type` destroys history. A **bi-temporal** fact has two clocks:
# **valid time** (`valid_from` / `valid_until`: when it was true in the world) and
# **ingestion time** (`ingested_at`: when the agent learned it). We keep one
# `ConfigVersion` node per version, and `valid_until = null` means "still true".
#
# **Pattern of this whole notebook:** Cypher feeds the skill, the skill computes, we write the
# result back. First the persisted history (timestamps stored as ISO strings: sortable and simple):

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
h.run("""
MATCH (s:Service {name:'checkout-api'})
UNWIND $rows AS r
CREATE (s)-[:HAD_CONFIG]->(:ConfigVersion {key:r.key, value:r.value, valid_from:r.vf, valid_until:r.vu,
                                           ingested_at:r.ing, link_type:r.lt, reason:r.reason})
""", rows=CONFIG)
h.show(h.run("""MATCH (:Service {name:'checkout-api'})-[:HAD_CONFIG]->(c:ConfigVersion)
RETURN c.key AS key, c.value AS value, c.valid_from AS valid_from, c.valid_until AS valid_until ORDER BY key, valid_from"""))

# %% [markdown]
# ### Anatomy of the skill: `create_edge` in, `TemporalEdge` out
#
# ```python
# create_edge(source, target, relationship, value=None, valid_from=None, ingested_at=None,
#             link_type="entity", weight=1.0, metadata=None)  -> TemporalEdge
# as_of(source, relationship, timestamp, edges)              -> TemporalEdge | None
# supersede(old_edge, new_value, reason, edges=None)         -> TemporalEdge   # closes old, opens new
# ```
#
# We rebuild `TemporalEdge` objects **from the Cypher rows** (not from the Python list above):

# %%
rows = h.run("""
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

print("Anatomy of one TemporalEdge (inst.to_dict()):")
for k, v in inst.to_dict().items():
    print(f"  {k:22s} {v}")
print("\n  value               -> the fact itself")
print("  valid_from/until    -> VALID time window (until=None: still true)")
print("  ingested_at         -> when the agent LEARNED it; ingestion_lag() = ingested_at - valid_from")
print("  link_type/weight    -> 'causal' edges get a higher effective_weight than plain 'entity' ones")
print(f"\nAt the outage: instance={inst.value}, deployment={dep.value}; on Mar 1 instance was {before.value}")
print(f"EBS ingestion lag: {ebs.ingestion_lag()}  (true since January, learned Mar 15)")

# Write the skill's OUTPUT back as a Finding and read it back.
h.run("""
MATCH (i:Incident {id:'INC-1001'})
MERGE (f:Finding {id:'cfg-at-outage'})
SET f.instance_type=$inst, f.deployment=$dep, f.as_of=$at
MERGE (f)-[:EXPLAINS]->(i)
""", inst=inst.value, dep=dep.value, at=OUTAGE.isoformat())
h.show(h.run("MATCH (f:Finding {id:'cfg-at-outage'})-[:EXPLAINS]->(i:Incident) RETURN f.instance_type AS instance, f.deployment AS deployment, i.id AS incident"))

# %% [markdown]
# ### The same "as of" question in pure Cypher, then `supersede`
#
# A point-in-time query is just an interval test: `valid_from <= t AND (valid_until IS NULL OR valid_until > t)`.
# ISO-8601 strings with the same offset compare correctly as text. The skill and the query must
# **agree** - that is a cheap cross-check you can always run.
#
# `supersede()` closes the old edge with a **mandatory reason** (no silent overwrite) and opens a
# new one; we persist both and link `(new)-[:SUPERSEDES]->(old)`.

# %%
AS_OF = """
MATCH (:Service {name:'checkout-api'})-[:HAD_CONFIG]->(c:ConfigVersion {key:$k})
WHERE c.valid_from <= $t AND (c.valid_until IS NULL OR c.valid_until > $t)
RETURN c.value AS value"""
cy_inst = h.run(AS_OF, k="instance_type", t=OUTAGE.isoformat())[0]["value"]
print("Cypher as-of instance_type:", cy_inst, "| skill:", inst.value)

current = bt.as_of("checkout-api", "instance_type", datetime.now(timezone.utc), edges)
new_edge = bt.supersede(current, "m5.2xlarge", "scale-up during INC-1001 mitigation", edges=edges)
h.run("""
MATCH (old:ConfigVersion {key:'instance_type', value:$old})
SET old.valid_until=$vu, old.reason=$reason
WITH old
MATCH (s:Service {name:'checkout-api'})
MERGE (s)-[:HAD_CONFIG]->(new:ConfigVersion {key:'instance_type', value:$new})
SET new.valid_from=$vf, new.ingested_at=$vf, new.link_type='entity'
MERGE (new)-[:SUPERSEDES]->(old)
""", old=current.value, new=new_edge.value, vu=current.valid_until.isoformat(),
      reason=current.invalidation_reason, vf=new_edge.valid_from.isoformat())

print("Version chain (OPTIONAL MATCH = left join: the current version has no successor, so null):")
h.show(h.run("""
MATCH (c:ConfigVersion {key:'instance_type'})
OPTIONAL MATCH (newer:ConfigVersion)-[:SUPERSEDES]->(c)
RETURN c.value AS version, c.valid_until IS NULL AS is_current, newer.value AS replaced_by
ORDER BY c.valid_from"""))
h.check("Cypher as-of agrees with the skill", cy_inst == inst.value == "m5.xlarge")
h.check("history survives supersede: outage state is still m5.xlarge",
        h.run(AS_OF, k="instance_type", t=OUTAGE.isoformat())[0]["value"] == "m5.xlarge")

# %% [markdown]
# ## 2. Hierarchical memory - core / recall / archival
#
# **Core** is small and always in the prompt; **recall** holds recent turns; **archival** is big
# and searched on demand. When core is full, the fact with the lowest *eviction score* moves to
# archival - **eviction is not deletion**. `durable` facts get a 10x protection factor.
#
# ```python
# mem = HierarchicalMemory(core_limit=8)
# mem.add_fact(content, durability)      # durability: DURABILITY_DURABLE | DURABILITY_SHORT_LIVED
# mem.query(q, top_k_per_tier=3)         # -> {"core": [...], "recall": [...], "archival": [...]}
# mem.diagnostics()                      # -> counts / health numbers
# ```
# Inputs come from Cypher: critical services are durable, the firing alarm is short-lived.

# %%
crit = h.run("MATCH (s:Service {tier:'critical'}) RETURN s.name AS name, s.owner AS owner ORDER BY name")
alarm = h.run("MATCH (al:Alarm)-[:FIRED_ON]->(s:Service) RETURN al.metric AS m, al.value AS v, s.name AS svc")[0]

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
    mem.add_fact(f"Tail after {cmd}: circuit-breaker open" if "psql" in cmd else f"Tail after {cmd}: ok",
                 hm.DURABILITY_SHORT_LIVED)

survivors = [d for d in durables if d in mem.core]
hit = mem.query("circuit-breaker")
print("Anatomy of mem.query():", {tier: len(items) for tier, items in hit.items()})
print("  one archival hit:", hit["archival"][0])
print(f"core={len(mem.core)}/8 archival={len(mem.archival)}; durable survivors {len(survivors)}/{len(durables)}")

# Write each tier back and aggregate with Cypher.
tier_rows = ([{"tier": "core", "c": f.content, "d": f.durability} for f in mem.core.values()] +
             [{"tier": "archival", "c": f.content, "d": f.durability} for f in mem.archival])
h.run("""
UNWIND $rows AS r
MERGE (t:MemTier {name:r.tier})
MERGE (f:MemFact {content:r.c}) SET f.durability=r.d
MERGE (t)-[:HOLDS]->(f)
""", rows=tier_rows)
h.show(h.run("""
MATCH (t:MemTier)-[:HOLDS]->(f:MemFact)
RETURN t.name AS tier, f.durability AS durability, count(f) AS facts ORDER BY tier, durability"""))
h.check("core never exceeds its limit", len(mem.core) <= 8)
h.check("durable facts stayed hot; evicted noise is still searchable", len(survivors) >= 3 and bool(hit["archival"]))

# %% [markdown]
# ## 3. Hindsight epistemic classifier - evidence vs belief
#
# `classify(text, metadata)` sorts a statement into one of four networks:
# **World** (objective fact, has an `external_ref`), **Experience** (what *I* did: `source`+`timestamp`),
# **Opinion** (hedged belief: `confidence` + `inferred_from`), **Observation** (synthesis: `derived_from`).
# Output `EpistemicFact` fields: `text`, `network`, `confidence`, `inferred_from`. When someone
# asks *"how do you know?"* the agent must cite provenance and never present an opinion as a world fact.
#
# World facts come from Cypher; the agent's own actions and belief are authored here, and the
# opinion's evidence is persisted as `:INFERRED_FROM` edges.

# %%
w1 = h.run("MATCH (al:Alarm)-[:FIRED_ON]->(s:Service) RETURN s.name+' p99 latency reached '+toString(al.value)+' ms' AS t")[0]["t"]
w2 = h.run("MATCH (d:Deployment)-[:CHANGED]->(x) RETURN d.id+' changed '+x.name+': '+d.change AS t")[0]["t"]
now_iso = datetime.now(timezone.utc).isoformat()
exp_text = "I retrieved the deployment log for deploy-481."
op_text = "I believe the root cause is deploy-481 shrinking the orders-db connection pool."
batch = [
    {"text": w1, "metadata": {"external_ref": "cloudwatch:alarm-77"}},
    {"text": w2, "metadata": {"external_ref": "deploy-log"}},
    {"text": exp_text, "metadata": {"source": "agent", "timestamp": now_iso}},
    {"text": op_text, "metadata": {"confidence": 0.7, "inferred_from": [w1, w2, exp_text]}},
    {"text": "In summary: deploy 08:30, p99 rising, alarm-77 fires, INC-1001 opened.",
     "metadata": {"derived_from": ["deploy-log", "cloudwatch", "incident-channel"]}},
]
facts = he.classify_batch(batch)
for f in facts:
    print(f"[{f.network:11s}] conf={f.confidence:.1f}  inferred_from={len(f.inferred_from)}  {f.text[:70]}")

h.run("UNWIND $f AS x MERGE (e:EpFact {text:x.text}) SET e.network=x.network, e.confidence=x.conf",
      f=[{"text": f.text, "network": f.network, "conf": f.confidence} for f in facts])
h.run("""
UNWIND $l AS l
MATCH (o:EpFact {text:l.op}), (e:EpFact {text:l.ev})
MERGE (o)-[:INFERRED_FROM]->(e)""", l=[{"op": f.text, "ev": ev} for f in facts for ev in f.inferred_from])
print("\nOpinion -> evidence chain read back from Neo4j:")
h.show(h.run("""
MATCH (o:EpFact {network:'Opinion'})-[:INFERRED_FROM]->(e:EpFact)
RETURN o.confidence AS confidence, e.network AS evidence_type, e.text AS evidence"""))
h.check("all four networks represented", {f.network for f in facts} == {"World", "Experience", "Opinion", "Observation"})
h.check("opinion is linked to graph evidence", len(h.run("MATCH (:EpFact {network:'Opinion'})-[:INFERRED_FROM]->(:EpFact) RETURN 1 AS x")) >= 2)

# %% [markdown]
# ## 4. Memory consolidation - episodes become knowledge
#
# ```python
# Episode(id, content, episode_type, created_at, metadata)
# consolidate(episodes, min_cluster_size=3, threshold=0.25) -> [ConsolidatedFact]
# provenance_of(fact, episodes)                              -> the source Episodes
# ```
# Similar episodes (Jaccard word overlap, single linkage) are merged; clusters smaller than
# `min_cluster_size` are skipped - not enough examples to generalize. Each
# `ConsolidatedFact` has `summary`, `confirmations` and `derived_from` (source episode ids),
# so every summary stays auditable. The real resolved `INC-0877` becomes an episode via Cypher
# string concatenation; a lone TLS episode should *not* be consolidated.

# %%
h.run("""
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
h.run("""
MATCH (i:Incident {id:'INC-0877'})-[:AFFECTS]->(s:Service)
CREATE (x:Episode {id:'ep-0877', type:'incident', created_at:'2026-02-20T09:00:00+00:00',
                   content: i.title + '. root cause: ' + i.root_cause + '. fix: ' + i.fix})
CREATE (x)-[:ABOUT]->(s)""")

records = h.run("MATCH (e:Episode) RETURN e.id AS id, e.content AS content, e.type AS type, e.created_at AS at ORDER BY id")
eps = [mc.Episode(id=r["id"], content=r["content"], episode_type=r["type"], created_at=ts(r["at"]), metadata={}) for r in records]
cfacts = mc.consolidate(eps, min_cluster_size=3, threshold=0.25)
cf = cfacts[0]
print("Anatomy of a ConsolidatedFact:", {k: (v if k != "summary" else v[:60] + "...") for k, v in cf.to_dict().items()
                                         if k in ("id", "summary", "confirmations", "derived_from")})
print("provenance ok:", len(mc.provenance_of(cf, eps)) == len(cf.derived_from))

h.run("""
UNWIND $f AS f
MERGE (c:ConsolidatedFact {id:f.id}) SET c.summary=f.summary, c.confirmations=f.conf
WITH c, f UNWIND f.eps AS eid
MATCH (e:Episode {id:eid}) MERGE (c)-[:DERIVED_FROM]->(e)""",
      f=[{"id": f.id, "summary": f.summary, "conf": f.confirmations, "eps": list(f.derived_from)} for f in cfacts])
leftover = h.run("""
MATCH (e:Episode) OPTIONAL MATCH (c:ConsolidatedFact)-[:DERIVED_FROM]->(e)
WITH e, c WHERE c IS NULL RETURN e.id AS id""")
print("Episodes NOT consolidated:", [r["id"] for r in leftover])
h.check("pool-exhaustion pattern includes the real INC-0877", len(cf.derived_from) >= 3 and "ep-0877" in cf.derived_from)
h.check("the TLS outlier stayed raw", "ep-5" not in cf.derived_from and any(r["id"] == "ep-5" for r in leftover))

# %% [markdown]
# ## 5. RRF hybrid retrieval - several rankers, one list
#
# ```python
# rrf_fuse(channels: {name: [doc_id, ...]}, k=60)          -> [(doc_id, score)]   # pure fusion
# hybrid_retrieve(query, channels, k, top_n_for_rerank, final_budget, get_tokens, metadata)
#                                                         -> [{doc_id, fusion_score, rerank_score, token_count}]
# ```
# **Reciprocal Rank Fusion**: `score(doc) = sum over channels of 1 / (k + rank)`. It uses only
# ranks (no score calibration), and documents that sit high in *several* lists rise. After
# fusion, a reranker reorders the top-N and a **token budget** trims to what fits the prompt.
#
# Our channels: word-overlap (*keyword*), character-trigram overlap (*semantic* stand-in; production
# uses embeddings - the documented `TODO` seam), recency (*temporal*), and a **graph** channel
# whose Cypher uses a variable-length path `[:DEPENDS_ON*1..3]`: "follow 1 to 3 hops".

# %%
docs = {r["id"]: r for r in h.run("MATCH (e:Episode) RETURN e.id AS id, e.content AS text, e.created_at AS at")}

def _tok(t): return set(t.lower().replace(",", " ").split())
def _tri(t): t = t.lower(); return {t[i:i + 3] for i in range(len(t) - 2)}
def _rank(q, f):
    s = {i: len(f(q) & f(d["text"])) / max(1, len(f(q) | f(d["text"]))) for i, d in docs.items()}
    return [i for i, v in sorted(s.items(), key=lambda kv: (-kv[1], kv[0])) if v > 0]

def graph_channel(q):
    rows = h.run("""
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

def memory_search(q, budget=60):
    return rr.hybrid_retrieve(q, channels, k=60, top_n_for_rerank=6, final_budget=budget,
                              get_tokens=lambda i: len(docs[i]["text"].split()),
                              metadata={i: d["text"] for i, d in docs.items()})

Q = "checkout latency connection pool"
hits = memory_search(Q)
h.show([{"doc": x["doc_id"], "fusion": round(x["fusion_score"], 4), "rerank": round(x["rerank_score"], 3),
         "tokens": x["token_count"]} for x in hits])
h.run("""
MERGE (q:Query {text:$q})
WITH q UNWIND $h AS h
MATCH (e:Episode {id:h.id}) MERGE (q)-[r:RETRIEVED]->(e) SET r.rank=h.rank, r.fused=h.fused""",
      q=Q, h=[{"id": x["doc_id"], "rank": i + 1, "fused": x["fusion_score"]} for i, x in enumerate(hits)])
h.check("fused retrieval ranks the TLS outlier below the pool-exhaustion episodes", hits[0]["doc_id"] != "ep-5")
h.check("token budget respected", sum(x["token_count"] for x in hits) <= 60)

# %% [markdown]
# ## 6. Cypher on the memory graph: a variable-length path and an aggregation
#
# Everything above now lives in one connected graph. Two questions (you know `MATCH`, `WHERE`,
# `RETURN` from tier 1, so only the new clauses are explained):
#
# 1. **Blast radius** - `[:DEPENDS_ON*1..4]` is a *variable-length* path (1 to 4 hops);
#    `min(length(path))` keeps the shortest distance per service (aggregation groups by the other
#    returned columns).
# 2. **Memory health** - `OPTIONAL MATCH` keeps services that have *no* episodes, and `count()` then
#    returns 0 instead of dropping the row.

# %%
print("Q1 blast radius of orders-db:")
h.show(h.run("""
MATCH path = (s:Service)-[:DEPENDS_ON*1..4]->(:Database {name:'orders-db'})
RETURN s.name AS service, s.tier AS tier, min(length(path)) AS hops
ORDER BY hops, service"""))
print("Q2 memory health per service:")
health = h.run("""
MATCH (s:Service)
OPTIONAL MATCH (s)<-[:ABOUT]-(e:Episode)
OPTIONAL MATCH (s)-[:HAD_CONFIG]->(c:ConfigVersion)
RETURN s.name AS service, count(DISTINCT e) AS episodes, count(DISTINCT c) AS config_versions
ORDER BY episodes DESC, service""")
h.show(health)

# %% [markdown]
# ## 7. 🧩 Your turn 1 - an "as of" query in Cypher
#
# **Task:** which `deployment_version` was valid on `2026-03-10T00:00:00+00:00`?
# Write the Cypher yourself first (reuse the interval test from section 1), then run the cell to
# compare with the reference solution.

# %%
def ex1():
    return h.run(AS_OF, k="deployment_version", t="2026-03-10T00:00:00+00:00")[0]["value"]

h.your_turn("deployment version as of Mar 10",
            "valid_from <= $t AND (valid_until IS NULL OR valid_until > $t), filtered on key='deployment_version'",
            ex1, "v3.4.1")

# %% [markdown]
# ## 8. 🧩 Your turn 2 - change a skill's input
#
# **Task:** call `rr.rrf_fuse` with two channels, `a = [x, y, z]` and `b = [y, z]`, `k=60`.
# Which doc comes first, and why? (Predict: `y` is 2nd in `a` and 1st in `b`; `x` is only 1st in `a`.)

# %%
def ex2():
    fused = rr.rrf_fuse({"a": ["x", "y", "z"], "b": ["y", "z"]}, k=60)
    print("  fused:", [(d, round(s, 5)) for d, s in fused])
    return fused[0][0]

h.your_turn("RRF winner", "1/(60+2)+1/(60+1) beats 1/(60+1) alone", ex2, "y")

# %% [markdown]
# ## 9. A hand-written pipeline - `checkout-api is slow`
#
# Chain three skills in plain Python (no agent framework; tier 3 wraps this in `MiniAgent`):
# **retrieve** past episodes (RRF) -> **recall** the config state at the outage (bi-temporal) ->
# **conclude** with an epistemic label. The output is a decision report an on-call engineer can read.

# %%
def triage(service, at):
    recent = h.run("MATCH (d:Deployment)-[:CHANGED]->(x) RETURN d.id AS deploy, d.change AS change, x.name AS target")[0]
    top = [x["doc_id"] for x in memory_search("checkout latency connection pool")[:3]]
    version = bt.as_of(service, "deployment_version", at, edges).value
    pattern = h.run("MATCH (c:ConsolidatedFact)-[:DERIVED_FROM]->(e:Episode) RETURN c.id AS id, count(e) AS n")[0]
    text = (f"I believe {service} is slow because {recent['deploy']} ({recent['change']}) hit {recent['target']}, "
            f"matching a pattern seen {pattern['n']} times.")
    verdict = he.classify(text, metadata={"confidence": 0.8, "inferred_from": [recent["deploy"], top[0]]})
    h.run("""MATCH (i:Incident {id:'INC-1001'}) MERGE (f:Finding {id:'pipeline-conclusion'})
             SET f.text=$t, f.network=$n, f.confidence=$c MERGE (f)-[:EXPLAINS]->(i)""",
          t=text, n=verdict.network, c=verdict.confidence)
    return {"service": service, "version_at_outage": version, "recent_change": recent, "similar_episodes": top,
            "pattern": pattern["id"], "conclusion": text, "label": f"{verdict.network} (conf {verdict.confidence})"}

report = triage("checkout-api", OUTAGE)
print("=== DECISION REPORT ===")
for k, v in report.items():
    print(f"{k:18s} {v}")
h.check("pipeline found the pool-exhaustion episodes and the v3.5.0 deployment",
        report["version_at_outage"] == "v3.5.0" and "ep-5" not in report["similar_episodes"])
h.check("conclusion is labelled Opinion, not World fact", report["label"].startswith("Opinion"))

# %% [markdown]
# ## 10. Trust a skill: run its CLI and its benchmark
#
# Every skill ships a `cli.py` (zero pip installs) and a `benchmark` subcommand of self-checks.
# That is *why you can trust a skill you did not write*: it verifies itself. Below, one CLI call
# and one benchmark. Tier 3 benchmarks every skill in the chapter.

# %%
cli = h.PROJECT_ROOT / "skills" / "memory" / "memory-consolidation" / "cli.py"
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
p = subprocess.run([sys.executable, str(cli), "scenario", "incident-consolidation"], capture_output=True, text=True,
                   encoding="utf-8", errors="replace", env=env, timeout=120)
print("$ python skills/memory/memory-consolidation/cli.py scenario   (exit", p.returncode, ")")
print("\n".join(p.stdout.splitlines()[:8]))
ok, first = h.run_benchmark("memory", "rrf-hybrid-retrieval")
print("benchmark rrf-hybrid-retrieval:", "PASS" if ok else "FAIL", "-", first)
h.check("CLI scenario exits 0", p.returncode == 0)
h.check("benchmark passes", ok)

# %% [markdown]
# ## 11. 🎯 Vibe-coding tip - make the AI add a graph-backed check
#
# **Copy-paste prompt for your AI assistant:**
#
# > In this repo, using `skills/memory/bi-temporal-edge/lib.py` (stdlib only), write a Python
# > function `config_changed_between(edges, source, relationship, t1, t2)` that returns True if the
# > value of `relationship` for `source` differs between `t1` and `t2`, using `as_of`. Handle the
# > case where `as_of` returns `None`. Do not edit any file under `skills/`. Add three assertions.
#
# **Acceptance check (run it - never trust unverified AI code):** the function must report a
# change for instance_type between Mar 1 and the outage, no change for `ebs_volume_gb`, and must
# not crash for a date before any record. The cell below is a reference implementation that passes it.

# %%
def config_changed_between(edges, source, rel, t1, t2):
    a, b = bt.as_of(source, rel, t1, edges), bt.as_of(source, rel, t2, edges)
    return (a.value if a else None) != (b.value if b else None)

t_mar1, t_jan1 = ts("2026-03-01T00:00:00+00:00"), ts("2025-01-01T00:00:00+00:00")
h.check("instance_type changed Mar 1 -> outage", config_changed_between(edges, "checkout-api", "instance_type", t_mar1, OUTAGE))
h.check("ebs_volume_gb unchanged Mar 1 -> outage", not config_changed_between(edges, "checkout-api", "ebs_volume_gb", ts("2026-02-01T00:00:00+00:00"), OUTAGE))
h.check("no crash before any record", config_changed_between(edges, "checkout-api", "instance_type", t_jan1, t_mar1))

# %% [markdown]
# ## Why this matters (trade-offs)
#
# - **Bi-temporal vs overwrite:** one node per version costs storage and a slightly harder query, but
#   buys auditability ("what did we believe on Mar 15?") and safe correction of late-arriving facts.
# - **Eviction vs deletion:** evicting core facts to archival keeps the prompt small without losing
#   data; the cost is an extra search step when an evicted fact is needed.
# - **Consolidation vs raw episodes:** summaries are cheap to retrieve but lose detail - keep
#   `DERIVED_FROM` edges so you can always drill back down. Run it off the hot path.
# - **RRF vs score blending:** ranks need no calibration across retrievers, but they throw away
#   score magnitude; a weak channel can still add noise.
#
# ## Common mistakes
#
# 1. Passing **naive datetimes** to the skills (they require timezone-aware ones).
# 2. Hand-building skill input instead of **deriving it from Cypher** - the graph and the skill drift apart.
# 3. Using `CREATE` where reruns happen; use `MERGE` for idempotent writes.
# 4. Forgetting `OPTIONAL MATCH`, so services with no episodes silently vanish from counts.
# 5. Presenting an **Opinion** as a **World** fact in the final answer.

# %%
def n(q, **kw): return h.run(q, **kw)[0]["n"]
h.check("config versions persisted (incl. the superseded chain)", n("MATCH (c:ConfigVersion) RETURN count(c) AS n") >= 6)
h.check("current instance type is the superseded-to value",
        bt.as_of("checkout-api", "instance_type", datetime.now(timezone.utc), edges).value == "m5.2xlarge")
h.check("memory tiers written", n("MATCH (:MemTier)-[:HOLDS]->(f:MemFact) RETURN count(f) AS n") >= 8)
h.check("epistemic facts written", n("MATCH (e:EpFact) RETURN count(e) AS n") == len(facts))
h.check("consolidated fact has provenance edges", n("MATCH (:ConsolidatedFact)-[r:DERIVED_FROM]->(:Episode) RETURN count(r) AS n") >= 4)
h.check("retrieval results written", n("MATCH (:Query)-[r:RETRIEVED]->(:Episode) RETURN count(r) AS n") == len(hits))
h.check("pipeline finding written", n("MATCH (:Finding {id:'pipeline-conclusion'})-[:EXPLAINS]->(:Incident) RETURN count(*) AS n") == 1)

# %% [markdown]
# ## Recap
#
# - **Bi-temporal edges** answer "what was true when" - verified two ways (skill and Cypher interval query).
# - **Hierarchical memory**, **epistemic classification**, **consolidation** and **RRF retrieval** each
#   follow `Cypher -> skill -> Cypher`: derive inputs from the graph, write outputs back as nodes/edges.
# - You wrote an "as of" query, predicted an RRF ranking, chained skills in a plain function, and
#   verified a skill via its CLI and benchmark.
#
# **Next:** the advanced tier adds the other three skills (graphiti updates, Letta failure modes,
# consistency models), a plan-act-observe `MiniAgent`, benchmarks for every skill, three vibe-coding
# corners and experiments: `../notebooks_advanced_simplified/ch4-memory.ipynb`.
