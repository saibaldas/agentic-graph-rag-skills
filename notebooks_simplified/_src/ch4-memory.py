# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > **Beginner (tier 1/3) ← you are here**  →  [Intermediate (tier 2/3)](../notebooks_intermediate_simplified/ch4-memory.ipynb)  →  [Advanced (tier 3/3)](../notebooks_advanced_simplified/ch4-memory.ipynb)

# %% [markdown]
# # Chapter 4 (Simplified) — Memory for agents, stored in a real Neo4j graph
#
# **Who this is for:** you are new to GraphRAG. Every concept is explained in plain language.
#
# **What you will learn**
# 1. **Bi-temporal edges** — remembering *what was true when*, and *when we learned it*.
# 2. **Incremental graph updates** — adding one new fact without rebuilding everything.
# 3. **Hierarchical memory** — a small "hot" memory, a bigger "warm" one and an archive.
# 4. **Epistemic classification** — separating *facts*, *actions*, *beliefs* and *summaries*.
# 5. **Failure-mode diagnosis** — a checklist of 8 ways agent memory goes wrong.
# 6. **Consolidation** — turning many noisy episodes into one lasting pattern with provenance.
# 7. **Hybrid retrieval (RRF)** — merging several search "channels" into one ranked list.
# 8. **Consistency models** — how fresh must a memory read be when several agents share it?
#
# **The story (same as every chapter):** the checkout API is slow. An agent with *memory* can
# answer "what changed just before the outage?" even weeks later. We store the memory itself
# **as nodes and relationships in Neo4j** and query it with **Cypher**.
#
# **Quick vocabulary**
# * **Node** = a thing (a service, a fact, an episode). **Relationship** (edge) = a link between two nodes.
# * **Property** = a key/value on a node or relationship (e.g. `valid_from`).
# * **Cypher** = Neo4j's query language: `(a)-[:DEPENDS_ON]->(b)` means "a depends on b".
# * **Skill** = a small Python file under `skills/` implementing one idea from the book. We import it with `g.load_skill`.
#
# > **Before running:** start Neo4j (`docker compose -f docker-compose/docker-compose.yml up -d`), then run cells top to bottom.

# %% [markdown]
# ## 0. Setup: connect and load the incident graph
#
# `load_devops_graph()` wipes the database and creates the incident story (services, a database,
# a deployment, an alarm and two incidents). From here on we *add memory structures* to it.

# %%
import json
from datetime import datetime, timezone, timedelta

import graph_helpers as g

print(g.load_devops_graph())


def ts(s):
    """ISO string like 2026-03-10T17:45:00Z -> timezone-aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def to_dt(v):
    """Neo4j datetime / ISO string / None -> python datetime / None."""
    if v is None:
        return None
    if hasattr(v, "to_native"):
        return v.to_native()
    return ts(v) if isinstance(v, str) else v


bt = g.load_skill("memory", "bi-temporal-edge")
print("Loaded bi-temporal-edge:", [n for n in dir(bt) if n in ("create_edge", "as_of", "history")])

# %% [markdown]
# ## 1. Bi-temporal edges — "what was true *then*?"
#
# Most databases only know the **current** value. But in an incident you ask:
# *"What instance type was checkout-api running **at 08:00 on 15 March**?"* — not today.
#
# A **bi-temporal** edge keeps two clocks:
# * `valid_from` / `valid_until` — when the fact was **true in the real world**.
# * `ingested_at` — when **we learned** it. The gap is the **ingestion lag**; a big lag means
#   our memory was wrong for a while.
#
# Instead of overwriting a value we **close** the old edge (set `valid_until` + a reason) and add a new one.
# Below we create the edges with the skill, then **store them in Neo4j** as `[:STATE]` relationships
# from the real `checkout-api` node to a `Config` node.

# %%
SRC = "service-checkout-api"
edges = []


def add(rel, value, vf, ing, **kw):
    e = bt.create_edge(SRC, "aws-ec2", rel, value=value, valid_from=ts(vf), ingested_at=ts(ing), **kw)
    edges.append(e)
    return e


t3 = add("instance_type", "t3.large", "2026-02-15T14:23:00Z", "2026-02-15T14:25:00Z",
         link_type="entity", metadata={"source": "terraform"})
m5 = add("instance_type", "m5.xlarge", "2026-03-10T17:45:00Z", "2026-03-10T17:46:00Z",
         link_type="entity", metadata={"source": "terraform"})
t3.valid_until = m5.valid_from  # close the old state exactly when the new one starts
t3.invalidation_reason = "next-gen migration to m5 family for steady-state throughput"

v341 = add("deployment_version", "v3.4.1", "2026-03-01T09:00:00Z", "2026-03-01T09:01:00Z", link_type="causal")
v350 = add("deployment_version", "v3.5.0", "2026-03-14T22:30:00Z", "2026-03-14T22:31:00Z", link_type="causal")
v341.valid_until = v350.valid_from
v341.invalidation_reason = "superseded by v3.5.0 release"

# This one was learned very late (56 days after it became true) -> big ingestion lag.
ebs = add("ebs_volume_gb", "500", "2026-01-18T10:00:00Z", "2026-03-15T09:10:00Z", link_type="entity")

print(f"Created {len(edges)} temporal edges in Python. Now persisting them in Neo4j...")

# %%
g.run("MATCH (c:Config {name:'aws-ec2'}) DETACH DELETE c")
for e in edges:
    g.run("""
        MATCH (s:Service {name:'checkout-api'})
        MERGE (c:Config {name:'aws-ec2'})
        CREATE (s)-[:STATE {rel:$rel, value:$value, link_type:$lt,
                            valid_from:$vf, valid_until:$vu, ingested_at:$ing,
                            invalidation_reason:$reason}]->(c)
    """, rel=e.relationship, value=str(e.value), lt=e.link_type, vf=e.valid_from, vu=e.valid_until,
          ing=e.ingested_at, reason=e.invalidation_reason)

g.show(g.run("""
    MATCH (:Service {name:'checkout-api'})-[r:STATE]->(:Config)
    RETURN r.rel AS what, r.value AS value, toString(r.valid_from) AS valid_from,
           toString(r.valid_until) AS valid_until, toString(r.ingested_at) AS ingested_at,
           r.invalidation_reason AS why_closed
    ORDER BY what, valid_from
"""))

# %% [markdown]
# **How to read this table:** each row is one edge. `valid_until = null` means *still true now*.
# Closed rows carry a `why_closed` — an audit trail instead of a silent overwrite.
#
# Now the key query: "what was true at the moment of the outage (2026-03-15 08:00 UTC)?"
# In Cypher this is simply *"valid_from ≤ T and (valid_until is null or valid_until > T)"*.

# %%
OUTAGE_TS = ts("2026-03-15T08:00:00Z")


def state_at(rel, when):
    rows = g.run("""
        MATCH (:Service {name:'checkout-api'})-[r:STATE {rel:$rel}]->(:Config)
        WHERE r.valid_from <= $t AND (r.valid_until IS NULL OR r.valid_until > $t)
        RETURN r.value AS value
    """, rel=rel, t=when)
    return rows[0]["value"] if rows else None


print("Cypher  @outage  instance_type:", state_at("instance_type", OUTAGE_TS))
print("Cypher  @outage  deployment   :", state_at("deployment_version", OUTAGE_TS))
print("Cypher  @03-01   instance_type:", state_at("instance_type", ts("2026-03-01T00:00:00Z")))

# %% [markdown]
# The skill answers the same question in Python (`bt.as_of`). To prove the graph really is the
# source of truth, we **rebuild the edges from Neo4j rows** and ask the skill.

# %%
rows = g.run("""
    MATCH (:Service {name:'checkout-api'})-[r:STATE]->(:Config)
    RETURN r.rel AS rel, r.value AS value, r.link_type AS lt, r.valid_from AS vf,
           r.valid_until AS vu, r.ingested_at AS ing, r.invalidation_reason AS why
""")
db_edges = []
for r in rows:
    e = bt.create_edge(SRC, "aws-ec2", r["rel"], value=r["value"], valid_from=to_dt(r["vf"]),
                       ingested_at=to_dt(r["ing"]), link_type=r["lt"])
    e.valid_until, e.invalidation_reason = to_dt(r["vu"]), r["why"]
    db_edges.append(e)

inst = bt.as_of(SRC, "instance_type", OUTAGE_TS, db_edges)
dep = bt.as_of(SRC, "deployment_version", OUTAGE_TS, db_edges)
old = bt.as_of(SRC, "instance_type", ts("2026-03-01T00:00:00Z"), db_edges)
print(f"At outage: instance={inst.value}, deploy={dep.value}; on 03-01 instance was {old.value}")

print("\nFull history of the instance type:")
for e in bt.history(SRC, db_edges, relationship="instance_type"):
    print(f"   {e.value:10s} {e.valid_from:%Y-%m-%d %H:%M} -> {e.valid_until or 'now'}")

# What changed in the 24h BEFORE the outage? (Cypher: valid_from inside a window)
changed = g.run("""
    MATCH (:Service {name:'checkout-api'})-[r:STATE]->(:Config)
    WHERE r.valid_from > $t - duration('PT24H') AND r.valid_from <= $t
    RETURN r.rel AS what, r.value AS new_value, toString(r.valid_from) AS at
""", t=OUTAGE_TS)
g.show(changed)

lag = [e for e in db_edges if e.relationship == "ebs_volume_gb"][0].ingestion_lag()
lag_days = lag.total_seconds() / 86400
print(f"EBS ingestion lag: {lag_days:.1f} days")

# --- Verification gate (same as the original notebook) ---------------------
assert inst.value == "m5.xlarge" and dep.value == "v3.5.0"
assert old.value == "t3.large"
assert len(changed) == 1 and changed[0]["new_value"] == "v3.5.0", "exactly one change in the 24h before outage"
assert lag_days > 55, "the EBS edge was learned ~56 days late"
assert state_at("instance_type", OUTAGE_TS) == inst.value, "Cypher and skill must agree"
print("[PASS] bi-temporal memory answers 'what was true then' and 'what changed before'.")

# %% [markdown]
# ### 1b. Cross-check against the evidence (log events as nodes)
#
# A memory is only trustworthy if it matches reality. We store the symptoms of the 08:00 spike as
# `LogEvent` nodes linked to `checkout-api`, then confirm the deploy recorded in memory (v3.5.0)
# was the *latest one before* the first error.

# %%
g.run("MATCH (e:LogEvent) DETACH DELETE e")
events = [("2026-03-15T07:45:00Z", "WARN", "latency p99 rising 1200ms"),
          ("2026-03-15T08:00:00Z", "ERROR", "504 gateway timeout"),
          ("2026-03-15T08:00:30Z", "ERROR", "db_pool_exhausted"),
          ("2026-03-15T08:01:00Z", "ERROR", "503 service unavailable")]
for t, lvl, msg in events:
    g.run("MATCH (s:Service {name:'checkout-api'}) CREATE (s)-[:EMITTED]->(:LogEvent {at:$t, level:$l, message:$m})",
          t=ts(t), l=lvl, m=msg)

g.show(g.run("""
    MATCH (:Service {name:'checkout-api'})-[:EMITTED]->(e:LogEvent)
    RETURN toString(e.at) AS at, e.level AS level, e.message AS message ORDER BY e.at
"""))
first_err = g.run("""MATCH (e:LogEvent {level:'ERROR'}) RETURN e.at AS at ORDER BY e.at LIMIT 1""")[0]["at"]
deploy_before = bt.as_of(SRC, "deployment_version", to_dt(first_err), db_edges)
now_inst = g.run("""MATCH (:Service {name:'checkout-api'})-[r:STATE {rel:'instance_type'}]->()
                    WHERE r.valid_until IS NULL RETURN r.value AS v""")[0]["v"]
print("Deploy valid at first error:", deploy_before.value, "| current instance in graph:", now_inst)
assert deploy_before.value == "v3.5.0"
assert now_inst == bt.as_of(SRC, "instance_type", datetime.now(timezone.utc), db_edges).value == "m5.xlarge"
print("[PASS] logs in the graph corroborate the bi-temporal memory.")

# %% [markdown]
# ## 2. Incremental graph updates (Graphiti-style)
#
# A new sentence arrives ("Sarah opened incident INC-1042…"). Naively you would re-extract the whole
# knowledge graph. **Incremental update** touches only what the episode mentions:
# 1. find entities in the text, 2. **resolve** each to an existing node (so "Sarah" and "s.chen" are
# *one* person — *entity resolution*), 3. add new relationships.
#
# The skill keeps the working graph in Python; after each episode we **upsert only the touched
# nodes and new edges into Neo4j** (`MERGE` = create if missing, otherwise reuse).

# %%
gi = g.load_skill("memory", "graphiti-incremental-update")
kg = gi.Graph()   # note: 'kg' (our knowledge graph), not 'g' (graph_helpers)

g.run("MATCH (n:MemEntity) DETACH DELETE n")
_seen_nodes, _seen_edges = {}, set()


def persist(kg):
    """Upsert only nodes whose content changed and edges not yet stored. Returns (nodes_written, edges_written)."""
    nodes = kg.nodes.items() if isinstance(kg.nodes, dict) else [(n.id, n) for n in kg.nodes]
    es = kg.edges.values() if isinstance(kg.edges, dict) else kg.edges
    nw = ew = 0
    for nid, n in nodes:
        sig = (n.name, tuple(sorted(n.aliases)))
        if _seen_nodes.get(nid) != sig:
            g.run("MERGE (m:MemEntity {uuid:$id}) SET m.name=$name, m.type=$type, m.aliases=$aliases",
                  id=str(nid), name=n.name, type=n.type, aliases=sorted(n.aliases))
            _seen_nodes[nid] = sig
            nw += 1
    for e in es:
        if e.id not in _seen_edges:
            g.run("""MATCH (a:MemEntity {uuid:$s}), (b:MemEntity {uuid:$t})
                     CREATE (a)-[:RELATES {relationship:$rel, episode_id:$ep, valid_from:datetime(), valid_to:null}]->(b)""",
                  s=str(e.source_id), t=str(e.target_id), rel=e.relationship, ep=e.episode_id)
            _seen_edges.add(e.id)
            ew += 1
    return nw, ew


episodes = [
    "person:Sarah(s.chen) opened incident:INC-1042 affecting service:checkout-api in region:us-east-1",
    "person:Marcus joined incident:INC-1042 as secondary on-call",
    "service:checkout-api depends_on service:payments-service",
    "deployment:v3.5.0 changed service:payments-service",
    "person:Sarah reviewed incident:INC-1042 postmortem",
    "incident:INC-1042 caused_by deployment:v3.5.0",
]
last = None
for i, text in enumerate(episodes, 1):
    last = gi.add_episode(text, kg, episode_id=f"e{i}")
    nw, ew = persist(kg)
    print(f"episode {i}: touched={last['touched_nodes']} nodes | Neo4j writes: {nw} nodes, {ew} edges | resolved via {last['resolution_method_counts']}")

print("Final in-memory graph:", len(kg.nodes), "nodes,", len(kg.edges), "edges")

# %% [markdown]
# **How to read this:** each line shows how few nodes an episode touched and how few writes hit Neo4j.
# Notice the 5th episode says plain `Sarah` — the resolver maps it back to the existing person
# (the `canonical` method), so we do **not** get a duplicate. Check it in the database:

# %%
g.show(g.run("MATCH (m:MemEntity {type:'person'}) RETURN m.name AS name, m.aliases AS aliases"))
g.show(g.run("""
    MATCH (a:MemEntity)-[r:RELATES]->(b:MemEntity)
    RETURN a.name AS source, r.relationship AS rel, b.name AS target, r.episode_id AS episode
    ORDER BY episode
"""))

# %% [markdown]
# **Locality test:** adding one episode should cost the same in a 5-node graph and a 500-node graph.

# %%
def make_graph(n):
    kg_ = gi.Graph()
    for i in range(n):
        kg_.add_node(f"filler-{i}", "service")
    return kg_


results = []
for size in (5, 500):
    kg_ = make_graph(size)
    before = kg_.snapshot()
    res = gi.add_episode("person:Eve opened incident:INC-9", kg_, episode_id="x")
    after = kg_.snapshot()
    results.append(res['touched_nodes'])
    print(f"graph of {size:>3} nodes -> episode touched {res['touched_nodes']} nodes; "
          f"locality report: {gi.verify_locality(before, after)}")

# --- Verification gate ------------------------------------------------------
sarahs = g.run("MATCH (m:MemEntity) WHERE m.name = 'Sarah' OR 'Sarah' IN m.aliases OR 's.chen' IN m.aliases RETURN count(m) AS c")[0]["c"]
assert sarahs == 1, "Sarah must be a single node"
assert len([n for n in (kg.nodes.values() if isinstance(kg.nodes, dict) else kg.nodes) if n.name == "Sarah"]) == 1
assert results[0] == results[1], "touched nodes must not depend on graph size"
resolved = gi.add_episode("person:Sarah reviewed incident:INC-1042 postmortem", kg, episode_id="e-re")
assert resolved["resolution_method_counts"].get("canonical", 0) >= 1
print("[PASS] one episode = a small, constant-size update; Sarah stays one node.")

# %% [markdown]
# ## 3. Hierarchical memory — hot, warm, archive
#
# An agent's context window is small. **Hierarchical memory** (the MemGPT/Letta idea) copies how
# computers work:
# * **Core** — a few facts always "in view" (limit 8 here).
# * **Recall** — recent interactions.
# * **Archival** — everything that was pushed out. **Eviction is not deletion**: it is still searchable.
#
# Each fact is **durable** (keep!) or **short-lived** (shell output, log tails). When core is full, the
# lowest-scoring fact is evicted — durable ones get a 10× protection.
#
# First we **derive durable facts from the real graph** with Cypher, so memory is grounded.

# %%
dep_row = g.run("""
    MATCH (s:Service {name:'checkout-api'})-[:DEPENDS_ON]->(d)
    RETURN s.name AS s, collect(d.name) AS deps
""")[0]
dep_fact = f"{dep_row['s']} depends on {', '.join(dep_row['deps'])}"
dr = g.run("MATCH (d:Deployment)-[:CHANGED]->(x) RETURN d.id AS d, x.name AS x, d.change AS change LIMIT 1")[0]
deploy_fact = f"{dr['d']} changed {dr['x']}: {dr['change']}"
print("Graph-derived durable facts:\n  -", dep_fact, "\n  -", deploy_fact)

# %%
hm = g.load_skill("memory", "hierarchical-memory")
mem = hm.HierarchicalMemory(core_limit=8)

durables = [
    "Production AWS account is 123456789012",
    "Primary region is us-east-1",
    "On-call rotation: Sarah primary, Marcus secondary",
    dep_fact,
    deploy_fact,
]
for d in durables:
    mem.add_fact(d, hm.DURABILITY_DURABLE)

short_lived = [
    "Running: aws ec2 describe-instances", "Tail: 07:59 timeout",
    "Running: kubectl get pods", "Tail: 08:00 504 gateway",
    "Running: gh pr view 1234", "Tail: 08:01 503",
    "Running: terraform plan", "Tail: 08:02 retry-storm",
    "Running: aws logs filter-log-events", "Tail: 08:03 circuit-breaker open",
    "Running: psql -c count", "Tail: 08:04 db-connection-pool full",
    "Running: redis-cli ping", "Tail: 08:05 latency p99 8200ms",
    "Running: aws sts get-caller-identity", "Tail: 08:06 oncall-page-fired",
]
for s in short_lived:
    mem.add_fact(s, hm.DURABILITY_SHORT_LIVED)

print(f"core size = {len(mem.core)} (limit 8), archival size = {len(mem.archival)}")
acct = mem.query("account")
cb = mem.query("circuit-breaker")
print("query('account') core hits:", [h["content"] for h in acct["core"]])
print("query('circuit-breaker') archival hits:", [h["content"] for h in cb["archival"]])

# %% [markdown]
# Now **persist the tiers as a graph**: one `MemTier` node per tier, and every fact a `MemFact` node linked
# with `[:HOLDS]`. `valid_from` = when the fact entered memory, `valid_to` = when it left the core tier
# (`null` = still in core), plus the eviction reason.

# %%
g.run("MATCH (n) WHERE n:MemFact OR n:MemTier DETACH DELETE n")
for tier in ("core", "archival"):
    g.run("CREATE (:MemTier {name:$n})", n=tier)


def store_fact(f, tier):
    g.run("""MATCH (t:MemTier {name:$tier})
             CREATE (t)-[:HOLDS]->(:MemFact {content:$c, durability:$d, tier:$tier,
                                             valid_from:$vf, valid_to:$vt, eviction_reason:$why})""",
          tier=tier, c=f.content, d=f.durability, vf=f.created_at, vt=f.evicted_at, why=f.eviction_reason)


for f in mem.core.values():
    store_fact(f, "core")
for f in mem.archival:
    store_fact(f, "archival")

g.show(g.run("""
    MATCH (t:MemTier)-[:HOLDS]->(f:MemFact)
    RETURN t.name AS tier, f.durability AS durability, count(f) AS facts ORDER BY tier, durability
"""))
g.show(g.run("""
    MATCH (:MemTier {name:'core'})-[:HOLDS]->(f:MemFact) RETURN f.content AS core_fact, f.durability AS durability
"""))
g.show(g.run("""
    MATCH (f:MemFact {tier:'archival'}) WHERE f.content CONTAINS 'circuit-breaker'
    RETURN f.content AS archived_fact, f.eviction_reason AS why_evicted
"""))

# --- Verification gate -------------------------------------------------------
survivors = [d for d in durables if d in mem.core]
assert len(survivors) >= 3, f"durable facts should resist eviction, only {len(survivors)} survived"
assert acct["core"], "the durable account fact should be a core hit"
assert cb["archival"], "an evicted short-lived fact must remain searchable in archival"
assert len(mem.core) <= 8
core_in_db = g.run("MATCH (:MemTier {name:'core'})-[:HOLDS]->(f) RETURN count(f) AS c")[0]["c"]
assert core_in_db == len(mem.core) <= 8, "Neo4j core tier mirrors the in-memory core"
print(f"[PASS] {len(survivors)}/{len(durables)} durable facts stayed hot; evicted noise stayed searchable.")

# %% [markdown]
# ## 4. Epistemic classification (Hindsight)
#
# **Epistemic** = "about knowledge". Agents mix up *what they observed* with *what they guess*.
# Hindsight sorts every statement into one of four **networks**:
# * **World** — objective facts. * **Experience** — what the agent itself did.
# * **Opinion** — a belief with a **confidence** and the evidence it came from.
# * **Observation** — a summary derived from several sources.
#
# We also add a World fact **from the graph** (the deployment recorded in Neo4j).

# %%
he = g.load_skill("memory", "hindsight-epistemic-classifier")
now_iso = datetime.now(timezone.utc).isoformat()

deploy_world = f"Deployment {dr['d']} changed {dr['x']} ({dr['change']})."
W1 = "The checkout API returned 503 errors between 08:00 and 08:30 UTC."
E2 = "I retrieved the deployment log for v3.5.0."
OP = "I believe the root cause is the v3.5.0 deployment."
postmortem = [
    {"text": W1, "metadata": None},
    {"text": "The production AWS account is 123456789012.", "metadata": {"external_ref": "iam-policy.json"}},
    {"text": deploy_world, "metadata": {"external_ref": "neo4j:Deployment"}},
    {"text": "I called CloudWatch GetLogEvents at 08:15.",
     "metadata": {"action_type": "cloudwatch_call", "source": "agent", "timestamp": now_iso}},
    {"text": E2, "metadata": {"source": "agent", "timestamp": now_iso}},
    {"text": OP, "metadata": {"confidence": 0.7, "inferred_from": [W1, E2]}},
    {"text": "In summary, the timeline is: deploy 22:30, latency rising 07:45, 503s at 08:00, rollback 08:32.",
     "metadata": {"derived_from": ["deploy-log", "cloudwatch", "rollback-log", "incident-channel"]}},
]
facts = he.classify_batch(postmortem)
for f in facts:
    extra = f" conf={f.confidence}" if f.confidence < 1.0 else ""
    print(f"   [{f.network:11s}]{extra}  {f.text}")
audit = he.network_audit(facts)
print("\nnetwork_audit:", audit["counts"], "| warnings:", audit["warnings"] or "none")

# %% [markdown]
# Now store each statement as a `Belief` node, and draw `[:INFERRED_FROM]` edges from opinions/observations
# to their evidence. "How do you know?" becomes a one-line graph traversal.

# %%
g.run("MATCH (b:Belief) DETACH DELETE b")
for f in facts:
    g.run("CREATE (:Belief {text:$t, network:$n, confidence:$c, valid_from:datetime(), valid_to:null})",
          t=f.text, n=f.network, c=float(f.confidence))
for f in facts:
    for ref in list(f.inferred_from or []) + list(f.derived_from or []):
        g.run("""MATCH (a:Belief {text:$a})
                 MERGE (b:Belief {text:$b}) ON CREATE SET b.network='Source', b.confidence=1.0
                 CREATE (a)-[:INFERRED_FROM]->(b)""", a=f.text, b=ref)

g.show(g.run("MATCH (b:Belief) RETURN b.network AS network, count(b) AS beliefs ORDER BY network"))
g.show(g.run("""
    MATCH (o:Belief {network:'Opinion'})-[:INFERRED_FROM]->(e:Belief)
    RETURN o.text AS opinion, o.confidence AS confidence, e.network AS evidence_type, e.text AS evidence
"""))

just = he.justify(facts, OP)
print(json.dumps(just, indent=2))

networks = {f.network for f in facts}
assert networks == {he.NETWORK_WORLD, he.NETWORK_EXPERIENCE, he.NETWORK_OPINION, he.NETWORK_OBSERVATION}
assert just.get("provenance") and len(just["provenance"]) >= 2
db_evidence = g.run("MATCH (:Belief {text:$t})-[:INFERRED_FROM]->(e) RETURN count(e) AS c", t=OP)[0]["c"]
assert db_evidence >= 2, "Neo4j holds the same provenance chain"
print("[PASS] evidence is separated from belief, and the opinion is justified in the graph.")

# %% [markdown]
# ## 5. Memory failure modes (Letta)
#
# Memory systems fail in predictable ways: silent overwrites, core clogged with noise, disconnected
# facts, linear-scan retrieval… The `letta-failure-modes` skill checks a **snapshot** (a dict of
# measurements about your memory) against **8 failure modes**.
#
# First tag all memory structures we built with a common `:Memory` label, connect them to an `:Agent`
# root, and **measure the graph** (nodes, edges, connectivity) with Cypher.

# %%
g.run("MATCH (a:Agent {name:'ops-agent'}) DETACH DELETE a")
g.run("""MATCH (n) WHERE n:MemEntity OR n:MemFact OR n:MemTier OR n:Belief OR n:Config
         SET n:Memory""")
g.run("CREATE (:Agent:Memory {name:'ops-agent'})")
g.run("""MATCH (a:Agent {name:'ops-agent'}), (n) WHERE n:MemTier OR n:Config OR n:MemEntity OR n:Belief
         MERGE (a)-[:REMEMBERS]->(n)""")

node_count = g.run("MATCH (n:Memory) RETURN count(n) AS c")[0]["c"]
edge_count = g.run("MATCH (:Memory)-[r]->(:Memory) RETURN count(r) AS c")[0]["c"]
pairs = g.run("MATCH (a:Memory)-[]->(b:Memory) RETURN elementId(a) AS a, elementId(b) AS b")
ids = {r["id"] for r in g.run("MATCH (n:Memory) RETURN elementId(n) AS id")}
parent = {i: i for i in ids}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for p in pairs:
    parent[find(p["a"])] = find(p["b"])
components = len({find(i) for i in ids})
avg_degree = round(2 * edge_count / node_count, 2)
print(f"memory graph: {node_count} nodes, {edge_count} edges, {components} connected component(s), avg degree {avg_degree}")

# %%
lf = g.load_skill("memory", "letta-failure-modes")
core_vals = list(mem.core.values())
clean_snapshot = {
    "core_size": len(mem.core), "core_limit": 8,
    "core_durable_count": sum(1 for f in core_vals if f.durability == hm.DURABILITY_DURABLE),
    "core_short_lived_count": sum(1 for f in core_vals if f.durability == hm.DURABILITY_SHORT_LIVED),
    "recall_size": 42, "archival_size": len(mem.archival), "archival_durable_count": 0,
    "node_count": node_count, "edge_count": edge_count,
    "disconnected_components": components, "avg_degree": avg_degree,
    "uses_bi_temporal_edges": True, "facts_without_created_at": 0,
    "facts_without_invalidation_reason_when_invalidated": 0,
    "retrieval_method": "hybrid", "retrieval_complexity_class": "O(log n)",
    "has_query_log": True, "extract_fn_present": True, "size_stress_test_passed": True,
}
broken_snapshot = {
    "core_size": 10, "core_limit": 10, "core_durable_count": 2, "core_short_lived_count": 8,
    "recall_size": 0, "archival_size": 200, "archival_durable_count": 15,
    "node_count": 50, "edge_count": 10, "disconnected_components": 30, "avg_degree": 0.4,
    "uses_bi_temporal_edges": False, "facts_without_created_at": 100,
    "facts_without_invalidation_reason_when_invalidated": 20,
    "retrieval_method": "linear-scan", "retrieval_complexity_class": "O(n)",
    "has_query_log": False, "extract_fn_present": False, "size_stress_test_passed": False,
}
clean_report = lf.diagnose(lf.MemorySnapshot(**clean_snapshot))
broken_report = lf.diagnose(lf.MemorySnapshot(**broken_snapshot))
clean_present = [m for m in clean_report.modes if m.status == "present"]
broken_present = [m for m in broken_report.modes if m.status == "present"]
print(f"Clean architecture : {len(clean_present)} / {len(clean_report.modes)} failure modes present")
print(f"Broken architecture: {len(broken_present)} / {len(broken_report.modes)} failure modes present\n")
for m in broken_present:
    print(f"   [sev {m.severity}] {m.name}: {m.fix}")

# %% [markdown]
# **Reading it:** `status == "present"` means that failure mode is happening. Our real memory graph
# (measured from Neo4j) triggers none; the deliberately bad "anti-pattern" snapshot lights up most.

# %%
assert len(clean_report.modes) == 8, "there are exactly eight Letta failure modes"
assert len(clean_present) == 0, f"composed architecture should be clean, got {[m.name for m in clean_present]}"
assert len(broken_present) >= 6
silent = next(m for m in broken_report.modes if m.name == "silent-overwrite")
assert silent.status == "present" and silent.severity == 3
print("[PASS] the composed memory is clean; the anti-pattern is caught.")

# %% [markdown]
# ## 6. Memory consolidation — many episodes → one lasting pattern
#
# An **episode** is one raw event ("incident X happened"). Keeping thousands makes retrieval noisy.
# **Consolidation** clusters similar episodes by topic, and when ≥3 agree it writes one durable
# **Pattern** with `confirmations` and `derived_from` (**provenance** = where it came from).
# Persist: `Episode` nodes, one `Pattern` node, `[:DERIVED_FROM]` edges.

# %%
mc = g.load_skill("memory", "memory-consolidation")
records = [
    {"id": "ep1", "episode_type": "Incident",
     "content": "checkout-api 503 errors after deploy v3.5.0 raised connection pool timeout"},
    {"id": "ep2", "episode_type": "Incident",
     "content": "checkout-api 503 spike following deploy v3.6.1 connection pool exhausted timeout"},
    {"id": "ep3", "episode_type": "Incident",
     "content": "checkout-api 503 errors traced to deploy v3.7.0 connection pool timeout under load"},
    {"id": "ep4", "episode_type": "Deployment",
     "content": "routine deploy of inventory-service v2.1.0 no errors observed"},
    {"id": "ep5", "episode_type": "Conversation",
     "content": "user asked about billing dashboard color scheme preferences"},
]
g.run("MATCH (n) WHERE n:Episode OR n:Pattern DETACH DELETE n")
for r in records:
    g.run("CREATE (:Episode:Memory {id:$id, type:$t, content:$c, valid_from:datetime()})",
          id=r["id"], t=r["episode_type"], c=r["content"])

# Read the episodes BACK from Neo4j and consolidate those.
db_records = g.run("MATCH (e:Episode) RETURN e.id AS id, e.type AS episode_type, e.content AS content ORDER BY e.id")
episodes_ = mc.episodes_from_records(db_records)
clusters = mc.cluster_by_topic(episodes_)
print("Topic clusters:", [[e.id for e in c] for c in clusters])

facts_c = mc.consolidate(episodes_)
for fct in facts_c:
    print(f"[{fct.knowledge_type}] {fct.summary}\n   confirmations={fct.confirmations} derived_from={fct.derived_from}")
    g.run("CREATE (:Pattern:Memory {id:$id, summary:$s, knowledge_type:$k, confirmations:$c, valid_from:datetime(), valid_to:null})",
          id=fct.id, s=fct.summary, k=fct.knowledge_type, c=fct.confirmations)
    for ep in fct.derived_from:
        g.run("MATCH (p:Pattern {id:$p}), (e:Episode {id:$e}) CREATE (p)-[:DERIVED_FROM {valid_from:datetime()}]->(e)",
              p=fct.id, e=ep)

# %% [markdown]
# **"How do you know this pattern?"** — follow `DERIVED_FROM` back to the raw episodes:

# %%
g.show(g.run("""
    MATCH (p:Pattern)-[:DERIVED_FROM]->(e:Episode)
    RETURN p.summary AS pattern, p.confirmations AS confirmations, e.id AS episode, e.content AS evidence
    ORDER BY episode
"""))
prov = mc.provenance_of(facts_c[0], episodes_)
print("skill provenance:", [e.id for e in prov])

inferences = mc.precompute_inferences(
    facts_c,
    inference_fn=lambda f: ["Risk: the next checkout-api deploy may exhaust the connection pool - pre-stage a runbook."])
print(json.dumps(inferences, indent=2))

assert len(facts_c) == 1
assert facts_c[0].confirmations == 3
assert sorted(facts_c[0].derived_from) == ["ep1", "ep2", "ep3"]
assert "ep4" not in facts_c[0].derived_from and "ep5" not in facts_c[0].derived_from
cnt = g.run("MATCH (:Pattern)-[r:DERIVED_FROM]->(:Episode) RETURN count(r) AS c")[0]["c"]
assert cnt == 3
print("[PASS] three noisy incidents became one provenance-backed Pattern in Neo4j.")

# %% [markdown]
# ## 7. Hybrid retrieval with Reciprocal Rank Fusion (RRF)
#
# No single search finds everything. We run several **channels** and **fuse** their ranked lists:
# * **semantic** — similar meaning (needs *embeddings*: lists of numbers capturing meaning; we reuse the
#   book's pre-computed ranking, since no model is needed here),
# * **keyword** — exact words, via Cypher `CONTAINS`,
# * **graph** — closeness in the graph (shortest path from the deploy), via Cypher,
# * **temporal** — newest first, via `valid_from`.
#
# **RRF** scores each document `Σ 1/(k + rank)` across channels, so a document that is *consistently
# good* beats one that is great in only one channel. Then a **reranker** re-scores the top few and a
# **token budget** trims the context to fit the LLM.
#
# First we store the candidate documents and their links in Neo4j.

# %%
g.run("MATCH (d:MemDoc) DETACH DELETE d")
metadata = {
    "INC-1042": "Checkout API 503 errors caused by v3.5.0 deploy connection-pool timeouts on payments",
    "INC-1043": "Inventory service degradation traced to v2.1.0 deploy",
    "INC-1044": "Payments region-specific failover test",
    "INC-1045": "Fraud-detection service introduced as checkout dependency",
    "INC-2024-11-23": "Historical checkout latency incident from last year",
    "deploy-v3.5.0": "Deployment v3.5.0 of checkout-api at 22:30 UTC",
    "service-payments-outage": "Payments dependency outage entry",
}
when = {"INC-1042": "2026-03-15", "INC-1043": "2026-03-12", "INC-1044": "2026-03-16", "INC-1045": "2026-03-17",
        "INC-2024-11-23": "2024-11-23", "deploy-v3.5.0": "2026-03-14", "service-payments-outage": "2026-03-10"}
for did, text in metadata.items():
    g.run("CREATE (:MemDoc:Memory {id:$id, text:$t, valid_from:datetime($d), valid_to:null})", id=did, t=text, d=when[did] + "T00:00:00Z")
for a, b in [("INC-1042", "deploy-v3.5.0"), ("INC-1042", "service-payments-outage"), ("INC-1042", "INC-1043"),
             ("deploy-v3.5.0", "INC-1044"), ("INC-1044", "INC-1045")]:
    g.run("MATCH (x:MemDoc {id:$a}), (y:MemDoc {id:$b}) CREATE (x)-[:RELATED]->(y)", a=a, b=b)

# %%
keyword = [r["id"] for r in g.run("""
    MATCH (d:MemDoc)
    WITH d, [t IN ['checkout','deploy','503','v3.5.0'] WHERE toLower(d.text) CONTAINS t] AS hits
    WHERE size(hits) > 0
    RETURN d.id AS id ORDER BY size(hits) DESC, d.id
""")]
graph_ch = [r["id"] for r in g.run("""
    MATCH (seed:MemDoc {id:'deploy-v3.5.0'}), (d:MemDoc)
    MATCH p = shortestPath((seed)-[:RELATED*0..3]-(d))
    RETURN d.id AS id ORDER BY length(p), d.id
""")]
temporal = [r["id"] for r in g.run("""
    MATCH (d:MemDoc) WHERE d.valid_from >= datetime('2026-03-01T00:00:00Z')
    RETURN d.id AS id ORDER BY d.valid_from DESC
""")]
channels = {
    "semantic": ["INC-1043", "INC-1042", "INC-2024-11-23", "INC-1044"],  # embedding stand-in from the book
    "keyword": keyword, "graph": graph_ch, "temporal": temporal,
}
for k, v in channels.items():
    print(f"{k:9s}", v)

# %%
rr = g.load_skill("memory", "rrf-hybrid-retrieval")
fused = rr.rrf_fuse(channels)
print("RRF fusion (top 5):")
for doc_id, score in fused[:5]:
    cov = sum(1 for c in channels.values() if doc_id in c)
    print(f"   {score:.5f}  {doc_id:24s} (in {cov}/4 channels)")

reranked = rr.cross_encoder_rerank(fused[:6], query="checkout latency from a recent deploy", metadata=metadata)
print("\nRerank (top 3):")
for doc_id, fs, rs in reranked[:3]:
    print(f"   fusion={fs:.5f} rerank={rs:.3f}  {doc_id}")

final = rr.token_budget_filter(reranked[:5], get_tokens=lambda d: len(metadata.get(d, d)), budget=200)
print(f"\nContext under a 200-token budget ({len(final)} items):")
for item in final:
    print(f"   {item['doc_id']}  ({item['token_count']} tokens)")

assert fused[0][0] == "INC-1042", fused[:3]
assert sum(i["token_count"] for i in final) <= 200
assert "INC-1042" in {r[0] for r in reranked[:3]}
print("[PASS] four channels fused, reranked and budget-fitted; the root-cause incident is on top.")

# %% [markdown]
# ## 8. Consistency models — how fresh must a shared memory read be?
#
# When **several agents** share a memory, one may read a stale copy. Stronger consistency = safer
# but slower. The `memory-consistency-model-selector` skill picks one of four models from a few 0-3
# ratings: **strong** (always latest), **causal** (respect cause→effect order),
# **read_your_writes** (see your own writes) and **eventual** (fine if briefly stale). It can also
# **detect cache divergence**: an agent whose cached read is older than the latest write.

# %%
ms = g.load_skill("memory", "memory-consistency-model-selector")
ops = {
    "deploy-rollback lock": ms.Operation(shared_authoritative_state=3, conflict_intolerance=3, staleness_budget=0,
                                          collaboration=2, self_session_only=0),
    "incident timeline sharing":     ms.Operation(shared_authoritative_state=0, conflict_intolerance=0, staleness_budget=1,
                                              collaboration=3, self_session_only=0),
    "agent's own session notes": ms.Operation(shared_authoritative_state=0, conflict_intolerance=1, staleness_budget=2,
                                              collaboration=0, self_session_only=3),
    "background log enrichment": ms.Operation(shared_authoritative_state=0, conflict_intolerance=0, staleness_budget=3,
                                              collaboration=0, self_session_only=0),
}
picked = {}
for name, op in ops.items():
    rec = ms.recommend_model(op)
    picked[name] = rec["recommended"]
    print(f"{name:28s} -> {rec['recommended']}")

# Cache divergence, using the pool-size change stored in the graph.
t0 = datetime(2026, 3, 15, 8, 0, tzinfo=timezone.utc)
writes = [{"key": "orders-db.pool_size", "timestamp": (t0 + timedelta(minutes=5)).timestamp(), "value": "20", "agent": "agent-A"}]
caches = [{"agent": "agent-B", "key": "orders-db.pool_size", "cached_at": t0.timestamp()}]
warnings = ms.detect_cache_divergence(agent_writes=writes, cache_snapshots=caches)
print("\nCache divergence warnings:", warnings)

assert picked["deploy-rollback lock"] == "strong"
assert picked["incident timeline sharing"] == "causal"
assert picked["agent's own session notes"] == "read_your_writes"
assert picked["background log enrichment"] == "eventual"
assert len(warnings) == 1
print("[PASS] the right consistency model is chosen per operation; stale cache is flagged.")

# %% [markdown]
# ## 9. Verdict — one temporal root-cause answer from all the memory
#
# We now combine every primitive: bi-temporal state, the pattern, the belief + evidence, and the
# hybrid-retrieved incident — with the memory health check on top.

# %%
verdict = {
    "instance_at_outage": inst.value,
    "deploy_at_outage": dep.value,
    "changes_24h_before": [c["new_value"] for c in changed],
    "known_pattern": facts_c[0].summary,
    "pattern_confirmations": facts_c[0].confirmations,
    "opinion": OP, "opinion_confidence": 0.7,
    "evidence_count": len(just["provenance"]),
    "top_retrieved_incident": fused[0][0],
    "memory_failure_modes_present": f"{len(clean_present)}/8",
}
print(json.dumps(verdict, indent=2))

graph_check = g.run("""
    MATCH (inc:Incident {id:'INC-1001'})-[:AFFECTS]->(svc:Service)-[:DEPENDS_ON*1..2]->(dep)<-[:CHANGED]-(d:Deployment)
    RETURN svc.name AS service, dep.name AS suspect, d.id AS deploy
""")
g.show(graph_check)

assert verdict["instance_at_outage"] == "m5.xlarge" and verdict["deploy_at_outage"] == "v3.5.0"
assert verdict["changes_24h_before"] == ["v3.5.0"]
assert verdict["pattern_confirmations"] == 3 and verdict["evidence_count"] >= 2
assert verdict["top_retrieved_incident"] == "INC-1042"
assert verdict["memory_failure_modes_present"] == "0/8"
assert graph_check, "the live incident graph still points at the same suspect"
print("[PASS] ch4 verdict: memory explains the outage with time, evidence and provenance.")

# %% [markdown]
# ## Recap
# * **Bi-temporal edges** keep *when it was true* and *when we learned it* — never overwrite, close and add.
# * **Incremental updates** touch only the entities in a new episode; entity resolution avoids duplicates.
# * **Hierarchical memory** keeps durable facts hot; eviction moves noise to a still-searchable archive.
# * **Epistemic classes** separate World / Experience / Opinion / Observation and keep provenance.
# * **Failure-mode diagnosis** checks eight ways memory breaks, using measurements taken from the graph.
# * **Consolidation** turns repeated episodes into one pattern with `DERIVED_FROM` evidence.
# * **RRF hybrid retrieval** fuses semantic, keyword, graph and temporal channels.
# * **Consistency models** match freshness guarantees to what each operation needs.
#
# **Next:** `ch5-reasoning-planning.ipynb`.
