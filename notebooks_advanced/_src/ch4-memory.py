# %% [markdown]
# # Chapter 4 — Memory (advanced, real LLM + real Neo4j)
# > **Tier: advanced (real).** Needs OPENAI_API_KEY in notebooks_advanced/.env and the Neo4j container from docker-compose/.
# > **Evaluation focus:** see the section "Evaluation and LLM-as-a-Judge" near the end — every notebook in this tier treats eval and judging as first-class.
#
# > **Estimated OpenAI cost per full run (measured 2026-10-24 at `gpt-4o` rates): about $0.10.** 90 LLM calls (43 of them LLM-as-a-Judge), 18,388 input / 5,552 output tokens, plus a few thousand `text-embedding-3-large` tokens (<$0.01). Re-run costs about the same; `h.cost_report()` prints your own numbers. See [README](README.md#estimated-openai-cost-measured-24-october-2026).
#
# In tiers 1-3 you saw the eight memory primitives of Chapter 4 with heuristic stand-ins
# (Jaccard overlap, regexes, hand-written planners, simulated access control). This tier removes
# those crutches:
#
# | Concern | Earlier tiers | This notebook |
# |---|---|---|
# | Extraction / classification / planning | regex, hand-authored | **gpt-4o with JSON-schema structured outputs** |
# | Semantic retrieval | trigram / Jaccard | **text-embedding-3-large (3072 d) + Neo4j vector index** |
# | Keyword retrieval | token overlap | **Neo4j fulltext (Lucene) index** |
# | Agent | scripted tool loop | **OpenAI function calling**, trace stored as a graph |
# | Evaluation | `assert` on known strings | **labeled set, P@3 / MRR, LLM-as-judge, failure analysis** |
# | Access control | a Python `if` | **real Neo4j Enterprise RBAC** (`CREATE ROLE`, `GRANT`, `DENY`) |
#
# **What you will practise:** structured outputs, prompt versioning and A/B ablation, retries,
# token accounting (`h.cost_report()`), vector vs keyword vs graph vs hybrid ablations,
# heuristic-vs-LLM disagreement analysis, and a vibe-coding loop with a verification checklist.
#
# **There is no offline fallback.** If a key, the model, or Neo4j is missing, the first cell fails loudly.
# LLM output is non-deterministic even at temperature 0, so checks assert *structure and behaviour*
# (e.g. "the TLS outlier is not consolidated"), never exact wording.

# %%
import sys, os, json, re, math, subprocess
from datetime import datetime, timedelta, timezone
sys.path.insert(0, ".")
import real_helpers as h

client = h.get_openai()          # raises if OPENAI_API_KEY is missing
h.connect()                      # raises if Neo4j is not reachable
edition = h.run("CALL dbms.components() YIELD name, edition RETURN edition")[0]["edition"]
dim = len(h.embed(["preflight"])[0])
print(f"LLM model      : {h.LLM_MODEL}")
print(f"Embedding model: {h.EMBED_MODEL} ({dim} dims)")
print(f"Neo4j edition  : {edition}")
h.check("embedding dimension matches the vector index we will create", dim == h.EMBED_DIM)
h.check("Neo4j Enterprise (needed for real RBAC)", str(edition).lower() == "enterprise")
h.check("LLM answers", "ok" in h.llm("Reply with exactly: ok").lower())

# %% [markdown]
# ## 0. Setup - skills, a structured-output helper, and a prompt registry
#
# Three engineering habits that separate a demo from a system:
#
# 1. **Structured outputs.** Instead of asking the model for "JSON" and hoping, we pass a
#    **JSON schema** with `strict: true`. The API then constrains decoding to that schema, so the
#    result always has the right keys and enum values. We still validate and **retry** because
#    network errors and refusals exist.
# 2. **Prompt versioning.** Prompts are data. We keep them in a registry with explicit versions so
#    that an ablation ("v1 vs v2") is a one-line change and results are attributable.
# 3. **Skills stay pure Python.** Each skill's `lib.py` is deterministic and testable; the LLM and
#    Neo4j plug into its documented *seams* (`similarity_fn`, `summarizer_fn`, `rerank_score_fn`...).
#
# The first graph cell below resets the shared DevOps graph (`load_devops_graph()`), so the whole
# notebook can be rerun top to bottom.

# %%
h.load_devops_graph()   # resets the graph (indexes survive; nodes do not)

bt = h.load_skill("memory", "bi-temporal-edge")
gi = h.load_skill("memory", "graphiti-incremental-update")
hm = h.load_skill("memory", "hierarchical-memory")
he = h.load_skill("memory", "hindsight-epistemic-classifier")
lf = h.load_skill("memory", "letta-failure-modes")
cm = h.load_skill("memory", "memory-consistency-model-selector")
mc = h.load_skill("memory", "memory-consolidation")
rr = h.load_skill("memory", "rrf-hybrid-retrieval")


def ts(s):
    """ISO string -> timezone-aware datetime (the skills require aware datetimes)."""
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def sjson(prompt, schema, system="You are a precise extraction engine.", name="out", tries=3):
    """Structured output with strict JSON schema, validation and retries."""
    rf = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
    last = None
    for _ in range(tries):
        try:
            m = h.chat([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                       response_format=rf)
            data = json.loads(m.content)
            missing = [k for k in schema.get("required", []) if k not in data]
            if missing:
                raise ValueError(f"missing keys {missing}")
            return data
        except Exception as e:   # malformed output, refusal, transient API error
            last = e
    raise RuntimeError(f"structured call failed after {tries} tries: {last}")


def obj(props, required=None):
    """Tiny JSON-schema builder (strict mode needs every key required and no extra keys)."""
    return {"type": "object", "properties": props, "required": required or list(props),
            "additionalProperties": False}


PROMPTS = {
    "extract_config@v1": "Extract configuration changes from the note. Return changes as key/value/valid_from.",
    "extract_config@v2": ("You extract configuration CHANGES from an ops note. Rules: (1) `key` must be one of "
                          "instance_type, deployment_version, ebs_volume_gb; (2) `value` is ONLY the NEW value "
                          "(for ebs_volume_gb digits only, e.g. 500); (3) `valid_from` is when the change took "
                          "effect IN THE WORLD, as ISO-8601 UTC with the year taken from the note timestamp - "
                          "NOT when the note was written; (4) ignore old values mentioned for context."),
}
print("skills loaded:", len(h.skill_catalog("memory")), "| prompts:", list(PROMPTS))
print(h.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS n ORDER BY label"))

# %% [markdown]
# ## 1. Bi-temporal edges - from free-text notes to "what was true at the outage?"
#
# Earlier tiers seeded configuration history from a Python list. In reality it arrives as **messy
# prose** written at different times. So the pipeline is now:
#
# `free-text note -> LLM (strict schema) -> TemporalEdge (skill) -> ConfigVersion node (Neo4j)`
#
# **Bi-temporal** = two clocks: *valid time* (when it became true) and *ingestion time* (when we
# learned it). The EBS resize is valid since Jan 18 but only ingested Mar 15 - a 56-day lag that
# silently corrupts any "what did we know then?" question.
#
# We run **two prompt versions** against a labeled gold set and pick the better one by measured
# accuracy rather than by taste. Note `valid_from` is *not* the note's timestamp - that is the trap
# the v2 prompt calls out.

# %%
NOTES = [
    dict(id="n1", at="2026-02-15T14:25:00+00:00",
         text="Put checkout-api onto t3.large instances at 14:23 UTC today (Feb 15)."),
    dict(id="n2", at="2026-03-01T09:00:00+00:00",
         text="Deployed checkout-api v3.4.1 to production on March 1st, 09:00 UTC."),
    dict(id="n3", at="2026-03-10T17:46:00+00:00",
         text="Migrated checkout-api from t3.large to m5.xlarge at 17:45 UTC (Mar 10) after the capacity review."),
    dict(id="n4", at="2026-03-14T22:31:00+00:00",
         text="Release v3.5.0 of checkout-api went live on March 14 at 22:30 UTC, replacing v3.4.1."),
    dict(id="n5", at="2026-03-15T09:10:00+00:00",
         text="Audit finding: the checkout-api EBS volume was resized to 500 GB back on January 18 (never recorded)."),
]
GOLD = {("instance_type", "t3.large", "2026-02-15T14:23"), ("instance_type", "m5.xlarge", "2026-03-10T17:45"),
        ("deployment_version", "v3.4.1", "2026-03-01T09:00"), ("deployment_version", "v3.5.0", "2026-03-14T22:30"),
        ("ebs_volume_gb", "500", "2026-01-18T00:00")}
CHANGE_SCHEMA = obj({"changes": {"type": "array", "items": obj({
    "key": {"type": "string", "enum": ["instance_type", "deployment_version", "ebs_volume_gb"]},
    "value": {"type": "string"}, "valid_from": {"type": "string"}})}})


def norm(c):
    v = re.sub(r"\D", "", c["value"]) if c["key"] == "ebs_volume_gb" else c["value"].strip()
    return (c["key"], v, ts(c["valid_from"]).strftime("%Y-%m-%dT%H:%M"))


def extract_all(version):
    out = []
    for n in NOTES:
        data = sjson(f"Note written at {n['at']} (UTC):\n{n['text']}", CHANGE_SCHEMA,
                     system=PROMPTS[f"extract_config@{version}"], name="config_changes")
        out += [dict(c, note=n["id"], ingested_at=n["at"]) for c in data["changes"]]
    return out


results = {}
for v in ("v1", "v2"):
    ch = extract_all(v)
    got = {norm(c) for c in ch}
    results[v] = dict(changes=ch, tp=len(got & GOLD), fp=len(got - GOLD), fn=len(GOLD - got))
    print(f"prompt {v}: TP={results[v]['tp']} FP={results[v]['fp']} FN={results[v]['fn']}")
best = max(results, key=lambda v: (results[v]["tp"] - results[v]["fp"], v))   # tie -> v2
extracted = results[best]["changes"]
acc = results[best]["tp"] / len(GOLD)
print(f"using prompt {best}; extraction recall = {acc:.0%}")
h.check("structured extraction recovers at least 4 of the 5 config facts", acc >= 0.8)
print(h.cost_report())

# %% [markdown]
# ### Build edges, answer `as_of`, persist
#
# For each `key` we sort versions by `valid_from` and **close the previous one when the next
# begins**, storing a *reason* (no silent overwrites - one of the Letta failure modes). The skill's
# `as_of()` then answers "what was true at time T?". We persist each version as a `:ConfigVersion`
# node (using `SET`, not map-literals, because `valid_until` is `null` for current versions).

# %%
by_key = {}
for c in extracted:
    by_key.setdefault(c["key"], []).append(c)
edges = []
for key, items in by_key.items():
    items.sort(key=lambda c: ts(c["valid_from"]))
    for i, c in enumerate(items):
        e = bt.create_edge("checkout-api", "aws-ec2", key, value=norm(c)[1], valid_from=ts(c["valid_from"]),
                           ingested_at=ts(c["ingested_at"]),
                           link_type="causal" if key == "deployment_version" else "entity")
        if i + 1 < len(items):
            e.valid_until = ts(items[i + 1]["valid_from"])
            e.invalidation_reason = f"superseded by {items[i + 1]['value']}"
        edges.append(e)

h.run("""
MATCH (s:Service {name:'checkout-api'})
UNWIND $rows AS r
CREATE (c:ConfigVersion) SET c.key=r.key, c.value=r.value, c.valid_from=r.vf, c.valid_until=r.vu,
       c.ingested_at=r.ing, c.reason=r.reason
CREATE (s)-[:HAD_CONFIG]->(c)""",
      rows=[dict(key=e.relationship, value=str(e.value), vf=e.valid_from.isoformat(),
                 vu=e.valid_until.isoformat() if e.valid_until else None,
                 ing=e.ingested_at.isoformat(), reason=e.invalidation_reason) for e in edges])

OUTAGE = ts("2026-03-15T08:00:00+00:00")
inst = bt.as_of("checkout-api", "instance_type", OUTAGE, edges)
dep = bt.as_of("checkout-api", "deployment_version", OUTAGE, edges)
print("at the outage:", inst.value if inst else None, dep.value if dep else None)
if acc == 1.0:   # only assert exact facts when extraction was perfect; otherwise the check above already flagged it
    h.check("as_of(outage) = m5.xlarge + v3.5.0", inst.value == "m5.xlarge" and dep.value == "v3.5.0")
h.run("""MATCH (i:Incident {id:'INC-1001'})
         MERGE (f:Finding {id:'cfg-at-outage'}) SET f.instance_type=$i, f.deployment=$d, f.as_of=$t
         MERGE (f)-[:EXPLAINS]->(i)""",
      i=inst.value if inst else None, d=dep.value if dep else None, t=OUTAGE.isoformat())

# %% [markdown]
# ### Closed-book vs raw-notes vs graph memory - judged by an LLM
#
# The same question answered three ways:
#
# 1. **Closed-book** - the model has no memory (what an agent without Chapter 4 does).
# 2. **Raw notes in the prompt** - "RAG over text"; the model must infer validity windows itself.
# 3. **Graph memory** - the answer is computed by `as_of()` plus the ingestion-lag, then verbalised.
#
# We grade with **`h.judge`** against a reference. LLM judges have biases (they favour long, confident
# answers), so we pass a *reference answer* in the rubric and read the reasons, not just the scores.
# We also compute the EBS lag twice - with the skill (`ingestion_lag()`) and with **Cypher durations**
# - because two independent implementations agreeing is a cheap correctness check.

# %%
Q = ("What instance type and deployment version was checkout-api running at 2026-03-15T08:00Z, "
     "and how many days late did we learn about the EBS resize?")
REF = ("Reference: instance m5.xlarge, version v3.5.0 (went live 2026-03-14 22:30Z), "
       "EBS resize valid 2026-01-18 but ingested 2026-03-15, about 56 days late.")
ebs = next((e for e in edges if e.relationship == "ebs_volume_gb"), None)
lag_skill = ebs.ingestion_lag().total_seconds() / 86400 if ebs else float("nan")
lag_cypher = h.run("""
MATCH (:Service {name:'checkout-api'})-[:HAD_CONFIG]->(c:ConfigVersion {key:'ebs_volume_gb'})
RETURN duration.inSeconds(datetime(c.valid_from), datetime(c.ingested_at)).seconds / 86400.0 AS d""")[0]["d"]
print(f"EBS lag: skill={lag_skill:.1f} d, cypher={lag_cypher:.1f} d")
h.check("two implementations agree on the ingestion lag (~56 days)", abs(lag_skill - lag_cypher) < 0.01 and 55 < lag_cypher < 57)

answers = {
    "closed-book": h.llm(Q),
    "raw notes": h.llm(Q + "\n\nNotes (with the time each was written):\n" +
                       "\n".join(f"[{n['at']}] {n['text']}" for n in NOTES)),
    "graph memory": (f"At the outage checkout-api ran {inst.value} with deployment {dep.value}; the EBS resize "
                     f"became valid {ebs.valid_from.date()} but was ingested {ebs.ingested_at.date()}, "
                     f"a lag of {lag_skill:.0f} days."),
}
scores = {}
for name, ans in answers.items():
    j = h.judge(Q, ans, "Score 1-5 for factual agreement with this reference; penalise invented facts. " + REF)
    scores[name] = j["score"]
    print(f"{name:13s} score={j['score']}  {j['reason']}")
h.check("graph memory is judged at least as good as closed-book", scores["graph memory"] >= scores["closed-book"])
h.check("closed-book model cannot know private config (judge <= 3)", scores["closed-book"] <= 3)

# %% [markdown]
# ## 2. Graphiti-style incremental updates - the LLM as the entity extractor
#
# `graphiti-incremental-update` resolves entities in three tiers (canonical -> alias -> fuzzy) and
# touches only the affected neighbourhood. Its built-in extractor only understands an *annotated*
# syntax, `person:Sarah(s.chen)`. Production code swaps in an LLM extractor; here we do the
# pragmatic version: the **LLM rewrites free text into the annotated syntax** (strict schema), and
# the skill does resolution and locality.
#
# The hard case is **coreference**: "Sarah Chen", "S. Chen" and "Sarah" are one person. We A/B two
# prompt versions and count how many Sarah nodes each leaves behind - a measurable symptom of bad
# entity resolution (duplicate entities fragment memory).

# %%
FREE = [
    "Sarah Chen opened incident INC-1042 affecting checkout-api in us-east-1.",
    "Release v3.5.0 of checkout-api went live at 22:30 UTC.",
    "v3.5.0 started causing 504 timeouts on payment-service calls.",
    "Marcus joined INC-1042 as secondary on-call.",
    "S. Chen rolled checkout-api back to v3.4.1.",
    "Sarah closed INC-1042 once the rollback succeeded.",
]
PROMPTS["annotate@v1"] = ("Rewrite the sentence, annotating each entity as type:Name. Types: person, service, "
                          "incident, deployment, region, team.")
PROMPTS["annotate@v2"] = ("Rewrite the sentence, annotating each entity as type:Name with NO spaces inside names. "
                          "Types: person, service, incident, deployment, region, team. People: use the first name "
                          "only, capitalised, and ALWAYS attach the alias (s.chen) to Sarah - 'Sarah Chen', "
                          "'S. Chen' and 'Sarah' are the same person, written person:Sarah(s.chen). "
                          "Versions are deployments, e.g. deployment:v3.5.0. Keep all other words unchanged.")
ANNOT = obj({"annotated": {"type": "string"}})


def run_stream(version):
    g = gi.Graph()
    log_rows, touched_rows = [], []
    for i, t in enumerate(FREE, 1):
        a = sjson(t, ANNOT, system=PROMPTS[f"annotate@{version}"], name="annotated")["annotated"]
        before = g.snapshot()
        log = gi.add_episode(a, g, episode_id=f"ep-{i:03d}")
        loc = gi.verify_locality(before, g.snapshot())
        log_rows.append((a, log["touched_nodes"], loc["percent_changed"]))
        touched_rows += [{"ep": f"ep-{i:03d}", "uid": u} for u in log["touched_ids"]]
    sarahs = [n for n in g.nodes.values() if "sarah" in n.name.lower() or "chen" in n.name.lower()]
    return g, sarahs, log_rows, touched_rows


runs = {}
for v in ("v1", "v2"):
    g_, sarahs, rows_, touched_ = run_stream(v)
    runs[v] = (g_, touched_)
    print(f"prompt {v}: {len(g_.nodes)} entities, Sarah nodes = {len(sarahs)} -> {[n.name for n in sarahs]}")
n_sarah = {v: len([n for n in runs[v][0].nodes.values() if "sarah" in n.name.lower() or "chen" in n.name.lower()])
           for v in runs}
# v1's free-form annotation often is not parseable by the skill at all (0 entities): a legitimate finding
h.check("v2 prompt resolves Sarah/S. Chen/Sarah Chen to exactly one node and extracts at least as many entities as v1",
        n_sarah["v2"] == 1 and len(runs["v2"][0].nodes) >= len(runs["v1"][0].nodes))
print(h.cost_report())

# %% [markdown]
# ### Persist the resolved memory graph and link it to the infrastructure
#
# The v2 run becomes `(:MemEpisode)-[:TOUCHED]->(:MemEntity)`. Then one Cypher `MERGE` links memory
# entities to real `:Service` nodes by name - so "what do we remember about checkout-api?" is a
# graph hop, not a string search. We also prove the **locality** claim: the same episode touches the
# same number of nodes whether the graph has 5 or 500 nodes.

# %%
g, touched_rows = runs["v2"]
h.run("UNWIND $e AS e MERGE (m:MemEntity {uid:e.uid}) SET m.name=e.name, m.type=e.type",
      e=[{"uid": u, "name": n.name, "type": n.type} for u, n in g.nodes.items()])
h.run("""UNWIND $t AS t MERGE (ep:MemEpisode {id:t.ep}) WITH ep, t
         MATCH (m:MemEntity {uid:t.uid}) MERGE (ep)-[:TOUCHED]->(m)""", t=touched_rows)
h.run("MATCH (m:MemEntity), (s:Service) WHERE toLower(m.name)=toLower(s.name) MERGE (m)-[:SAME_AS]->(s)")
print(h.run("""MATCH (ep:MemEpisode)-[:TOUCHED]->(m:MemEntity)
               WITH m, count(ep) AS mentions WHERE mentions > 1
               RETURN m.name AS entity, m.type AS type, mentions ORDER BY mentions DESC, entity"""))
small, big = gi.Graph(), gi.Graph()
for k in range(5): small.add_node(f"svc-{k}", "service")
for k in range(500): big.add_node(f"svc-{k}", "service")
t_small = gi.add_episode("person:Eve opened incident:INC-9", small)["touched_nodes"]
t_big = gi.add_episode("person:Eve opened incident:INC-9", big)["touched_nodes"]
print(f"same episode touches {t_small} nodes in a 5-node graph and {t_big} in a 500-node graph")
h.check("updates are local (touch count independent of graph size)", t_small == t_big)
linked = h.run("MATCH (:MemEntity)-[:SAME_AS]->(s:Service) RETURN count(*) AS n")[0]["n"]
print(f"{linked} memory entities linked to real services")

# %% [markdown]
# ## 3. Hierarchical memory - let the LLM decide what is *durable*
#
# **Core** memory (small, always in the prompt), **recall** (recent turns), **archival** (large,
# searched on demand). When core is full, the fact with the lowest *eviction score* moves to
# archival; durable facts get a 10x protection factor. The skill needs each fact labelled `durable`
# or `short-lived` - which is a **classification task** an LLM can do. We measure it against hand
# labels, and compare with the naive baseline "everything is short-lived" (which protects nothing).
#
# Short-lived noise is added *first*, durable facts *later*, so the durable ones have to fight their
# way into a full core.

# %%
FACTS = [
    ("Running: kubectl get pods", "short-lived"), ("Tail after psql: circuit-breaker open", "short-lived"),
    ("Alarm firing right now: p99_latency_ms=2400 on checkout-api", "short-lived"),
    ("Currently waiting for terraform plan to finish", "short-lived"),
    ("Last command returned exit code 0", "short-lived"), ("User just asked for a coffee break", "short-lived"),
    ("Production AWS account is 123456789012", "durable"), ("Primary region is us-east-1", "durable"),
    ("On-call rotation: Sarah primary, Marcus secondary", "durable"),
    ("checkout-api is a critical-tier service owned by team-payments", "durable"),
    ("checkout-api calls payment-service over synchronous HTTP", "durable"),
    ("Never deploy to production on Fridays after 15:00", "durable"),
]
DUR = obj({"labels": {"type": "array", "items": obj({
    "index": {"type": "integer"}, "durability": {"type": "string", "enum": ["durable", "short-lived"]}})}})
listing = "\n".join(f"{i}: {t}" for i, (t, _) in enumerate(FACTS))
lab = sjson("Classify each memory fact. durable = still true and useful weeks from now (identities, policies, "
            "topology); short-lived = only relevant to the current moment or command.\n" + listing, DUR,
            name="durability")["labels"]
pred = {x["index"]: x["durability"] for x in lab}
llm_acc = sum(pred.get(i) == g_ for i, (_, g_) in enumerate(FACTS)) / len(FACTS)
naive_acc = sum(g_ == "short-lived" for _, g_ in FACTS) / len(FACTS)
print(f"durability accuracy: LLM={llm_acc:.0%}  naive all-short-lived={naive_acc:.0%}")
h.check("LLM classifier beats the naive baseline", llm_acc > naive_acc and llm_acc >= 0.75)

mem = hm.HierarchicalMemory(core_limit=6)
for i, (t, _) in enumerate(FACTS):
    mem.add_fact(t, pred.get(i, "short-lived"))
gold_durable = [t for t, g_ in FACTS if g_ == "durable"]
survivors = [t for t in gold_durable if t in mem.core]
print(f"core={len(mem.core)}/6 archival={len(mem.archival)}; truly-durable facts still in core: {len(survivors)}/{len(gold_durable)}")
h.check("core limit respected and most durable facts stayed hot", len(mem.core) <= 6 and len(survivors) >= 3)

# %% [markdown]
# ### Archival search that understands meaning
#
# The skill's `query()` is lexical. Ask "*who gets paged first when something breaks?*" and no word
# overlaps with "On-call rotation: Sarah primary". We persist the tiers as `:MemFact` nodes, embed
# them, and create a **Neo4j vector index** (`CREATE VECTOR INDEX ... OPTIONS {indexConfig: {...}}`,
# cosine similarity). `db.index.vector.queryNodes(index, k, vector)` returns nearest nodes. Note the
# embedding is stored with `db.create.setNodeVectorProperty` (compact, validated vector storage).

# %%
tier_rows = ([{"tier": "core", "c": f.content, "d": f.durability} for f in mem.core.values()] +
             [{"tier": "archival", "c": f.content, "d": f.durability} for f in mem.archival])
h.run("""UNWIND $rows AS r MERGE (t:MemTier {name:r.tier})
         MERGE (f:MemFact {content:r.c}) SET f.durability=r.d MERGE (t)-[:HOLDS]->(f)""", rows=tier_rows)
h.ensure_vector_index("ch4_fact_vec", "MemFact")
print("embedded MemFact nodes:", h.embed_nodes("MemFact", text_prop="content"))

QV_CACHE = {}
def qvec(q):
    if q not in QV_CACHE:
        QV_CACHE[q] = h.embed([q])[0]
    return QV_CACHE[q]

def vsearch(index, q, k=3, ret="node.content AS text"):
    return h.run(f"CALL db.index.vector.queryNodes($i, $k, $v) YIELD node, score RETURN {ret}, score",
                 i=index, k=k, v=qvec(q))

Q = "who gets paged first when something breaks?"
lex = mem.query("paged")
vec = vsearch("ch4_fact_vec", Q, 5)
print("skill (lexical) :", [x["content"] for x in lex["core"] + lex["archival"]])
print("vector search   :", [(x["text"], round(x["score"], 3)) for x in vec])
h.check("vector search finds the on-call fact that keyword search misses",
        any("On-call" in x["text"] for x in vec) and not (lex["core"] or lex["archival"]))

# %% [markdown]
# ## 4. Hindsight epistemic classifier - LLM vs heuristic, with disagreement analysis
#
# Four **networks** of belief: *World* (objective), *Experience* (what I did), *Opinion* (hedged
# belief with confidence), *Observation* (synthesis of several sources). The skill's heuristic is
# regex-based: first-person verbs, hedging words, "in summary". That misses opinions phrased with
# **no hedge** ("Rolling back is the safest fix") - a classic failure.
#
# We label ten statements by hand and score both classifiers. The LLM sees **only the text**; the
# heuristic also gets the metadata a real pipeline would attach (source=agent, timestamps), so the
# comparison is, if anything, tilted *toward* the heuristic. Disagreements are the interesting part.

# %%
now_iso = datetime.now(timezone.utc).isoformat()
AGENT = {"source": "agent", "timestamp": now_iso}
HS = [
    ("p99 latency on checkout-api reached 2400 ms", "World", {}),
    ("deploy-481 changed orders-db: db connection pool 100 -> 20", "World", {}),
    ("checkout-api depends on payment-service", "World", {}),
    ("I called CloudWatch GetMetricData for checkout-api.", "Experience", AGENT),
    ("I restarted the orders-service pods at 08:41.", "Experience", AGENT),
    ("I believe the root cause is deploy-481 shrinking the pool.", "Opinion", {"confidence": 0.7}),
    ("Rolling back is the safest fix here.", "Opinion", {}),
    ("I think we should page team-orders.", "Opinion", {}),
    ("In summary, three past incidents all traced back to pool exhaustion.", "Observation", {"derived_from": ["a", "b", "c"]}),
    ("Across INC-0877 and INC-1001 the same pool configuration is implicated.", "Observation", {"derived_from": ["INC-0877", "INC-1001"]}),
]
NETS = ["World", "Experience", "Opinion", "Observation"]
EPI = obj({"items": {"type": "array", "items": obj({
    "index": {"type": "integer"}, "network": {"type": "string", "enum": NETS},
    "confidence": {"type": "number"}})}})
PROMPTS["epistemic@v1"] = ("Classify each statement into an epistemic network. World = objective fact about the "
                           "system. Experience = an action the speaker itself took. Opinion = a belief, judgement "
                           "or recommendation (even if stated confidently). Observation = a synthesis across "
                           "several sources. confidence in [0,1] = how certain the statement claims to be.")
out = sjson("\n".join(f"{i}: {t}" for i, (t, _, _) in enumerate(HS)), EPI, system=PROMPTS["epistemic@v1"],
            name="epistemic")["items"]
llm_net = {x["index"]: x for x in out}
heur = [he.classify(t, metadata=md) for t, _, md in HS]
rows = []
for i, (t, gold, _) in enumerate(HS):
    rows.append(dict(i=i, gold=gold, llm=llm_net.get(i, {}).get("network"), heuristic=heur[i].network, text=t[:55]))
a_llm = sum(r["llm"] == r["gold"] for r in rows) / len(rows)
a_he = sum(r["heuristic"] == r["gold"] for r in rows) / len(rows)
h.show(rows)
print(f"accuracy: LLM={a_llm:.0%} heuristic={a_he:.0%}")
print("disagreements (LLM != heuristic):")
h.show([r for r in rows if r["llm"] != r["heuristic"]])
h.check("LLM classifier is at least 70% accurate on the labeled set", a_llm >= 0.7)

# %% [markdown]
# ### Persist beliefs with provenance
#
# Facts become `:EpFact` nodes; an opinion gets `:INFERRED_FROM` edges to its evidence. We use the
# **LLM's** classification (the better one here, but check the numbers above on *your* run). The
# skill's `network_audit()` then summarises the distribution - a quick "is my agent mostly guessing?"
# health check. Provenance matters: when someone asks "how do you know?", the answer is a graph
# traversal, not a re-generation.

# %%
facts = [dict(text=t, network=llm_net.get(i, {}).get("network", "World"),
              conf=float(llm_net.get(i, {}).get("confidence", 1.0))) for i, (t, _, _) in enumerate(HS)]
h.run("UNWIND $f AS x MERGE (e:EpFact {text:x.text}) SET e.network=x.network, e.confidence=x.conf", f=facts)
h.run("""MATCH (o:EpFact {text:$op}) UNWIND $ev AS t MATCH (e:EpFact {text:t}) MERGE (o)-[:INFERRED_FROM]->(e)""",
      op=HS[5][0], ev=[HS[0][0], HS[1][0], HS[3][0]])
print(h.run("""MATCH (o:EpFact {text:$op})-[:INFERRED_FROM]->(e:EpFact)
               RETURN o.network AS opinion_network, o.confidence AS confidence, e.network AS evidence_type, e.text AS evidence""",
            op=HS[5][0]))
audit = he.network_audit(heur)
print("heuristic network_audit:", audit)
n_links = h.run("MATCH (:EpFact)-[r:INFERRED_FROM]->(:EpFact) RETURN count(r) AS n")[0]["n"]
h.check("opinion is linked to three evidence facts", n_links == 3)

# %% [markdown]
# ## 5. Memory consolidation - embeddings decide what belongs together
#
# Raw episodes pile up; the agent needs **patterns**. `consolidate()` clusters similar episodes
# (single-linkage at a similarity threshold), keeps clusters of at least `min_cluster_size`, and
# emits a `ConsolidatedFact` whose `derived_from` ids keep every summary **auditable**.
#
# Two seams are now real:
# - `similarity_fn` = **cosine similarity of text-embedding-3-large vectors** (vs the default Jaccard
#   word overlap, which cannot see that "ran out of pooled connections" and "connection pool exhausted"
#   mean the same thing);
# - `summarizer_fn` = **gpt-4o** (vs concatenating the cluster).
#
# We add three *paraphrased* episodes (ep-6..8) that share almost no words with the others, plus the
# real resolved incident INC-0877 built from the graph, plus one unrelated TLS outlier (ep-5).

# %%
h.run("""
UNWIND $eps AS e MATCH (s:Service {name:e.about})
CREATE (x:Episode {id:e.id, content:e.content, type:'incident', created_at:e.at})
CREATE (x)-[:ABOUT]->(s)""", eps=[
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
    dict(id="ep-6", about="checkout-api", at="2026-03-14T10:00:00+00:00",
         content="Shoppers saw very slow page loads because the orders database had no free pooled connections left"),
    dict(id="ep-7", about="orders-service", at="2026-03-14T12:00:00+00:00",
         content="After the release the order store hit its connection limit and requests queued up, delaying purchases"),
    dict(id="ep-8", about="payment-service", at="2026-03-14T15:00:00+00:00",
         content="Payments stalled while waiting for database sessions that the order store could not hand out"),
])
h.run("""
MATCH (i:Incident {id:'INC-0877'})-[:AFFECTS]->(s:Service)
CREATE (x:Episode {id:'ep-0877', type:'incident', created_at:'2026-02-20T09:00:00+00:00',
                   content: i.title + '. root cause: ' + i.root_cause + '. fix: ' + i.fix})
CREATE (x)-[:ABOUT]->(s)""")

h.ensure_vector_index("ch4_episode_vec", "Episode")
h.run("CREATE FULLTEXT INDEX ch4_episode_ft IF NOT EXISTS FOR (e:Episode) ON EACH [e.content]")
h.run("CALL db.awaitIndexes(60)")
print("embedded episodes:", h.embed_nodes("Episode", text_prop="content"))

records = h.run("MATCH (e:Episode) RETURN e.id AS id, e.content AS content, e.type AS type, e.created_at AS at ORDER BY id")
eps = [mc.Episode(id=r["id"], content=r["content"], episode_type=r["type"], created_at=ts(r["at"]), metadata={})
       for r in records]
TEXT = {e.id: e.content for e in eps}
EMB = dict(zip([e.content for e in eps], h.embed([e.content for e in eps])))


def cos(a, b):
    va, vb = EMB[a], EMB[b]
    return sum(x * y for x, y in zip(va, vb)) / (math.sqrt(sum(x * x for x in va)) * math.sqrt(sum(y * y for y in vb)))


print("cosine(ep-1, ep-6 paraphrase) =", round(cos(TEXT["ep-1"], TEXT["ep-6"]), 3),
      "| cosine(ep-1, ep-5 TLS) =", round(cos(TEXT["ep-1"], TEXT["ep-5"]), 3))

# %% [markdown]
# ### Jaccard vs embeddings, and a real summarizer
#
# We run `consolidate()` twice. The **threshold is not portable**: 0.25 is a Jaccard number; for
# embedding cosine we use 0.45 (picked from the two cosines printed above: related ~high, unrelated
# ~low). Choosing thresholds from *measured separation* on your own data - not by copying a number -
# is part of the job. The LLM summarizer is told to use **only** the supplied episodes, which we then
# verify with a judge.

# %%
def llm_summary(texts):
    return h.llm("Summarize the recurring pattern in these incident episodes in ONE sentence (max 35 words). "
                 "Use only facts stated in them; do not invent causes, numbers or names.\n- " + "\n- ".join(texts))


jac = mc.consolidate(eps, min_cluster_size=3, threshold=0.25)
emb = mc.consolidate(eps, min_cluster_size=3, threshold=0.45, similarity_fn=cos, summarizer_fn=llm_summary)
for name, fs in (("jaccard", jac), ("embedding+LLM", emb)):
    print(f"{name:14s}: {[(f.id, sorted(f.derived_from)) for f in fs]}")
best = max(emb, key=lambda f: len(f.derived_from)) if emb else None
jbest = max(jac, key=lambda f: len(f.derived_from)) if jac else None
print("\nLLM summary:", best.summary if best else None)
paraphrases = {"ep-6", "ep-7", "ep-8"}
recovered = paraphrases & set(best.derived_from) if best else set()
print(f"paraphrase episodes recovered: jaccard={len(paraphrases & set(jbest.derived_from)) if jbest else 0}/3, "
      f"embeddings={len(recovered)}/3")
h.check("embedding consolidation produced a fact and never swallowed the TLS outlier",
        best is not None and "ep-5" not in best.derived_from and len(best.derived_from) >= 3)

# %% [markdown]
# ### Faithfulness, provenance and sleep-time inference
#
# - **Provenance:** `provenance_of()` maps a fact back to its source episodes; we check it equals the
#   `derived_from` list.
# - **Faithfulness:** an LLM judge verifies the summary against the cited episodes. A hallucinated
#   summary is *worse than none* because it looks authoritative and is now long-term memory.
# - **Sleep-time compute:** `precompute_inferences()` runs an LLM during idle time to derive likely
#   follow-up questions *before* anyone asks. The result is stored on the fact node.

# %%
INF = obj({"inferences": {"type": "array", "items": {"type": "string"}}})


def infer(fact):
    srcs = "\n".join("- " + TEXT[i] for i in fact.derived_from)
    return sjson("Given this consolidated fact and its evidence, list up to 3 short, actionable inferences an "
                 "on-call engineer would want precomputed (e.g. what to check first). Fact: " + fact.summary +
                 "\nEvidence:\n" + srcs, INF, name="inferences")["inferences"][:3]


inferences = mc.precompute_inferences(emb, infer)
prov_ok = all(len(mc.provenance_of(f, eps)) == len(f.derived_from) for f in emb)
verdict = h.judge("Summarize the recurring incident pattern.", best.summary,
                  "5 = every claim is supported by these episodes and nothing is invented: " +
                  " | ".join(TEXT[i] for i in best.derived_from))
print("faithfulness:", verdict)
print("inferences:", inferences.get(best.id))

h.run("""UNWIND $f AS f MERGE (c:ConsolidatedFact {id:f.id}) SET c.summary=f.summary, c.confirmations=f.conf, c.inferences=f.inf
         WITH c, f UNWIND f.eps AS eid MATCH (e:Episode {id:eid}) MERGE (c)-[:DERIVED_FROM]->(e)""",
      f=[{"id": f.id, "summary": f.summary, "conf": f.confirmations, "eps": list(f.derived_from),
          "inf": inferences.get(f.id, [])} for f in emb])
left = h.run("""MATCH (e:Episode) OPTIONAL MATCH (c:ConsolidatedFact)-[:DERIVED_FROM]->(e)
                WITH e, c WHERE c IS NULL RETURN e.id AS id""")
print("episodes left raw:", [r["id"] for r in left])
h.check("provenance resolves for every consolidated fact", prov_ok)
h.check("judge finds the summary faithful (>= 4)", verdict["score"] >= 4)
h.check("TLS outlier stayed raw in the graph", any(r["id"] == "ep-5" for r in left))

# %% [markdown]
# ## 6. RRF hybrid retrieval - four real channels, then measure them
#
# **Reciprocal Rank Fusion** merges ranked lists using ranks only: `score(d) = sum 1/(k + rank_c(d))`
# over channels `c`. It needs no score calibration - which matters because a cosine similarity, a
# Lucene BM25 score and a hop count are not comparable. Our channels:
#
# | channel | implementation | good at |
# |---|---|---|
# | **vector** | `db.index.vector.queryNodes` on `ch4_episode_vec` | paraphrase, meaning |
# | **keyword** | `db.index.fulltext.queryNodes` on `ch4_episode_ft` (Lucene) | exact ids and rare terms |
# | **graph** | Cypher `[:DEPENDS_ON*0..3]` from services named in the query | topology-near evidence |
# | **temporal** | `ORDER BY created_at DESC` | recency |
#
# After fusion, a **reranker** reorders the top-N. The skill's `cross_encoder_rerank` accepts a
# `score_fn(query, doc_id)`; ours is **gpt-4o relevance scores, computed in one batched call per
# query** and cached (so the skill's per-document loop costs no extra API calls).
#
# Fulltext queries are sanitised: Lucene treats `- : ( ) "` as operators, so we keep alphanumeric
# terms and drop stop-words, joined with `OR`.

# %%
DOCS = {r["id"]: r for r in h.run("MATCH (e:Episode) RETURN e.id AS id, e.content AS text, e.created_at AS at")}
STOP = {"the", "a", "an", "of", "on", "in", "to", "did", "do", "was", "is", "why", "what", "how", "and", "for", "with", "us"}


def ch_vector(q):
    return [r["id"] for r in vsearch("ch4_episode_vec", q, 8, ret="node.id AS id")]


def ch_keyword(q):
    terms = [t for t in re.findall(r"[A-Za-z0-9]+", q) if t.lower() not in STOP]
    if not terms:
        return []
    rows = h.run("CALL db.index.fulltext.queryNodes('ch4_episode_ft', $q) YIELD node, score "
                 "RETURN node.id AS id ORDER BY score DESC LIMIT 8", q=" OR ".join(terms))
    return [r["id"] for r in rows]


def ch_graph(q):
    rows = h.run("""
        MATCH (a:Service) WHERE toLower($q) CONTAINS toLower(a.name)
        MATCH (e:Episode)-[:ABOUT]->(s:Service)
        OPTIONAL MATCH p = (a)-[:DEPENDS_ON*0..3]-(s)
        WITH e, min(length(p)) AS hops WHERE hops IS NOT NULL
        RETURN e.id AS id ORDER BY hops, id""", q=q)
    return [r["id"] for r in rows]


def ch_temporal(q):
    return [d["id"] for d in sorted(DOCS.values(), key=lambda d: d["at"], reverse=True)]


CHANNELS = {"vector": ch_vector, "keyword": ch_keyword, "graph": ch_graph, "temporal": ch_temporal}
SCORE = obj({"scores": {"type": "array", "items": obj({"id": {"type": "string"}, "score": {"type": "integer"}})}})
RER = {}


def llm_rerank_scores(q):
    if not any(k[0] == q for k in RER):
        listing = "\n".join(f"{i}: {d['text']}" for i, d in DOCS.items())
        out = sjson(f"Query: {q}\nScore each memory 0-10 for how directly it helps answer the query.\n{listing}",
                    SCORE, system="You are a strict relevance judge.", name="rerank")["scores"]
        for x in out:
            RER[(q, x["id"])] = float(x["score"])


TOK = lambda i: len(DOCS[i]["text"].split())
META = {i: d["text"] for i, d in DOCS.items()}


def hybrid(q, rerank=True):
    if rerank:
        llm_rerank_scores(q)
    return [x["doc_id"] for x in rr.hybrid_retrieve(
        q, CHANNELS, k=60, top_n_for_rerank=len(DOCS), final_budget=10_000, get_tokens=TOK,
        rerank_score_fn=(lambda qq, d: RER.get((qq, d), 0.0)) if rerank else None, metadata=META)]


q0 = "why did checkout get slow"
for n_, f_ in CHANNELS.items():
    print(f"{n_:9s}", f_(q0)[:5])
print("hybrid+rerank", hybrid(q0)[:5])
fused = rr.rrf_fuse({n: f(q0) for n, f in CHANNELS.items()}, k=60)
print("pure RRF top-3:", [(d, round(s, 4)) for d, s in fused[:3]])
h.check("hybrid top-3 is dominated by pool-exhaustion episodes, not the TLS outlier", "ep-5" not in hybrid(q0)[:3])

# %% [markdown]
# ### An evaluation harness: labeled queries, P@3 and MRR
#
# "It looks better" is not evidence. We write five queries with the **set of truly relevant
# episodes**, then score each retrieval strategy with
#
# - **P@3** - fraction of the top 3 that are relevant (precision at 3);
# - **MRR** - mean reciprocal rank of the first relevant hit (`1/rank`, 0 if absent).
#
# The set is deliberately mixed: paraphrase queries (vector should win), an exact-term query
# (keyword should win), and a service-anchored query (graph can help). Five queries is tiny - treat
# differences of one query as noise, and extend the set before drawing conclusions.

# %%
POOL = {"ep-1", "ep-2", "ep-3", "ep-4", "ep-6", "ep-7", "ep-8", "ep-0877"}
LABELED = [
    ("why did checkout get slow", POOL),
    ("the database ran out of connections", POOL),
    ("certificate expiry problems", {"ep-5"}),
    ("how did we fix the pool exhaustion", {"ep-0877", "ep-3"}),
    ("anything wrong on cart-service", {"ep-5"}),
]


def p_at(ranked, rel, k=3):
    return sum(d in rel for d in ranked[:k]) / k


def mrr(ranked, rel):
    return next((1 / (i + 1) for i, d in enumerate(ranked) if d in rel), 0.0)


METHODS = {
    "vector": ch_vector, "keyword": ch_keyword, "graph": ch_graph,
    "rrf (4 channels)": lambda q: [d for d, _ in rr.rrf_fuse({n: f(q) for n, f in CHANNELS.items()}, k=60)],
    "rrf + LLM rerank": lambda q: hybrid(q, rerank=True),
}
table, ranked_cache = [], {}
for name, fn in METHODS.items():
    ps, ms = [], []
    for q, rel in LABELED:
        r = fn(q)
        ranked_cache[(name, q)] = r
        ps.append(p_at(r, rel))
        ms.append(mrr(r, rel))
    table.append(dict(method=name, P_at_3=round(sum(ps) / len(ps), 2), MRR=round(sum(ms) / len(ms), 2)))
h.show(table)
print(h.cost_report())
scores_by = {t["method"]: t for t in table}
h.check("hybrid+rerank reaches MRR >= 0.6 on the labeled set", scores_by["rrf + LLM rerank"]["MRR"] >= 0.6)
h.check("hybrid is not much worse than the best single channel",
        scores_by["rrf + LLM rerank"]["MRR"] >= max(scores_by[m]["MRR"] for m in ("vector", "keyword", "graph")) - 0.2)

# %% [markdown]
# ### Failure analysis, then persist what the agent retrieved
#
# Averages hide *why* a method fails. For each query where the best hybrid ranking is imperfect we
# print the expected set vs what came back, and each single channel's first hit. Typical causes you
# should look for: (1) the **graph channel returns nothing** when no service name appears in the query
# (it has no semantic anchor - an honest limitation, fixable by an LLM entity-linking step);
# (2) **keyword** misses paraphrases; (3) **temporal** is a prior, not evidence. We then record the
# retrieval as `(:Query)-[:RETRIEVED {rank}]->(:Episode)` - a query log, which the Letta diagnosis
# later credits as `has_query_log`.

# %%
for q, rel in LABELED:
    r = ranked_cache[("rrf + LLM rerank", q)]
    if mrr(r, rel) < 1.0 or p_at(r, rel) < 1.0:
        print(f"\nQUERY: {q}\n  relevant : {sorted(rel)}\n  hybrid   : {r[:4]}")
        for n_ in ("vector", "keyword", "graph"):
            print(f"  {n_:8s} first={ranked_cache[(n_, q)][:1]}")
top = ranked_cache[("rrf + LLM rerank", LABELED[0][0])][:5]
h.run("""MERGE (q:Query {text:$q}) WITH q UNWIND $h AS x
         MATCH (e:Episode {id:x.id}) MERGE (q)-[r:RETRIEVED]->(e) SET r.rank=x.rank""",
      q=LABELED[0][0], h=[{"id": d, "rank": i + 1} for i, d in enumerate(top)])
h.check("query log persisted", h.run("MATCH (:Query)-[r:RETRIEVED]->() RETURN count(r) AS n")[0]["n"] == len(top))

# %% [markdown]
# ## 7. Consistency models and *real* access control
#
# With several agents sharing memory, pick a **consistency model**: *strong*, *causal*,
# *read-your-writes*, *eventual*. The skill scores an `Operation` described by five 0-3 weights
# (shared authoritative state, conflict intolerance, staleness budget, collaboration, self-session
# only). Hand-picking those weights is the subjective part - so an **LLM maps a plain-English
# operation description to weights** (strict schema, integers 0-3), and we compare the resulting
# *recommendation* against a human reference. Disagreement here is a prompt to read the rationale,
# not an automatic bug in either side.

# %%
OPS = {
    "shared prod config (every agent must see writes)": dict(w=(3, 3, 0, 2, 0), ref="strong"),
    "team notebook (collaborative, tolerant of lag)": dict(w=(1, 1, 2, 3, 0), ref="causal"),
    "private scratchpad (own session only)": dict(w=(0, 0, 3, 0, 3), ref="read_your_writes"),
}
W = obj({k: {"type": "integer"} for k in ("shared_authoritative_state", "conflict_intolerance",
                                          "staleness_budget", "collaboration", "self_session_only")})
rows = []
for desc, spec in OPS.items():
    w = sjson("Rate this memory operation on five 0-3 axes (0 = not at all, 3 = essential): "
              "shared_authoritative_state (agents act on one shared source of truth), conflict_intolerance "
              "(conflicting concurrent decisions are unacceptable), staleness_budget (how much stale data is "
              "tolerable), collaboration (agents build on each other's writes), self_session_only (single agent "
              "needs continuity over its own writes).\nOperation: " + desc, W, name="op_weights")
    w = {k: max(0, min(3, int(v))) for k, v in w.items()}
    llm_rec = cm.recommend_model(cm.Operation(**w))
    hand_rec = cm.recommend_model(cm.Operation(*spec["w"]))
    rows.append(dict(operation=desc[:34], llm_weights=tuple(w.values()), hand_weights=spec["w"],
                     llm_model=llm_rec["recommended"], hand_model=hand_rec["recommended"],
                     agree=llm_rec["recommended"] == hand_rec["recommended"]))
h.show(rows)
h.check("every recommendation is a valid consistency model",
        all(r["llm_model"] in cm.MODELS and r["hand_model"] in cm.MODELS for r in rows))
h.check("LLM-weighted and hand-weighted recommendations agree on the strictest case",
        rows[0]["llm_model"] in ("strong", "causal"))

# %% [markdown]
# ### Stale-cache detection on a real write log
#
# Writes and cache reads are stored as `(:Agent)-[:WROTE|:CACHED]->(:MemKey)`. We read them back, run
# `detect_cache_divergence()` (any agent whose cache predates the latest write is stale), and write
# `:STALE_ON {gap_s}` edges. `FOREACH (_ IN CASE WHEN ... THEN [1] ELSE [] END | ...)` is the idiom for
# a *conditional* write inside a single `UNWIND`.

# %%
T0 = ts("2026-03-15T08:00:00+00:00")
h.run("""
UNWIND $w AS w
MERGE (ag:Agent {name:w.agent}) MERGE (k:MemKey {key:w.key})
FOREACH (_ IN CASE WHEN w.kind='write' THEN [1] ELSE [] END | MERGE (ag)-[:WROTE {at:w.at, value:w.value}]->(k))
FOREACH (_ IN CASE WHEN w.kind='cache' THEN [1] ELSE [] END | MERGE (ag)-[:CACHED {at:w.at}]->(k))""", w=[
    dict(agent="planner", key="orders-db.pool_size", kind="write", at=T0.isoformat(), value="20"),
    dict(agent="executor", key="orders-db.pool_size", kind="write", at=(T0 + timedelta(minutes=10)).isoformat(), value="100"),
    dict(agent="reporter", key="orders-db.pool_size", kind="cache", at=(T0 + timedelta(minutes=2)).isoformat(), value=None),
    dict(agent="planner", key="orders-db.pool_size", kind="cache", at=(T0 + timedelta(minutes=11)).isoformat(), value=None),
])
writes = [{"agent": r["agent"], "key": r["key"], "timestamp": ts(r["at"]).timestamp(), "value": r["value"]}
          for r in h.run("MATCH (ag:Agent)-[w:WROTE]->(k:MemKey) RETURN ag.name AS agent, k.key AS key, w.at AS at, w.value AS value")]
caches = [{"agent": r["agent"], "key": r["key"], "cached_at": ts(r["at"]).timestamp()}
          for r in h.run("MATCH (ag:Agent)-[c:CACHED]->(k:MemKey) RETURN ag.name AS agent, k.key AS key, c.at AS at")]
warnings = cm.detect_cache_divergence(writes, caches)
for w in warnings:
    gap = w["staleness_gap"]
    h.run("""MATCH (ag:Agent {name:$ag}), (k:MemKey {key:$k}) MERGE (ag)-[r:STALE_ON]->(k) SET r.gap_s=$gap""",
          ag=w["agent"], k=w["key"], gap=gap.total_seconds() if hasattr(gap, "total_seconds") else float(gap))
print("stale agents:", [(w["agent"], str(w["staleness_gap"])) for w in warnings])
h.check("only the reporter holds a stale cache", [w["agent"] for w in warnings] == ["reporter"])

# %% [markdown]
# ### Real Neo4j RBAC - a read-only reporting agent
#
# Earlier tiers *simulated* access control with a Python `if`. Neo4j Enterprise enforces it in the
# database. We create a **role** that may read only `:Episode` and `:ConsolidatedFact` nodes and a
# **user** holding that role, then connect *as that user* (`h.run_as`) and verify:
#
# 1. it can read episodes;
# 2. `:ConfigVersion` is **invisible** (`DENY MATCH` - counts return 0, the data does not even exist
#    from its point of view);
# 3. a write is rejected with a `Forbidden` error.
#
# Finally an LLM that only receives what the reporter can see is asked a question whose answer lives
# in the hidden data - the correct behaviour is to say it does not know. RBAC is a **hard** boundary;
# prompt instructions ("don't reveal config") are only a soft one.
#
# The cell is rerunnable (drops first, `IF NOT EXISTS` creates) and cleans up in `finally`.

# %%
PW = "Ch4-reporter-pw-1"
try:
    h.sysrun("DROP USER ch4_reporter IF EXISTS")
    h.sysrun("DROP ROLE ch4_memory_reader IF EXISTS")
    h.sysrun("CREATE ROLE ch4_memory_reader IF NOT EXISTS")
    h.sysrun("CREATE USER ch4_reporter IF NOT EXISTS SET PASSWORD $pw CHANGE NOT REQUIRED", pw=PW)
    h.sysrun("GRANT ROLE ch4_memory_reader TO ch4_reporter")
    h.sysrun("GRANT ACCESS ON DATABASE neo4j TO ch4_memory_reader")
    h.sysrun("GRANT MATCH {*} ON GRAPH neo4j NODES Episode, ConsolidatedFact TO ch4_memory_reader")
    h.sysrun("DENY MATCH {*} ON GRAPH neo4j NODES ConfigVersion TO ch4_memory_reader")

    n_ep = h.run_as("ch4_reporter", PW, "MATCH (e:Episode) RETURN count(e) AS n")[0]["n"]
    n_cfg = h.run_as("ch4_reporter", PW, "MATCH (c:ConfigVersion) RETURN count(c) AS n")[0]["n"]
    print(f"as ch4_reporter: episodes visible={n_ep}, config versions visible={n_cfg}")
    try:
        h.run_as("ch4_reporter", PW, "CREATE (:Episode {id:'forged'})")
        write_blocked, msg = False, ""
    except Exception as e:
        write_blocked, msg = True, f"{type(e).__name__}: {str(e)[:90]}"
    print("write attempt ->", msg or "ALLOWED (unexpected)")

    visible = h.run_as("ch4_reporter", PW, "MATCH (e:Episode) RETURN e.content AS c LIMIT 6")
    ans = h.llm("Using ONLY this memory, what instance type is checkout-api running? If the memory does not say, "
                "answer exactly 'unknown'.\n" + "\n".join(x["c"] for x in visible))
    j = h.judge("What instance type is checkout-api running?", ans,
                "The memory contains no instance type. 5 = says it is unknown/not available; 1 = states a specific type.")
    print("reporter agent answer:", ans[:100], "| judge:", j)
    h.check("reporter can read episodes", n_ep >= 8)
    h.check("ConfigVersion is invisible to the reporter (DENY enforced)", n_cfg == 0)
    h.check("write is rejected by the database", write_blocked)
    h.check("restricted agent does not hallucinate hidden config", j["score"] >= 4)
finally:
    h.sysrun("DROP USER ch4_reporter IF EXISTS")
    h.sysrun("DROP ROLE ch4_memory_reader IF EXISTS")
print("cleanup done; remaining ch4 users:", [r["user"] for r in h.sysrun("SHOW USERS YIELD user WHERE user STARTS WITH 'ch4_' RETURN user")])

# %% [markdown]
# ## 8. A real function-calling agent over the memory
#
# Now everything is a **tool** the model can call: `as_of_config` (bi-temporal skill),
# `search_memory` (vector + keyword + graph via RRF) and `blast_radius` (Cypher variable-length path).
# `h.ToolAgent` uses OpenAI **function calling**: the model picks tools and arguments, we execute,
# and we feed results (or errors) back until it answers.
#
# Every call is wrapped so it is **stored in Neo4j** as `(:Run)-[:STEP]->(:ToolCall)-[:NEXT]->...`.
# That makes agent behaviour queryable (tool usage, failures, latency of reasoning chains) and is the
# raw material for the self-evolution chapter. We run three tasks, one deliberately *unanswerable*
# (no instance type was recorded before Feb 15) to see whether the agent admits ignorance or invents
# a value, and grade each answer with an LLM judge.

# %%
def t_as_of(key, timestamp):
    e = bt.as_of("checkout-api", key, ts(timestamp), edges)
    return ({"value": e.value, "valid_from": e.valid_from.isoformat(), "ingested_at": e.ingested_at.isoformat()}
            if e else "no fact recorded for that key at that time")


def t_search(query):
    fused = rr.rrf_fuse({n: CHANNELS[n](query) for n in ("vector", "keyword", "graph")}, k=60)
    return [{"id": d, "text": DOCS[d]["text"]} for d, _ in fused[:3]]


def t_blast(component):
    rows = h.run("MATCH (d)-[:DEPENDS_ON*1..4]->(x {name:$n}) RETURN DISTINCT d.name AS name ORDER BY name", n=component)
    return [r["name"] for r in rows] or "nothing depends on it"


RAW = {
    "as_of_config": (t_as_of, obj({"key": {"type": "string", "enum": ["instance_type", "deployment_version", "ebs_volume_gb"]},
                                   "timestamp": {"type": "string", "description": "ISO-8601 UTC"}}),
                     "Value of a checkout-api configuration key that was valid at a given time."),
    "search_memory": (t_search, obj({"query": {"type": "string"}}), "Search past incident episodes."),
    "blast_radius": (t_blast, obj({"component": {"type": "string"}}),
                     "Services that transitively depend on a component such as orders-db."),
}
SYSTEM = ("You are an SRE memory agent. Use tools to answer; never guess values the tools do not return. "
          "If a tool reports no record, say you have no record.")


def traced(name, fn, trace):
    def wrapper(**kw):
        try:
            out = fn(**kw)
            trace.log(name, kw, json.dumps(out, default=str), True)
            return out
        except Exception as e:
            trace.log(name, kw, f"{type(e).__name__}: {e}", False)
            raise
    return wrapper


TASKS = [
    ("What instance type was checkout-api on at 2026-03-15T08:00:00Z, and which services are hit if orders-db degrades?",
     "Correct: m5.xlarge; services depending on orders-db (orders-service, payment-service, checkout-api, cart-service)."),
    ("What instance type was checkout-api running on 2026-01-01T00:00:00Z?",
     "No instance type was recorded before 2026-02-15. 5 = says there is no record; 1 = invents a value."),
    ("Have we seen database connection pool problems before? Summarize briefly.",
     "Past incidents (e.g. INC-0877, several episodes) were caused by connection pool exhaustion on orders-db."),
]
agent_runs = []
for goal, rubric in TASKS:
    trace = h.AgentTrace(goal)
    tools = {n: (traced(n, f, trace), p, d) for n, (f, p, d) in RAW.items()}
    ans = h.ToolAgent(tools, SYSTEM, max_steps=6, trace=trace).run(goal, verbose=False)
    j = h.judge(goal, ans, "Score 1-5 against: " + rubric)
    agent_runs.append(dict(goal=goal[:50], run_id=trace.run_id, tools=[s["tool"] for s in trace.timeline()],
                           judge=j["score"], reason=j["reason"][:80]))
h.show(agent_runs)
print(h.cost_report())
h.check("every run used at least one tool and left a trace in Neo4j", all(r["tools"] for r in agent_runs))
h.check("agent used the bi-temporal tool for the time-based questions",
        "as_of_config" in agent_runs[0]["tools"] and "as_of_config" in agent_runs[1]["tools"])

# %% [markdown]
# ### Query the trace graph, and read the failures
#
# The trace is plain graph data, so we can ask questions of it with Cypher: which tools are used
# most, which calls failed, and what each run's chain looked like. Then we do a **failure analysis**
# on the lowest-scoring run: was the problem the *tool* (bad data), the *model* (invented value), or
# the *judge* (harsh/lenient)? Always read at least one low-scoring transcript before trusting an
# aggregate.

# %%
print(h.run("""MATCH (r:Run)-[:STEP]->(c:ToolCall)
               RETURN r.goal AS goal, count(c) AS calls, collect(c.tool) AS chain, sum(CASE WHEN c.ok THEN 0 ELSE 1 END) AS failures
               ORDER BY goal"""))
print(h.run("MATCH (c:ToolCall) RETURN c.tool AS tool, count(*) AS calls ORDER BY calls DESC"))
worst = min(agent_runs, key=lambda r: r["judge"])
print("\nlowest-scored run:", worst["goal"], "| judge", worst["judge"], "|", worst["reason"])
print(h.run("MATCH (:Run {id:$id})-[:STEP]->(c) RETURN c.seq AS seq, c.tool AS tool, c.args AS args, c.result AS result ORDER BY seq",
            id=worst["run_id"]))
unanswerable = agent_runs[1]
h.check("on the unanswerable question the agent admits it has no record (judge >= 4)", unanswerable["judge"] >= 4)

# %% [markdown]
# ## 9. Letta failure modes - diagnose the memory we just built, twice
#
# The Letta leaderboard names **eight** failure modes (no retrieval, hierarchy collapse, in-conversation
# misses, volume degradation, silent overwrite, cross-reference failure, temporal blur, threshold
# collapse). `diagnose()` consumes a `MemorySnapshot` of *measurements*, which we now take from the
# live system: tier stats from section 3, node/edge/isolated-node counts from Cypher, bi-temporal and
# query-log flags from what we actually built, and retrieval method `hybrid` because section 6 measured it.
#
# The skill's verdict is a heuristic score: 0 = production-ready, < 10 review, < 18 ship at risk,
# otherwise do not ship. We ask an **LLM for an independent verdict** on the same snapshot and print
# where they differ - a deliberate check on both. A naive design is diagnosed for contrast.

# %%
stats = h.run("""MATCH (n) WITH count(n) AS nodes
                 MATCH ()-[r]->() WITH nodes, count(r) AS edges
                 OPTIONAL MATCH (iso) WHERE NOT (iso)--()
                 RETURN nodes, edges, count(iso) AS isolated""")[0]
snap = lf.MemorySnapshot(
    core_size=len(mem.core), core_limit=6,
    core_durable_count=sum(f.durability == hm.DURABILITY_DURABLE for f in mem.core.values()),
    core_short_lived_count=sum(f.durability == hm.DURABILITY_SHORT_LIVED for f in mem.core.values()),
    recall_size=len(mem.recall), archival_size=len(mem.archival),
    archival_durable_count=sum(f.durability == hm.DURABILITY_DURABLE for f in mem.archival),
    node_count=stats["nodes"], edge_count=stats["edges"], disconnected_components=stats["isolated"],
    avg_degree=round(2 * stats["edges"] / stats["nodes"], 2),
    uses_bi_temporal_edges=h.run("MATCH (c:ConfigVersion) RETURN count(c) AS n")[0]["n"] > 0,
    facts_without_created_at=0, facts_without_invalidation_reason_when_invalidated=0,
    retrieval_method="hybrid", retrieval_complexity_class="O(log n)",
    has_query_log=h.run("MATCH (:Query)-[:RETRIEVED]->() RETURN count(*) AS n")[0]["n"] > 0,
    extract_fn_present=True, size_stress_test_passed=True)
naive = lf.MemorySnapshot(
    core_size=50, core_limit=50, core_durable_count=2, core_short_lived_count=48, node_count=40, edge_count=10,
    disconnected_components=30, avg_degree=0.5, uses_bi_temporal_edges=False, facts_without_created_at=20,
    facts_without_invalidation_reason_when_invalidated=5, retrieval_method="linear-scan", retrieval_complexity_class="O(n)")


def band(score):
    return "production-ready" if score == 0 else "review" if score < 10 else "ship-at-risk" if score < 18 else "do-not-ship"


VERDICT = obj({"verdict": {"type": "string", "enum": ["production-ready", "review", "ship-at-risk", "do-not-ship"]},
               "top_risks": {"type": "array", "items": {"type": "string"}}})
report = []
for name, s in (("ours", snap), ("naive", naive)):
    rep = lf.diagnose(s)
    score = lf.total_score(rep)
    v = sjson("You review an AI agent memory system. Given these measurements, judge whether it is ready to "
              "ship and list up to 3 top risks.\n" + json.dumps(s.to_dict()), VERDICT, name="verdict")
    report.append(dict(design=name, heuristic_score=score, heuristic_band=band(score), llm_verdict=v["verdict"],
                       agree=band(score) == v["verdict"], risks=v["top_risks"][:2]))
    if name == "ours":
        print("failing modes:", [(m.name, m.status) for m in rep.modes if m.status != "ok"])
h.show(report)
sc = {r["design"]: r["heuristic_score"] for r in report}
h.check("graph-backed memory scores better than the naive design", sc["ours"] < sc["naive"])
h.check("LLM also ranks the naive design as no better than ours",
        ["production-ready", "review", "ship-at-risk", "do-not-ship"].index(report[1]["llm_verdict"]) >=
        ["production-ready", "review", "ship-at-risk", "do-not-ship"].index(report[0]["llm_verdict"]))

# %% [markdown]
# ## 10. Skill benchmarks and CLIs
#
# Every skill ships a CLI whose `benchmark` subcommand runs its own deterministic self-checks, and
# whose `--help` prints the description from `SKILL.md`. These are the **contract tests** for the
# pure-Python parts we plugged the LLM and Neo4j into - if one of them is red, fix it before
# blaming the model. (Subprocesses use `PYTHONIOENCODING=utf-8` so Windows consoles do not break on
# non-ASCII output.)

# %%
env = dict(os.environ, PYTHONIOENCODING="utf-8")
bench = []
for s in h.skill_catalog("memory"):
    ok, first = h.run_benchmark("memory", s["slug"])
    help_p = subprocess.run([sys.executable, str(h.SKILLS_DIR / "memory" / s["slug"] / "cli.py"), "--help"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=60)
    bench.append(dict(skill=s["slug"], benchmark_ok=ok, help_exit=help_p.returncode, first_line=first[:60]))
h.show(bench)
h.check("all eight memory skills exist", len(bench) == 8)
h.check("all benchmarks pass and --help exits 0", all(b["benchmark_ok"] and b["help_exit"] == 0 for b in bench))

# %% [markdown]
# ## 11. Vibe coding - directing an AI assistant to build and verify memory features
#
# "Vibe coding" works when *you* own the specification and the verification. Prompts you could give
# an AI coding assistant for this chapter:
#
# 1. *"Using `skills/memory/bi-temporal-edge/lib.py`, write a function `as_of_report(service, edges, at)`
#    that returns every config key's value at time `at`, plus its ingestion lag. Use only the stdlib,
#    timezone-aware datetimes, and add three unit tests including one for a key with no value at `at`."*
# 2. *"Add an LLM-backed `similarity_fn` for `memory-consolidation` with an embedding cache so that
#    each text is embedded once. Show how to choose the threshold from labeled related/unrelated pairs."*
# 3. *"Write a read-only Neo4j 5.26 Cypher query returning, per ConfigVersion key of checkout-api, the
#    ingestion lag in days, sorted descending. The schema is in the comment below. Do not use APOC."*
#
# **Verification checklist - never merge AI-written memory code until all are true:**
#
# - [ ] Cypher is **read-only** where it should be (no `CREATE`/`SET`/`DELETE`/`MERGE`/`CALL {...} IN TRANSACTIONS`) and `EXPLAIN` accepts it.
# - [ ] It returns the **same rows as a trusted reference** (an independent implementation or a known number).
# - [ ] Every datetime is timezone-aware; no `datetime.now()` without a tz.
# - [ ] No silent overwrite: invalidated facts keep a reason and a valid-until.
# - [ ] LLM calls use a **schema**, temperature 0, bounded input, retries, and token accounting.
# - [ ] Tests cover the **negative** case (no record, unauthorised, empty result), not just the happy path.
# - [ ] No secret in the diff (`.env` stays ignored).
#
# The cell below runs this loop for prompt 3: gpt-4o writes the Cypher, we **guard** it (read-only
# keywords), run `EXPLAIN`, feed any error back to the model (up to three attempts), then compare its
# rows with the reference result computed by the skill in section 1.

# %%
SCHEMA_HINT = ("Graph schema: (:Service {name})-[:HAD_CONFIG]->(:ConfigVersion {key, value, valid_from, "
               "valid_until, ingested_at}). Timestamps are ISO-8601 strings with offset; use datetime(x). "
               "ConfigVersion.key is one of instance_type, deployment_version, ebs_volume_gb.")
TASK = ("Return, per ConfigVersion key of Service 'checkout-api', the maximum ingestion lag in days "
        "(ingested_at minus valid_from) as columns key and lag_days, sorted by lag_days descending.")
WRITE_WORDS = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD|FOREACH|CALL)\b", re.I)
CY = obj({"cypher": {"type": "string"}})
feedback, good, attempts = "", None, 0
for attempts in range(1, 4):
    cy = sjson(f"{SCHEMA_HINT}\nTask: {TASK}\nNeo4j 5.26 Cypher, read-only, no APOC.{feedback}", CY,
               system="You write Neo4j Cypher.", name="cypher")["cypher"].strip().rstrip(";")
    if WRITE_WORDS.search(cy):
        feedback = f"\nYour previous query was rejected because it contains a write/procedure keyword: {cy}"
        continue
    try:
        h.run("EXPLAIN " + cy)
        got = h.run(cy)
        good = (cy, got)
        break
    except Exception as e:
        feedback = f"\nYour previous query failed: {cy}\nError: {str(e)[:200]}"
print(f"attempts={attempts}")
print(good[0] if good else "no valid query produced")
ref = sorted(((e.relationship, round(e.ingestion_lag().total_seconds() / 86400, 1)) for e in edges),
             key=lambda x: -x[1])
ref_max = {}
for k, v in ref:
    ref_max[k] = max(v, ref_max.get(k, 0))
if good:
    got_map = {r.get("key"): round(float(r.get("lag_days", 0)), 1) for r in good[1]}
    print("generated:", got_map, "\nreference:", ref_max, "\nmatch:", got_map == ref_max)
h.check("generated Cypher passed the read-only guard and EXPLAIN within three attempts", good is not None)

# %% [markdown]
# ## 12. Experiments to try
#
# 1. **Prompt ablation.** Add `extract_config@v3` that also asks for a `confidence`; does recall or the
#    false-positive count change? Keep the registry, keep the gold set, change one thing.
# 2. **Threshold sweep.** Run `consolidate()` with embedding thresholds 0.35 / 0.45 / 0.55 and plot the
#    cluster membership. Where does the TLS outlier first join? Where do paraphrases drop out?
# 3. **Add an LLM entity linker to the graph channel** so "checkout got slow" anchors on `checkout-api`;
#    re-run the harness and see whether graph MRR moves.
# 4. **Grow the labeled set** to 20 queries and bootstrap a confidence interval for MRR.
# 5. **Judge bias.** Grade the same answers with two rubrics (strict vs lenient) and with the answers
#    shuffled; how much does the score move?
# 6. **RBAC.** Add a second role that may read `:ConfigVersion` but not `:Run`, and verify with `run_as`.
# 7. **Swap the model** with `OPENAI_LLM_MODEL=gpt-4o-mini` and compare cost and accuracy on sections 1, 4 and 6.

# %%
print(h.cost_report())
print("Neo4j node counts by label:")
h.show(h.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS n ORDER BY n DESC"))

# %% [markdown]
# ## Evaluation and LLM-as-a-Judge — how you know it works
#
# Sections 1-11 each ended in a check, but a memory layer fails *quietly*: a stale fact, a leaked field or a
# fabricated summary all look like fluent answers. This section shows how to **measure** the memory-backed answers
# with a layered evaluation, then how to **audit the judge itself** before trusting it.
#
# | Layer | What it checks | Cost | Weakness |
# |---|---|---|---|
# | 1. Deterministic | Cypher/Python assertions on the graph (versions exist, lag is ~56 d, outlier stayed raw, provenance edges) | free | only what you thought to assert |
# | 2. Reference metric | token-overlap F1 against a reference answer | free | sees words, not meaning |
# | 3. LLM judge | rubric with 1-vs-5 anchors, graded against a reference | 1 call per item | biased, needs calibration |
# | 4. Human | labelled gold set, spot-checks | expensive | does not scale; it is the ground truth |
#
# **Known judge biases**: *position* (prefers the first or second answer), *verbosity* (long and polished beats short
# and right), *leniency* (everything is a 4), *self-preference* (a model likes text that sounds like itself) and
# *instability* (same input, different score). We probe each one below instead of assuming it away.
#
# ### E.1 A gold set and human-labelled candidate answers
#
# Eight questions about *this* notebook's memory, each with a reference answer, stored as `(:Ch4EvalGold)` nodes.
# Thirteen candidate answers carry a **human** score 1-5 that we wrote *before* looking at any judge output:
# correct, subtly wrong (stale or off-by-weeks), fluffy, unsafe and hallucinated ones.

# %%
EVAL_GOLD = [
    {"id": "G1", "question": "What instance type and deployment version was checkout-api running at 2026-03-15T08:00Z?",
     "reference": "m5.xlarge (since 2026-03-10 17:45 UTC) and v3.5.0 (live since 2026-03-14 22:30 UTC)."},
    {"id": "G2", "question": "How many days late did we learn about the checkout-api EBS volume resize?",
     "reference": "About 56 days: the resize was valid from 2026-01-18 but was only ingested on 2026-03-15."},
    {"id": "G3", "question": "What instance type was checkout-api running before 2026-02-15?",
     "reference": "Unknown. The earliest recorded instance_type is t3.large from 2026-02-15 14:23 UTC; memory holds "
                  "nothing earlier, so the agent must say it does not know rather than guess."},
    {"id": "G4", "question": "What recurring pattern do the checkout latency incident episodes share?",
     "reference": "orders-db connection pool exhaustion, repeated across episodes ep-1 to ep-4 and the paraphrased "
                  "episodes ep-6 to ep-8, consolidated into one fact."},
    {"id": "G5", "question": "Is the TLS certificate expiry on cart-service part of the pool-exhaustion pattern?",
     "reference": "No. ep-5 is an unrelated outlier; it is not in the consolidated fact and stays a raw episode."},
    {"id": "G6", "question": "Can the read-only reporting agent see checkout-api's instance type?",
     "reference": "No. ConfigVersion is denied to the reporter role by Neo4j RBAC, so it sees nothing and must answer "
                  "unknown. This is a database boundary, not a prompt instruction."},
    {"id": "G7", "question": "Is 'I believe the root cause is deploy-481 shrinking the pool' a fact?",
     "reference": "No. It is an Opinion (confidence about 0.7), inferred from evidence facts; store it with "
                  "INFERRED_FROM links rather than as a World fact."},
    {"id": "G8", "question": "How can we audit why a consolidated memory fact exists?",
     "reference": "ConsolidatedFact nodes keep DERIVED_FROM edges to the source episodes, so every summary can be "
                  "traced back to the evidence it was built from."},
]
EVAL_CANDS = [
    ("C1", "G1", "At that time checkout-api ran m5.xlarge on version v3.5.0 (m5.xlarge since 10 March, v3.5.0 since 14 March 22:30 UTC).",
     5, "correct, specific, bi-temporally right"),
    ("C2", "G1", "It was running a fairly recent version on a standard instance size.", 2, "vague, no values"),
    ("C3", "G1", "checkout-api ran t3.large with v3.4.1.", 1, "stale: reads the latest *ingested* facts as of the old date"),
    ("C4", "G2", "About 56 days: valid from 18 January, but we only ingested it on 15 March.", 5, "correct"),
    ("C5", "G2", "About 25 days late; the resize was recorded in February.", 2, "subtly wrong number and date"),
    ("C6", "G3", "I do not know: memory starts at t3.large on 15 February 2026 and has nothing earlier.", 5, "calibrated"),
    ("C7", "G3", "It was a t2.medium instance.", 1, "hallucinated value"),
    ("C8", "G4", "Incidents show recurring database resource contention, which suggests the team should keep improving "
                 "capacity planning and monitoring.", 3, "fluffy: right direction, no specifics"),
    ("C9", "G5", "No, TLS problems are always minor so they are simply ignored and kept apart for that reason.", 3,
     "right decision, wrong mechanism"),
    ("C10", "G7", "No. It is an Opinion with about 0.7 confidence, linked to its evidence by INFERRED_FROM.", 5, "correct"),
    ("C11", "G6", "Yes, it can read the instance type if you ask the agent politely.", 1, "unsafe: confuses prompts with RBAC"),
    ("C12", "G8", "Follow the DERIVED_FROM edges from the ConsolidatedFact to the source episodes.", 5, "correct, grounded"),
    ("C13", "G4", "Orders database connection pool exhaustion keeps recurring across several incident episodes.", 4,
     "correct but no episode ids"),
]
EVAL_BY_ID = {g["id"]: g for g in EVAL_GOLD}
EVAL_CAND = {c[0]: {"gold": c[1], "answer": c[2], "human": c[3], "why": c[4]} for c in EVAL_CANDS}
EVAL_HUMAN = [c[3] for c in EVAL_CANDS]

h.run("MATCH (g:Ch4EvalGold) DETACH DELETE g")
h.run("UNWIND $rows AS r CREATE (:Ch4EvalGold {id:r.id, question:r.question, reference:r.reference})", rows=EVAL_GOLD)
stored = h.run("MATCH (g:Ch4EvalGold) RETURN g.id AS id, size(g.reference) AS ref_chars ORDER BY id")
h.show(stored)
h.check("gold set has >= 8 cases stored in the graph", len(stored) >= 8)
h.check("at least 8 human-labelled candidates covering bad, subtle and fluffy answers",
        len(EVAL_CANDS) >= 8 and min(EVAL_HUMAN) == 1 and max(EVAL_HUMAN) == 5 and 2 in EVAL_HUMAN and 3 in EVAL_HUMAN)

# %% [markdown]
# ### E.2 Layer 1 and 2: deterministic checks, then a reference-based metric
#
# **Layer 1** asserts the facts the references cite really are in the graph, and flags candidate answers that name an
# instance type, version or service the memory has never seen (a cheap **hallucination detector**). No LLM, no cost.
#
# **Layer 2** scores each candidate against its reference with token-overlap F1 rescaled to 1-5. It is free and objective
# but only sees words: watch where it disagrees with the humans.

# %%
EVAL_FACTS = {
    "cfg_instance": [r["v"] for r in h.run("MATCH (:Service {name:'checkout-api'})-[:HAD_CONFIG]->(c:ConfigVersion {key:'instance_type'}) "
                                           "RETURN c.value AS v ORDER BY c.valid_from")],
    "cfg_version": [r["v"] for r in h.run("MATCH (:Service {name:'checkout-api'})-[:HAD_CONFIG]->(c:ConfigVersion {key:'deployment_version'}) "
                                          "RETURN c.value AS v ORDER BY c.valid_from")],
    "tls_raw": h.run("MATCH (e:Episode {id:'ep-5'}) WHERE NOT (:ConsolidatedFact)-[:DERIVED_FROM]->(e) RETURN count(e) AS c")[0]["c"],
    "derived_edges": h.run("MATCH (:ConsolidatedFact)-[r:DERIVED_FROM]->(:Episode) RETURN count(r) AS c")[0]["c"],
    "opinion": h.run("MATCH (e:EpFact {network:'Opinion'}) WHERE (e)-[:INFERRED_FROM]->() RETURN count(e) AS c")[0]["c"],
}
print(EVAL_FACTS, "lag_cypher=", round(lag_cypher, 1))
h.check("G1 reference holds: m5.xlarge and v3.5.0 are the latest recorded values",
        EVAL_FACTS["cfg_instance"][-1:] == ["m5.xlarge"] and EVAL_FACTS["cfg_version"][-1:] == ["v3.5.0"])
h.check("G2 reference holds: lag is about 56 days", 55 < lag_cypher < 57)
h.check("G3 reference holds: the earliest instance_type is t3.large", EVAL_FACTS["cfg_instance"][:1] == ["t3.large"])
h.check("G5 reference holds: the TLS outlier ep-5 stayed a raw episode", EVAL_FACTS["tls_raw"] == 1)
h.check("G8 reference holds: DERIVED_FROM provenance edges exist", EVAL_FACTS["derived_edges"] >= 3)
h.check("G7 reference holds: an Opinion carries INFERRED_FROM evidence", EVAL_FACTS["opinion"] >= 1)

KNOWN_INST = {"t3.large", "m5.xlarge"}
KNOWN_VER = {"v3.4.1", "v3.5.0"}
INST_RE = re.compile(r"\b[a-z]\d[a-z]?\.(?:nano|micro|small|medium|large|\d*xlarge)\b")
VER_RE = re.compile(r"\bv\d+\.\d+\.\d+\b")


def invented(text: str) -> list:
    return sorted((set(INST_RE.findall(text)) - KNOWN_INST) | (set(VER_RE.findall(text)) - KNOWN_VER))


def eval_tokens(t: str) -> set:
    return set(re.findall(r"[a-z0-9_\-\.]+", t.lower()))


def f1_to_5(answer: str, reference: str) -> int:
    a, r = eval_tokens(answer), eval_tokens(reference)
    overlap = len(a & r)
    if not overlap:
        return 1
    p, rc = overlap / len(a), overlap / len(r)
    return int(round(1 + 4 * (2 * p * rc / (p + rc))))


for cid, c in EVAL_CAND.items():
    c["invented"] = invented(c["answer"])
    c["f1_score"] = f1_to_5(c["answer"], EVAL_BY_ID[c["gold"]]["reference"])
h.show([{"id": cid, "human": c["human"], "overlap_1to5": c["f1_score"], "invented": c["invented"], "why": c["why"]}
        for cid, c in EVAL_CAND.items()])
print("layer-2 overlap vs human:", h.agreement(EVAL_HUMAN, [EVAL_CAND[c[0]]["f1_score"] for c in EVAL_CANDS]))
h.check("layer 1 flags the hallucinated instance type and not the correct answers",
        EVAL_CAND["C7"]["invented"] == ["t2.medium"] and not EVAL_CAND["C1"]["invented"] and not EVAL_CAND["C6"]["invented"])

# %% [markdown]
# ### E.3 Layer 3: a rubric judge with explicit 1-vs-5 anchors
#
# Vague rubrics produce lenient scores. Each criterion states what a **1** and a **5** look like, and the judge sees the
# **reference**, so it grades against ground truth. `h.judge_rubric` returns per-criterion scores, a mean and reasons.
# We judge all 13 candidates once (13 judge calls).

# %%
EVAL_CRIT = {
    "correctness": "5 = every fact agrees with the reference; 1 = contradicts the reference, or states a value the reference does not support.",
    "grounding": "5 = cites the specific memory facts (values, dates, episode ids, node or edge names) that the reference cites; "
                 "1 = cites none or invents some.",
    "completeness": "5 = covers every point in the reference; 1 = covers none of them (generic or off-topic).",
    "calibration": "5 = states uncertainty or refusal exactly where the reference does and never guesses; 1 = confidently asserts "
                   "something the memory cannot support, or refuses when the memory can answer.",
}


def to_5(mean: float) -> int:
    return min(5, max(1, int(mean + 0.5)))


for cid, c in EVAL_CAND.items():
    g = EVAL_BY_ID[c["gold"]]
    c["rubric"] = h.judge_rubric(g["question"], c["answer"], EVAL_CRIT, reference=g["reference"])
    c["judge"] = to_5(c["rubric"]["mean"])
h.show([{"id": cid, "human": c["human"], "judge": c["judge"], **c["rubric"]["scores"]} for cid, c in EVAL_CAND.items()])
worst_c = min(EVAL_CAND.items(), key=lambda kv: kv[1]["rubric"]["mean"])
print(f"lowest-scored candidate {worst_c[0]}: {str(worst_c[1]['rubric']['reasons'])[:160]}")
h.check("judge returned a 1-5 score for every candidate", all(1 <= c["judge"] <= 5 for c in EVAL_CAND.values()))
print(h.cost_report())

# %% [markdown]
# ### E.4 Calibration: does the judge agree with the humans?
#
# The step most teams skip. **exact / within1** say how often the judge matches us; **Cohen's kappa** corrects exact
# agreement for chance (0.4-0.6 moderate, 0.6-0.8 substantial); **Spearman** asks whether both *rank* the answers the same
# way. With 13 items these are noisy, so the check is **tolerant**: a weak judge is a finding to act on, not a crash.

# %%
EVAL_JUDGE = [EVAL_CAND[c[0]]["judge"] for c in EVAL_CANDS]
cal = h.agreement(EVAL_HUMAN, EVAL_JUDGE)
print("judge vs human:", cal)
h.show([{"id": cid, "human": c["human"], "judge": c["judge"], "delta": c["judge"] - c["human"]} for cid, c in EVAL_CAND.items()])
lenient = [cid for cid, c in EVAL_CAND.items() if c["judge"] - c["human"] >= 2]
if cal["kappa"] < 0.4 or cal["spearman"] < 0.7:
    print("WARNING: the judge is weakly calibrated against our labels. Do not use its scores for decisions yet. Options: "
          "tighten the anchors on criteria with the biggest deltas, add few-shot 1/3/5 examples, try a stronger or "
          "different-family judge, or re-check the human labels.")
else:
    print("OK: kappa and Spearman clear the bar (kappa >= 0.4, spearman >= 0.7). Re-check when the model or rubric changes.")
if lenient:
    print("Leniency: the judge scored 2+ points above the human on", lenient)
h.check("calibration was computed over all candidates", cal["n"] == len(EVAL_CANDS))

# %% [markdown]
# ### E.5 Bias probe 1 — position bias
#
# `h.judge_pairwise` asks twice with the order swapped; if the verdicts disagree it reports `consistent=False`. Two pairs
# (4 calls): a real pair (C1 correct vs C2 vague: the better answer should win in **both** orders) and an **identical**
# pair (the only honest verdict is a tie, so any winner is pure position bias).

# %%
PAIR_G = EVAL_BY_ID["G1"]
PAIR_CRIT = "Prefer the answer that is more correct, better grounded in the reference facts, and better calibrated."
real_pair = h.judge_pairwise(PAIR_G["question"], EVAL_CAND["C1"]["answer"], EVAL_CAND["C2"]["answer"], PAIR_CRIT,
                             reference=PAIR_G["reference"])
same_pair = h.judge_pairwise(PAIR_G["question"], EVAL_CAND["C1"]["answer"], EVAL_CAND["C1"]["answer"], PAIR_CRIT,
                             reference=PAIR_G["reference"])
print("real pair (C1 vs C2):", real_pair)
print("identical pair      :", same_pair)
print("PASS" if real_pair["winner"] == "a" and real_pair["consistent"] else
      "WARNING: the judge did not clearly and consistently prefer the better answer in both orders.")
print("PASS" if same_pair["winner"] == "tie" and same_pair["consistent"] else
      "WARNING: position bias on identical answers; judge both orders and average, or use absolute rubrics.")
h.check("pairwise judge returned verdicts for both probes",
        real_pair["winner"] in ("a", "b", "tie") and same_pair["winner"] in ("a", "b", "tie"))

# %% [markdown]
# ### E.6 Bias probe 2 — verbosity bias
#
# Wrap the weak answer C2 in confident, contentless filler (no new facts). A well-behaved judge scores it about the same;
# a verbosity-biased judge scores it higher. A rise of more than 0.75 mean points is flagged (1 extra judge call).

# %%
FLUFF = ("Great question, and an important one for any modern engineering organisation. In today's fast-moving, "
         "cloud-native landscape, holistic observability and a robust, forward-looking, best-practice-driven mindset "
         "are absolutely essential. With that context in mind: ")
FLUFF_TAIL = (" Ultimately, by leveraging synergies across teams and embracing a culture of continuous improvement, "
              "organisations can unlock significant long-term value and operational excellence.")
padded = FLUFF + EVAL_CAND["C2"]["answer"] + FLUFF_TAIL
padded_res = h.judge_rubric(PAIR_G["question"], padded, EVAL_CRIT, reference=PAIR_G["reference"])
rise = padded_res["mean"] - EVAL_CAND["C2"]["rubric"]["mean"]
print(f"C2 plain mean={EVAL_CAND['C2']['rubric']['mean']:.2f}  padded mean={padded_res['mean']:.2f}  rise={rise:+.2f}  "
      f"(answer grew {len(padded) / len(EVAL_CAND['C2']['answer']):.1f}x)")
print("PASS: padding did not buy points." if rise <= 0.75 else
      "WARNING: verbosity bias. Add 'do not reward length' with a 1-anchor for filler, or a conciseness criterion.")
h.check("verbosity probe produced a score", 1 <= padded_res["mean"] <= 5)

# %% [markdown]
# ### E.7 Bias probe 3 — self-consistency, and a sabotage control
#
# `h.judge_repeat` scores the *same* answer `n=3` times at temperature 0.7; a `spread` of 2+ means the judge is unreliable
# on that item (6 calls on C2 and the borderline C9). **Sabotage control:** take the best answer and corrupt its *facts*
# while keeping its style (here: the classic bi-temporal mistake of answering with stale, as-ingested data). A fact-checking
# judge must score it clearly lower. This one is a hard `h.check`: a judge that cannot tell a true answer from a fluent
# lie makes all its other scores meaningless (1 call).

# %%
ONE_LINE = ("Score 1-5 against the reference: 5 = correct, specific, grounded in the memory facts and calibrated; "
            "3 = right direction but vague; 1 = wrong, invented, or empty.")
rep = {}
for cid in ("C2", "C9"):
    g = EVAL_BY_ID[EVAL_CAND[cid]["gold"]]
    rep[cid] = h.judge_repeat(g["question"], EVAL_CAND[cid]["answer"], f"{ONE_LINE} Reference: {g['reference']}", n=3)
    print(f"{cid}: scores={rep[cid]['scores']} mean={rep[cid]['mean']:.2f} spread={rep[cid]['spread']}")
    if rep[cid]["spread"] >= 2:
        print(f"  WARNING: unstable judge on {cid}; average several runs or tighten the rubric for this kind of item.")

SABOTAGED = ("At that time checkout-api ran t3.large with v3.4.1; the instance was only moved to m5.xlarge and v3.5.0 "
             "later, after the audit on 15 March.")
sab = h.judge_rubric(PAIR_G["question"], SABOTAGED, EVAL_CRIT, reference=PAIR_G["reference"])
print(f"good C1 mean={EVAL_CAND['C1']['rubric']['mean']:.2f}  sabotaged mean={sab['mean']:.2f}")
h.check("sabotaged (fluent but stale) answer scores at least 1 point below the good one",
        sab["mean"] <= EVAL_CAND["C1"]["rubric"]["mean"] - 1)
print(h.cost_report())

# %% [markdown]
# ### E.8 Judge vs heuristic: where do they disagree, and why?
#
# Section 4 compared the skill's rule-based `he.classify` with an LLM. Here the **rules** and the **LLM judge** face the
# same 10 statements: the heuristic score is 5 if the rules named the gold network and 1 otherwise; the judge grades
# *the rules' verdict* against the gold network's definition (10 calls, partial credit for a neighbouring network).
# Rows that differ by 2+ points are a heuristic blind spot or a judge quirk; we read each one.

# %%
NET_DEF = {
    "World": "an objective fact about the system",
    "Experience": "an action the speaker itself took",
    "Opinion": "a belief, judgement or recommendation, even if stated confidently",
    "Observation": "a synthesis across several sources",
}
cmp_rows = []
for i, (t, gold, _) in enumerate(HS):
    pred = heur[i].network
    j = h.judge(f"Which epistemic network does this statement belong to? {t}", f"Rules verdict: {pred}",
                f"The correct network is {gold} ({NET_DEF[gold]}). Score 5 if the verdict names {gold}, 3 if it names a "
                "closely related network, 1 if it names an unrelated one.")
    hs = 5 if pred == gold else 1
    cmp_rows.append({"i": i, "gold": gold, "rules": pred, "heuristic": hs, "judge": int(j["score"]),
                     "judge_reason": str(j["reason"])[:80]})
h.show([{k: v for k, v in r.items() if k != "judge_reason"} for r in cmp_rows])
print("heuristic vs judge:", h.agreement([r["heuristic"] for r in cmp_rows], [r["judge"] for r in cmp_rows]))
diffs = [r for r in cmp_rows if abs(r["heuristic"] - r["judge"]) >= 2]
print(f"{len(diffs)} disagreement(s) of >= 2 points")
for r in diffs:
    meaning = ("heuristic miss, judge gives partial credit: a keyword blind spot or judge leniency toward a near-miss"
               if r["heuristic"] < r["judge"] else
               "judge stricter than exact match: the rules hit the gold label but the judge disputes it; re-read the item")
    print(f"  stmt {r['i']} rules={r['rules']} gold={r['gold']} heuristic={r['heuristic']} judge={r['judge']}: "
          f"{meaning}. Judge said: {r['judge_reason']}")
h.check("compared the heuristic and the judge on all 10 statements", len(cmp_rows) == len(HS) == 10)

# %% [markdown]
# ### E.9 Regression harness: change the retrieval, prove it did not get worse
#
# Evaluation earns its keep when something *changes*. `h.run_eval` scores a system on cases and stores
# `(:EvalRun)-[:HAS_RESULT]->(:EvalResult)` in Neo4j under a name and **version**; `h.compare_runs` compares versions with
# one Cypher query, and a **CI-style gate** turns it into PASS/FAIL. Here `v1` is keyword-only retrieval and `v2` is the
# four-channel RRF fusion from section 6, scored by reciprocal rank on the five labeled queries. The scoring is
# deterministic (no judge in the loop), so a flaky judge cannot hide a regression. The gate prints and never raises.

# %%
EVAL_NAME = "ch4_retrieval"
h.run("MATCH (r:EvalRun) WHERE r.name STARTS WITH 'ch4_' OPTIONAL MATCH (r)-[:HAS_RESULT]->(x) DETACH DELETE x, r")
RET_CASES = [{"id": f"q{i + 1}", "input": q, "gold": sorted(rel)} for i, (q, rel) in enumerate(LABELED)]
RET_SYSTEMS = {"v1": "keyword", "v2": "rrf (4 channels)"}


def score_retrieval(case, out):
    s = mrr(out, set(case["gold"]))
    return {"score": s, "passed": s >= 0.5, "notes": f"gold={case['gold']} top3={out[:3]}"}


summaries = {}
for ver, method in RET_SYSTEMS.items():
    summaries[ver] = h.run_eval(EVAL_NAME, RET_CASES, lambda c, m=method: ranked_cache[(m, c["input"])], score_retrieval, version=ver)
    print(f"{ver} ({method}): pass_rate={summaries[ver]['pass_rate']:.0%} failures={summaries[ver]['failures']}")
runs = h.compare_runs(EVAL_NAME)
h.show(runs)
h.check("both versions were stored and are comparable in Neo4j", {r["version"] for r in runs} == {"v1", "v2"})

TOLERANCE = 0.10          # candidate may be at most this much worse than baseline (0-1 scale)
by_ver = {r["version"]: r for r in runs}
base, cand = by_ver["v1"]["mean_score"], by_ver["v2"]["mean_score"]
verdict_gate = "PASS" if cand >= base - TOLERANCE else "FAIL"
print(f"CI GATE: baseline v1={base:.2f}  candidate v2={cand:.2f}  tolerance={TOLERANCE:.2f}  ->  {verdict_gate}")
if verdict_gate == "FAIL":
    print("  Regression: inspect (:EvalResult) rows with passed=false for the candidate version before shipping.")
h.show(h.run("MATCH (:EvalRun {name:$n, version:'v2'})-[:HAS_RESULT]->(x:EvalResult) WHERE NOT x.passed "
             "RETURN x.case AS failed_case, x.notes AS notes ORDER BY x.case", n=EVAL_NAME) or [{"failed_case": "none"}])

# %% [markdown]
# ### E.10 Judge report card, and a prompt to grow the eval set
#
# **What I would trust this judge for** (after reading the calibration, bias and sabotage results above):
#
# - **Ranking and regression-spotting** between versions, using the *same* judge, rubric and reference each time.
# - **Catching clearly bad and fluent-but-false answers**, because reference-anchored criteria plus the sabotage control held.
# - **Triage**: deciding which answers a human should read first.
#
# **What I would not trust it for:**
#
# - **Absolute quality claims** ("our memory is 4.3/5") or comparing numbers across different judges or rubrics.
# - **Access control.** Whether an agent *can* read config is a Neo4j RBAC fact (section 7), never a judge opinion.
# - **Time-travel facts without a reference**: bi-temporal correctness belongs in layer-1 Cypher assertions.
# - **Anything after the judge model, rubric or task changes**, until re-calibrated on fresh human labels.
# - **Self-preference**: the judge and the answering model are both OpenAI models here, so expect flattery we did not measure.
#
# **Clean-up for this section:** the next cell removes the `Ch4EvalGold` and `ch4_` `EvalRun` data it created.
#
# **Vibe-coding prompt: extend the eval set**
# ```text
# Context: section "Evaluation and LLM-as-a-Judge" in notebooks_advanced/_src/ch4-memory.py, and real_helpers.py
# (judge_rubric, judge_pairwise, judge_repeat, agreement, run_eval, compare_runs).
# Task: add 6 gold cases and 8 human-labelled candidates about memory *staleness* (cache divergence, valid-time vs
# ingestion-time), including two subtly wrong answers (valid time confused with ingestion time) and one fluffy answer.
# Re-run the calibration.
# Constraints: reuse the EVAL_CRIT dict; <= 20 extra judge calls; every Cypher query parameterised; Neo4j labels
# namespaced with ch4_/Ch4; do not weaken any existing h.check.
# Acceptance: print agreement (kappa, spearman) before and after; if kappa drops by more than 0.1, list the new
# items with the largest judge-human delta and propose a rubric change. Do not change human labels to fit the judge.
# ```

# %%
h.run("MATCH (g:Ch4EvalGold) DETACH DELETE g")
h.run("MATCH (r:EvalRun) WHERE r.name STARTS WITH 'ch4_' OPTIONAL MATCH (r)-[:HAS_RESULT]->(x) DETACH DELETE x, r")
eval_left = h.run("MATCH (n) WHERE n:Ch4EvalGold OR n:EvalRun OR n:EvalResult RETURN count(n) AS c")[0]["c"]
h.check("evaluation section left no Ch4EvalGold / EvalRun / EvalResult nodes behind", eval_left == 0)
print(h.cost_report())

# %% [markdown]
# ## Cost, limits and what to swap next
#
# **Cost.** `h.cost_report()` above is a *token* count, not dollars (prices change). Roughly, this
# notebook makes a few dozen gpt-4o calls with small inputs and a few hundred embeddings; rerunning
# it is cheap but not free. Rerank scoring is the largest repeat cost, which is why scores are
# computed once per query and cached.
#
# **What is still approximate - be honest about it:**
#
# - **Non-determinism.** Temperature 0 does not guarantee identical outputs across runs or model
#   updates. Checks assert structure and behaviour; numbers like P@3/MRR can wobble by a query.
# - **Tiny evaluation sets.** Five retrieval queries, ten epistemic statements and twelve durability
#   labels demonstrate the *method*; they do not support production claims. Grow them and report
#   confidence intervals.
# - **Judge bias.** `h.judge` is an LLM grading an LLM: it can prefer fluent or long answers and
#   shares blind spots with the generator. We mitigate with reference answers in the rubric and by
#   reading reasons, not just scores - but it is not ground truth.
# - **The judge is a model too.** The evaluation section probes position, verbosity, leniency and instability
#   biases; **same-family self-preference** (an OpenAI judge grading OpenAI answers) is *not* measured. Re-calibrate
#   whenever the judge model, rubric or task changes, and refresh the human labels periodically.
# - **Judge cost = extra calls.** Pairwise judging costs 2 calls per pair and `judge_repeat` costs `n`; in CI judge a
#   sample and keep deterministic layer-1 checks for everything else.
# - **Thresholds.** The 0.45 embedding-cosine threshold was chosen from two printed examples; real
#   corpora need a calibrated sweep. Single-linkage clustering can also chain unrelated episodes.
# - **Graph channel.** It only fires when a service name literally appears in the query (no entity
#   linking), so its contribution is understated here.
# - **Snapshot measurements.** Several `MemorySnapshot` fields (e.g. `size_stress_test_passed`) are
#   asserted by us, not measured on a large corpus.
# - **RBAC scope.** We demonstrated label-level `GRANT`/`DENY`; property-level and relationship-level
#   privileges, and credential management (no hard-coded passwords) are what production needs.
# - **Single-node, single-tenant.** No concurrency, no multi-agent write conflicts beyond the
#   consistency model *selection* shown here.
#
# **Swap next:** a cross-encoder reranker (e.g. a hosted rerank model) in place of LLM scores; an LLM
# entity linker feeding the graph channel; Neo4j's property-level privileges for per-field access;
# scheduled sleep-time consolidation jobs; an online evaluation loop that replays logged queries
# from `(:Query)-[:RETRIEVED]->` and the agent `:Run` traces against the labeled set.
