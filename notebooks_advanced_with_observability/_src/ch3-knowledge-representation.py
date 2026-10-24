# %% [markdown]
# # Chapter 3 — Knowledge Representation (advanced, real LLM + real Neo4j)
# > **Tier: advanced (real).** Needs OPENAI_API_KEY in notebooks_advanced/.env and the Neo4j container from docker-compose/.
# > **Evaluation focus:** see the section "Evaluation and LLM-as-a-Judge" near the end — every notebook in this tier treats eval and judging as first-class.
#
# > **Estimated OpenAI cost per full run (measured 2026-10-24 at `gpt-4o` rates): about $0.13.** 95 LLM calls (30 of them LLM-as-a-Judge), 26,124 input / 6,197 output tokens, plus a few thousand `text-embedding-3-large` tokens (<$0.01). Re-run costs about the same; `h.cost_report()` prints your own numbers. See [README](README.md#estimated-openai-cost-measured-24-october-2026).
#
# > **Observability (this folder):** set `OBSERVABILITY_BACKEND=phoenix`, `deepeval`, `both` or `none` in `.env`. Phoenix traces every LLM call, judge and agent step at http://localhost:6006; DeepEval re-scores a sample of the judged answers in a local container (dashboard http://localhost:8765). See the final section of this notebook and the README.
#
# You finished the earlier tiers: you know the eight Chapter 3 skills and what each one decides. In the simplified tier
# every "smart" seam was a heuristic (token similarity, keyword matching, a hand-set profile). Here the seams are real:
#
# | Seam | Earlier tiers | This notebook |
# |---|---|---|
# | Reading requirements / profiles | hand-set numbers | **gpt-4o** with strict JSON-schema outputs |
# | Entity resolution, retrieval | token similarity | **text-embedding-3-large** + **Neo4j vector index** |
# | Extraction | none / canned | **LLM extraction**, prompt v1 vs v2, scored against a labeled set |
# | Agent | scripted loop | **OpenAI function-calling agent** with a trace graph stored in Neo4j |
# | Access control | a gate in Python | the gate **plus real Neo4j RBAC** (`CREATE USER/ROLE`, `GRANT/DENY`) |
#
# There is **no offline fallback** anywhere: if the key or the database is missing, the first cell fails loudly.
# The skills still do the deterministic part (validation, scoring, routing, authorization); the LLM does the
# fuzzy part; and an evaluation harness tells us when each is right or wrong.

# %% [markdown]
# ## 0. Preflight — fail loudly
# **Why:** every later cell assumes a live model, a live embedding endpoint and Neo4j Enterprise. A silent fallback
# would make the results meaningless, so we check all three up front and print the exact model names.

# %%
import sys
sys.path.insert(0, ".")
import json, math, os, re, subprocess, time
import real_helpers as h

h.obs_start("ch3-knowledge-representation")                       # observability: reads OBSERVABILITY_BACKEND from .env (phoenix | deepeval | both | none)

client = h.get_openai()                      # raises if OPENAI_API_KEY is missing
h.connect()                                  # raises if Neo4j is not reachable
print("chat model     :", h.LLM_MODEL)
print("embedding model:", h.EMBED_MODEL, h.EMBED_DIM, "dims")
print("neo4j          :", h.run("RETURN 1 AS ok")[0])
print("edition        :", h.run("CALL dbms.components() YIELD edition RETURN edition")[0])
h.check("preflight: Neo4j answers", h.run("RETURN 1 AS ok")[0]["ok"] == 1)
h.check("preflight: embedding dims 3072", h.EMBED_DIM == 3072)

# %% [markdown]
# ## 1. Load the eight skills and reset the graph
# **Why:** `load_skill` imports each skill's `lib.py` exactly as an agent harness would. The first graph cell of every
# notebook resets the DevOps graph so the notebook is rerunnable top-to-bottom.

# %%
CH = "knowledge-representation"
SLUGS = ["graph-model-selector", "three-graph-router", "entity-resolution-strategy-selector",
         "kg-extraction-approach-selector", "schema-pattern-selector", "homoiconic-meta-schema",
         "knowledge-organization-classifier", "capability-authorization-gate"]
gms, tgr, er, kea, sps, hms, koc, cag = [h.load_skill(CH, s) for s in SLUGS]
KR = h.SKILLS_DIR / CH
info = h.load_devops_graph()
print(info)
h.show(h.run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS n ORDER BY n DESC"))
h.check("8 skills loaded", len(SLUGS) == 8)
h.check("services present", h.run("MATCH (s:Service) RETURN count(s) AS n")[0]["n"] >= 4)

# %% [markdown]
# ## 2. The LLM toolkit: prompt registry, strict structured output, retries, repair loop
# **Why:** production LLM code is mostly *contracts*. Three habits are worth building once:
#
# 1. **Versioned prompts** — a prompt is code; give it an id/version and log which one produced each result, so a
#    regression can be traced to a prompt change.
# 2. **Strict JSON schema** (`response_format=json_schema`, `strict: true`) — the API itself guarantees the shape;
#    strict mode needs `additionalProperties: false` and every field listed in `required`.
# 3. **Retry + repair** — transient API errors get exponential backoff; *semantic* errors get a **repair loop**: feed
#    the deterministic validator's error messages (which come from the skills!) back to the model.

# %%
PROMPTS = {
    "profile.v1": "Read the workload description and score each dimension 0 (not needed) to 3 (critical). "
                  "Be conservative; score only what the text states.",
    "extract.v1": "Extract factual claims from the log line. Return every entity you see.",
    "extract.v2": "Extract factual claims from the log line. Rules: only claim what the text states; "
                  "each claim's `mention` must be a phrase copied from the text; if the text has no factual claim "
                  "about infrastructure return an empty list; do not invent entities.",
}

def structured(name, schema, prompt, system="You are a precise assistant.", retries=3, prompt_id=None):
    """Strict JSON-schema call with exponential backoff. Logs a :PromptRun node."""
    rf = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
    last = None
    for attempt in range(retries):
        try:
            msg = h.chat([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                         response_format=rf)
            out = json.loads(msg.content)
            h.run("CREATE (:PromptRun {prompt_id:$p, schema:$s, attempt:$a, at:datetime()})",
                  p=prompt_id or name, s=name, a=attempt + 1)
            return out
        except Exception as e:                      # API/transient or JSON errors
            last = e
            time.sleep(1.5 ** attempt)
    raise RuntimeError(f"structured({name}) failed after {retries} tries: {last}")

def repair_loop(task_prompt, schema_name, schema, validate, max_tries=3, prompt_id=None):
    """Generate -> validate with a deterministic skill validator -> feed errors back. Returns (obj, history)."""
    history, prompt = [], task_prompt
    for i in range(1, max_tries + 1):
        obj = structured(schema_name, schema, prompt, prompt_id=prompt_id)
        res = validate(obj)
        history.append({"try": i, "valid": res["valid"], "errors": res.get("errors", [])})
        if res["valid"]:
            return obj, history
        prompt = (task_prompt + "\n\nYour previous answer was:\n" + json.dumps(obj) +
                  "\nIt failed validation with these errors:\n- " + "\n- ".join(res["errors"]) +
                  "\nReturn a corrected answer.")
    return obj, history

def cos(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0

def vsearch(index, text, k=3):
    v = h.embed([text])[0]
    return h.run("CALL db.index.vector.queryNodes($i, $k, $v) YIELD node, score "
                 "RETURN coalesce(node.id, node.name) AS id, node.name AS name, score",
                 i=index, k=k, v=v)

h.check("prompt registry versioned", all("." in k for k in PROMPTS))

# %% [markdown]
# ## 3. Make the graph richer, then label it with the three-graph layers
# **Why:** retrieval and entity-resolution experiments need more than four services and no text. We add three more
# dependents of `checkout-api` (so criticality rules have something to count) and eight log lines — the unstructured
# evidence. Structured nodes get `:Resource` (Domain layer); log lines get `:Lexical`.

# %%
h.run("""
UNWIND ['web-frontend','mobile-bff','partner-api'] AS nm
MERGE (s:Service {name: nm}) SET s.tier = 'medium', s.owner = 'team-edge'
WITH s MATCH (c:Service {name:'checkout-api'}) MERGE (s)-[:DEPENDS_ON]->(c)""")
LOGS = [
    "08:41 checkout-api p99 latency spiked from 220ms to 4.8s; error budget burning fast.",
    "08:43 orders-db connection pool is saturated: 20 of 20 connections in use, requests are queueing.",
    "08:30 deploy v2.14.0 reduced max_connections on the orders database from 100 to 20 (deploy-481).",
    "08:44 cart-service retries against checkout are piling up and amplifying the load.",
    "08:40 payments svc health checks are green; no errors in the last hour.",
    "08:45 session-cache hit ratio steady at 97 percent.",
    "Runbook: on pool exhaustion either raise max_connections or roll back the deploy. A database restart is a "
    "Supervisor-level action.",
    "08:46 web frontend error rate is 12 percent because checkout returns HTTP 504.",
]
h.run("""UNWIND range(0, size($logs)-1) AS i
MERGE (l:LogLine {id: 'log-' + i}) SET l:Lexical, l.text = $logs[i], l.source = 'ops-feed'""", logs=LOGS)
h.run("""MATCH (n) WHERE n:Service OR n:Database OR n:Cache
SET n:Resource, n:Domain, n.text = n.name + ' ' + head(labels(n))""")
print(h.run("MATCH (r:Resource) RETURN count(r) AS resources")[0], h.run("MATCH (l:Lexical) RETURN count(l) AS logs")[0])
h.check("8 log lines", h.run("MATCH (l:Lexical) RETURN count(l) AS n")[0]["n"] == 8)
h.check("7 services", h.run("MATCH (s:Service) RETURN count(s) AS n")[0]["n"] == 7)

# %% [markdown]
# ## 4. Real embeddings and real vector indexes
# **Why:** `text-embedding-3-large` (3072 dims) is stored on the nodes and indexed with Neo4j's native HNSW vector index
# (cosine). Two indexes: `resource_idx` (domain entities — used for entity resolution) and `lexical_idx` (log chunks —
# used for retrieval). Index creation is idempotent, embedding is re-done each run (8+7 short strings, cents).

# %%
h.ensure_vector_index("resource_idx", "Resource")
h.ensure_vector_index("lexical_idx", "Lexical")
n1 = h.embed_nodes("Resource", text_prop="text")
n2 = h.embed_nodes("Lexical", text_prop="text")
print("embedded", n1, "resources and", n2, "log lines")
time.sleep(2)  # let the index catch up
hits = vsearch("resource_idx", "the main checkout microservice", k=3)
h.show(hits)
h.check("vector index returns checkout-api first for a paraphrase", hits and hits[0]["name"] == "checkout-api")

# %% [markdown]
# ## 5. Skill 1 — graph-model-selector, with the LLM reading the requirements
# **Why:** `recommend_model` is deterministic but needs a `Requirements` object whose five 0–3 scores someone must
# judge from a prose description. That judgement is exactly what an LLM is good at. We compare it with a human-set
# profile and *report where it disagrees* — the skill is only as good as its input.

# %%
WORKLOAD = ("We need to trace which services are affected when the orders database degrades: fast multi-hop "
            "traversals dominate. The team already runs Neo4j and has Cypher skills. Deployments involve several "
            "participants (service, database, version, operator) in one event. We have no requirement for formal "
            "logical inference, but we want simple property constraints such as tier and owner.")
PROFILE_SCHEMA = lambda keys: {"type": "object", "additionalProperties": False, "required": keys + ["why"],
                               "properties": {**{k: {"type": "integer", "minimum": 0, "maximum": 3} for k in keys},
                                              "why": {"type": "string"}}}
REQ_KEYS = ["formal_reasoning", "n_ary_relations", "performance", "tool_ecosystem", "constraint_expr"]
llm_req = structured("requirements", PROFILE_SCHEMA(REQ_KEYS),
                     PROMPTS["profile.v1"] + "\nDimensions: " + ", ".join(REQ_KEYS) + "\n\nWorkload:\n" + WORKLOAD,
                     prompt_id="profile.v1")
human_req = gms.Requirements(formal_reasoning=0, n_ary_relations=2, performance=3, tool_ecosystem=3, constraint_expr=1)
r_llm = gms.recommend_model(gms.Requirements(**{k: llm_req[k] for k in REQ_KEYS}))
r_hum = gms.recommend_model(human_req)
print("LLM profile  :", {k: llm_req[k] for k in REQ_KEYS}, "->", r_llm["recommended"], r_llm["scores"])
print("human profile:", human_req, "->", r_hum["recommended"], r_hum["scores"])
diff = {k: (getattr(human_req, k), llm_req[k]) for k in REQ_KEYS if getattr(human_req, k) != llm_req[k]}
print("dimensions where LLM and human disagree (human, llm):", diff or "none")
h.check("model recommendation is a valid model", r_llm["recommended"] in {"property_graph", "rdf", "hypergraph"})
h.check("skill picks property graph for the human profile", r_hum["recommended"] == "property_graph")

# %% [markdown]
# **Independent LLM opinion and judge.** We ask the model to choose the model *without* the skill, then use
# `h.judge` to grade the skill's own rationale against a rubric. If the two choices differ, that is a flag to read the
# workload again, not an automatic error — the LLM may weigh "no formal reasoning" differently from the score table.

# %%
indep = structured("model_choice", {"type": "object", "additionalProperties": False, "required": ["model", "reason"],
                   "properties": {"model": {"type": "string", "enum": ["property_graph", "rdf", "hypergraph"]},
                                  "reason": {"type": "string"}}},
                   "Which graph data model fits best? Options: property_graph, rdf, hypergraph.\n\n" + WORKLOAD)
print("independent LLM choice:", indep["model"], "|", indep["reason"][:140])
print("skill choice (LLM profile):", r_llm["recommended"], "| agree:", indep["model"] == r_llm["recommended"])
j = h.judge("Is this model recommendation justified for the workload?\n" + WORKLOAD,
            f"{r_hum['recommended']}: {r_hum['rationale']}",
            "Score 1-5: 5 = rationale clearly matches the workload needs (traversal speed, tooling).")
print("judge:", j)
h.check("judge returned a numeric score", isinstance(j.get("score"), (int, float)))

# %% [markdown]
# ### Blast radius and n-ary reification
# A deployment has four participants — `HyperEdge` models that natively, but a property graph needs an auxiliary
# node. `representation_cost` quantifies it, and we **write the reified form to Neo4j** so the choice is concrete.

# %%
edge = gms.HyperEdge(type="DEPLOYMENT", nodes={"service": "orders-service", "database": "orders-db",
                                               "version": "v2.14.0", "operator": "ci-bot"},
                     attributes={"at": "2026-10-24T08:30:00Z"})
print(gms.representation_cost(edge))
h.run("""MERGE (f:Fact:Reified {id:'deploy-481-fact'}) SET f.type = $t, f.at = $at
WITH f UNWIND $pairs AS p
MERGE (e:Entity {name: p.v}) MERGE (f)-[:PARTICIPANT {role: p.r}]->(e)""",
      t=edge.type, at=edge.attributes["at"], pairs=[{"r": k, "v": v} for k, v in edge.nodes.items()])
radius = h.run("""MATCH p = (d:Database {name:'orders-db'})<-[:DEPENDS_ON*1..3]-(s:Service)
RETURN s.name AS service, min(length(p)) AS hops ORDER BY service""")
h.show(radius)
h.check("reified fact has 4 participants",
        h.run("MATCH (:Fact:Reified {id:'deploy-481-fact'})-[p:PARTICIPANT]->() RETURN count(p) AS n")[0]["n"] == 4)
h.check("blast radius of orders-db includes checkout-api", any(r["service"] == "checkout-api" for r in radius))

# %% [markdown]
# ## 6. Skill 3 — entity-resolution-strategy-selector, then a real resolution bake-off
# **Why:** names in logs never match the canonical graph names. First the LLM derives the `StrategyProfile`; then we
# build a **labeled alias set** and compare three resolvers: the skill's `resolve_match` (name feature), an embedding
# cosine threshold, and an LLM that chooses among vector-retrieved candidates. Labeled data turns "feels right" into a
# measurement.

# %%
ER_KEYS = ["high_stakes", "needs_explainability", "adversarial", "needs_determinism", "cultural_variation",
           "has_training_examples"]
ER_TEXT = ("We link alert and log mentions to our service catalog to decide who gets paged. Wrong merges page the wrong "
           "team, and auditors must see why a link was made. Inputs come from our own tooling, so no adversarial "
           "spoofing. Results must be reproducible. Names are English only. We have no labeled training pairs.")
llm_sp = structured("strategy_profile", PROFILE_SCHEMA(ER_KEYS), PROMPTS["profile.v1"] + "\nDimensions: " +
                    ", ".join(ER_KEYS) + "\n\n" + ER_TEXT, prompt_id="profile.v1")
hand_sp = er.StrategyProfile(high_stakes=2, needs_explainability=3, adversarial=0, needs_determinism=3,
                             cultural_variation=0, has_training_examples=0)
rec_llm = er.recommend_strategy(er.StrategyProfile(**{k: llm_sp[k] for k in ER_KEYS}))
rec_hand = er.recommend_strategy(hand_sp)
print("LLM profile ->", rec_llm["recommended"], "| hand profile ->", rec_hand["recommended"])
print("LLM rationale:", llm_sp["why"][:160])
h.check("strategy recommendations exist", bool(rec_llm["recommended"]) and bool(rec_hand["recommended"]))

# %%
ALIASES = {  # mention -> gold canonical name (None = no such entity in the catalog)
    "Checkout API (prod)": "checkout-api", "chkout-api": "checkout-api", "payments svc": "payment-service",
    "orders db primary": "orders-db", "redis session store": "session-cache", "cart": "cart-service",
    "billing-service": None, "checkout-api-canary": None,
}
canon = [r["name"] for r in h.run("MATCH (r:Resource) RETURN r.name AS name ORDER BY name")]

def resolve_skill(m):
    best = max(canon, key=lambda c: er.resolve_match({"name": m}, {"name": c})["confidence"])
    conf = er.resolve_match({"name": m}, {"name": best})["confidence"]
    return (best if er.classify_edge(conf) == "RESOLVED" else None), conf

def resolve_embed(m, thr=0.55):
    top = vsearch("resource_idx", m, k=1)[0]
    return (top["name"] if top["score"] >= thr else None), top["score"]

CAND_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["choice", "reason"],
               "properties": {"choice": {"type": ["string", "null"]}, "reason": {"type": "string"}}}
def resolve_llm(m):
    cands = [c["name"] for c in vsearch("resource_idx", m, k=3)]
    o = structured("er_choice", CAND_SCHEMA,
                   f"Mention from an ops log: {m!r}\nCandidate catalog entries: {cands}\n"
                   "Return the single candidate this mention denotes, or null if none is clearly the same "
                   "entity (a canary/variant or an absent service is NOT a match).")
    return (o["choice"] if o["choice"] in cands else None), o["reason"]

rows = []
for m, gold in ALIASES.items():
    s, sc = resolve_skill(m); e, es = resolve_embed(m); l, why = resolve_llm(m)
    rows.append({"mention": m, "gold": gold, "skill": s, "embed": e, "llm": l, "skill_conf": round(sc, 2),
                 "embed_cos": round(es, 2), "llm_why": why[:60]})
h.show(rows)
acc = {k: sum(r[k] == r["gold"] for r in rows) / len(rows) for k in ("skill", "embed", "llm")}
print("accuracy:", acc)
h.check("LLM resolver beats or ties the string skill resolver", acc["llm"] >= acc["skill"])
h.check("accuracies are valid fractions", all(0 <= v <= 1 for v in acc.values()))

# %% [markdown]
# ### Failure analysis and two-key edge writing
# Where do they disagree and why? String similarity misses abbreviations (`payments svc`) but is conservative;
# embeddings catch paraphrases but can over-link a `-canary` variant; the LLM is best on semantics but non-deterministic.
# So the write path uses **two keys**: `RESOLVED` only if the skill's explainable confidence is ≥ 0.85 *and* the LLM
# agrees; if only the LLM agrees, the edge is `POSSIBLY_RELATED` with `needs_review`. `flag_edge_cases` adds the
# skill's own warnings.

# %%
for r in rows:
    if len({r["skill"], r["embed"], r["llm"]}) > 1:
        print(f"DISAGREE {r['mention']!r}: gold={r['gold']} skill={r['skill']} embed={r['embed']} llm={r['llm']}")
h.run("MATCH (m:Mention) DETACH DELETE m")
written = {"RESOLVED": 0, "POSSIBLY_RELATED": 0, "none": 0}
for r in rows:
    if r["llm"] is None:
        written["none"] += 1
        continue
    conf = er.resolve_match({"name": r["mention"]}, {"name": r["llm"]})["confidence"]
    kind = "RESOLVED" if er.classify_edge(conf) == "RESOLVED" else "POSSIBLY_RELATED"
    written[kind] += 1
    h.run(f"""MERGE (m:Mention {{text:$m}}) WITH m MATCH (e:Resource {{name:$e}})
              MERGE (m)-[x:{kind}]->(e) SET x.confidence=$c, x.needs_review = $nr""",
          m=r["mention"], e=r["llm"], c=conf, nr=(kind != "RESOLVED"))
print("edges written:", written)
print("edge-case flags for the canary:", er.flag_edge_cases({"name": "checkout-api-canary"}, {"name": "checkout-api"}))
h.check("two-key policy wrote at least one reviewable edge", written["POSSIBLY_RELATED"] + written["RESOLVED"] >= 3)

# %% [markdown]
# ## 7. Skills 2 & 4 — extraction approach and three-graph routing, with real extraction
# **Why:** `recommend_approach` tells us *which* extraction family to use; then we actually run one. The LLM profile
# is compared with a hand-set one, and `incremental_cost` shows why incremental pipelines matter. Then we extract claims
# from the log lines with two prompt versions and score both on a labeled gold set.

# %%
text_src = ("Source: free-text operational logs, arriving continuously (a few dozen per day). Each line is short and "
            "self-contained. The schema is still evolving. Reproducibility is nice but not strict. Volume is small.")
prof_schema = {"type": "object", "additionalProperties": False,
               "required": ["source_type", "incremental", "document_context_needed", "schema_stability",
                            "determinism_need", "volume"],
               "properties": {"source_type": {"type": "string", "enum": ["structured", "unstructured", "mixed"]},
                              "incremental": {"type": "boolean"}, "document_context_needed": {"type": "boolean"},
                              "schema_stability": {"type": "integer", "enum": [0, 1, 2, 3]},
                              "determinism_need": {"type": "integer", "enum": [0, 1, 2, 3]},
                              "volume": {"type": "integer"}}}
llm_prof = structured("kea_profile", prof_schema, "Describe this extraction workload. schema_stability and "
                      "determinism_need are integers 0..3 (0 = none/evolving, 3 = very high); volume is the "
                      "approximate document count.\n" + text_src)
print("LLM-read profile:", llm_prof)
try:
    kp = kea.Profile(**llm_prof)
    rec = kea.recommend_approach(kp)
    print("recommended approach:", rec["recommended"])
    h.check("approach recommended", bool(rec["recommended"]))
except TypeError as e:  # field names/values are the skill's contract; show the mismatch honestly
    print("Profile contract mismatch:", e)
    h.check("profile contract mismatch is visible, not hidden", True)
print("incremental cost for 5 new docs of 500:", kea.incremental_cost(5, 500, "itext2kg"))

# %%
GOLD = {"log-0": {"checkout-api"}, "log-1": {"orders-db"}, "log-2": {"orders-db"},
        "log-3": {"cart-service", "checkout-api"}, "log-4": {"payment-service"}, "log-5": {"session-cache"},
        "log-6": set(), "log-7": {"web-frontend", "checkout-api"}}
CLAIM_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["claims"], "properties": {"claims": {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
                               "required": ["mention", "predicate", "statement", "confidence"],
                               "properties": {"mention": {"type": "string"}, "predicate": {"type": "string"},
                                              "statement": {"type": "string"},
                                              "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}}}}
def extract(version, lid, text):
    sysmsg = PROMPTS[version]
    if version == "extract.v2":
        sysmsg += " Known catalog entities (use as hints, do not force): " + ", ".join(canon)
    return structured("claims", CLAIM_SCHEMA, f"Log line:\n{text}", system=sysmsg, prompt_id=version)["claims"]

chunks = {r["id"]: r["text"] for r in h.run("MATCH (l:Lexical) RETURN l.id AS id, l.text AS text ORDER BY id")}
EXTRACTED = {v: {lid: extract(v, lid, t) for lid, t in chunks.items()} for v in ("extract.v1", "extract.v2")}
for v, d in EXTRACTED.items():
    print(v, "claims per chunk:", {k: len(c) for k, c in d.items()})
h.check("v2 claims have a mention copied from the text (>=80%)",
        sum(c["mention"].lower() in chunks[l].lower() for l, cs in EXTRACTED["extract.v2"].items() for c in cs)
        >= 0.8 * max(1, sum(len(cs) for cs in EXTRACTED["extract.v2"].values())))

# %% [markdown]
# ### Linking mentions: skill string similarity vs embedding `sim_fn`
# `link_subject_to_domain` accepts a pluggable `sim_fn` — we inject embedding cosine and **sweep the threshold**, then
# score P/R/F1 at the chunk level against `GOLD`. The string baseline uses the skill's default (threshold 0.85).

# %%
EMB = {}
def emb(s):
    if s not in EMB:
        EMB[s] = h.embed([s])[0]
    return EMB[s]
for name in canon:
    emb(name)
all_m = sorted({c["mention"] for d in EXTRACTED.values() for cs in d.values() for c in cs})
for s, v in zip(all_m, h.embed(all_m) if all_m else []):
    EMB[s] = v
domain_map = {n: n for n in canon}

def link_all(version, thr, sim_fn=None):
    pred = {}
    for lid, claims in EXTRACTED[version].items():
        got = set()
        for c in claims:
            corr = tgr.link_subject_to_domain(c["mention"], domain_map, threshold=thr, sim_fn=sim_fn)
            if corr.linked:
                got.add(corr.domain_id)
        pred[lid] = got
    return pred

def prf(pred):
    tp = sum(len(pred[k] & GOLD[k]) for k in GOLD); fp = sum(len(pred[k] - GOLD[k]) for k in GOLD)
    fn = sum(len(GOLD[k] - pred[k]) for k in GOLD)
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return round(p, 2), round(r, 2), round(2 * p * r / (p + r), 2) if p + r else 0.0

table = []
for v in EXTRACTED:
    table.append({"prompt": v, "linker": "string@0.85", "P/R/F1": prf(link_all(v, 0.85))})
    for thr in (0.4, 0.5, 0.6, 0.7):
        table.append({"prompt": v, "linker": f"embed@{thr}", "P/R/F1": prf(link_all(v, thr, lambda a, b: cos(emb(a), emb(b))))})
h.show(table, max_rows=20)
best = max(table, key=lambda r: r["P/R/F1"][2])
print("best:", best)
f1 = {(r["prompt"], r["linker"]): r["P/R/F1"][2] for r in table}
h.check("embedding linker at its best >= string linker (v2)",
        max(f1[("extract.v2", f"embed@{t}")] for t in (0.4, 0.5, 0.6, 0.7)) >= f1[("extract.v2", "string@0.85")])

# %% [markdown]
# **Failure analysis.** Which gold links were missed / invented at the best setting? Typical causes: v1 invents
# entities from the runbook line (`log-6` has none), "payments svc" falls under a string threshold, and "web frontend"
# vs `web-frontend` only matches semantically. Read the printout — this is the part a score can't give you.

# %%
ver, lk = best["prompt"], best["linker"]
thr = float(lk.split("@")[1]); fn = None if lk.startswith("string") else (lambda a, b: cos(emb(a), emb(b)))
pb = link_all(ver, thr, fn)
for k in GOLD:
    if pb[k] != GOLD[k]:
        print(f"{k}: missed={sorted(GOLD[k]-pb[k])} extra={sorted(pb[k]-GOLD[k])}")
h.check("analysis ran over all chunks", len(pb) == 8)

# %% [markdown]
# ### Route each record to its graph, write Subject claims with provenance
# `route(Record(...))` encodes the book's rule: structured → **Domain**, raw text → **Lexical** (needs provenance),
# LLM extraction → **Subject** (needs a confidence, entity not yet resolved). Claims are linked to their source chunk
# (`EXTRACTED_FROM`) and to the domain entity they resolve to (`CORRESPONDS_TO`).

# %%
print(tgr.route(tgr.Record(payload={"name": "checkout-api"}, origin="structured")))
print(tgr.route(tgr.Record(payload={"text": LOGS[0]}, origin="raw_text", has_provenance=True)))
h.run("MATCH (c:Claim) DETACH DELETE c")
best_fn = fn
for lid, claims in EXTRACTED["extract.v2"].items():
    for i, c in enumerate(claims):
        dec = tgr.route(tgr.Record(payload=c, origin="extraction", confidence=float(c["confidence"]),
                                   has_provenance=True))
        corr = tgr.link_subject_to_domain(c["mention"], domain_map, threshold=0.6,
                                          sim_fn=lambda a, b: cos(emb(a), emb(b)))
        h.run("""MATCH (l:Lexical {id:$lid})
                 CREATE (s:Subject:Claim {id:$cid, statement:$st, predicate:$pr, confidence:$cf, mention:$mn})
                 CREATE (s)-[:EXTRACTED_FROM]->(l)
                 WITH s OPTIONAL MATCH (e:Resource {name:$dom}) WHERE $linked
                 FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END | CREATE (s)-[:CORRESPONDS_TO]->(e))""",
              lid=lid, cid=f"{lid}#{i}", st=c["statement"], pr=c["predicate"], cf=float(c["confidence"]),
              mn=c["mention"], dom=corr.domain_id, linked=bool(corr.linked))
assert dec["graph"] == "subject"
h.show(h.run("""MATCH (s:Subject:Claim)-[:EXTRACTED_FROM]->(l:Lexical)
OPTIONAL MATCH (s)-[:CORRESPONDS_TO]->(e:Resource)
RETURN l.id AS chunk, s.statement AS claim, e.name AS resolves_to, round(s.confidence*100)/100 AS conf
ORDER BY chunk LIMIT 12"""))
h.check("claims were written with provenance",
        h.run("MATCH (:Claim)-[:EXTRACTED_FROM]->(:Lexical) RETURN count(*) AS n")[0]["n"] >= 6)
h.check("some claims resolved to domain entities",
        h.run("MATCH (:Claim)-[:CORRESPONDS_TO]->(:Resource) RETURN count(*) AS n")[0]["n"] >= 4)

# %% [markdown]
# ## 8. Retrieval ablation — vector vs graph vs hybrid
# **Why:** the claim "graph structure helps retrieval" should be *measured*. Five labeled questions, gold chunks,
# metrics recall@3 and MRR. **Vector** = embed the question, search `lexical_idx`. **Graph** = LLM extracts the entity,
# vector-resolve it to a Resource, expand one `DEPENDS_ON` hop in either direction, return chunks whose claims point
# at those resources. **Hybrid** = reciprocal-rank fusion (RRF) of both.

# %%
QA = [("Why did checkout latency spike?", {"log-0", "log-1", "log-2"}),
      ("Which callers are suffering because of checkout?", {"log-3", "log-7"}),
      ("How do we remedy connection pool exhaustion?", {"log-6", "log-2"}),
      ("Is the payment path healthy?", {"log-4"}),
      ("What changed on the orders database?", {"log-2", "log-1"})]

def r_vector(q, k=3):
    return [x["id"] for x in vsearch("lexical_idx", q, k)]

def r_graph(q, k=3):
    ent = structured("ent", {"type": "object", "additionalProperties": False, "required": ["entity"],
                             "properties": {"entity": {"type": "string"}}},
                     "Name the main infrastructure component this question is about (short noun phrase).\n" + q)["entity"]
    name = vsearch("resource_idx", ent, 1)[0]["name"]
    rows = h.run("""MATCH (e:Resource {name:$n}) MATCH (e)-[:DEPENDS_ON*0..1]-(x:Resource)
    MATCH (c:Lexical)<-[:EXTRACTED_FROM]-(:Subject)-[:CORRESPONDS_TO]->(x)
    RETURN c.id AS id, count(*) AS support ORDER BY support DESC, id LIMIT $k""", n=name, k=k)
    return [r["id"] for r in rows]

def rrf(*lists, k=60):
    score = {}
    for lst in lists:
        for rank, d in enumerate(lst):
            score[d] = score.get(d, 0) + 1 / (k + rank + 1)
    return [d for d, _ in sorted(score.items(), key=lambda x: -x[1])][:3]

def metrics(ranked, gold):
    rec = len(set(ranked) & gold) / len(gold)
    rr = next((1 / (i + 1) for i, d in enumerate(ranked) if d in gold), 0.0)
    return rec, rr

abl = {"vector": [], "graph": [], "hybrid": []}
for q, gold in QA:
    v, g = r_vector(q), r_graph(q)
    for name, ranked in (("vector", v), ("graph", g), ("hybrid", rrf(v, g))):
        abl[name].append(metrics(ranked, gold))
summary = [{"method": m, "recall@3": round(sum(x[0] for x in vs) / len(vs), 2),
            "MRR": round(sum(x[1] for x in vs) / len(vs), 2)} for m, vs in abl.items()]
h.show(summary)
h.check("every method produced a score for each question", all(len(v) == len(QA) for v in abl.values()))
h.check("hybrid recall is not worse than the weaker single method",
        next(s for s in summary if s["method"] == "hybrid")["recall@3"] >=
        min(s["recall@3"] for s in summary if s["method"] != "hybrid") - 1e-9)

# %% [markdown]
# **Reading the ablation.** With five questions each question is 20% of the score — differences below ~0.2 are noise.
# Vector search wins on paraphrases ("why spike?"), graph wins on relational questions ("callers of checkout"), and
# hybrid is usually the safe choice. This is a *teaching-sized* evaluation; see the closing section for what a real
# one needs.

# %% [markdown]
# ## 9. Skill 5 — schema-pattern-selector: heuristic vs LLM classification
# **Why:** `recommend_pattern` uses keyword matching. We write six scenario descriptions that deliberately avoid the
# obvious keywords, then compare the heuristic and an LLM classifier against gold labels.

# %%
PATTERNS = ["event_centric", "contextual_boundary", "multi_perspective", "capability_model"]
SCEN = [
    ("A rollout touched a database at 08:30, an engineer approved it, and the pool shrank minutes later.", "event_centric"),
    ("The SRE dashboard says the service is healthy while the customer-support team insists it is failing; we must "
     "keep both views and who said what.", "multi_perspective"),
    ("In EU accounts a retention limit applies to logs; in US accounts it does not; the same field means different "
     "things per region.", "contextual_boundary"),
    ("Which actions may the on-call bot take, and on which resources, depending on its role?", "capability_model"),
    ("The incident began when the alarm fired, then the page went out, then the rollback finished.", "event_centric"),
    ("Two vendors report different severities for the same vulnerability, each with a confidence.", "multi_perspective"),
]
def llm_pattern(txt):
    o = structured("pattern", {"type": "object", "additionalProperties": False, "required": ["pattern"],
                               "properties": {"pattern": {"type": "string", "enum": PATTERNS}}},
                   "Pick the knowledge-graph schema pattern. event_centric: things that happen with participants and "
                   "time; contextual_boundary: validity depends on context/region; multi_perspective: competing "
                   "claims by different sources; capability_model: who/what may do which actions.\n\n" + txt)
    return o["pattern"]
res = []
for txt, gold in SCEN:
    hp = sps.recommend_pattern(txt)["recommended"]
    res.append({"gold": gold, "heuristic": hp, "llm": llm_pattern(txt), "text": txt[:50]})
h.show(res)
ha = sum(r["heuristic"] == r["gold"] for r in res) / len(res); la = sum(r["llm"] == r["gold"] for r in res) / len(res)
print(f"heuristic accuracy={ha:.2f}  llm accuracy={la:.2f}")
print("disagreements:", [r["text"] for r in res if r["heuristic"] != r["llm"]])
h.check("LLM classifier at least as accurate as keyword heuristic", la >= ha)

# %% [markdown]
# ### Generate a schema-conformant instance with a repair loop
# We ask the LLM for an `event_centric` instance of deploy-481. The first prompt **does not state the contract**, so the
# first try often fails `validate_instance` — and the validator's error messages drive the repair. The history shows how
# many tries were needed (a real measure of prompt quality).

# %%
EVENT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["name", "relationships"],
                "properties": {"name": {"type": "string"},
                               "relationships": {"type": "object", "additionalProperties": False,
                                                 "required": ["hasParticipant", "hasStartTime"],
                                                 "properties": {"hasParticipant": {"type": "array", "items": {"type": "string"}},
                                                                "hasStartTime": {"type": "array", "items": {"type": "string"}}}}}}
inst, hist = repair_loop("Model the deployment deploy-481 (v2.14.0 of orders-service, changed orders-db, at "
                         "2026-10-24T08:30:00Z) as an event instance.",
                         "event_instance", EVENT_SCHEMA,
                         lambda o: sps.validate_instance("event_centric", o), max_tries=3, prompt_id="event.v1")
print(json.dumps(hist, indent=1))
final = sps.validate_instance("event_centric", inst)
print("final validation:", final)
h.check("repair loop ended with a valid instance or an honest error list",
        final["valid"] or len(final["errors"]) > 0)
h.check("history recorded", 1 <= len(hist) <= 3)

# %% [markdown]
# **Bi-temporal write-back.** Events carry two times: `valid_at` (when it happened) and `recorded_at` (when the graph
# learned of it). Separating them lets you answer "what did we believe at 08:50?" — essential for post-mortems.

# %%
h.run("""MATCH (d:Deployment {id:'deploy-481'})
SET d.valid_at = datetime('2026-10-24T08:30:00Z'), d.recorded_at = datetime()""")
h.show(h.run("MATCH (d:Deployment {id:'deploy-481'}) RETURN d.id AS id, toString(d.valid_at) AS valid_at, "
             "d.recorded_at IS NOT NULL AS recorded"))
h.check("bi-temporal properties stored",
        h.run("MATCH (d:Deployment {id:'deploy-481'}) RETURN d.valid_at IS NOT NULL AS ok")[0]["ok"])

# %% [markdown]
# ## 10. Skill 6 — homoiconic-meta-schema: the schema *is* data, written by the LLM
# **Why:** in a homoiconic design, entity types and rules are nodes in the graph. We let the LLM *propose* an
# `EntityType` from sample nodes, validate it with the skill, and validate real data against it. Types must be
# in `{string, date, int, float, bool, list}` — the validator enforces this, whatever the LLM says.

# %%
samples = h.run("MATCH (s:Service) RETURN s.name AS name, s.tier AS tier, s.owner AS owner LIMIT 5")
ET_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["name", "properties"],
             "properties": {"name": {"type": "string"},
                            "properties": {"type": "array", "items": {
                                "type": "object", "additionalProperties": False, "required": ["name", "type", "required"],
                                "properties": {"name": {"type": "string"}, "type": {"type": "string"},
                                               "required": {"type": "boolean"}}}}}}
def to_def(o):
    return {"name": o["name"], "properties": [{"name": p["name"], "type": p["type"], "required": p["required"]}
                                              for p in o["properties"]]}
et_obj, et_hist = repair_loop(
    "Propose an entity type definition named Service from these sample rows. Property types must be one of "
    "string, date, int, float, bool, list. Mark name as required.\n" + json.dumps(samples),
    "entity_type", ET_SCHEMA, lambda o: hms.validate_entity_type(to_def(o)), max_tries=3, prompt_id="entitytype.v1")
et_def = to_def(et_obj)
print(et_def, "| tries:", len(et_hist))
vres = [hms.validate_data_against_type(et_def, {k: v for k, v in r.items() if v is not None})
        for r in h.run("MATCH (s:Service) RETURN s.name AS name, s.tier AS tier, s.owner AS owner")]
print("rows valid against the proposed type:", sum(v["valid"] for v in vres), "/", len(vres))
h.check("entity type definition passes the skill validator", hms.validate_entity_type(et_def)["valid"])

# %% [markdown]
# ### Natural-language policy → executable Rule → graph write-back
# The LLM translates a policy into a `Rule`; the skill's `validate_rule` is the gate (the action needs parseable
# `WHEN var > N THEN SET s.field = 'X'` clauses plus `ELSE`). We then count dependents in Neo4j, call `evaluate_rule`
# and write `criticality` back. The rule itself is stored as a node — knowledge as data.

# %%
POLICY = ("A service with more than 3 dependents is Critical, with more than 1 dependent is Important, otherwise "
          "Standard. The input variable is called dependents and the output field is criticality.")
RULE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["name", "condition", "action"],
               "properties": {k: {"type": "string"} for k in ("name", "condition", "action")}}
rule, rhist = repair_loop(
    "Translate this policy into a Rule. `action` must use exactly this grammar, tiers in descending order: "
    "WHEN dependents > 3 THEN SET s.criticality = 'Critical' WHEN dependents > 1 THEN SET s.criticality = 'Important' "
    "ELSE SET s.criticality = 'Standard' (adapt numbers to the policy).\nPolicy: " + POLICY,
    "rule", RULE_SCHEMA, hms.validate_rule, max_tries=3, prompt_id="rule.v1")
print(rule, "| tries:", len(rhist))
h.check("rule passes validate_rule", hms.validate_rule(rule)["valid"])
h.run("MERGE (r:Rule {name:$n}) SET r.condition=$c, r.action=$a", n=rule["name"], c=rule["condition"], a=rule["action"])
deps = h.run("""MATCH (s:Service) OPTIONAL MATCH (d:Service)-[:DEPENDS_ON]->(s)
RETURN s.name AS name, count(d) AS dependents""")
for r in deps:
    out = hms.evaluate_rule(rule, {"dependents": r["dependents"]}) or {}
    h.run("MATCH (s:Service {name:$n}) SET s.criticality = $c", n=r["name"], c=out.get("criticality"))
h.show(h.run("MATCH (s:Service) RETURN s.name AS name, s.criticality AS criticality ORDER BY name"))
h.check("checkout-api (4 dependents) is Critical",
        h.run("MATCH (s:Service {name:'checkout-api'}) RETURN s.criticality AS c")[0]["c"] == "Critical")

# %% [markdown]
# ## 11. Skill 7 — knowledge-organization-classifier: spectrum, LLM opinion, ontology
# **Why:** the controlled-vocabulary spectrum (pick list → taxonomy → thesaurus → ontology) is a decision with a cost
# of being wrong. The skill classifies from feature flags; the LLM reads a prose description and sets the flags. We
# compare the final classification with gold labels.

# %%
VOCABS = [("A flat dropdown of environment names: prod, staging, dev.", "pick_list"),
          ("Cloud services arranged in a parent/child tree; a service has one parent category.", "taxonomy"),
          ("Terms with synonyms and related-term links, so 'RDS' and 'relational database' find the same documents.",
           "thesaurus"),
          ("Classes with properties, class hierarchy and rules that let a reasoner infer that a Service running on a "
           "Critical host is itself high-risk.", "ontology")]
SPEC_KEYS = ["values", "has_hierarchy", "has_synonyms", "has_associative", "has_classes", "has_properties", "has_inference"]
spec_schema = {"type": "object", "additionalProperties": False, "required": SPEC_KEYS,
               "properties": {k: ({"type": "integer"} if k == "values" else {"type": "boolean"}) for k in SPEC_KEYS}}
vrows = []
for txt, gold in VOCABS:
    spec = structured("vocab_spec", spec_schema, "Fill the feature flags for this vocabulary (values = approx number "
                      "of terms).\n" + txt)
    c = koc.classify(spec)["classification"]
    vrows.append({"gold": gold, "classified": c, "upgrade_next": koc.recommend_upgrade(spec).get("next"), "text": txt[:45]})
h.show(vrows)
kacc = sum(r["gold"] == r["classified"] for r in vrows) / len(vrows)
print("LLM-flags -> skill accuracy:", kacc)
print("cost of a cross-cutting instance in a taxonomy:", koc.migration_cost("taxonomy", "new_instance_cross_cutting"))
h.check("classification accuracy is a sane fraction", 0 <= kacc <= 1)
h.check("most vocabularies classified correctly", kacc >= 0.5)

# %% [markdown]
# **LLM-proposed ontology, deterministically validated.** The LLM writes a tiny ontology for our domain; the skill checks
# that every relationship endpoint is a declared class. A second, *deliberately broken* ontology shows the validator
# catching a dangling endpoint — the point of separating generation from validation.

# %%
ONT_SCHEMA = {"type": "object", "additionalProperties": False,
              "required": ["classes", "subclasses", "individuals", "axioms", "relationships"],
              "properties": {"classes": {"type": "array", "items": {"type": "string"}},
                             "subclasses": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                             "individuals": {"type": "array", "items": {"type": "string"}},
                             "axioms": {"type": "array", "items": {"type": "string"}},
                             "relationships": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}}}
ont_prompt = ("Propose a small ontology for DevOps latency investigation (Service, Database, Cache, Deployment, "
              "Incident). subclasses are [child, parent] pairs; relationships are [source_class, name, target_class]. "
              "Every relationship endpoint must be one of the classes.")
def to_ont(o):
    return {"classes": [{"name": c} for c in o["classes"]],
            "subclasses": [{"name": s[0], "parent": s[1]} for s in o["subclasses"] if len(s) >= 2],
            "individuals": [{"name": i} for i in o["individuals"]],
            "axioms": list(o["axioms"]),
            "relationships": [{"from": r[0], "name": r[1], "to": r[2]} for r in o["relationships"] if len(r) >= 3]}
ont, ohist = repair_loop(ont_prompt, "ontology", ONT_SCHEMA,
                         lambda o: koc.validate_ontology_components(to_ont(o)), max_tries=3, prompt_id="ontology.v1")
print("tries:", len(ohist), "| classes:", ont["classes"])
bad = dict(ont, relationships=ont["relationships"] + [["Service", "RUNS_ON", "Spaceship"]])
bad_res = koc.validate_ontology_components(to_ont(bad))
print("deliberately broken ->", bad_res)
h.check("broken ontology is rejected", not bad_res["valid"])
h.check("LLM ontology ended valid or reported errors", koc.validate_ontology_components(to_ont(ont))["valid"] or len(ohist) == 3)

# %% [markdown]
# ## 12. Skill 8 — capability-authorization-gate, with a trace graph and real Neo4j RBAC
# **Why:** an agent that can write Cypher can also write destructive Cypher. Two independent layers:
# 1. **Capability gate** (skill) — decides *per action* whether the agent's level allows, escalates or denies.
# 2. **Database RBAC** (Neo4j Enterprise) — even if the gate is bypassed or the LLM is prompt-injected, the database
#    refuses the write and hides `owner`. The gate explains; the database enforces. Neither replaces the other.
#
# First the gate, with an audit trail stored as a graph.

# %%
spec = json.loads((KR / "capability-authorization-gate" / "sample_devops_agent.json").read_text(encoding="utf-8"))
agent_spec = cag.agent_from_spec(spec)
ATTEMPTS = [("read_metrics", None), ("query_logs", None), ("describe_infrastructure", None),
            ("restart_instance", None), ("scale_autoscaling_group", 25)]
h.run("MATCH (a:AgentProfile) DETACH DELETE a")
h.run("MERGE (a:AgentProfile {id:$i}) SET a.level = $l", i=spec.get("id", "devops-latency-investigator"),
      l=str(spec.get("granted_level", "User")))
decisions = []
for cap, amt in ATTEMPTS:
    d = cag.authorize(agent_spec, cap, amount=amt)
    decisions.append((cap, d["decision"]))
    h.run("""MATCH (a:AgentProfile) MERGE (c:Capability {name:$c})
             CREATE (a)-[:ATTEMPTED {decision:$d, reasons:$r, at:datetime()}]->(c)""",
          c=cap, d=d["decision"], r="; ".join(d.get("reasons", [])))
print(decisions)
h.check("read-only capabilities allowed", all(d == "allow" for c, d in decisions[:3]))
h.check("restart is not simply allowed for a User-level agent", dict(decisions)["restart_instance"] != "allow")
h.check("audit trail holds 5 attempts",
        h.run("MATCH (:AgentProfile)-[r:ATTEMPTED]->() RETURN count(r) AS n")[0]["n"] == 5)

# %%
RB_ROLE, RB_USER, RB_PW = "kr_investigator", "kr_agent", "Kr-agent-pw-1"
h.sysrun(f"CREATE ROLE {RB_ROLE} IF NOT EXISTS")
h.sysrun(f"CREATE USER {RB_USER} IF NOT EXISTS SET PASSWORD '{RB_PW}' CHANGE NOT REQUIRED")
h.sysrun(f"GRANT ROLE {RB_ROLE} TO {RB_USER}")
for stmt in (f"GRANT ACCESS ON DATABASE neo4j TO {RB_ROLE}",
             f"GRANT MATCH {{*}} ON GRAPH neo4j NODES * TO {RB_ROLE}",
             f"GRANT MATCH {{*}} ON GRAPH neo4j RELATIONSHIPS * TO {RB_ROLE}",
             f"DENY READ {{owner}} ON GRAPH neo4j NODES Service TO {RB_ROLE}"):
    h.sysrun(stmt)
print(list(h.sysrun(f"SHOW ROLE {RB_ROLE} PRIVILEGES AS COMMANDS")))
reads = h.run_as(RB_USER, RB_PW, "MATCH (s:Service {name:'checkout-api'}) RETURN s.name AS name, s.tier AS tier, s.owner AS owner")
print("as kr_agent:", reads)
try:
    h.run_as(RB_USER, RB_PW, "CREATE (:Hacked {x:1})")
    write_blocked = False
except Exception as e:
    write_blocked = True
    print("write refused:", type(e).__name__, str(e)[:100])
h.check("RBAC: read works", bool(reads) and reads[0]["name"] == "checkout-api")
h.check("RBAC: denied property is hidden", reads and reads[0].get("owner") is None)
h.check("RBAC: write is refused by the database", write_blocked)
h.check("RBAC: nothing was created", h.run("MATCH (n:Hacked) RETURN count(n) AS n")[0]["n"] == 0)

# %% [markdown]
# ## 13. A real function-calling agent over everything built so far
# **Why:** all previous cells are tools in disguise. `h.ToolAgent` uses OpenAI function calling: the model chooses
# which tool to call and with what arguments. Every tool call is (a) executed with the **restricted** `kr_agent` user
# (reads only, `owner` hidden), and (b) logged to Neo4j as a `(:Run)-[:STEP]->(:ToolCall)-[:NEXT]->…` trace graph, so we
# can query *how the agent reasoned* afterwards. `authorize_capability` is the skill gate in tool form.

# %%
trace = h.AgentTrace("checkout-api is slow: dependents, likely root cause, may I restart orders-db?")

def traced(name, fn):
    def wrapper(**kw):
        try:
            out = fn(**kw)
            trace.log(name, kw, str(out)[:120], True)
            return json.dumps(out, default=str)[:3000]
        except Exception as e:
            trace.log(name, kw, f"{type(e).__name__}: {str(e)[:100]}", False)
            return f"ERROR: {type(e).__name__}: {str(e)[:200]}"
    return wrapper

def t_cypher(query):
    if re.search(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|CALL)\b", query, re.I):
        raise ValueError("read-only tool: write clauses are refused before reaching the database")
    return h.run_as(RB_USER, RB_PW, query)[:25]

def t_entity(mention):
    return vsearch("resource_idx", mention, 3)

def t_evidence(question):
    return [{"id": x["id"], "text": chunks.get(x["id"], "")[:140]} for x in vsearch("lexical_idx", question, 3)]

def t_authorize(capability, amount=None):
    d = cag.authorize(agent_spec, capability, amount=amount)
    return {"decision": d["decision"], "reasons": d.get("reasons"), "required_level": str(d.get("required_level"))}

TOOLS = {
    "cypher_read": (traced("cypher_read", t_cypher),
                    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                    "Run a read-only Cypher query. Schema: (:Service|Database|Cache {name,tier})-[:DEPENDS_ON]->(...), "
                    "(:Deployment)-[:CHANGED]->(:Database), (:Incident)."),
    "find_entity": (traced("find_entity", t_entity),
                    {"type": "object", "properties": {"mention": {"type": "string"}}, "required": ["mention"]},
                    "Resolve a free-text mention to catalog entities via vector search."),
    "search_evidence": (traced("search_evidence", t_evidence),
                        {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
                        "Retrieve the most relevant log lines for a question."),
    "authorize_capability": (traced("authorize_capability", t_authorize),
                             {"type": "object", "properties": {"capability": {"type": "string"},
                                                               "amount": {"type": ["number", "null"]}},
                              "required": ["capability"]},
                             "Ask the capability gate whether you may perform an action, e.g. restart_instance."),
}
SYSTEM = ("You are a DevOps latency investigator. Use tools; never guess. Resolve names with find_entity, query the "
          "graph with cypher_read, cite log evidence, and ALWAYS call authorize_capability before recommending any "
          "action. If the gate says escalate or deny, say you must escalate and do not recommend doing it yourself.")
agent = h.ToolAgent(TOOLS, SYSTEM, max_steps=8, trace=trace)
answer = agent.run("checkout-api is slow. Which services depend on it, what is the likely root cause, and may I "
                   "restart orders-db to fix it?")
print("\nFINAL:", answer)

# %% [markdown]
# ### Evaluate the agent: behaviour checks + LLM-as-judge + trace query
# Behavioural checks are tolerant (no exact wording): it must have called ≥2 tools, must have consulted the gate, and
# must not claim the restart is permitted. The judge adds a graded view with a rubric; its limits are in the last section.

# %%
steps = h.run("MATCH (r:Run)-[:STEP]->(t:ToolCall) WHERE r.goal STARTS WITH 'checkout-api is slow' "
              "RETURN t.tool AS tool, t.ok AS ok ORDER BY t.seq") if True else []
if not steps:
    steps = h.run("MATCH (:Run)-[:STEP]->(t:ToolCall) RETURN t.tool AS tool, t.ok AS ok LIMIT 20")
h.show(steps)
used = [s["tool"] for s in steps]
low = answer.lower()
h.check("agent used >= 2 tool calls", len(used) >= 2)
h.check("agent consulted the capability gate", "authorize_capability" in used)
h.check("agent did not assert that restart is permitted",
        not re.search(r"\byou (may|can) (go ahead and )?restart\b", low) or "escalat" in low)
jr = h.judge("checkout-api is slow. Dependents, root cause, may I restart orders-db?", answer,
             "Score 1-5. 5 = names dependents (cart-service, web-frontend, ...), points to the orders-db connection "
             "pool / deploy v2.14.0, and clearly says a restart needs escalation (Supervisor) rather than doing it.")
print("judge:", jr)
h.check("judge produced a score", isinstance(jr.get("score"), (int, float)))
print(trace.timeline())

# %% [markdown]
# ## 14. Cleanup of real RBAC objects
# **Why:** users and roles live in the `system` database and survive `clear_graph`. We always remove what we created.

# %%
for stmt in (f"DROP USER {RB_USER} IF EXISTS", f"DROP ROLE {RB_ROLE} IF EXISTS"):
    h.sysrun(stmt)
left = [r for r in h.sysrun("SHOW USERS YIELD user RETURN user") if r["user"] == RB_USER]
h.check("RBAC user dropped", not left)
h.run("MATCH (h:Hacked) DETACH DELETE h")

# %% [markdown]
# ## 15. Skill benchmarks, one CLI call, and the bill
# **Why:** every skill ships a `benchmark` self-check; running all eight confirms the library layer still behaves
# while we were wrapping it in LLM code. One CLI call shows the same skill used from cron/CI with zero Python glue.
# `h.cost_report()` then accounts every token spent in this notebook.

# %%
bench = {s: h.run_benchmark(CH, s) for s in SLUGS}
for s, (ok, first) in bench.items():
    print(("PASS " if ok else "FAIL ") + s, "|", first[:70])
env = dict(os.environ, PYTHONIOENCODING="utf-8")
cp = subprocess.run([sys.executable, str(KR / "graph-model-selector" / "cli.py"), "--help"],
                    capture_output=True, text=True, env=env, encoding="utf-8")
print("CLI --help exit:", cp.returncode, "|", cp.stdout.strip().splitlines()[0][:90] if cp.stdout.strip() else "")
h.check("all 8 skill benchmarks pass", all(ok for ok, _ in bench.values()))
h.check("CLI exits 0", cp.returncode == 0)
print(h.cost_report())

# %% [markdown]
# ## 16. Evaluation summary
# Everything measured above in one place. Values change run-to-run (LLM sampling at temperature 0 is *mostly* but not
# perfectly stable); what matters is the **shape**: which resolver/linker/retriever wins, and where they disagree.

# %%
h.show([
    {"experiment": "entity resolution accuracy", "result": str(acc)},
    {"experiment": "extraction best linker (P/R/F1)", "result": f"{best['prompt']} {best['linker']} {best['P/R/F1']}"},
    {"experiment": "retrieval vector/graph/hybrid", "result": str({s['method']: s['recall@3'] for s in summary})},
    {"experiment": "pattern classification heur/llm", "result": f"{ha:.2f}/{la:.2f}"},
    {"experiment": "vocabulary classification", "result": f"{kacc:.2f}"},
    {"experiment": "agent judge score", "result": str(jr.get("score"))},
])
h.check("summary produced", True)

# %% [markdown]
# ## 17. Vibe coding: prompts to give an AI assistant, and how to verify the result
# **Prompts that work** (note each states inputs, outputs, constraints and the test):
#
# 1. *"Write `link_mentions(claims, domain_names, sim_fn, threshold)` returning `{chunk_id: set(canonical)}`. Use
#    `three-graph-router.link_subject_to_domain`; do not re-implement similarity. Add a test with a mention that must
#    NOT link (`checkout-api-canary`)."*
# 2. *"Add a strict JSON schema to the extraction call; every field required, `additionalProperties: false`. On a
#    validation error retry with the error text appended, max 3 tries, and log the prompt version in Neo4j."*
# 3. *"Create a read-only Neo4j role for the agent with `GRANT MATCH` and `DENY READ {owner}`; write a test that a
#    `CREATE` as that user is refused; always `DROP USER/ROLE ... IF EXISTS` in a `finally`."*
#
# **Verification checklist** (never accept the assistant's first answer):
# - [ ] Does every skill function it calls exist? `grep -n "def " skills/<chapter>/<slug>/lib.py`
# - [ ] Is every Cypher parameterised (no string-concatenated user text) and valid on Neo4j 5.26 (`elementId`, not `id()`)?
# - [ ] Is the write path idempotent (`MERGE`) so a re-run does not duplicate nodes?
# - [ ] Does the test fail when you break the code (mutate a threshold)? A test that cannot fail proves nothing.
# - [ ] Are checks on LLM output about *structure/behaviour*, not exact wording?
# - [ ] Is cleanup (users, roles, temp nodes) guaranteed even when a cell errors?
# - [ ] Did you read `h.cost_report()` — does the number match your expectation?

# %% [markdown]
# ## Evaluation and LLM-as-a-Judge — how you know it works
#
# Everything above *ran*, and sections 5-13 already scored a few things against labels. None of that proves the
# system is **right** in general. Evaluation is how you know a knowledge-representation pipeline works, and how you
# know a change made it better instead of merely different. Think of it as a **pyramid**: cheap, objective, numerous
# checks at the bottom; expensive, subjective, scarce ones at the top.
#
# | Layer | What | Cost | What it can / cannot do |
# |---|---|---|---|
# | 1. Deterministic checks | Cypher / schema / skill-validator assertions | ~free | Exact, but only for what you can state as a rule |
# | 2. Reference-based metrics | overlap, accuracy vs a **gold set** | cheap | Objective, but blind to paraphrase and to *why* an answer is wrong |
# | 3. LLM-as-judge | rubric scoring, pairwise preference | 1+ LLM call per item | Handles free text, but is a **model with biases** |
# | 4. Judge calibration | compare the judge with **human labels** | a one-off labelling effort | Tells you whether layer 3 deserves any trust |
# | 5. Human spot checks | read a sample yourself | scarce | The only ground truth; keeps layer 4 honest |
#
# **A judge is itself a model that must be evaluated.** Known biases, and the probe below that targets each:
#
# - **Position bias**: prefers whichever answer is shown first. *Probe:* pairwise judging in both orders, and an
#   identical-answers pair that must tie (`h.judge_pairwise`).
# - **Verbosity bias**: rewards length and polish. *Probe:* pad a weak answer with fluff; its score must not rise much.
# - **Self-preference**: favours text that looks like its own family's output. *Mitigation:* judge with a different model
#   family than the generator for decisions that matter (`h.judge_rubric(..., model=...)`); we flag it, we cannot measure it here.
# - **Leniency / inconsistency**: hands out 4s and 5s, or changes its mind between runs. *Probes:* calibration against
#   human labels and `h.judge_repeat` (self-consistency).
# - **Fooled by fluent wrongness**: *Probe:* a sabotaged answer with the same style but wrong facts **must** score lower.
#
# Scope: the gold set is about *this chapter*: choosing a graph model, reifying n-ary facts, schema patterns, entity
# resolution, homoiconic schemas, bi-temporal events and RBAC. `h.judge*` helpers ask for the **reason before the
# score** and use temperature 0 except in the repeat probe.

# %% [markdown]
# ### Eval 1. A gold set, written down and stored in the graph
#
# 8 questions whose references come from decisions and facts this notebook established, and 10 **candidate answers that
# we scored by hand (1-5)**: fully correct ones, a vague-but-right one, wrong ones, *subtly* wrong ones (a lossy
# modelling choice, an unexplainable resolver, a prompt-only guardrail) and a **fluffy-but-empty** one. The labels are
# ours and arguable; that is why Eval 4 measures agreement.

# %%
GOLD = [
    {"id": "G1", "question": "Which graph data model fits tracing which services are affected when orders-db degrades, and why?",
     "reference": "A property graph: multi-hop traversal speed dominates, the team already has Neo4j and Cypher skills, "
                  "and simple property constraints suffice. RDF is not needed because there is no formal-reasoning requirement."},
    {"id": "G2", "question": "A deployment has four participants (service, database, version, operator). How do you model it in a property graph?",
     "reference": "Reify it: a Fact node for the deployment with one PARTICIPANT relationship per participant carrying a role. "
                  "A plain relationship connects only two nodes."},
    {"id": "G3", "question": "The SRE dashboard says a service is healthy while support says it is failing, and we must keep both views and who said what. Which schema pattern?",
     "reference": "multi_perspective: store each competing claim with its source (and confidence) instead of overwriting one with the other."},
    {"id": "G4", "question": "Log lines say 'payments svc' but the catalog says payments-api, and wrong merges page the wrong team. Which entity-resolution approach?",
     "reference": "Because it is high-stakes, needs an audit trail and reproducibility, use deterministic, explainable matching "
                  "(name rules, canonical IDs), with uncertain cases sent to a human; a bare embedding threshold or an LLM alone is not auditable."},
    {"id": "G5", "question": "What does a homoiconic meta-schema mean and what is the benefit?",
     "reference": "Entity types and rules are stored as nodes in the graph itself and validated by the same machinery, so you can add or "
                  "change types as data, without a code deploy, and the schema is queryable and auditable."},
    {"id": "G6", "question": "May a read-only advisor agent create or delete nodes?",
     "reference": "No. The restriction is enforced by Neo4j RBAC (a read-only role with DENY on writes), not by a prompt instruction."},
    {"id": "G7", "question": "On deploy-481, what is the difference between valid_at and recorded_at?",
     "reference": "valid_at is when the event happened in the world (08:30); recorded_at is when the graph learned it. Keeping both "
                  "(bi-temporal) answers 'what did we believe at 08:50?'."},
    {"id": "G8", "question": "In the three-graph layout, where do log lines and services live?",
     "reference": "Log lines are Lexical nodes (unstructured evidence); services are Resource/Domain entities; mentions are linked from the lexical to the domain layer."},
]
CANDIDATES = [  # (id, gold id, answer, HUMAN label 1-5, why we gave that label)
    ("C1", "G1", "A property graph. The workload is dominated by fast multi-hop traversals from orders-db, the team already "
                 "runs Neo4j and Cypher, and simple property constraints are enough. RDF would add formal reasoning we do not need.",
     5, "correct, grounded, complete"),
    ("C2", "G1", "Use a graph database since the relationships between services matter.",
     3, "right direction, no model choice, no reasons"),
    ("C3", "G1", "An RDF triple store, because it is the standard and gives automatic inference; a property graph is too limited for dependencies.",
     1, "wrong choice, wrong reasons"),
    ("C4", "G2", "Reify the deployment: create a Fact node and connect service, database, version and operator to it with "
                 "PARTICIPANT relationships that each carry a role, because one relationship can only join two nodes.",
     5, "correct and explains why"),
    ("C5", "G2", "Add one DEPLOYED edge from the operator to the version and put the service and database names inside the edge as a "
                 "comma-separated string property, so no extra node is needed.",
     2, "subtly wrong: participants become opaque strings, not queryable entities"),
    ("C6", "G5", "Meta-schemas are powerful because they unify structure and meaning into a flexible, holistic foundation that "
                 "unlocks the full potential of modern knowledge systems.",
     1, "fluffy but empty: no mechanism, no benefit"),
    ("C7", "G4", "Embed every mention and auto-merge anything with cosine similarity above 0.8; it is fast and needs no labels.",
     2, "fast but not auditable or reproducible for a high-stakes link"),
    ("C8", "G5", "The entity types and rules are themselves nodes in the graph, validated by the same code as the data, so you can "
                 "add a type without a code deploy and query or audit the schema.",
     5, "correct and complete"),
    ("C9", "G6", "No, it should refuse. I would add a line to the system prompt telling it never to modify data.",
     3, "right decision, wrong mechanism (prompted, not enforced)"),
    ("C10", "G7", "valid_at is when the deployment actually happened (08:30) and recorded_at is when the graph learned about it, so "
                  "you can reconstruct what we believed at any earlier moment.",
     5, "correct and complete"),
]
GOLD_BY_ID = {g["id"]: g for g in GOLD}
CAND = {c[0]: {"gold": c[1], "answer": c[2], "human": c[3], "why": c[4]} for c in CANDIDATES}
HUMAN = [CAND[c[0]]["human"] for c in CANDIDATES]

h.run("MATCH (g:Ch3EvalGold) DETACH DELETE g")
h.run("UNWIND $rows AS r CREATE (:Ch3EvalGold {id:r.id, question:r.question, reference:r.reference})", rows=GOLD)
stored = h.run("MATCH (g:Ch3EvalGold) RETURN g.id AS id, size(g.reference) AS ref_chars ORDER BY id")
h.show(stored)
h.check("gold set has >= 8 cases stored in the graph", len(stored) >= 8)
h.check("at least 8 human-labelled candidates, covering bad, subtle and fluffy answers",
        len(CANDIDATES) >= 8 and min(HUMAN) == 1 and max(HUMAN) == 5 and any(v in (2, 3) for v in HUMAN))

# %% [markdown]
# ### Eval 2. Layer 1 and 2: deterministic checks, then a reference-based metric
#
# **Layer 1** asserts what must be true of the *artefacts*: the facts our references cite really are in the graph, the
# skills agree with the references, and no candidate names a component the graph does not contain (a cheap
# **hallucination detector**). No LLM, no cost. **Layer 2** scores each candidate against its reference with
# token-overlap F1 rescaled to 1-5: objective and free, but it only sees *words*, so watch where it disagrees with the humans.

# %%
facts = {
    "participants": h.run("MATCH (:Fact:Reified {id:'deploy-481-fact'})-[p:PARTICIPANT]->() RETURN count(p) AS c")[0]["c"],
    "bitemporal": h.run("MATCH (d:Deployment {id:'deploy-481'}) RETURN d.valid_at IS NOT NULL AND d.recorded_at IS NOT NULL AS ok")[0]["ok"],
    "lexical": h.run("MATCH (l:Lexical) RETURN count(l) AS c")[0]["c"],
    "resources": h.run("MATCH (r:Resource) RETURN count(r) AS c")[0]["c"],
    "model": gms.recommend_model(human_req)["recommended"],
}
print(facts)
h.check("reference G2 holds: the reified deployment has four participants", facts["participants"] == 4)
h.check("reference G7 holds: deploy-481 carries valid_at and recorded_at", facts["bitemporal"])
h.check("reference G8 holds: log lines are Lexical and services are Resource nodes", facts["lexical"] == 8 and facts["resources"] >= 7)
h.check("reference G1 agrees with the graph-model-selector skill", facts["model"] == "property_graph")

KNOWN = {r["name"] for r in h.run("MATCH (n:Resource) RETURN n.name AS name")}
ENTITY_RE = re.compile(r"\b[a-z]+(?:-[a-z0-9]+)*-(?:db|api|service|cache|gateway)\b")
for cid, c in CAND.items():
    c["invented"] = sorted(set(ENTITY_RE.findall(c["answer"])) - KNOWN)


def tokens(t: str) -> set:
    return set(re.findall(r"[a-z0-9_\-\.]+", t.lower()))


def f1_to_5(answer: str, reference: str) -> int:
    a, r = tokens(answer), tokens(reference)
    overlap = len(a & r)
    if not overlap:
        return 1
    p, rc = overlap / len(a), overlap / len(r)
    return int(round(1 + 4 * (2 * p * rc / (p + rc))))


for cid, c in CAND.items():
    c["f1_score"] = f1_to_5(c["answer"], GOLD_BY_ID[c["gold"]]["reference"])
h.show([{"id": cid, "human": c["human"], "overlap_1to5": c["f1_score"], "invented_entities": c["invented"], "why": c["why"]}
        for cid, c in CAND.items()])
print("layer-2 overlap vs human:", h.agreement(HUMAN, [CAND[c[0]]["f1_score"] for c in CANDIDATES]))
h.check("layer 1 finds no invented component names in the correct answers", not CAND["C1"]["invented"] and not CAND["C10"]["invented"])

# %% [markdown]
# ### Eval 3. Layer 3: a rubric judge with explicit 1-vs-5 anchors
#
# Vague rubrics ("is it good?") produce vague, lenient scores. Each criterion says exactly what a **1** and a **5** look
# like, and the judge sees the **reference**, so it grades against ground truth rather than its own beliefs.
# `h.judge_rubric` asks for a one-sentence reason first and returns per-criterion scores plus their mean.
# We judge all 10 candidates once (10 judge calls) and keep the numbers for calibration.

# %%
CRITERIA = {
    "correctness": "5 = every modelling decision and fact agrees with the reference; 1 = contradicts the reference or recommends an unsound design.",
    "grounding": "5 = names the specific mechanism or entities the reference names (e.g. Fact node, PARTICIPANT, valid_at, RBAC); "
                 "1 = names none, or names things that are not in the reference.",
    "completeness": "5 = covers every point the reference makes; 1 = covers none of them (generic or off-topic).",
    "actionability": "5 = states the correct decision and the right mechanism to implement it; 1 = no usable decision, or a harmful one.",
}


def to_5(mean: float) -> int:
    return min(5, max(1, int(mean + 0.5)))


for cid, c in CAND.items():
    g = GOLD_BY_ID[c["gold"]]
    c["rubric"] = h.judge_rubric(g["question"], c["answer"], CRITERIA, reference=g["reference"])
    c["judge"] = to_5(c["rubric"]["mean"])
h.show([{"id": cid, "human": c["human"], "judge": c["judge"], **c["rubric"]["scores"]} for cid, c in CAND.items()])
worst = min(CAND.items(), key=lambda kv: kv[1]["rubric"]["mean"])
print(f"lowest-scored candidate {worst[0]}: {str(worst[1]['rubric']['reasons'].get('correctness', ''))[:140]}")
h.check("judge returned a 1-5 score for every candidate", all(1 <= c["judge"] <= 5 for c in CAND.values()))
print(h.cost_report())

# %% [markdown]
# ### Eval 4. Calibration: does the judge agree with the humans?
#
# The step most teams skip. Compare the judge's 1-5 scores with our human labels on the **same items**:
#
# - **exact / within1**: how often the judge matches exactly, or is off by at most one point. Within-1 near 1.0 with
#   low exact means "right ordering, different scale".
# - **Cohen's kappa**: exact agreement *corrected for chance*. Rough guide: < 0.2 poor, 0.2-0.4 fair, 0.4-0.6 moderate,
#   0.6-0.8 substantial, > 0.8 near-perfect. Be suspicious of high exact agreement with kappa near 0 (the judge may just say "3").
# - **Spearman**: do the two *rank* answers the same way? For picking a winner this matters more than exact scores.
#
# With 10 items these numbers are noisy (one item moves exact agreement by 10 points): a smoke alarm, not a
# certificate. The check is deliberately **tolerant**: a weak judge is a finding to act on, not a crash.

# %%
JUDGE = [CAND[c[0]]["judge"] for c in CANDIDATES]
cal = h.agreement(HUMAN, JUDGE)
print("judge vs human:", cal)
h.show([{"id": cid, "human": c["human"], "judge": c["judge"], "delta": c["judge"] - c["human"]} for cid, c in CAND.items()])
lenient = [cid for cid, c in CAND.items() if c["judge"] - c["human"] >= 2]
if cal["kappa"] < 0.4 or cal["spearman"] < 0.7:
    print("WARNING: the judge is weakly calibrated against our labels. Do not use its scores for decisions yet. Options: "
          "(1) tighten the rubric anchors on the criteria with the biggest deltas, (2) add 2-3 few-shot examples of "
          "1/3/5 answers to the judge prompt, (3) switch to a stronger or different-family judge model, "
          "(4) re-label items where you suspect the human label is the wrong one.")
else:
    print("OK: kappa and Spearman clear the bar (kappa >= 0.4, spearman >= 0.7). Still re-check when the model or rubric changes.")
if lenient:
    print("Leniency: the judge scored these >= 2 points above the human label:", lenient)
h.check("calibration was computed over all candidates", cal["n"] == len(CANDIDATES))

# %% [markdown]
# ### Eval 5. Bias probe 1 — position bias
#
# **What this tests:** whether the verdict depends on *which answer is shown first*. `h.judge_pairwise` asks twice with
# the order swapped; if the verdicts disagree it reports `consistent=False` (counted as a tie). Two pairs (4 judge calls):
# a real pair (C4 correct reification vs C5 lossy edge-property: the better answer should win in **both** orders) and an
# **identical** pair (the only honest verdict is a tie, so any "winner" is pure position bias).

# %%
PAIR_Q, PAIR_REF = GOLD_BY_ID["G2"]["question"], GOLD_BY_ID["G2"]["reference"]
PAIR_CRIT = "Prefer the answer that is more correct, better grounded in the reference facts, and more actionable."
real_pair = h.judge_pairwise(PAIR_Q, CAND["C4"]["answer"], CAND["C5"]["answer"], PAIR_CRIT, reference=PAIR_REF)
same_pair = h.judge_pairwise(PAIR_Q, CAND["C4"]["answer"], CAND["C4"]["answer"], PAIR_CRIT, reference=PAIR_REF)
print("real pair (C4 vs C5):", real_pair)
print("identical pair      :", same_pair)
print("PASS" if real_pair["winner"] == "a" and real_pair["consistent"] else
      "WARNING: the better answer did not win consistently in both orders.")
print("PASS" if same_pair["winner"] == "tie" and same_pair["consistent"] else
      "WARNING: position bias detected on identical answers; always judge both orders and average, or use absolute rubrics.")
h.check("pairwise judge returned verdicts for both probes", real_pair["winner"] in ("a", "b", "tie") and same_pair["winner"] in ("a", "b", "tie"))

# %% [markdown]
# ### Eval 6. Bias probe 2 — verbosity bias
#
# **What this tests:** whether *length and polish* buy points. We wrap the weak answer C2 in confident, contentless
# filler (no new facts). A well-behaved judge scores it about the same; a verbosity-biased one scores it higher.
# One extra judge call; we compare with the C2 score from Eval 3 (a rise above 0.75 mean points is flagged).

# %%
FLUFF = ("Great question, and an important one for any modern data organisation. In today's fast-moving, "
         "AI-driven landscape, holistic knowledge architecture and a robust, forward-looking, best-practice-driven mindset "
         "are absolutely essential. With that context in mind: ")
FLUFF_TAIL = (" Ultimately, by leveraging synergies across teams and embracing a culture of continuous improvement, "
              "organisations can unlock significant long-term value and semantic excellence.")
padded = FLUFF + CAND["C2"]["answer"] + FLUFF_TAIL
g2 = GOLD_BY_ID[CAND["C2"]["gold"]]
padded_res = h.judge_rubric(g2["question"], padded, CRITERIA, reference=g2["reference"])
rise = padded_res["mean"] - CAND["C2"]["rubric"]["mean"]
print(f"C2 plain mean={CAND['C2']['rubric']['mean']:.2f}  padded mean={padded_res['mean']:.2f}  rise={rise:+.2f}  "
      f"(answer grew {len(padded) / len(CAND['C2']['answer']):.1f}x)")
print("PASS: padding did not buy points." if rise <= 0.75 else
      "WARNING: verbosity bias. Add 'do not reward length' with a 1-anchor for filler, or add a conciseness criterion.")
h.check("verbosity probe produced a score", 1 <= padded_res["mean"] <= 5)

# %% [markdown]
# ### Eval 7. Bias probe 3 — self-consistency, and a sabotage control
#
# **Self-consistency.** `h.judge_repeat` scores the *same* answer `n=3` times at temperature 0.7. A `spread` (max - min)
# of 0 or 1 is fine; 2+ means the judge is unreliable *on that item*. We test the mid-quality C2 and the
# subtly-wrong C9 (prompt-only guardrail), because borderline answers are where judges wobble (6 judge calls).
#
# **Sabotage control.** Take the best reification answer (C4) and corrupt its *facts* while keeping its style: right
# shape, wrong content. A judge that checks facts must score it clearly lower. This one is a hard `h.check`: if a judge
# cannot tell a true answer from a fluent lie, none of its other scores mean anything (1 judge call).

# %%
ONE_LINE = ("Score 1-5 against the reference: 5 = correct, specific, grounded in the reference facts and actionable; "
            "3 = right direction but vague; 1 = wrong, invented, or empty.")
rep = {}
for cid in ("C2", "C9"):
    g = GOLD_BY_ID[CAND[cid]["gold"]]
    rep[cid] = h.judge_repeat(g["question"], CAND[cid]["answer"], f"{ONE_LINE} Reference: {g['reference']}", n=3)
    print(f"{cid}: scores={rep[cid]['scores']} mean={rep[cid]['mean']:.2f} spread={rep[cid]['spread']}")
    if rep[cid]["spread"] >= 2:
        print(f"  WARNING: unstable judge on {cid}; average several runs or tighten the rubric for this kind of item.")

SABOTAGED = ("Do not reify: keep the deployment as a single DEPLOYED relationship between the operator and the version, and "
             "store the service and database as a comma-separated string on that relationship, because a relationship "
             "can hold any number of participants.")
g2b = GOLD_BY_ID["G2"]
sab = h.judge_rubric(g2b["question"], SABOTAGED, CRITERIA, reference=g2b["reference"])
print(f"good C4 mean={CAND['C4']['rubric']['mean']:.2f}  sabotaged mean={sab['mean']:.2f}")
h.check("sabotaged (fluent but false) answer scores at least 1 point below the good one",
        sab["mean"] <= CAND["C4"]["rubric"]["mean"] - 1)
print(h.cost_report())

# %% [markdown]
# ### Eval 8. Judge vs heuristic: where do they disagree, and why?
#
# Section 9 compared the skill's keyword rules (`recommend_pattern`) with an LLM classifier against gold labels. Here
# the **rules** and the **LLM judge** face the same 6 scenarios: the heuristic score is 5 if the rules named the gold
# pattern and 1 otherwise; the judge grades *the rules' verdict* against the pattern definitions (6 judge calls; partial
# credit for a closely related pattern). Rows that differ by 2+ points are the interesting ones: each is either a
# heuristic blind spot or a judge quirk, and you decide which by reading the item.

# %%
PATTERN_DEFS = {
    "event_centric": "things that happen, with participants and time",
    "contextual_boundary": "validity or meaning depends on context such as region",
    "multi_perspective": "competing claims by different sources, kept with provenance",
    "capability_model": "who or what may perform which actions on which resources",
}
cmp_rows = []
for i, (txt, gold) in enumerate(SCEN):
    pred = sps.recommend_pattern(txt)["recommended"]
    heur = 5 if pred == gold else 1
    j8 = h.judge(f"Which knowledge-graph schema pattern fits this scenario? {txt}",
                 f"Rules verdict: {pred}",
                 f"The correct pattern is {gold} ({PATTERN_DEFS[gold]}). Score 5 if the verdict names that pattern, 3 if it "
                 "names a closely related pattern, 1 if it names an unrelated pattern or abstains.")
    cmp_rows.append({"id": f"S{i + 1}", "gold": gold, "rules": pred, "heuristic": heur,
                     "judge": int(j8["score"]), "judge_reason": str(j8["reason"])[:80]})
h.show([{k: v for k, v in r.items() if k != "judge_reason"} for r in cmp_rows])
print("heuristic vs judge:", h.agreement([r["heuristic"] for r in cmp_rows], [r["judge"] for r in cmp_rows]))
diffs = [r for r in cmp_rows if abs(r["heuristic"] - r["judge"]) >= 2]
print(f"{len(diffs)} disagreement(s) of >= 2 points")
for r in diffs:
    if r["heuristic"] < r["judge"]:
        meaning = ("heuristic miss, judge gives partial credit: the rules chose a neighbouring pattern or abstained, "
                   "so this is a keyword blind spot (or judge leniency toward a near-miss)")
    else:
        meaning = "judge stricter than exact match: the rules hit the gold label but the judge disputes it; re-read the item"
    print(f"  {r['id']} rules={r['rules']} gold={r['gold']} heuristic={r['heuristic']} judge={r['judge']}: {meaning}. "
          f"Judge said: {r['judge_reason']}")
h.check("compared the heuristic and the judge on all scenarios", len(cmp_rows) == len(SCEN))

# %% [markdown]
# ### Eval 9. Regression harness: change a prompt, prove it did not get worse
#
# Evaluation earns its keep when something *changes*. `h.run_eval` runs a system function over a case list, scores
# each case, and stores `(:EvalRun)-[:HAS_RESULT]->(:EvalResult)` in Neo4j under a name and **version**. One Cypher
# query (`h.compare_runs`) compares versions, and a **CI-style gate** turns the comparison into PASS/FAIL. The system
# is the schema-pattern classifier on the six scenarios from section 9: `v1` is a bare prompt with no pattern
# definitions, `v2` defines the four patterns (12 LLM calls, deterministic scoring, so a flaky judge cannot hide a
# regression). The gate prints PASS or FAIL and never raises.

# %%
EVAL_NAME = "ch3_pattern_classification"
h.run("MATCH (r:EvalRun) WHERE r.name STARTS WITH 'ch3_' OPTIONAL MATCH (r)-[:HAS_RESULT]->(x) DETACH DELETE x, r")
PATTERN_CASES = [{"id": f"S{i + 1}", "input": txt, "gold": gold} for i, (txt, gold) in enumerate(SCEN)]
PATTERN_ENUM_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["pattern"],
                       "properties": {"pattern": {"type": "string", "enum": PATTERNS}}}
PATTERN_PROMPTS = {
    "v1": "Pick the knowledge-graph schema pattern for this scenario.\n\n",
    "v2": "Pick the knowledge-graph schema pattern. " + "; ".join(f"{k}: {v}" for k, v in PATTERN_DEFS.items()) + "\n\n",
}


def classify_pattern(case, ver):
    return structured("pattern_eval", PATTERN_ENUM_SCHEMA, PATTERN_PROMPTS[ver] + case["input"], prompt_id=f"pattern.{ver}")["pattern"]


def score_pattern(case, out):
    ok = out == case["gold"]
    return {"score": 1.0 if ok else 0.0, "passed": ok, "notes": f"gold={case['gold']} got={out}"}


summaries = {}
for ver in ("v1", "v2"):
    summaries[ver] = h.run_eval(EVAL_NAME, PATTERN_CASES, lambda c, v=ver: classify_pattern(c, v), score_pattern, version=ver)
    print(f"{ver}: pass_rate={summaries[ver]['pass_rate']:.0%} failures={summaries[ver]['failures']}")
runs = h.compare_runs(EVAL_NAME)
h.show(runs)
h.check("both versions were stored and are comparable in Neo4j", {r["version"] for r in runs} == {"v1", "v2"})

TOLERANCE = 0.10          # candidate may be at most this much worse than baseline (on a 0-1 scale)
by_ver = {r["version"]: r for r in runs}
base, cand_score = by_ver["v1"]["mean_score"], by_ver["v2"]["mean_score"]
verdict = "PASS" if cand_score >= base - TOLERANCE else "FAIL"
print(f"CI GATE: baseline v1={base:.2f}  candidate v2={cand_score:.2f}  tolerance={TOLERANCE:.2f}  ->  {verdict}")
if verdict == "FAIL":
    print("  Regression: inspect (:EvalResult) rows with passed=false for the candidate version before shipping.")
h.show(h.run("MATCH (:EvalRun {name:$n, version:'v2'})-[:HAS_RESULT]->(x:EvalResult) WHERE NOT x.passed "
             "RETURN x.case AS failed_case, x.notes AS notes ORDER BY x.case", n=EVAL_NAME) or [{"failed_case": "none"}])

# %% [markdown]
# ### Eval 10. Judge report card, and a prompt to grow the eval set
#
# **What I would trust this judge for** (after reading the calibration, bias and sabotage results above):
#
# - **Ranking and regression-spotting** between versions, using the *same* judge, rubric and reference each time.
# - **Catching clearly bad and fluent-but-false modelling advice**, because reference-anchored criteria plus the sabotage control passed.
# - **Triage**: deciding which 10% of answers a human should read first.
#
# **What I would not trust it for:**
#
# - **Absolute quality claims** ("our schema advice is 4.3/5") or comparing numbers across different judges or rubrics.
# - **Safety and permissions.** Whether an agent *can* write is a Neo4j RBAC fact (section 12), never a judge opinion.
# - **Structural validity.** Whether an instance conforms to a schema is the skills' validators' job (layer 1), not the judge's.
# - **Anything after the judge model, rubric or task changes**, until it is re-calibrated on fresh human labels.
# - **Self-preference**: if the judge and the answering model share a family, expect flattery we did not measure here.
#
# **Clean-up for this section:** the next cell removes the `Ch3EvalGold` nodes and the `ch3_` `EvalRun` data it created.
#
# **Vibe-coding prompt: extend the eval set**
# ```text
# Context: section "Evaluation and LLM-as-a-Judge" in notebooks_advanced/_src/ch3-knowledge-representation.py, and
# real_helpers.py (judge_rubric, judge_pairwise, judge_repeat, agreement, run_eval, compare_runs).
# Task: add 6 new gold cases and 8 new human-labelled candidates about entity resolution (alias linking), including two
# subtly wrong answers (an auto-merge threshold with no review, a resolver that cannot explain its link) and one fluffy
# answer. Re-run the calibration.
# Constraints: reuse the CRITERIA dict; <= 20 extra judge calls; every Cypher query parameterised; Neo4j labels
# namespaced with ch3_/Ch3; do not weaken any existing h.check.
# Acceptance: print agreement (kappa, spearman) before and after; if kappa drops by more than 0.1, list the new
# items with the largest judge-human delta and propose a rubric change. Do not change the human labels to fit the judge.
# ```

# %%
h.run("MATCH (g:Ch3EvalGold) DETACH DELETE g")
h.run("MATCH (r:EvalRun) WHERE r.name STARTS WITH 'ch3_' OPTIONAL MATCH (r)-[:HAS_RESULT]->(x) DETACH DELETE x, r")
left = h.run("MATCH (n) WHERE n:Ch3EvalGold OR n:EvalRun OR n:EvalResult RETURN count(n) AS c")[0]["c"]
h.check("evaluation section left no Ch3EvalGold / EvalRun / EvalResult nodes behind", left == 0)
print(h.cost_report())

# %% [markdown]
# ## 18. Cost, limits and what to swap next
# **Cost.** One full run is roughly a few dozen gpt-4o calls with tiny inputs plus ~60 short embeddings — typically well
# under a US dollar; read `h.cost_report()` for the exact number. Embeddings are recomputed every run for simplicity;
# cache them by content hash before scaling.
#
# **What is still approximate — be honest about it:**
# - **Non-determinism.** `temperature=0` reduces but does not eliminate variation; a rerun may flip a borderline link.
#   Our checks are structural for this reason. For gating decisions pin a model snapshot and store prompt versions.
# - **Tiny labeled sets.** 8 aliases, 8 chunks, 5 questions, 6 scenarios: differences under ~20 points are noise. Use
#   hundreds of examples, confidence intervals and a held-out split before choosing a threshold (our 0.4–0.7 sweep is
#   tuned *on the test set* — optimistic by construction).
# - **LLM-as-judge bias.** The same model family writes and judges; judges favour verbose, confident answers. Use
#   a different judge model, pairwise comparison, and periodic human spot checks.
# - **The judge is a model too.** It has position, verbosity and leniency biases and same-family self-preference
#   (gpt-4o answers, gpt-4o judges); the probes in the evaluation section reduce them but do not remove them.
# - **Judge cost is extra calls.** Every judged item is at least one more LLM call (pairwise = two, repeats = n); budget
#   it, cap the loops, and judge a sample in CI rather than everything.
# - **Re-calibrate periodically.** Re-check the judge against fresh human labels whenever the judge model, rubric or task changes.
# - **Heuristic vs LLM "disagreements"** are signals to inspect, not verdicts; we showed both and labeled data decided.
# - **RBAC scope.** One role, one database; we did not test property-level `DENY` on relationships, multi-database
#   setups or credential rotation. Passwords here are demo values — use a secrets manager.
# - **Vector index** is HNSW (approximate) and eventually consistent after writes — hence the short sleep.
#
# **Swap next:** a cross-encoder reranker on the hybrid retriever; a fine-tuned or distilled resolver once you have
# labeled pairs; fulltext + vector hybrid inside Neo4j; an eval service that replays this suite in CI on every prompt
# change; and move the trace graph into your observability stack. **Next chapter:** `ch4-memory`.

# %% [markdown]
# ## Observability — what you can look at now
#
# Everything above ran with the backend chosen by `OBSERVABILITY_BACKEND` in `.env`:
#
# - **`phoenix`** (Arize Phoenix, http://localhost:6006): open project *"ch3-knowledge-representation"*. Every chat/embedding call is a span
#   with its prompt, answer, tokens and latency. `ToolAgent.run` is an **AGENT** span with one **TOOL** span per tool call and the
#   LLM calls nested inside it; every judge verdict is an **EVALUATOR** span carrying `eval.mean` / `eval.scores`; every
#   `h.run_eval` case is a span with `eval.score` and `eval.passed`. Use it to answer "why was this step slow / wrong / expensive?".
# - **`deepeval`** (local container, http://localhost:8765): logs each LLM call and judge verdict, then the cell below asks
#   **DeepEval** (GEval *Correctness* and *AnswerRelevancy*) to re-score a small sample of the answers *this notebook's own
#   judge already scored*. Two independent judges agreeing is reassuring; disagreement tells you which verdicts to read by hand.
# - **`both`** / **`none`**: use both, or switch everything off (the notebook behaves exactly like `notebooks_advanced`).

# %%
obs_result = h.obs_finish()
print(h.cost_report())
