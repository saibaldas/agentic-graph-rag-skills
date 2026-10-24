# %% [markdown]
# # Chapter 2 — Architecture Foundations (advanced, real LLM + real Neo4j)
# > **Tier: advanced (real).** Needs OPENAI_API_KEY in notebooks_advanced/.env and the Neo4j container from docker-compose/.
# > **Evaluation focus:** see the section "Evaluation and LLM-as-a-Judge" near the end — every notebook in this tier treats eval and judging as first-class.
#
# > **Estimated OpenAI cost per full run (measured 2026-10-24 at `gpt-4o` rates): about $0.14.** 114 LLM calls (45 of them LLM-as-a-Judge), 25,788 input / 7,407 output tokens, plus a few thousand `text-embedding-3-large` tokens (<$0.01). Re-run costs about the same; `h.cost_report()` prints your own numbers. See [README](README.md#estimated-openai-cost-measured-24-october-2026).
#
# You finished the beginner, intermediate and "advanced, explained" tiers. Those tiers were honest about their limits:
# a keyword router, a deterministic planner, a trigram stand-in for embeddings and *simulated* access control.
# **Here every one of those seams is replaced by the real thing**:
#
# | Concern | Earlier tiers | This notebook |
# |---|---|---|
# | Routing / classification | keyword scorer | `gpt-4o` with strict JSON-schema output, **compared** with the skill's heuristic |
# | Retrieval | Cypher only / trigram | `text-embedding-3-large` (3072-d) + a **real Neo4j vector index** + fulltext + graph expansion |
# | Agent | scripted plan | a real **OpenAI function-calling agent**; every call stored as a trace graph |
# | Access control | a Python `if` | **Neo4j Enterprise RBAC** (`CREATE USER/ROLE`, `GRANT/DENY`, `h.run_as`) |
# | Evaluation | `assert`s | a labeled set, accuracy tables, failure analysis, **LLM-as-judge** next to heuristic scores |
#
# **Skills used (all 3 of the chapter):** `dual-graph-router`, `harness-node-splitter`, `eight-pillar-readiness-map`.
# **Maps to the book:** Ch2 *"The Dual-Graph Architecture"*, *"Defining the harness"*, *"Splitting a workflow into nodes"*, *"The Eight Pillars"*.
#
# > Run top to bottom. The first graph cell **resets the Neo4j graph**; RBAC cells create users/roles with `IF NOT EXISTS` and drop them at the end.
# > LLM calls use `temperature=0` and tiny inputs; token usage is reported by `h.cost_report()`.

# %%
# Preflight: fail loudly. There is NO offline fallback anywhere in this notebook.
import sys, json, re, subprocess
sys.path.insert(0, ".")
import real_helpers as h

h.get_openai()                      # raises with instructions if OPENAI_API_KEY is missing
h.connect()                         # raises with docker-compose instructions if Neo4j is down
print("LLM model       :", h.LLM_MODEL)
print("Embedding model :", h.EMBED_MODEL, f"({h.EMBED_DIM} dims)")

ping = h.llm("Reply with the single word: ready")
vec = h.embed(["preflight"])[0]
edition = h.run("CALL dbms.components() YIELD edition RETURN edition")[0]["edition"]
me = h.sysrun("SHOW CURRENT USER")              # system database must be reachable for RBAC
print("LLM ping        :", ping)
print("Neo4j edition   :", edition, "| connected as", me[0]["user"])
h.check("LLM answers", isinstance(ping, str) and len(ping) > 0)
h.check("embeddings are 3072-dimensional", len(vec) == h.EMBED_DIM)
h.check("Neo4j is Enterprise (needed for RBAC)", edition.lower() == "enterprise")

# %% [markdown]
# ## 1. Setup, skills, and four engineering habits
#
# Before the architecture, four habits that separate a demo from something you can trust. We set them up once and reuse them everywhere:
#
# 1. **Structured outputs.** We never parse free text. `gpt-4o` is called with `response_format={"type": "json_schema", "strict": True}`,
#    so the reply *must* match a schema (enums, required fields). This removes a whole class of "the model wrapped it in markdown" bugs.
# 2. **Prompt versioning.** Prompts live in a registry keyed `(name, version)`. When quality changes you can say *which* prompt did it, and
#    you can A/B `v1` against `v2` on the same labeled set (we do exactly that for routing).
# 3. **Retries with feedback.** A schema guarantees *shape*, not *meaning*. Our wrapper also runs a semantic `validate` callback; if it fails,
#    the error is fed back to the model for another attempt (bounded: 3 tries).
# 4. **Cost accounting.** `h.cost_report()` shows tokens used so far. We print it at section ends — cost is a design constraint, not an afterthought.
#
# **Cypher so far:** `CREATE (n:Label {p: v})` makes a node, `(a)-[:TYPE]->(b)` an arrow, `MERGE` = create-if-missing, `MATCH` finds patterns.

# %%
h.load_devops_graph()                                           # resets the graph and loads the DevOps scenario
dg = h.load_skill("architecture", "dual-graph-router")
splitter = h.load_skill("architecture", "harness-node-splitter")
pillars = h.load_skill("architecture", "eight-pillar-readiness-map")

catalog = h.skill_catalog("architecture")
for row in catalog:
    print(f"- {row['slug']}: {row['description'][:90]}...")
h.check("chapter exposes exactly 3 skills", len(catalog) == 3)

# ---- reusable helpers: strict schemas, prompt registry, retrying structured call
def obj(props: dict) -> dict:
    """Strict-mode object schema: every property required, no extras."""
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

STR, INT, BOOL = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
def enum(*vals): return {"type": "string", "enum": list(vals)}
def arr(item): return {"type": "array", "items": item}

PROMPTS = {
    ("route", "v1"): "You route requests for a DevOps agent. 'vertical' = a lookup answerable from a knowledge graph of facts; "
                     "'horizontal' = a multi-step process/action; 'both' = a process that needs graph facts; "
                     "'unroutable' = neither. Answer with the schema.",
    ("route", "v2"): "You route requests for a DevOps agent that owns TWO graphs. VERTICAL = a knowledge graph of facts "
                     "(services, owners, dependencies, deployments, incidents). HORIZONTAL = a workflow of steps (investigate, decide, act). "
                     "Rules: (1) A pure question about entities/relationships, however phrased, is 'vertical'. "
                     "(2) A request to DO something or to decide/plan, with no need for facts, is 'horizontal'. "
                     "(3) A request that needs a process AND facts from the graph is 'both'. "
                     "(4) Small talk or off-topic questions (math, weather) are 'unroutable'. "
                     "Judge by meaning, not by keywords.",
}
PROMPT_LOG = []     # (name, version) of every structured call, so results are attributable

def structured(name: str, version: str, schema_name: str, schema: dict, user: str, validate=None, retries: int = 3) -> dict:
    """Strict JSON-schema call + semantic validation + feedback retries. Raises after `retries` failures."""
    system = PROMPTS[(name, version)]
    feedback, last = "", None
    for attempt in range(1, retries + 1):
        msg = h.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user + feedback}],
            response_format={"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
        )
        try:
            out = json.loads(msg.content)
            if validate:
                validate(out)
            PROMPT_LOG.append((name, version, attempt))
            return out
        except Exception as exc:
            last = exc
            feedback = f"\n\nYour previous answer was rejected: {exc}. Fix it and answer again."
    raise RuntimeError(f"structured({name}@{version}) failed after {retries} tries: {last}")

h.check("prompt registry has two route versions", ("route", "v1") in PROMPTS and ("route", "v2") in PROMPTS)

# %% [markdown]
# ## 2. The vertical graph — facts with *time* on the edge
#
# A vector store can say two paragraphs are *similar*; it cannot say `checkout-api` **DEPENDS_ON** `orders-db`. A knowledge graph stores that
# as an explicit, typed edge — and we put **time** on the edge (`since`), because "what was true *then*" is a different question from "what is true now".
#
# **Bi-temporal** (defined): *valid time* (when it was true in the world: `since`) vs *record time* (when we learned it: `recorded_at`).
# Chapter 4 goes deep; here we just store both so point-in-time queries are possible.
#
# * `SET r.since = CASE d.name WHEN ... END` is an if/else inside a query.
# * `-[:DEPENDS_ON*1..3]->` is a **variable-length path**; `min(length(p))` aggregates over paths.
# * **Blast radius** (defined): everything that breaks or slows down if one component fails.

# %%
h.run("""
    MATCH (c:Service {name:'checkout-api'})-[r:DEPENDS_ON]->(d)
    SET r.since = CASE d.name WHEN 'payment-service' THEN '2024-01-15'
                              WHEN 'orders-service'  THEN '2024-03-01' ELSE '2024-05-10' END,
        r.recorded_at = '2026-10-24T08:00:00Z'
""")
h.run("MATCH (c:Service {name:'checkout-api'}) MERGE (c)-[:USES_LIBRARY {version:'5.4.0'}]->(:Library {name:'stripe-python', version:'5.4.0'})")

asof = h.run("""
    MATCH (:Service {name:'checkout-api'})-[r:DEPENDS_ON]->(d)
    WHERE r.since <= $as_of RETURN d.name AS dependency, r.since AS since ORDER BY since
""", as_of="2024-02-01")
h.show(asof)

blast = h.run("""
    MATCH p = (s:Service)-[:DEPENDS_ON*1..3]->(:Database {name:'orders-db'})
    RETURN s.name AS service, min(length(p)) AS hops_to_db ORDER BY hops_to_db, service
""")
h.show(blast)
hops = {r["service"]: r["hops_to_db"] for r in blast}
h.check("as of 2024-02-01 only payment-service was a dependency", [r["dependency"] for r in asof] == ["payment-service"])
h.check("checkout-api is 2 hops and cart-service 3 hops from orders-db", hops["checkout-api"] == 2 and hops["cart-service"] == 3)

# %% [markdown]
# ## 3. Real embeddings on a real vector index (plus fulltext)
#
# The graph answers *structural* questions; people ask *fuzzy* ones ("has this happened before?"). To let fuzzy questions find graph entry points we give
# every important node a short natural-language `text` and embed it with `text-embedding-3-large` (3072 dims).
#
# **Why three kinds of index?**
# * **Vector index** — semantic similarity ("latency after a change" ≈ "connection pool 100 -> 20").
# * **Fulltext index** (Lucene/BM25) — exact words and identifiers (`deploy-481`, `redis`) that embeddings blur.
# * **Graph expansion** — once you have an entry node, follow edges for *relationships* neither index can see.
#
# We tag nodes with a shared `:Searchable` label so one index covers services, databases, deployments, incidents and alarms.
# `h.embed_nodes` embeds only nodes that still lack a vector, so re-running is cheap.

# %%
h.run("MATCH (n:Service)    SET n:Searchable, n.text = 'Service ' + n.name + ' (tier ' + n.tier + ', owned by ' + n.owner + ')'")
h.run("MATCH (n:Database)   SET n:Searchable, n.text = 'Database ' + n.name + ' engine ' + n.engine + ' max_connections ' + toString(n.max_connections)")
h.run("MATCH (n:Cache)      SET n:Searchable, n.text = 'Cache ' + n.name + ' engine ' + n.engine + ', used for sessions'")
h.run("MATCH (n:Deployment) SET n:Searchable, n.text = 'Deployment ' + n.id + ' ' + n.version + ' at ' + n.at + ': ' + n.change")
h.run("""MATCH (n:Incident)   SET n:Searchable,
         n.text = 'Incident ' + n.id + ': ' + n.title + ' (status ' + n.status + ')'
                  + coalesce('. Root cause: ' + n.root_cause, '') + coalesce('. Fix: ' + n.fix, '')""")
h.run("MATCH (n:Alarm)      SET n:Searchable, n.text = 'Alarm ' + n.id + ' on metric ' + n.metric + ' value ' + toString(n.value) + ' above threshold ' + toString(n.threshold)")

h.ensure_vector_index("arch_vec", "Searchable")
h.run("CREATE FULLTEXT INDEX arch_text IF NOT EXISTS FOR (n:Searchable) ON EACH [n.text]")
h.run("CALL db.awaitIndexes(60)")
embedded = h.embed_nodes("Searchable")
stats = h.run("MATCH (n:Searchable) RETURN count(n) AS nodes, count(n.embedding) AS embedded")[0]
print(f"embedded {embedded} nodes now;", stats)
print(h.cost_report())
h.check("every Searchable node has a 3072-d vector", stats["nodes"] >= 10 and stats["nodes"] == stats["embedded"])

# %% [markdown]
# Quick sanity query: a *paraphrase* with no shared words should still land on the right node. If this does not, nothing later will work.

# %%
def vec_search(question: str, k: int = 3) -> list[dict]:
    qv = h.embed([question])[0]
    return h.run("CALL db.index.vector.queryNodes('arch_vec', $k, $v) YIELD node, score "
                 "RETURN coalesce(node.name, node.id) AS key, round(score, 3) AS score ORDER BY score DESC", k=k, v=qv)

top = vec_search("someone shrank the database connection limit and then things slowed down")
h.show(top)
h.check("paraphrase retrieves deploy-481 in the top 3", "deploy-481" in [r["key"] for r in top])

# %% [markdown]
# ## 4. `dual-graph-router` — LLM vs the skill's keyword scorer
#
# The harness first decides **what kind of request** it got:
# * a *lookup* ("which services depend on …?") is one traversal → **vertical** graph (the *map*);
# * a *process* ("classify, plan, decide on rollback") is a workflow → **horizontal** graph (the *route*);
# * a *diagnosis* needs a workflow whose steps query the map → **both** (*where the two graphs meet*).
#
# The skill is a transparent keyword scorer: cheap, free, explainable — and brittle. The chapter says the **contract**
# (`vertical/horizontal/both/unroutable` + rationale) is the seam where you swap in an LLM classifier. We do that now and measure it.
#
# **Evaluation harness (small but real).** Twelve labeled requests: some are easy, some are *traps* — paraphrases with no signal word,
# or words like "what" in an off-topic question. We score three routers: the skill, LLM prompt `v1`, LLM prompt `v2`.

# %%
LABELED = [
    ("R01", "Which services depend on the orders-db and were deployed in the last 24 hours?", "vertical"),
    ("R02", "Classify the alert severity, plan the remediation, and decide whether to roll back.", "horizontal"),
    ("R03", "Diagnose why the checkout API is slow: which services does it depend on and what changed recently?", "both"),
    ("R04", "Hello there, nice weather", "unroutable"),
    ("R05", "Who owns the cart service?", "vertical"),
    ("R06", "Tell me the owner team of the payment service", "vertical"),
    ("R07", "Investigate the latency spike and work out what to do about it", "horizontal"),
    ("R08", "Is the checkout flow sitting on top of the same database as payments?", "vertical"),
    ("R09", "Figure out why orders are failing and which upstream components are involved", "both"),
    ("R10", "Please roll back the last deployment", "horizontal"),
    ("R11", "What is 2+2?", "unroutable"),
    ("R12", "How should we respond to a database connection exhaustion alert, given what the topology looks like?", "both"),
]
TARGETS = ["vertical", "horizontal", "both", "unroutable"]
ROUTE_SCHEMA = obj({"target": enum(*TARGETS), "rationale": STR})

def route_skill(text: str) -> str:
    return dg.route(text).target

def make_llm_router(version: str):
    def _route(text: str) -> str:
        return structured("route", version, "route", ROUTE_SCHEMA, f"Request: {text}")["target"]
    return _route

ROUTERS = {"skill-keywords": route_skill, "llm-v1": make_llm_router("v1"), "llm-v2": make_llm_router("v2")}

def evaluate(router, items=LABELED) -> dict:
    preds = {rid: router(text) for rid, text, _ in items}
    acc = sum(preds[rid] == gold for rid, _, gold in items) / len(items)
    return {"accuracy": acc, "preds": preds}

results = {name: evaluate(fn) for name, fn in ROUTERS.items()}
for name, r in results.items():
    print(f"{name:<15} accuracy = {r['accuracy']:.2f}")
print(h.cost_report())
h.check("every router returned a valid target for every request",
        all(p in TARGETS for r in results.values() for p in r["preds"].values()))
h.check("LLM v2 is at least 70% accurate (tolerant: behaviour, not exact labels)", results["llm-v2"]["accuracy"] >= 0.7)

# %% [markdown]
# ### Confusion view and failure analysis
#
# Accuracy hides *where* a router is wrong. We print every request where any router disagrees with the label, then **categorise the skill's mistakes**:
# * `lexical_gap` — the request has no signal word, so the scorer says `unroutable` (e.g. *"owner team"*, *"roll back"* as two tokens);
# * `false_positive_keyword` — a generic word (`what`, `how`) triggers a route on a request that is not about the graph;
# * `ambiguous_label` — the gold label itself is debatable.
#
# We let the LLM propose the category (strict schema, enum), then **also** ask the LLM-as-judge to grade the skill's own *rationale* text.
# Where the keyword score and the judge disagree is exactly where you would not trust the heuristic.

# %%
rows = []
for rid, text, gold in LABELED:
    p = {n: results[n]["preds"][rid] for n in ROUTERS}
    if any(v != gold for v in p.values()):
        rows.append({"id": rid, "gold": gold, **p})
h.show(rows)

skill_errors = [(rid, text, gold) for rid, text, gold in LABELED if results["skill-keywords"]["preds"][rid] != gold]
CAT_SCHEMA = obj({"items": arr(obj({"id": STR, "category": enum("lexical_gap", "false_positive_keyword", "ambiguous_label"), "why": STR}))})
PROMPTS[("failure", "v1")] = "You analyse failures of a keyword-based request router. Categorise each failure using the schema."
payload = [{"id": rid, "request": text, "gold": gold, "skill_said": results["skill-keywords"]["preds"][rid],
            "matched_vertical": dg.route(text).matched_vertical, "matched_horizontal": dg.route(text).matched_horizontal}
           for rid, text, gold in skill_errors]
cats = structured("failure", "v1", "failures", CAT_SCHEMA, json.dumps(payload))["items"] if payload else []
h.show(cats)

print("\nLLM-as-judge on the SKILL's rationale for each of its mistakes (1 = rationale is wrong for this request):")
judged = {}
for rid, text, gold in skill_errors:
    d = dg.route(text)
    j = h.judge(text, f"Route: {d.target}. {d.rationale}", f"Correct route is '{gold}'. Score 5 only if the route and rationale are right for the request; 1 if clearly wrong.")
    judged[rid] = j["score"]
    print(f"  {rid} skill={d.target:<12} gold={gold:<10} judge={j['score']}  {j['reason'][:80]}")
h.check("the skill's mistakes were categorised", len(cats) == len(skill_errors))
h.check("judge scores are valid 1-5 integers", all(isinstance(s, int) and 1 <= s <= 5 for s in judged.values()))

# %% [markdown]
# ### Store the decisions in the graph — and audit them with Cypher
#
# Routing decisions are data. We write each request and each router's verdict as `(:Request)-[:ROUTED_TO {router}]->(:Route)`.
# Then regression-testing is a **query**, not a script: `sum(CASE WHEN ... THEN 1 ELSE 0 END)` counts agreements per router.

# %%
h.run("UNWIND $rows AS r CREATE (:Request {id:r[0], text:r[1], expected:r[2]})", rows=[list(x) for x in LABELED])
for name, r in results.items():
    h.run("""
        UNWIND $pairs AS p
        MATCH (q:Request {id:p[0]})
        MERGE (k:Route {kind:p[1]})
        MERGE (q)-[x:ROUTED_TO {router:$router}]->(k)
    """, pairs=[[rid, pred] for rid, pred in r["preds"].items()], router=name)

audit = h.run("""
    MATCH (q:Request)-[x:ROUTED_TO]->(k:Route)
    RETURN x.router AS router, count(q) AS n, sum(CASE WHEN k.kind = q.expected THEN 1 ELSE 0 END) AS correct
    ORDER BY router
""")
h.show(audit)
h.check("graph audit equals the Python evaluation",
        all(a["correct"] == round(results[a["router"]]["accuracy"] * len(LABELED)) for a in audit))

meet = dg.explain_meeting_point(dg.route(LABELED[2][1]))
print("\nWhere the graphs meet (skill):")
for k, v in meet.items():
    print(f"  {k}: {v[:90]}...")
h.check("only a 'both' request has a meeting point", dg.explain_meeting_point(dg.route(LABELED[0][1])) == {})

# %% [markdown]
# ## 5. `harness-node-splitter` — nodes differ by *tool surface*, not by prompt
#
# The harness executes the workflow. The chapter's rule: **before adding a node, list the tools it calls.** If the tool list overlaps ≥ 80 % with an
# existing node *of the same type*, it is a prompt variation → merge. Otherwise it is a new role → split. **Jaccard overlap** (defined): `|A ∩ B| / |A ∪ B|`.
# Why split by tools? Each node then has a **least-privilege** permission surface — and in section 9 we will turn that surface into *real* database roles.
#
# **Real component #1 — LLM extraction.** In real life nobody hands you `Operation` objects; you have a *runbook paragraph*. We have `gpt-4o`
# extract operations (strict schema; `tools` restricted to an **enum** of tools that really exist, so it cannot invent one) and feed the result straight to the skill.

# %%
RUNBOOK = ("When an alarm fires, first classify the alert's severity and type. Then look up the affected services and their dependencies in the "
           "knowledge graph. Next pull live p99 latency from CloudWatch metrics, and in parallel pull CPU and memory from CloudWatch metrics for the "
           "same services, and run a CloudWatch Logs Insights query for error patterns. Correlate all the evidence to find the root cause, and finally "
           "write a structured incident report and record the root cause back into the knowledge graph.")
ALLOWED_TOOLS = ["graph_read", "graph_write", "cloudwatch_get_metric_data", "cloudwatch_logs_insights_query", "filesystem"]
OPS_SCHEMA = obj({"operations": arr(obj({
    "id": STR, "task": STR, "node_type": enum(*splitter.NODE_TYPES),
    "tools": arr(enum(*ALLOWED_TOOLS)), "reads": arr(STR), "writes": arr(STR), "input_contract": STR, "output_schema": STR}))})
PROMPTS[("extract_ops", "v1")] = ("You turn an operations runbook into workflow operations. One operation per distinct step. "
                                  "List ONLY the tools a step really calls (reasoning and generation steps usually call none). "
                                  "Use short snake_case ids. Steps that run in parallel are separate operations.")

def _valid_ops(out):
    ids = [o["id"] for o in out["operations"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate operation ids")
    if len(ids) < 5:
        raise ValueError("expected at least 5 operations")

llm_ops_raw = structured("extract_ops", "v1", "ops", OPS_SCHEMA, RUNBOOK, validate=_valid_ops)["operations"]
llm_ops = splitter.operations_from_dicts(llm_ops_raw)
llm_split = splitter.split_nodes(llm_ops)
h.show([{"id": o.id, "type": o.node_type, "tools": ",".join(o.tools)} for o in llm_ops])
print(f"LLM-extracted {len(llm_ops)} operations -> {len(llm_split.nodes)} harness nodes after the skill's overlap rule")
h.check("extracted operations are well-typed", all(o.node_type in splitter.NODE_TYPES for o in llm_ops))
h.check("splitting never creates more nodes than operations", len(llm_split.nodes) <= len(llm_ops))

# %% [markdown]
# ### Heuristic vs LLM judgement on every merge/split decision
#
# The skill's decision is a number (overlap) against a threshold. We now use the **gold** operations from the skill's `sample-workflow.json`
# (so the graph below is deterministic), have the skill make every pairwise decision, and ask the LLM — *without showing it the overlap number* —
# whether each pair is "just a prompt variation" or "a distinct role". Disagreements are the interesting output.

# %%
sample = h.SKILLS_DIR / "architecture" / "harness-node-splitter" / "sample-workflow.json"
op_dicts = json.loads(sample.read_text(encoding="utf-8"))["operations"]
sample_ops = splitter.operations_from_dicts(op_dicts)
res = splitter.split_nodes(sample_ops)
by_id = {o.id: o for o in sample_ops}

def desc(o): return f"{o.id}: {o.task} | type={o.node_type} tools={sorted(o.tool_set())} output={o.output_schema}"
PAIR_SCHEMA = obj({"judgements": arr(obj({"i": INT, "same_role": BOOL, "reason": STR}))})
PROMPTS[("pairs", "v1")] = ("For each pair decide: are the two operations merely prompt variations of one workflow node (same capabilities, same kind of output), "
                            "or distinct roles needing different capabilities/permissions/output contracts? same_role=true only for prompt variations.")
pair_payload = [{"i": i, "a": desc(by_id[d["candidate"]]), "b": desc(by_id[d["compared_to"]])} for i, d in enumerate(res.decisions)]
verdicts = structured("pairs", "v1", "pairs", PAIR_SCHEMA, json.dumps(pair_payload))["judgements"]
llm_same = {v["i"]: v["same_role"] for v in verdicts}

table = []
for i, d in enumerate(res.decisions):
    skill_merge = d["action"] == "merge"
    table.append({"pair": f"{d['candidate']} vs {d['compared_to']}", "overlap": d["overlap"], "skill": d["action"],
                  "llm": "merge" if llm_same.get(i) else "split", "agree": skill_merge == bool(llm_same.get(i))})
h.show(table)
dis = [t for t in table if not t["agree"]]
print(f"{len(table) - len(dis)}/{len(table)} decisions agree; disagreements: {[t['pair'] for t in dis]}")
h.check("LLM judged every pair", len(verdicts) == len(res.decisions))
h.check("get_metrics vs get_cpu: skill says merge", any(t["pair"] == "get_cpu vs get_metrics" and t["skill"] == "merge" for t in table))
h.check("majority of decisions agree", len(table) - len(dis) >= len(table) * 0.6)

# %% [markdown]
# **Reading the disagreements.** Where the LLM says *merge* and the skill says *split* (or the reverse) you have a design question, not a bug:
# the skill keys on **tool surface only** (the chapter's rule — cheap, auditable, enforceable as permissions), while a language model also weighs the
# *semantics* of the task. Prefer the tool-surface rule as the **gate** (it maps to real permissions) and use the LLM as an advisor that flags suspicious pairs.
#
# Next we store the operations, the split result and each harness node's scope in Neo4j, and run a threshold ablation (no LLM needed).

# %%
h.run("UNWIND $rows AS r CREATE (o:Operation) SET o = r WITH o UNWIND o.tools AS tool MERGE (t:Tool {name: tool}) MERGE (o)-[:USES_TOOL]->(t)", rows=op_dicts)
for n in res.nodes:
    sc = splitter.node_scope(n)
    h.run("""
        MERGE (x:HarnessNode {id:$id}) SET x.node_type=$type, x.merged=$merged, x.output_schema=$out
        WITH x UNWIND $merged_from AS op_id MATCH (o:Operation {id:op_id}) MERGE (x)-[:MERGES]->(o)
    """, id=sc["id"], type=sc["node_type"], merged=sc["is_merged"], out=sc["output_schema"], merged_from=n.merged_from)
    h.run("MATCH (x:HarnessNode {id:$id}) UNWIND $tools AS tl MERGE (t:Tool {name:tl}) MERGE (x)-[:CAN_CALL]->(t)", id=sc["id"], tools=sc["tool_surface"])
    h.run("MATCH (x:HarnessNode {id:$id}) UNWIND $s AS s MERGE (m:MemorySlice {name:s}) MERGE (x)-[:MAY_READ]->(m)", id=sc["id"], s=sc["memory_reads"])
    h.run("MATCH (x:HarnessNode {id:$id}) UNWIND $s AS s MERGE (m:MemorySlice {name:s}) MERGE (x)-[:MAY_WRITE]->(m)", id=sc["id"], s=sc["memory_writes"])

overlap_pairs = h.run("""
    MATCH (x:HarnessNode)-[:CAN_CALL]->(t:Tool)<-[:CAN_CALL]-(y:HarnessNode) WHERE x.id < y.id
    RETURN x.id AS a, y.id AS b, t.name AS tool
""")
print("harness nodes sharing a tool:", overlap_pairs)
sweep = {th: len(splitter.split_nodes(sample_ops, threshold=th).nodes) for th in (0.8, 0.5, 0.2, 0.0)}
print("threshold ablation (nodes):", sweep)
h.check("9 operations -> 8 harness nodes; only get_metrics+get_cpu merged",
        len(res.nodes) == 8 and [set(n.merged_from) for n in res.nodes if n.is_merged] == [{"get_metrics", "get_cpu"}])
h.check("no two harness nodes share a tool (least privilege)", overlap_pairs == [])
h.check("threshold 0.0 collapses same-type nodes (role separation lost)", sweep[0.0] < sweep[0.8])

# %% [markdown]
# ## 6. The horizontal graph as a **DAG** — and the seam to the vertical graph
#
# **DAG** (defined): *directed acyclic graph* — "must run before" arrows with **no loops**. A step runs when its inputs are ready; independent branches run in parallel.
# We add `PRECEDES` edges, then ask Cypher for cycles (must be 0) and for each step's **level** (longest path from a start step = which wave it runs in).
#
# Then the **seam**: the `get_metrics` node does not guess which services to check — the *vertical* graph says so (the service and its dependencies).
# The "monitoring tool" writes `MetricPoint` nodes (a stand-in data source, not a stand-in for AI); the latency jumps after `deploy-481` (08:30).

# %%
edges = [("classify", "query_kg"), ("query_kg", "get_metrics"), ("query_kg", "analyze"), ("get_metrics", "analyze"), ("analyze", "report")]
h.run("UNWIND $e AS e MATCH (x:Operation {id:e[0]}), (y:Operation {id:e[1]}) MERGE (x)-[:PRECEDES]->(y)", e=[list(e) for e in edges])
cycles = h.run("MATCH p=(n:Operation)-[:PRECEDES*1..10]->(n) RETURN count(p) AS cycles")[0]["cycles"]
levels = h.run("""
    MATCH (root:Operation) WHERE NOT ()-[:PRECEDES]->(root) AND (root)-[:PRECEDES]->()
    MATCH p = (root)-[:PRECEDES*0..]->(n:Operation)
    OPTIONAL MATCH (m)-[:PRECEDES]->(n)
    RETURN n.id AS step, max(length(p)) AS level, count(DISTINCT m) AS waits_on_inputs ORDER BY level, step
""")
h.show(levels)
lv = {r["step"]: r for r in levels}

h.run("""
    MATCH (c:Service {name:'checkout-api'})
    UNWIND [['08:20',42.0],['08:25',45.0],['08:31',3100.0],['08:36',3300.0],['08:41',2400.0]] AS p
    CREATE (c)-[:HAS_METRIC]->(:MetricPoint {metric:'p99_latency_ms', at:p[0], value:p[1]})
""")
ba = h.run("""
    MATCH (d:Deployment {id:'deploy-481'}) WITH substring(d.at, 11, 5) AS hhmm
    MATCH (:Service {name:'checkout-api'})-[:HAS_METRIC]->(m:MetricPoint)
    RETURN round(avg(CASE WHEN m.at <  hhmm THEN m.value END)) AS before_avg_ms,
           round(avg(CASE WHEN m.at >= hhmm THEN m.value END)) AS after_avg_ms
""")[0]
print(ba)
h.check("workflow has no cycles and analyze is a 2-input join", cycles == 0 and lv["analyze"]["waits_on_inputs"] == 2)
h.check("latency jumped >10x after the deployment", ba["after_avg_ms"] / ba["before_avg_ms"] > 10)

# %% [markdown]
# ## 7. `eight-pillar-readiness-map` — LLM extraction, skill assessment, LLM second opinion
#
# The dual graph is the *structure*; the **eight pillars** are the *build order*, and they are **layered** (memory needs a knowledge graph to remember *into*).
# The skill takes a dict `{pillar: present|partial|missing}` and returns readiness %, layering violations, unresolved flaws and a roadmap.
# Its weak point is the *input*: someone has to say what each system actually has. That is a reading-comprehension task — a perfect LLM job.
#
# We first model pillars/flaws in Neo4j (`REQUIRES`, `SOLVED_BY`), then:
# 1. **LLM extraction**: free-text system description → the 8 statuses (strict schema, enum per pillar). Scored against hand labels.
# 2. **Skill** `assess()` on the extracted statuses (deterministic).
# 3. **LLM second opinion**: its own readiness % and next pillar, compared with the skill's.

# %%
h.run("""UNWIND $ps AS p MERGE (x:Pillar {key:p.key}) SET x.name=p.name, x.chapter=p.chapter, x.order=p.order, x.production_only=p.production_only""",
      ps=[dict(key=p.key, name=p.name, chapter=p.chapter, order=p.order, production_only=p.production_only) for p in pillars.PILLARS.values()])
h.run("UNWIND $d AS d MATCH (x:Pillar {key:d[0]}), (y:Pillar {key:d[1]}) MERGE (x)-[:REQUIRES]->(y)",
      d=[[p.key, dep] for p in pillars.PILLARS.values() for dep in p.depends_on])
h.run("UNWIND $fs AS f MERGE (fl:Flaw {name:f[0]}) WITH fl, f UNWIND f[1] AS pk MATCH (p:Pillar {key:pk}) MERGE (fl)-[:SOLVED_BY]->(p)",
      fs=[[flaw, list(ps)] for flaw, ps in pillars.FLAW_TO_PILLARS.items()])

foundation = h.run("""
    MATCH (p:Pillar) OPTIONAL MATCH (q:Pillar)-[:REQUIRES]->(p)
    RETURN p.key AS pillar, count(q) AS dependents ORDER BY dependents DESC, pillar LIMIT 3
""")
h.show(foundation)
h.check("knowledge_representation is the most depended-on pillar", foundation[0]["pillar"] == "knowledge_representation")

# %%
KEYS = list(pillars.PILLAR_ORDER)
SYS_SCHEMA = obj({"statuses": obj({k: enum("present", "partial", "missing") for k in KEYS}),
                  "llm_readiness_pct": INT, "llm_next_pillar": enum(*KEYS, "none"), "rationale": STR})
PROMPTS[("pillars", "v1")] = ("You assess an agent system against the eight pillars of agentic GraphRAG: " + ", ".join(KEYS) + ". "
                              "present = a real working implementation; partial = exists but incomplete or informal; missing = absent. "
                              "A plain vector store is NOT a knowledge graph. Be strict. Also give your own overall readiness percentage "
                              "and the single pillar to build next (respect layering: lower pillars first).")
SYSTEMS = {
    "S1-prototype": ("Prototype: embeds PDFs into a vector store and answers with the top-k chunks in a single LLM call. No graph, no memory beyond the prompt.",
                     dict(zip(KEYS, ["missing"] * 8))),
    "S2-graph-first": ("Has a Neo4j knowledge graph of services and dependencies with an extraction pipeline and Cypher tools. Sessions have no memory, the agent is "
                       "single-shot with no planner, and tools are called with raw JSON strings without schema validation.",
                       dict(knowledge_representation="present", memory="missing", reasoning="partial", planning="missing",
                            tool_orchestration="partial", structured_output="missing", self_evolution="missing", optimization="missing")),
    "S3-mature": ("Knowledge graph plus bi-temporal memory, multi-hop graph reasoning, an explicit planner, and a typed tool registry with schema-validated outputs. "
                  "No evaluation-driven self-improvement and no caching or cost controls.",
                  dict(knowledge_representation="present", memory="present", reasoning="present", planning="present",
                       tool_orchestration="present", structured_output="present", self_evolution="missing", optimization="missing")),
}
extracted, total, correct = {}, 0, 0
for name, (text, gold) in SYSTEMS.items():
    out = structured("pillars", "v1", "pillars", SYS_SCHEMA, text)
    extracted[name] = out
    hits = sum(out["statuses"][k] == gold[k] for k in KEYS)
    total, correct = total + len(KEYS), correct + hits
    print(f"{name}: {hits}/8 statuses match the labels; wrong: {[k for k in KEYS if out['statuses'][k] != gold[k]]}")
print(f"overall extraction accuracy {correct}/{total}")
h.check("LLM status extraction is at least 70% accurate vs hand labels", correct / total >= 0.7)

# %% [markdown]
# Now the skill runs on the **LLM-extracted** statuses; we store them as `HAS_STATUS` edges and the verdict as `NEXT_TO_BUILD`, then compare with the LLM's own opinion.
# Note the two *different* kinds of estimate: the skill's readiness is a **weighted count** (present = 1, partial = 0.5); the LLM's is a holistic guess.
# We also run a deliberately **badly layered** hand-written system (`legacy-bot`: memory claimed without a knowledge graph) to prove the layering check catches it.

# %%
def store_assess(system: str, statuses: dict):
    h.run("MERGE (:System {name:$n})", n=system)
    h.run("UNWIND $it AS it MATCH (s:System {name:$n}), (p:Pillar {key:it[0]}) MERGE (s)-[x:HAS_STATUS]->(p) SET x.status = it[1]",
          n=system, it=[[k, v] for k, v in statuses.items()])
    rep = pillars.assess(statuses)
    if rep.next_pillar:
        h.run("MATCH (s:System {name:$n}), (p:Pillar {key:$k}) MERGE (s)-[:NEXT_TO_BUILD]->(p)", n=system, k=rep.next_pillar)
    for v in rep.dependency_violations:
        for dep in v["missing_dependencies"]:
            h.run("MATCH (p:Pillar {key:$p}), (d:Pillar {key:$d}) MERGE (p)-[x:VIOLATES]->(d) SET x.system=$n", p=v["pillar"], d=dep, n=system)
    return rep

rows = []
for name, out in extracted.items():
    rep = store_assess(name, out["statuses"])
    rows.append({"system": name, "skill_pct": rep.readiness_pct, "llm_pct": out["llm_readiness_pct"],
                 "skill_next": rep.next_pillar, "llm_next": out["llm_next_pillar"],
                 "next_agrees": rep.next_pillar == out["llm_next_pillar"], "violations": len(rep.dependency_violations)})
h.show(rows)

legacy = store_assess("legacy-bot", {"memory": "present", "reasoning": "partial", **{k: "missing" for k in KEYS if k not in ("memory", "reasoning")}})
viol = h.run("MATCH (p:Pillar)-[x:VIOLATES {system:'legacy-bot'}]->(d:Pillar) RETURN p.key AS claimed, collect(d.key) AS missing_prerequisites ORDER BY claimed")
h.show(viol)

cy_unresolved = h.run("""
    MATCH (f:Flaw)-[:SOLVED_BY]->(p:Pillar)
    OPTIONAL MATCH (:System {name:'S2-graph-first'})-[x:HAS_STATUS]->(p)
    WITH f, collect(coalesce(x.status, 'missing')) AS st WHERE NOT 'present' IN st RETURN f.name AS flaw
""")
skill_unres = {f["flaw"] for f in pillars.assess(extracted["S2-graph-first"]["statuses"]).unresolved_flaws}
h.check("Cypher and skill agree on unresolved flaws for S2", {r["flaw"] for r in cy_unresolved} == skill_unres)
h.check("layering check flags legacy-bot (memory/reasoning without a knowledge graph)",
        {v["pillar"] for v in legacy.dependency_violations} == {"memory", "reasoning"})
print(h.cost_report())

# %% [markdown]
# ## 8. Where does the roadmap advice come from? Judge it.
#
# The skill's roadmap is a *list of pillars*. A human wants advice that fits *their* system. We have the LLM write a short, grounded recommendation,
# then grade it with an **LLM-as-judge using a rubric** — and compare against a cheap *heuristic grounding score* (does the advice mention the skill's `next_pillar`?).
# **Judge bias warning:** judges prefer longer, confident prose and tend to rate their own family's output highly; we use a rubric, temperature 0 and a *negative control*
# (a wrong recommendation that sounds fluent) to see whether the judge can tell the difference at all.

# %%
def advise(name: str) -> str:
    rep = pillars.assess(extracted[name]["statuses"])
    return h.llm(f"System: {SYSTEMS[name][0]}\nSkill roadmap (in order): {rep.roadmap}\nNext pillar: {rep.next_pillar}\n"
                 "Write 3 sentences of concrete advice that follow the roadmap order and explain WHY the next pillar comes first.")

RUBRIC = ("Score 5 if the advice recommends building the next pillar from the roadmap first and explains the layering reason; "
          "3 if reasonable but vague or reorders pillars; 1 if it recommends skipping a foundation layer.")
advice = advise("S2-graph-first")
bad = ("Skip the memory work. The best next investment for this system is self-evolution and cost optimisation, because those deliver the most visible "
       "impact quickly, and foundations can be retrofitted later without risk.")
j_good = h.judge("What should the S2 graph-first system build next?", advice, RUBRIC)
j_bad = h.judge("What should the S2 graph-first system build next?", bad, RUBRIC)
nxt = pillars.assess(extracted["S2-graph-first"]["statuses"]).next_pillar
heur = lambda text: int(nxt.replace("_", " ") in text.lower() or nxt in text.lower())
print("advice:", advice)
print(f"judge(good)={j_good['score']} heuristic_mentions_next={heur(advice)} | judge(bad)={j_bad['score']} heuristic_mentions_next={heur(bad)}")
h.check("the judge separates the layered advice from the layer-skipping advice", j_good["score"] > j_bad["score"])

# %% [markdown]
# ## 9. Real access control: harness-node scopes become Neo4j roles
#
# So far "least privilege" was a *query over our own graph*. Now we make the database enforce it. We **derive** the needed permissions from the harness
# nodes in the graph — which nodes may `graph_read`, which `MAY_WRITE` the `vertical_kg` memory slice — and create real Neo4j roles:
#
# | Role | Derived from | Privileges |
# |---|---|---|
# | `ch2_retrieval` | nodes that call `graph_read` | `MATCH` on the domain labels + all relationships; **`DENY READ {root_cause}`** on `Incident` |
# | `ch2_reporting` | the node that writes `vertical_kg` | everything retrieval has, plus `CREATE` of `CAUSED_BY` edges (and the right to introduce that relationship type) |
#
# **Neo4j privilege syntax in one minute:** `GRANT MATCH {*} ON GRAPH neo4j NODES A, B TO role` allows traversing *and* reading all properties of nodes labelled A or B;
# `DENY` always beats `GRANT`; and — important — a user lacking `MATCH` on a label simply **does not see** those nodes (queries return empty, they do not error),
# while writes without privilege **do** raise `Forbidden`. We prove all three behaviours.

# %%
mapping = h.run("""
    MATCH (x:HarnessNode)
    OPTIONAL MATCH (x)-[:CAN_CALL]->(t:Tool {name:'graph_read'})
    OPTIONAL MATCH (x)-[:MAY_WRITE]->(m:MemorySlice {name:'vertical_kg'})
    RETURN x.id AS node, count(DISTINCT t) > 0 AS reads_graph, count(DISTINCT m) > 0 AS writes_graph ORDER BY node
""")
h.show([m for m in mapping if m["reads_graph"] or m["writes_graph"]])
h.check("exactly query_kg reads the graph and report writes it",
        [m["node"] for m in mapping if m["reads_graph"]] == ["query_kg"] and [m["node"] for m in mapping if m["writes_graph"]] == ["report"])

USERS = {"ch2_retriever": ("ch2_retrieval", "Ch2-retr-Passw0rd!"), "ch2_reporter": ("ch2_reporting", "Ch2-rept-Passw0rd!")}
DOMAIN = "Service, Database, Cache, Deployment, Incident, Alarm, Library, MetricPoint, Searchable"
stmts = []
for role in ("ch2_retrieval", "ch2_reporting"):
    stmts += [f"CREATE ROLE {role} IF NOT EXISTS",
              f"GRANT ACCESS ON DATABASE neo4j TO {role}",
              f"GRANT MATCH {{*}} ON GRAPH neo4j NODES {DOMAIN} TO {role}",
              f"GRANT MATCH {{*}} ON GRAPH neo4j RELATIONSHIPS * TO {role}"]
stmts += ["DENY READ {root_cause} ON GRAPH neo4j NODES Incident TO ch2_retrieval",
          "GRANT CREATE ON GRAPH neo4j RELATIONSHIPS CAUSED_BY TO ch2_reporting",
          "GRANT CREATE NEW RELATIONSHIP TYPE ON DATABASE neo4j TO ch2_reporting"]
for user, (role, pw) in USERS.items():
    stmts += [f"CREATE USER {user} IF NOT EXISTS SET PASSWORD '{pw}' CHANGE NOT REQUIRED", f"GRANT ROLE {role} TO {user}"]
for s in stmts:
    h.sysrun(s)
roles = {r["role"] for r in h.sysrun("SHOW ROLES YIELD role RETURN role")}
users = {r["user"] for r in h.sysrun("SHOW USERS YIELD user RETURN user")}
h.check("roles and users exist", {"ch2_retrieval", "ch2_reporting"} <= roles and set(USERS) <= users)

# %%
RET, REP = ("ch2_retriever", USERS["ch2_retriever"][1]), ("ch2_reporter", USERS["ch2_reporter"][1])

def attempt(who, cypher, **p):
    try:
        return True, h.run_as(*who, cypher, **p)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:110]}"

h.run("CREATE (:AuditSecret {note:'payroll export'})")                          # an unrelated label nobody granted

admin_n = h.run("MATCH (s:Service) RETURN count(s) AS n")[0]["n"]
ok1, r1 = attempt(RET, "MATCH (s:Service) RETURN count(s) AS n")
ok2, r2 = attempt(RET, "MATCH (s:AuditSecret) RETURN count(s) AS n")
ok3, r3 = attempt(RET, "MATCH (i:Incident {id:'INC-0877'}) RETURN i.fix AS fix, i.root_cause AS root_cause")
ok4, r4 = attempt(RET, "MATCH (i:Incident {id:'INC-1001'}), (d:Deployment {id:'deploy-481'}) CREATE (i)-[:CAUSED_BY]->(d)")
ok5, r5 = attempt(REP, "MATCH (i:Incident {id:'INC-1001'}), (d:Deployment {id:'deploy-481'}) CREATE (i)-[:CAUSED_BY]->(d) RETURN 1 AS ok")
print("1) retriever sees services      :", r1, "| admin sees", admin_n)
print("2) retriever, unauthorised label:", r2, "<- silently empty, not an error")
print("3) retriever, denied property   :", r3, "<- root_cause is hidden (null)")
print("4) retriever tries to write     :", r4)
print("5) reporter writes CAUSED_BY    :", r5)
h.run("MATCH ()-[x:CAUSED_BY]->() DELETE x")                                    # keep the graph clean for the agent
h.check("retriever reads domain data", ok1 and r1[0]["n"] == admin_n)
h.check("unauthorised label is invisible", ok2 and r2[0]["n"] == 0)
h.check("denied property is null but granted property is visible", ok3 and r3[0]["root_cause"] is None and r3[0]["fix"] is not None)
h.check("write without privilege is refused", (not ok4))
h.check("reporter may create CAUSED_BY", ok5 and r5[0]["ok"] == 1)

# %% [markdown]
# **Why this beats "simulated" access control:** the guarantee no longer depends on our Python code being correct. Even if a prompt-injected agent
# tried `close_incident` or `DETACH DELETE`, *Neo4j itself* says no, and the *refusal* is something the agent must handle — which is what we test next.

# %% [markdown]
# ## 10. A real function-calling agent, with a stored trace and real RBAC
#
# Now the agent is `gpt-4o` choosing tools via OpenAI **function calling** (`h.ToolAgent`). Its tools are Cypher functions that run **as the Neo4j users above**:
# read tools as `ch2_retriever`, the write tool as `ch2_reporter`. We deliberately also give it a `close_incident` tool wired to the *retriever* identity —
# a tool that exists but which the database will refuse. Each call is logged by `AgentTrace` into the graph (`(:Run)-[:STEP]->(:ToolCall)-[:NEXT]->...`),
# including failures, so the whole run is auditable with Cypher.
#
# Two design points: (1) the system prompt tells the model to *only claim facts returned by tools* (grounding); (2) tool errors are returned to the model as text,
# so it can recover or report honestly instead of crashing.

# %%
GOAL = "Diagnose why the checkout API is slow: which services does it depend on and what changed recently?"
trace = h.AgentTrace(GOAL)

def q_as(who, cypher, **p): return h.run_as(*who, cypher, **p)

def t_search(query, k=3):
    return vec_search(query, int(k))
def t_deps(service):
    return [r["n"] for r in q_as(RET, "MATCH (:Service {name:$s})-[:DEPENDS_ON*1..2]->(d) RETURN DISTINCT d.name AS n ORDER BY n", s=service)]
def t_changes(service):
    return q_as(RET, "MATCH (:Service {name:$s})-[:DEPENDS_ON*0..3]->(x)<-[:CHANGED]-(d:Deployment) "
                     "RETURN d.id AS deployment, d.change AS change, d.at AS at, x.name AS changed_component ORDER BY d.at DESC", s=service)
def t_peak(service):
    return q_as(RET, "MATCH (:Service {name:$s})-[:HAS_METRIC]->(m:MetricPoint) RETURN max(m.value) AS peak_p99_ms", s=service)[0]
def t_similar(component):
    return q_as(RET, "MATCH (old:Incident {status:'resolved'})-[:ROOT_CAUSE]->(c {name:$c}) RETURN old.id AS incident, old.title AS title, old.fix AS fix", c=component)
def t_record(incident, deployment):
    q_as(REP, "MATCH (i:Incident {id:$i}), (d:Deployment {id:$d}) MERGE (i)-[:CAUSED_BY]->(d)", i=incident, d=deployment)
    return f"{incident} -CAUSED_BY-> {deployment}"
def t_close(incident):
    q_as(RET, "MATCH (i:Incident {id:$i}) SET i.status = 'resolved'", i=incident)
    return f"{incident} closed"

def traced(name, fn):
    def inner(**kw):
        try:
            out = fn(**kw)
        except Exception as exc:
            trace.log(name, kw, f"{type(exc).__name__}: {exc}", ok=False)
            raise
        trace.log(name, kw, json.dumps(out, default=str), ok=True)
        return out
    return inner

def P(**props): return {"type": "object", "properties": {k: {"type": "string"} for k in props}, "required": list(props)}
TOOLS = {
    "search_graph_nodes": (traced("search_graph_nodes", t_search), {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                           "Semantic (vector) search over graph nodes; returns node keys with similarity scores."),
    "kg_dependencies": (traced("kg_dependencies", t_deps), P(service=1), "List what a service depends on (up to 2 hops)."),
    "recent_changes": (traced("recent_changes", t_changes), P(service=1), "Deployments that changed the service or anything it depends on (up to 3 hops)."),
    "metrics_peak": (traced("metrics_peak", t_peak), P(service=1), "Peak p99 latency (ms) recorded for a service."),
    "similar_past_incident": (traced("similar_past_incident", t_similar), P(component=1), "Resolved incidents whose root cause was the given component."),
    "record_root_cause": (traced("record_root_cause", t_record), P(incident=1, deployment=1), "Write a CAUSED_BY edge from an incident to a deployment once you are confident."),
    "close_incident": (traced("close_incident", t_close), P(incident=1), "Mark an incident resolved."),
}
SYSTEM = ("You are a DevOps investigation agent. Use the tools; do not guess. Only state facts that tool results contain. "
          "You MUST call kg_dependencies, recent_changes, metrics_peak (for the changed component, e.g. orders-db, and for checkout-api) and similar_past_incident "
          "before answering; never answer after only one or two tool calls. "
          "Workflow: find dependencies, find recent changes, check the metric peak, look for similar past incidents; if the evidence is clear, "
          "record the root cause for INC-1001. If a tool returns an error, say so plainly in your answer instead of pretending it worked. "
          "Finish with at most 4 sentences naming the deployment, the changed component, the evidence number, and any precedent.")
agent = h.ToolAgent(TOOLS, SYSTEM, max_steps=10, trace=trace)
answer = agent.run(GOAL + " Then close incident INC-1001 if you can.")

# %% [markdown]
# ### Audit the run from Neo4j
#
# The trace is a graph, so auditing is Cypher: a per-run summary (aggregation), the ordered chain (`NEXT*` path ending at a node with no outgoing `NEXT`),
# the **write-back** edge, and — the RBAC proof — the incident status *as the admin sees it*. The agent was **asked** to close the incident; the database decides.

# %%
rid = trace.run_id
summary = h.run("""
    MATCH (r:Run {id:$rid})-[:STEP]->(c:ToolCall)
    RETURN r.goal AS goal, count(c) AS steps, sum(CASE WHEN c.ok THEN 1 ELSE 0 END) AS ok_steps,
           sum(CASE WHEN c.ok THEN 0 ELSE 1 END) AS failed_steps, collect(DISTINCT c.tool) AS tools
""", rid=rid)[0]
h.show([{k: (v if k != "goal" else v[:40] + "...") for k, v in summary.items()}])
chain = h.run("""
    MATCH (:Run {id:$rid})-[:STEP]->(f:ToolCall {seq:1})
    OPTIONAL MATCH p = (f)-[:NEXT*]->(l:ToolCall) WHERE NOT (l)-[:NEXT]->()
    RETURN CASE WHEN p IS NULL THEN [f.tool] ELSE [n IN nodes(p) | n.tool] END AS chain
""", rid=rid)[0]["chain"]
print("call chain:", " -> ".join(chain))
wb = h.run("MATCH (i:Incident {id:'INC-1001'}) OPTIONAL MATCH (i)-[:CAUSED_BY]->(d:Deployment) RETURN i.status AS status, d.id AS caused_by")[0]
print("INC-1001 as admin sees it:", wb)
failed = h.run("MATCH (:Run {id:$rid})-[:STEP]->(c:ToolCall {ok:false}) RETURN c.tool AS tool, c.result AS result", rid=rid)
h.show(failed)
print(h.cost_report())
h.check("agent used several distinct tools", len(summary["tools"]) >= 3)
h.check("trace chain length equals number of logged steps", len(chain) == summary["steps"])
h.check("INC-1001 is still open: Neo4j enforced least privilege", wb["status"] == "open")
h.check("if the agent tried close_incident the refusal was recorded as a failed step",
        ("close_incident" not in summary["tools"]) or any(f["tool"] == "close_incident" for f in failed))

# %% [markdown]
# ## 11. Evaluating the agent's answer: judge + heuristic, and where they disagree
#
# An agent's final text is the one thing users read, so we evaluate it with two independent signals:
# * a **heuristic fact-coverage score**: which of the four ground-truth facts appear as substrings (`deploy-481`, `orders-db`, `20`, `INC-0877`);
# * an **LLM-as-judge** score against a rubric.
#
# We score the agent's real answer next to a *plausible-but-wrong* answer (blames `payment-service`) and a *vague* one. If the judge cannot rank the wrong answer
# below the right one, it is useless as an evaluator. We also **re-judge the same answer** to see judge noise (even at temperature 0).

# %%
FACTS = ["deploy-481", "orders-db", "20", "inc-0877"]
def coverage(text: str) -> float:
    t = text.lower()
    return sum(f in t for f in FACTS) / len(FACTS)

RUBRIC = ("Ground truth: deployment deploy-481 changed orders-db (connection pool 100 -> 20), latency peaked above 3000 ms, and a similar past incident INC-0877 "
          "was fixed by raising the pool size. Score 5 if the answer names deploy-481 and orders-db with the evidence and precedent; 3 if partly right or vague; "
          "1 if it blames the wrong component or invents facts.")
CANDIDATES = {
    "agent": answer,
    "wrong-but-fluent": "The slowdown is caused by the payment-service, which was upgraded last week and now adds roughly 3 seconds of latency to every checkout call.",
    "vague": "Checkout is slow, probably because something downstream changed recently. Worth looking at the dependencies.",
}
rows = []
for name, text in CANDIDATES.items():
    j = h.judge(GOAL, text, RUBRIC)
    rows.append({"candidate": name, "coverage": round(coverage(text), 2), "judge": j["score"], "reason": j["reason"][:70]})
h.show(rows)
rejudge = h.judge(GOAL, answer, RUBRIC)["score"]
sc = {r["candidate"]: r for r in rows}
print(f"re-judging the agent answer: first={sc['agent']['judge']} second={rejudge}  (any difference is judge noise)")
for r in rows:
    gap = abs(r["coverage"] * 4 + 1 - r["judge"])
    if gap >= 2:
        print(f"  DISAGREE on {r['candidate']}: heuristic coverage={r['coverage']} vs judge={r['judge']}")
h.check("judge ranks the correct agent answer above the fluent wrong answer", sc["agent"]["judge"] > sc["wrong-but-fluent"]["judge"])
h.check("heuristic coverage also ranks the agent above the vague answer", sc["agent"]["coverage"] > sc["vague"]["coverage"])

# %% [markdown]
# ## 12. Ablation: vector vs fulltext vs graph vs hybrid retrieval
#
# Which retrieval strategy actually serves the *architecture*? We build six labeled questions whose gold answers are graph nodes, then compare four strategies on **recall**
# (fraction of gold nodes returned) and **context size** (how much the LLM would have to read):
#
# 1. **vector** — top-3 by embedding similarity.
# 2. **fulltext** — top-3 by Lucene/BM25 over node text.
# 3. **graph** — the LLM *links entities* (choosing from the real node keys via an enum), then a Cypher **expansion** follows `DEPENDS_ON`, `CHANGED`, `ROOT_CAUSE`, `AFFECTS`.
# 4. **hybrid** — Reciprocal Rank Fusion of vector + fulltext (top 2 seeds) **plus** the same graph expansion.
#
# **RRF** (defined): each retriever contributes `1 / (60 + rank)` per result; summing rewards items several retrievers agree on without needing comparable scores.
# Expect vector to shine on paraphrase, graph on multi-hop (Q6), and hybrid to be the most robust — but it will cost more context.

# %%
QA = [
    ("What changed recently that could slow down checkout?", {"deploy-481"}),
    ("Has this kind of latency problem happened before, and how was it resolved?", {"INC-0877"}),
    ("Which database do the payment and orders services rely on?", {"orders-db"}),
    ("Which cache holds checkout session data?", {"session-cache"}),
    ("Who owns the cart service?", {"cart-service"}),
    ("If the orders database fails, which services are affected?", {"orders-service", "payment-service", "checkout-api", "cart-service"}),
]
ALL_KEYS = [r["k"] for r in h.run("MATCH (n:Searchable) RETURN coalesce(n.name, n.id) AS k ORDER BY k")]
STOP = set("the a an of in on is are do does did which what who how that this it to and or for by with from has have was were".split())

def ft_search(q: str, k: int = 3) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in STOP]
    if not words:
        return []
    rows = h.run("CALL db.index.fulltext.queryNodes('arch_text', $q) YIELD node, score "
                 "RETURN coalesce(node.name, node.id) AS key ORDER BY score DESC LIMIT $k", q=" OR ".join(words), k=k)
    return [r["key"] for r in rows]

def expand(seeds: list[str]) -> list[str]:
    if not seeds:
        return []
    r = h.run("""
        MATCH (n:Searchable) WHERE coalesce(n.name, n.id) IN $seeds
        OPTIONAL MATCH (n)<-[:DEPENDS_ON*1..3]-(up:Service)
        OPTIONAL MATCH (n)-[:DEPENDS_ON]->(down)
        OPTIONAL MATCH (n)<-[:CHANGED|ROOT_CAUSE|AFFECTS]-(ev)
        OPTIONAL MATCH (n)-[:DEPENDS_ON*1..3]->(dep)<-[:CHANGED]-(ev2)
        RETURN collect(DISTINCT coalesce(n.name, n.id)) + collect(DISTINCT coalesce(up.name, up.id)) + collect(DISTINCT coalesce(down.name, down.id))
               + collect(DISTINCT coalesce(ev.name, ev.id)) + collect(DISTINCT coalesce(ev2.name, ev2.id)) AS keys
    """, seeds=seeds)[0]["keys"]
    return list(dict.fromkeys(r))

LINK_SCHEMA = obj({"entities": arr(enum(*ALL_KEYS))})
PROMPTS[("link", "v1")] = "Pick the graph entities (from the allowed keys) that the question is about. Return an empty list if none."

def s_vector(q): return [r["key"] for r in vec_search(q, 3)]
def s_fulltext(q): return ft_search(q, 3)
def s_graph(q): return expand(structured("link", "v1", "link", LINK_SCHEMA, q)["entities"])
def s_hybrid(q):
    score = {}
    for ranked in ([r["key"] for r in vec_search(q, 5)], ft_search(q, 5)):
        for rank, key in enumerate(ranked, 1):
            score[key] = score.get(key, 0) + 1 / (60 + rank)
    seeds = [k for k, _ in sorted(score.items(), key=lambda kv: -kv[1])[:2]]
    return list(dict.fromkeys(seeds + expand(seeds)))

STRATS = {"vector": s_vector, "fulltext": s_fulltext, "graph": s_graph, "hybrid": s_hybrid}
grid, summary_rows = [], {}
for name, fn in STRATS.items():
    recs, sizes = [], []
    for q, gold in QA:
        got = set(fn(q))
        recs.append(len(gold & got) / len(gold)); sizes.append(len(got))
        grid.append({"strategy": name, "q": q[:48], "recall": round(recs[-1], 2), "context_nodes": len(got)})
    summary_rows[name] = {"strategy": name, "mean_recall": round(sum(recs) / len(recs), 2), "avg_context_nodes": round(sum(sizes) / len(sizes), 1)}
h.show(list(summary_rows.values()))
print(h.cost_report())
h.check("every strategy produced a non-trivial result", all(r["mean_recall"] > 0.3 for r in summary_rows.values()))
h.check("hybrid is not worse than vector alone (tolerance 0.05)", summary_rows["hybrid"]["mean_recall"] >= summary_rows["vector"]["mean_recall"] - 0.05)

# %% [markdown]
# ### Failure analysis of the ablation
#
# Look at the per-question grid: which strategy lost recall on which question, and *why*? Typical findings — and what to do about each:
# * **vector misses Q6** (multi-hop): similarity finds `orders-db` but nothing links it to `cart-service` three hops away → add **graph expansion**.
# * **fulltext misses paraphrases** (Q1/Q2 share few words with the node text) → keep the **vector** leg in the hybrid.
# * **graph depends on entity linking**: if the LLM picks the wrong seed, the expansion is confidently wrong → keep a similarity leg as a cross-check.
# * **hybrid returns more nodes** → higher recall but more tokens to read; a reranker or an answer-time budget is the usual next step.

# %%
worst = sorted([g for g in grid if g["recall"] < 1.0], key=lambda g: g["recall"])
h.show(worst[:10])
by_q = {}
for g in grid:
    by_q.setdefault(g["q"], {})[g["strategy"]] = g["recall"]
h.show([{"question": q, **v} for q, v in by_q.items()])
q6 = [g for g in grid if g["q"].startswith("If the orders database fails")]
r6 = {g["strategy"]: g["recall"] for g in q6}
print("multi-hop question recall:", r6)
h.check("the multi-hop question is solved by a graph-aware strategy", max(r6["graph"], r6["hybrid"]) >= 0.75)

# %% [markdown]
# ## 13. Skill CLIs and self-benchmarks (the zero-dependency path)
#
# Every skill is also a command-line tool for cron, CI or any agent harness — no LLM, no pip installs. `benchmark` runs the skill's self-checks and exits non-zero on failure.
# This is the **cheap regression gate** that sits *under* all the LLM evaluation above: if a deterministic skill regresses, you want to know before spending tokens.

# %%
bench = {}
for row in catalog:
    ok, line = h.run_benchmark("architecture", row["slug"])
    bench[row["slug"]] = ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {row['slug']:<28} {line[:70]}")
h.check("all 3 chapter benchmarks pass", all(bench.values()) and len(bench) == 3)

import os
env = dict(os.environ, PYTHONIOENCODING="utf-8")
cli = h.SKILLS_DIR / "architecture" / "dual-graph-router" / "cli.py"
helptext = subprocess.run([sys.executable, str(cli), "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
routed = subprocess.run([sys.executable, str(cli), "route", GOAL, "--json"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
cli_decision = json.loads(routed.stdout)
h.check("--help exits 0 and prints text", helptext.returncode == 0 and len(helptext.stdout) > 50)
h.check("CLI route agrees with the library (both)", cli_decision.get("target") == "both" == dg.route(GOAL).target)

# %% [markdown]
# ## 14. 🎯 Vibe-coding corner — prompts, a *real* assistant, and a verification checklist
#
# *Vibe coding* = describe what you want to an AI assistant and steer by running checks. A good prompt has **context** (what exists), **constraints**
# (what must not happen) and an **acceptance check** (how you will know). Then **you verify** — never trust unrun code. Here the assistant is `gpt-4o`
# itself, and the *verification is enforced by Neo4j*: generated Cypher runs as the read-only `ch2_retriever`, so even a bad suggestion cannot write.
#
# **Prompt A — a new Cypher query** (used live below):
# ```text
# Context: Neo4j has (:Service {name})-[:DEPENDS_ON]->(x), and (:Database {name}) nodes.
# Task: ONE read-only Cypher query returning, for every Service, its name and how many distinct Databases it reaches within 3 DEPENDS_ON hops
#       (use OPTIONAL MATCH so services with none show 0). Order by count desc.
# Constraints: no writes, no APOC, no markdown fences, return exactly two columns.
# Acceptance: checkout-api = 1 and cart-service = 1; session-cache must not appear.
# ```
# **Prompt B — a new skill:**
# ```text
# Create skills/architecture/workflow-cycle-guard/ following the anatomy of skills/architecture/dual-graph-router/:
#  SKILL.md (7 sections + Source Attribution to Ch2 "The Horizontal Workflow Graph"), stdlib-only lib.py with detect_cycles(edges) and topo_levels(edges),
#  argparse cli.py with `benchmark` (>= 4 self-checks, exit 1 on failure). No network, no pip installs, do not touch other skills.
# Acceptance: `cli.py --help` exits 0, `benchmark` passes, and topo_levels reproduces the levels Cypher computed in section 6.
# ```
# **Prompt C — a prompt-regression test:**
# ```text
# Write a pytest that loads the 12 labeled routing requests, calls the router, and fails if accuracy drops below the last recorded value (store it in a JSON file).
# Constraint: the LLM call must be injectable so the test can run offline with a recorded fixture.
# ```
#
# **Verification checklist (do all of it, every time):**
# 1. **Run it** — never accept code you have not executed.  2. **Read-only first** — run generated Cypher under a role that cannot write.
# 3. **`EXPLAIN` before `PROFILE`** — check the plan, not just the rows.  4. **Check the acceptance line literally** (the numbers you wrote down before asking).
# 5. **Parameterise** — any value from a user or an LLM goes in `$params`; never f-string it into Cypher.  6. **Look for hidden cost** — cartesian products, unbounded `*` paths.
# 7. **Keep the test, not the chat** — commit the assertion that guards the behaviour.

# %%
PROMPTS[("cypher", "v1")] = "You write Neo4j 5 Cypher. Output ONLY the query text, no markdown, no explanation."
VIBE_PROMPT = ("Context: Neo4j has (:Service {name})-[:DEPENDS_ON]->(x), and (:Database {name}) nodes.\n"
               "Task: ONE read-only Cypher query returning, for every Service, its name and how many distinct Databases it reaches within 3 DEPENDS_ON hops "
               "(use OPTIONAL MATCH so services with none show 0). Order by count desc.\n"
               "Constraints: no writes, no APOC, no markdown fences, return exactly two columns.")
FORBIDDEN = re.compile(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|CALL|LOAD|FOREACH)\b", re.I)

feedback, generated, result_rows = "", None, None
for attempt in range(1, 4):
    raw = h.chat([{"role": "system", "content": PROMPTS[("cypher", "v1")]}, {"role": "user", "content": VIBE_PROMPT + feedback}]).content
    generated = re.sub(r"^```(?:cypher)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        if FORBIDDEN.search(generated):
            raise ValueError("query contains a write/procedure keyword")
        h.run_as(*RET, "EXPLAIN " + generated)                      # does it even compile?
        result_rows = h.run_as(*RET, generated)                      # run it under the read-only role
        break
    except Exception as exc:
        feedback = f"\n\nYour previous query failed: {str(exc)[:200]}\nQuery was:\n{generated}\nFix it."
        print(f"attempt {attempt} rejected: {str(exc)[:100]}")
print("generated Cypher:\n", generated)
counts = {list(r.values())[0]: list(r.values())[1] for r in (result_rows or [])}
print("result:", counts)
h.check("acceptance: checkout-api and cart-service each reach 1 database", counts.get("checkout-api") == 1 and counts.get("cart-service") == 1)
h.check("acceptance: session-cache is not a Service row", "session-cache" not in counts)

# %% [markdown]
# **Worked example you own** — a helper the AI might write badly. Cypher cannot parameterise the `*1..N` bound, so a naive prompt tempts string-building;
# we force `int()` and bound it. The *tests* encode the values you derived by hand in section 2 — they are yours, not the assistant's.

# %%
def blast_radius(component: str, max_hops: int = 3) -> dict:
    n = int(max_hops)                                    # bound cannot be a Cypher parameter; force an int
    if not 1 <= n <= 6:
        raise ValueError("max_hops must be between 1 and 6")
    rows = h.run("MATCH p = (s:Service)-[:DEPENDS_ON*1.." + str(n) + "]->(x {name:$c}) RETURN s.name AS service, min(length(p)) AS hops", c=component)
    return {r["service"]: r["hops"] for r in rows}

assert blast_radius("orders-db", 3) == {"orders-service": 1, "payment-service": 1, "checkout-api": 2, "cart-service": 3}
assert blast_radius("orders-db", 1) == {"orders-service": 1, "payment-service": 1}
try:
    blast_radius("orders-db", 99)
    raise SystemExit("guard missing")
except ValueError:
    pass
sizes = {n: len(blast_radius("orders-db", n)) for n in (1, 2, 3)}
h.check("blast radius grows 2 -> 3 -> 4 with hops, and the input guard works", sizes == {1: 2, 2: 3, 3: 4})

# %% [markdown]
# ## 15. Clean up the real RBAC objects
#
# Users and roles are *server-level* state that survives a notebook run — leaving them behind is how test credentials end up in production.
# We drop them (and the unrelated `:AuditSecret` node) and verify they are gone.

# %%
h.run("MATCH (n:AuditSecret) DETACH DELETE n")
for user in USERS:
    h.sysrun(f"DROP USER {user} IF EXISTS")
for role in ("ch2_retrieval", "ch2_reporting"):
    h.sysrun(f"DROP ROLE {role} IF EXISTS")
left_users = {r["user"] for r in h.sysrun("SHOW USERS YIELD user RETURN user")}
left_roles = {r["role"] for r in h.sysrun("SHOW ROLES YIELD role RETURN role")}
h.check("test users and roles are dropped", not (set(USERS) & left_users) and not ({"ch2_retrieval", "ch2_reporting"} & left_roles))
print("prompt versions used:", sorted({(n, v) for n, v, _ in PROMPT_LOG}))
retried = [p for p in PROMPT_LOG if p[2] > 1]
print("structured calls that needed a retry:", len(retried), "of", len(PROMPT_LOG))
print(h.cost_report())

# %% [markdown]
# ## 16. Evaluation and LLM-as-a-Judge — how you know it works
#
# Everything above *ran*. None of it proves the architecture's decisions are **right**. Evaluation is how you know a
# dual-graph harness works, and how you know a change made it better instead of merely different. Think of it as a
# **pyramid**: cheap, objective, numerous checks at the bottom; expensive, subjective, scarce ones at the top.
#
# | Layer | What | Cost | What it can / cannot do |
# |---|---|---|---|
# | 1. Deterministic checks | Cypher / schema / graph assertions | ~free | Exact, but only for what you can state as a rule |
# | 2. Reference-based metrics | overlap, router accuracy vs a **gold set** | cheap | Objective, but blind to paraphrase and to *why* an answer is wrong |
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
# Scope here: the gold set is about *this chapter's* architecture: routing between the vertical and horizontal graph,
# blast radius and time on the edge, node splitting by tool surface, and RBAC-enforced harness scopes.

# %% [markdown]
# ### 16.1 A gold set, written down and stored in the graph
#
# A gold set is *questions + reference answers + human labels*. Ours has 8 questions whose references come from facts this
# notebook established (hop counts from section 2, the time-on-edge query, the 80% Jaccard rule, the RBAC roles), and
# 10 **candidate answers that we scored by hand (1-5)**: correct ones, a vague-but-right one, clearly wrong ones, *subtly*
# wrong ones (wrong hop count, a prompt-only guardrail) and a **fluffy-but-empty** one. The labels are ours and arguable,
# which is why section 16.4 measures agreement.

# %%
GOLD = [
    {"id": "G1", "question": "Where should the request 'Diagnose why the checkout API is slow: which services does it depend on "
                             "and what changed recently?' be routed, and why?",
     "reference": "To 'both': a diagnosis is a process (the horizontal graph) whose steps need facts from the vertical "
                  "knowledge graph (dependencies and recent deployments). This is where the two graphs meet."},
    {"id": "G2", "question": "If orders-db saturates, how many services are in the blast radius and how far is cart-service?",
     "reference": "Four services are in the blast radius; checkout-api is 2 hops from orders-db and cart-service is 3 hops."},
    {"id": "G3", "question": "As of 2024-02-01, what did checkout-api depend on, and how does the graph know?",
     "reference": "Only payment-service. The DEPENDS_ON edges carry a 'since' property (valid time), so a point-in-time query "
                  "filters on r.since; orders-service was added on 2024-03-01."},
    {"id": "G4", "question": "A proposed new workflow node uses 90% of the same tools as an existing node of the same type. "
                             "Split or merge?",
     "reference": "Merge. Tool overlap of 80% or more (Jaccard) means it is a prompt variation, not a new role; split only "
                  "when the tool surface really differs."},
    {"id": "G5", "question": "Why is a dependency or blast-radius question a graph problem rather than a vector-similarity problem?",
     "reference": "The answer is a chain of typed relationships (DEPENDS_ON paths to orders-db); similar text finds documents "
                  "about databases but cannot traverse dependencies to find upstream impact."},
    {"id": "G6", "question": "Should the retrieval node be allowed to read Incident.root_cause, and how is that enforced?",
     "reference": "No. The ch2_retrieval role has DENY READ on Incident.root_cause, enforced by Neo4j RBAC for that "
                  "identity rather than by a prompt instruction."},
    {"id": "G7", "question": "The agent wants to close an incident using the read-only retriever identity. What happens?",
     "reference": "The write is rejected by Neo4j because the retriever identity has no write privilege; closing needs the "
                  "separate reporting identity and, for real actions, a human approval."},
    {"id": "G8", "question": "Is a keyword router enough to decide vertical vs horizontal for all requests?",
     "reference": "No. A keyword scorer misses paraphrases with no signal word (a lexical gap) and misfires on generic words; "
                  "keep the same route contract but swap in an LLM router and measure it on a labeled set."},
]
CANDIDATES = [  # (id, gold id, answer, HUMAN label 1-5, why we gave that label)
    ("C1", "G1", "Route it to 'both'. It is a diagnosis, so it is a workflow (investigate, decide), but the steps need "
                 "facts from the knowledge graph about dependencies and recent deployments; that is where the two graphs meet.",
     5, "correct, complete, explains the meeting point"),
    ("C2", "G1", "It involves some steps and some facts, so it probably needs both kinds of graph, though I cannot say "
                 "exactly how they connect.", 3, "right direction, thin on mechanism"),
    ("C3", "G1", "Unroutable: 'slow' is just a comment about performance, not something either graph can answer.",
     1, "wrong decision"),
    ("C4", "G2", "Four services are in the blast radius; checkout-api is 1 hop from orders-db and cart-service is 2 hops.",
     2, "subtly wrong: right count, wrong hops"),
    ("C5", "G3", "As of 2024-02-01 only payment-service was a dependency of checkout-api; orders-service came later, on "
                 "2024-03-01. The since property on the DEPENDS_ON edge makes point-in-time queries possible.",
     5, "correct, grounded, explains the mechanism"),
    ("C6", "G5", "Graphs are powerful because they capture rich, interconnected knowledge and enable holistic, "
                 "context-aware insights that unlock the full potential of modern AI systems.",
     1, "fluffy but empty: no fact, no mechanism"),
    ("C7", "G4", "Merge them. 90% overlap is above the 80% Jaccard threshold, so it is a prompt variation rather than "
                 "a new role.", 5, "correct and complete"),
    ("C8", "G6", "No, it should not read it. I would add a line to the system prompt telling the retrieval node never "
                 "to reveal root causes.", 3, "right decision, wrong mechanism (prompted, not enforced)"),
    ("C9", "G7", "It works: the agent decides on its own to close the incident and the tool simply runs, since the "
                 "database trusts whatever the agent sends.", 1, "unsafe and wrong: ignores RBAC"),
    ("C10", "G8", "No. A keyword scorer has lexical gaps (paraphrases with no signal word) and false positives on generic "
                  "words. Keep the vertical/horizontal/both/unroutable contract, swap in an LLM router, and measure both on "
                  "a labeled set.", 5, "correct, grounded in section 4"),
]
GOLD_BY_ID = {g["id"]: g for g in GOLD}
CAND = {c[0]: {"gold": c[1], "answer": c[2], "human": c[3], "why": c[4]} for c in CANDIDATES}
HUMAN = [CAND[c[0]]["human"] for c in CANDIDATES]

# store the gold set in the graph under a namespaced label so it can be queried and cleaned up
h.run("MATCH (g:Ch2EvalGold) DETACH DELETE g")
h.run("UNWIND $rows AS r CREATE (:Ch2EvalGold {id:r.id, question:r.question, reference:r.reference})", rows=GOLD)
stored = h.run("MATCH (g:Ch2EvalGold) RETURN g.id AS id, size(g.reference) AS ref_chars ORDER BY id")
h.show(stored)
h.check("gold set has >= 8 cases stored in the graph", len(stored) >= 8)
h.check("at least 8 human-labelled candidates, covering bad, subtle and fluffy answers",
        len(CANDIDATES) >= 8 and {1, 2, 3, 5} <= set(HUMAN))

# %% [markdown]
# ### 16.2 Layer 1 and 2: deterministic checks, then a reference-based metric
#
# **Layer 1** asserts what must be true of the *artefacts* with Cypher and plain Python: the facts our references cite
# really are in the graph, the router labels are well formed, and no candidate answer names a component the graph does
# not contain (a cheap **hallucination detector**). No LLM, no cost.
#
# **Layer 2** scores each candidate against its reference with token-overlap F1, rescaled to 1-5. It is objective and
# free, but it only sees *words*: watch where it disagrees with the human labels (it cannot tell "1 hop" from "3 hops").

# %%
facts = {
    "blast": len(blast),
    "cycles": h.run("MATCH p = (s:Service)-[:DEPENDS_ON*1..6]->(s) RETURN count(p) AS c")[0]["c"],
    "asof": [r["dependency"] for r in asof],
    "dated_edges": h.run("MATCH (:Service {name:'checkout-api'})-[r:DEPENDS_ON]->() WHERE r.since IS NOT NULL RETURN count(r) AS c")[0]["c"],
    "targets": sorted({gold for _, _, gold in LABELED}),
}
print(facts)
h.check("reference G2 matches the graph (4 services, checkout-api 2 hops, cart-service 3 hops)",
        facts["blast"] == 4 and hops["checkout-api"] == 2 and hops["cart-service"] == 3)
h.check("reference G3 matches the graph (only payment-service as of 2024-02-01)", facts["asof"] == ["payment-service"])
h.check("every checkout-api DEPENDS_ON edge carries a 'since' (time on the edge)", facts["dated_edges"] >= 3)
h.check("DEPENDS_ON is acyclic (blast-radius answers are well defined)", facts["cycles"] == 0)
h.check("the routing gold labels cover all four route targets", facts["targets"] == sorted(TARGETS))

KNOWN = {r["n"] for r in h.run("MATCH (n) WHERE n:Service OR n:Database OR n:Cache RETURN n.name AS n")}
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
overlap_agree = h.agreement(HUMAN, [CAND[c[0]]["f1_score"] for c in CANDIDATES])
print("layer-2 overlap vs human:", overlap_agree)
h.check("layer 1 finds no invented component names in the correct answers",
        not CAND["C1"]["invented"] and not CAND["C5"]["invented"] and not CAND["C7"]["invented"])

# %% [markdown]
# ### 16.3 Layer 3: a rubric judge with explicit 1-vs-5 anchors
#
# Vague rubrics ("is it good?") produce vague, lenient scores. Each criterion below says exactly what a **1** and a **5**
# look like, and the judge sees the **reference**, so it grades against ground truth rather than its own beliefs.
# `h.judge_rubric` asks for a one-sentence reason first and returns per-criterion scores plus their mean.
# We judge all 10 candidates once (10 judge calls) and keep the numbers for calibration.

# %%
CRITERIA = {
    "correctness": "5 = every fact and decision agrees with the reference; 1 = contradicts the reference or gives an unsafe decision.",
    "grounding": "5 = cites specific architecture facts (routes, hop counts, dates, roles, thresholds) that match the reference; "
                 "1 = cites none, or states facts that are not in the reference.",
    "completeness": "5 = covers every point the reference makes; 1 = covers none of them (generic or off-topic).",
    "actionability": "5 = states the correct decision and the right mechanism; 1 = offers no usable decision, or a harmful one.",
}


def to_5(mean: float) -> int:
    return min(5, max(1, int(mean + 0.5)))


for cid, c in CAND.items():
    g = GOLD_BY_ID[c["gold"]]
    c["rubric"] = h.judge_rubric(g["question"], c["answer"], CRITERIA, reference=g["reference"])
    c["judge"] = to_5(c["rubric"]["mean"])
h.show([{"id": cid, "human": c["human"], "judge": c["judge"], **c["rubric"]["scores"]} for cid, c in CAND.items()])
worst = min(CAND.items(), key=lambda kv: kv[1]["rubric"]["mean"])
print(f"lowest-scored candidate {worst[0]}: {worst[1]['rubric']['reasons'].get('correctness', '')[:140]}")
h.check("judge returned a 1-5 score for every candidate", all(1 <= c["judge"] <= 5 for c in CAND.values()))
print(h.cost_report())

# %% [markdown]
# ### 16.4 Calibration: does the judge agree with the humans?
#
# This is the step most teams skip. Compare the judge's 1-5 scores with our human labels on the **same items**:
#
# - **exact / within1**: how often the judge matches us exactly, or is off by at most one point. Within-1 near 1.0 with
#   low exact means "right ordering, different scale".
# - **Cohen's kappa**: exact agreement *corrected for chance*. Rough guide: < 0.2 poor, 0.2-0.4 fair, 0.4-0.6 moderate,
#   0.6-0.8 substantial, > 0.8 near-perfect. Be suspicious of high exact agreement with kappa near 0.
# - **Spearman**: do the two *rank* answers the same way? For picking a winner, this matters more than exact scores.
#
# With only 10 items these numbers are noisy (one item moves exact agreement by 10 points). Treat them as a smoke
# alarm, not a certificate. The check is deliberately **tolerant**: a weak judge is a finding to act on, not a crash.

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
# ### 16.5 Bias probe 1 — position bias
#
# **What this tests:** whether the verdict depends on *which answer is shown first*. `h.judge_pairwise` asks twice with
# the order swapped; if the verdicts disagree it reports `consistent=False` (counted as a tie). Two pairs (4 judge calls):
# a real pair (C1 correct vs C2 vague: the better answer should win in **both** orders) and an **identical** pair
# (the same text twice: the only honest verdict is a tie, so any "winner" is pure position bias).

# %%
PAIR_Q, PAIR_REF = GOLD_BY_ID["G1"]["question"], GOLD_BY_ID["G1"]["reference"]
PAIR_CRIT = "Prefer the answer that is more correct, better grounded in the reference facts, and more actionable."
real_pair = h.judge_pairwise(PAIR_Q, CAND["C1"]["answer"], CAND["C2"]["answer"], PAIR_CRIT, reference=PAIR_REF)
same_pair = h.judge_pairwise(PAIR_Q, CAND["C1"]["answer"], CAND["C1"]["answer"], PAIR_CRIT, reference=PAIR_REF)
print("real pair (C1 vs C2):", real_pair)
print("identical pair      :", same_pair)
print("PASS" if real_pair["winner"] == "a" and real_pair["consistent"] else
      "WARNING: the judge did not clearly and consistently prefer the better answer in both orders.")
print("PASS" if same_pair["winner"] == "tie" and same_pair["consistent"] else
      "WARNING: position bias detected on identical answers; always judge both orders and average, or use absolute rubrics.")
h.check("pairwise judge returned verdicts for both probes", real_pair["winner"] in ("a", "b", "tie") and same_pair["winner"] in ("a", "b", "tie"))

# %% [markdown]
# ### 16.6 Bias probe 2 — verbosity bias
#
# **What this tests:** whether *length and polish* buy points. We take the weak answer C2 and wrap it in confident,
# contentless filler (no new facts). A well-behaved judge scores it about the same; a verbosity-biased judge scores it
# higher. One extra judge call; we compare with the C2 score from 16.3. (A rise of more than 0.75 mean points is flagged.)

# %%
FLUFF = ("Great question, and an important one for any modern engineering organisation. In today's fast-moving, "
         "cloud-native landscape, holistic architecture and a robust, forward-looking, best-practice-driven mindset "
         "are absolutely essential. With that context in mind: ")
FLUFF_TAIL = (" Ultimately, by leveraging synergies across teams and embracing a culture of continuous improvement, "
              "organisations can unlock significant long-term value and operational excellence.")
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
# ### 16.7 Bias probe 3 — self-consistency, and a sabotage control
#
# **Self-consistency.** `h.judge_repeat` scores the *same* answer `n=3` times at temperature 0.7. A `spread` (max - min)
# of 0 or 1 is fine; 2+ means the judge is unreliable *on that item* and a single score should not be trusted. We test
# the mid-quality answer C2 and the prompt-only C8, because borderline answers are where judges wobble (6 judge calls).
#
# **Sabotage control.** Take a good answer and corrupt its *facts* while keeping its style: right shape, wrong content.
# A judge that is checking facts must score it clearly lower than the original. This one is a hard `h.check`:
# if a judge cannot tell a true answer from a fluent lie, none of its other scores mean anything (1 judge call).

# %%
ONE_LINE = ("Score 1-5 against the reference: 5 = correct, specific, grounded in the architecture facts and actionable; "
            "3 = right direction but vague or wrong mechanism; 1 = wrong, invented, or empty.")
rep = {}
for cid in ("C2", "C8"):
    g = GOLD_BY_ID[CAND[cid]["gold"]]
    rep[cid] = h.judge_repeat(g["question"], CAND[cid]["answer"], f"{ONE_LINE} Reference: {g['reference']}", n=3)
    print(f"{cid}: scores={rep[cid]['scores']} mean={rep[cid]['mean']:.2f} spread={rep[cid]['spread']}")
    if rep[cid]["spread"] >= 2:
        print(f"  WARNING: unstable judge on {cid}; average several runs or tighten the rubric for this kind of item.")

SABOTAGED = ("As of 2024-02-01 only inventory-service was a dependency of checkout-api; payment-service came much later, "
             "on 2024-09-01. The since property on the DEPENDS_ON edge makes point-in-time queries possible.")
g5 = GOLD_BY_ID["G3"]
sab = h.judge_rubric(g5["question"], SABOTAGED, CRITERIA, reference=g5["reference"])
print(f"good C5 mean={CAND['C5']['rubric']['mean']:.2f}  sabotaged mean={sab['mean']:.2f}  invented entities: "
      f"{sorted(set(ENTITY_RE.findall(SABOTAGED)) - KNOWN)}")
h.check("sabotaged (fluent but false) answer scores at least 1 point below the good one",
        sab["mean"] <= CAND["C5"]["rubric"]["mean"] - 1)
print(h.cost_report())

# %% [markdown]
# ### 16.8 Judge vs heuristic: where do they disagree, and why?
#
# Section 4 used the skill's keyword scorer to route requests. Here we put the **rules** and the **LLM judge** in front of
# the same 12 requests: the heuristic score is 5 if the skill's route equals the gold route and 1 otherwise; the judge
# grades *the skill's route and rationale* against the gold route (12 judge calls; partial credit allowed for a plausible
# neighbouring route). Rows where they differ by 2+ points are the interesting ones: each is either a heuristic blind
# spot or a judge quirk, and you decide which by reading the request.

# %%
cmp_rows = []
for rid, text, gold in LABELED:
    d = dg.route(text)
    j = h.judge(f"Which route should this request take (vertical / horizontal / both / unroutable)? {text}",
                f"Route: {d.target}. {d.rationale}",
                f"The correct route is '{gold}'. Score 5 if the route is '{gold}', 3 if it is a plausible neighbouring "
                "route (for example 'both' vs 'vertical' when graph facts are involved), 1 if it is clearly wrong or "
                "abstains ('unroutable') on a request that is routable.")
    heur = 5 if d.target == gold else 1
    cmp_rows.append({"id": rid, "gold": gold, "skill": d.target, "heuristic": heur,
                     "judge": int(j["score"]), "judge_reason": j["reason"][:80]})
h.show([{k: v for k, v in r.items() if k != "judge_reason"} for r in cmp_rows])
print("heuristic vs judge:", h.agreement([r["heuristic"] for r in cmp_rows], [r["judge"] for r in cmp_rows]))
diffs = [r for r in cmp_rows if abs(r["heuristic"] - r["judge"]) >= 2]
print(f"{len(diffs)} disagreement(s) of >= 2 points")
for r in diffs:
    if r["heuristic"] < r["judge"]:
        meaning = ("heuristic miss, judge gives partial credit: the skill abstained or chose a neighbouring route, "
                   "so this is a keyword blind spot (or judge leniency toward a near-miss)")
    else:
        meaning = "judge stricter than exact match: the skill hit the gold label but the judge disputes it; re-read the request"
    print(f"  {r['id']} skill={r['skill']} gold={r['gold']} heuristic={r['heuristic']} judge={r['judge']}: {meaning}. "
          f"Judge said: {r['judge_reason']}")
h.check("compared the heuristic and the judge on all 12 requests", len(cmp_rows) == len(LABELED) == 12)

# %% [markdown]
# ### 16.9 Regression harness: change a prompt, prove it did not get worse
#
# Evaluation earns its keep when something *changes*. `h.run_eval` runs a system function over a case list, scores
# each case, and stores `(:EvalRun)-[:HAS_RESULT]->(:EvalResult)` in Neo4j under a name and **version**. Then one
# Cypher query (`h.compare_runs`) compares versions, and a **CI-style gate** turns the comparison into PASS/FAIL.
# Here the system is the LLM router from section 4, prompt `v1` (lazy) vs `v2` (explicit rules), on the 12 labeled
# requests: 24 LLM calls, exact-match scoring, so a flaky judge cannot hide a regression. The gate prints PASS or FAIL
# and never raises, so you can wire its verdict into a pipeline however you like.

# %%
EVAL_NAME = "ch2_routing"
h.run("MATCH (r:EvalRun) WHERE r.name STARTS WITH 'ch2_' OPTIONAL MATCH (r)-[:HAS_RESULT]->(x) DETACH DELETE x, r")
ROUTE_CASES = [{"id": rid, "input": text, "gold": gold} for rid, text, gold in LABELED]


def score_route(case, out):
    ok = out == case["gold"]
    return {"score": 1.0 if ok else 0.0, "passed": ok, "notes": f"gold={case['gold']} got={out}"}


summaries = {}
for ver in ("v1", "v2"):
    router = make_llm_router(ver)
    summaries[ver] = h.run_eval(EVAL_NAME, ROUTE_CASES, lambda c, r=router: r(c["input"]), score_route, version=ver)
    print(f"{ver}: pass_rate={summaries[ver]['pass_rate']:.0%} failures={summaries[ver]['failures']}")
runs = h.compare_runs(EVAL_NAME)
h.show(runs)
h.check("both versions were stored and are comparable in Neo4j", {r["version"] for r in runs} == {"v1", "v2"})

TOLERANCE = 0.10          # candidate may be at most this much worse than baseline (on a 0-1 scale)
by_ver = {r["version"]: r for r in runs}
base, cand = by_ver["v1"]["mean_score"], by_ver["v2"]["mean_score"]
verdict = "PASS" if cand >= base - TOLERANCE else "FAIL"
print(f"CI GATE: baseline v1={base:.2f}  candidate v2={cand:.2f}  tolerance={TOLERANCE:.2f}  ->  {verdict}")
if verdict == "FAIL":
    print("  Regression: inspect (:EvalResult) rows with passed=false for the candidate version before shipping.")
h.show(h.run("MATCH (:EvalRun {name:$n, version:'v2'})-[:HAS_RESULT]->(x:EvalResult) WHERE NOT x.passed "
             "RETURN x.case AS failed_case, x.notes AS notes ORDER BY x.case", n=EVAL_NAME) or [{"failed_case": "none"}])

# %% [markdown]
# ### 16.10 Judge report card, and a prompt to grow the eval set
#
# **What I would trust this judge for** (after reading the calibration, bias and sabotage results above):
#
# - **Ranking and regression-spotting** between versions, using the *same* judge, rubric and reference each time (relative change).
# - **Catching clearly bad and fluent-but-false answers**, because reference-anchored criteria plus the sabotage control passed.
# - **Triage**: deciding which 10% of answers a human should read first.
#
# **What I would not trust it for:**
#
# - **Absolute quality claims** ("our router is 4.3/5") or comparing numbers across different judges or rubrics.
# - **Safety and permissions.** Whether a harness node *can* read `root_cause` is a Neo4j RBAC fact (section 9), never a judge opinion.
# - **Subtle factual errors without a reference** (wrong hop counts, wrong dates): use layer 1 Cypher assertions for those.
# - **Anything after the judge model, rubric or task changes**, until it is re-calibrated on fresh human labels.
# - **Self-preference**: if the judge and the answering model share a family, expect flattery we did not measure here.
#
# **Clean-up for this section:** the next cell removes the `Ch2EvalGold` and `ch2_` `EvalRun` data it created.
#
# **Vibe-coding prompt: extend the eval set**
# ```text
# Context: section "Evaluation and LLM-as-a-Judge" in notebooks_advanced/_src/ch2-architecture-foundations.py, and
# real_helpers.py (judge_rubric, judge_pairwise, judge_repeat, agreement, run_eval, compare_runs).
# Task: add 6 new gold cases and 8 new human-labelled candidates about the eight-pillar readiness map, including two
# subtly wrong answers (a wrong pillar status, a wrong recommended next step) and one fluffy answer. Re-run the calibration.
# Constraints: reuse the CRITERIA dict; <= 20 extra judge calls; every Cypher query parameterised; Neo4j labels
# namespaced with ch2_ / Ch2; do not weaken any existing h.check.
# Acceptance: print agreement (kappa, spearman) before and after; if kappa drops by more than 0.1, list the new
# items with the largest judge-human delta and propose a rubric change. Do not change the human labels to fit the judge.
# ```

# %%
h.run("MATCH (g:Ch2EvalGold) DETACH DELETE g")
h.run("MATCH (r:EvalRun) WHERE r.name STARTS WITH 'ch2_' OPTIONAL MATCH (r)-[:HAS_RESULT]->(x) DETACH DELETE x, r")
left = h.run("MATCH (n) WHERE n:Ch2EvalGold OR (n:EvalRun AND n.name STARTS WITH 'ch2_') RETURN count(n) AS c")[0]["c"]
h.check("evaluation section left no Ch2EvalGold / EvalRun / EvalResult nodes behind", left == 0)
print(h.cost_report())

# %% [markdown]
# ## 17. Cost, limits and what to swap next
#
# **What this notebook made real:** an LLM router and extractor (strict schemas, versioned prompts, retries), 3072-d embeddings on a Neo4j vector index with a fulltext
# index and graph expansion, a function-calling agent with a stored trace graph, and database-enforced RBAC. Every skill's heuristic was run *and* compared with an LLM judgement.
#
# **What is still approximate — be honest about it:**
# * **LLM non-determinism.** `temperature=0` reduces variance, not eliminates it. Our checks assert *behaviour and ranges* (accuracy ≥ 0.7, judge ranks right > wrong), not exact wording.
#   A flaky check usually means the threshold is too tight or the labeled set too small — twelve routing requests and six retrieval questions are a *smoke test*, not a benchmark.
# * **Judge bias.** The LLM judge shares blind spots with the agent (same model family), favours fluent prose, and is noisy (see the re-judge). A negative control and a deterministic
#   coverage score are the minimum; production needs human-labeled samples and a *different* judge model.
# * **The judge is a model too.** It has position, verbosity and leniency biases, and same-family self-preference (the
  judge and the answering model are both `gpt-4o` here, which we flagged but did not measure); the probes in the evaluation
  section reduce these biases but do not remove them.
* **Judge cost is extra calls.** Every judged item is at least one more LLM call (pairwise = two, repeats = n); budget it,
  cap the loops, and judge a sample in CI rather than everything.
* **Re-calibrate periodically.** Re-check the judge against fresh human labels whenever the judge model, rubric or task changes.
* **Small labeled sets.** Accuracy of 0.83 on 12 items has a huge confidence interval. Treat prompt `v1` vs `v2` differences of one item as noise.
# * **Cost.** Every run spends tokens (see the last `cost_report()`); the biggest line items are the retrieval ablation (entity linking + embeddings per question) and the agent loop.
#   Cache embeddings, batch judge calls and gate LLM evaluation behind the free CLI benchmarks.
# * **The graph is tiny.** Six services and ten Searchable nodes make every strategy look good; the *ordering* of strategies, not the absolute scores, is the lesson.
# * **RBAC scope.** We protected labels, one property and one relationship type on a single database. Real deployments also need per-user audit logs, secret rotation, and
#   procedure-level privileges (we avoided calling vector procedures as restricted users on purpose).
# * **Tool-surface merging** is a design heuristic, not a proof of safety: two nodes with disjoint tools can still collude through shared memory slices.
#
# **Swap next (in order of value):** (1) grow the labeled sets and track accuracy per prompt version in a file; (2) add a reranker between retrieval and the LLM;
# (3) run the agent over a batch of incidents and aggregate trace statistics with Cypher; (4) move the judge to a second model; (5) real CloudWatch instead of `MetricPoint` nodes
# (Chapter 6 covers tool orchestration); (6) bi-temporal memory for the agent itself (Chapter 4).
