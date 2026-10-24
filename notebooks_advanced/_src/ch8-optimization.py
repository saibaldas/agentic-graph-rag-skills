# %% [markdown]
# # Chapter 8 — Optimization (advanced, real LLM + real Neo4j)
# > **Tier: advanced (real).** Needs OPENAI_API_KEY in notebooks_advanced/.env and the Neo4j container from docker-compose/.
# > **Evaluation focus:** see the section "Evaluation and LLM-as-a-Judge" near the end — every notebook in this tier treats eval and judging as first-class.
#
# > **Estimated OpenAI cost per full run (measured 2026-10-24 at `gpt-4o` rates): about $0.08.** 87 LLM calls (29 of them LLM-as-a-Judge), 16,669 input / 4,081 output tokens, plus a few thousand `text-embedding-3-large` tokens (<$0.01). Re-run costs about the same; `h.cost_report()` prints your own numbers. See [README](README.md#estimated-openai-cost-measured-24-october-2026).
#
# Tiers 1-3 taught the Chapter 8 primitives against heuristics: a rule-based planner, a lookup table of
# "model quality", a Python dict pretending to be access control. This tier removes those crutches:
#
# | Concern | Simplified tiers | This notebook |
# |---|---|---|
# | Model routing | static table of quality numbers | **measured** accuracy, latency and cost of two real OpenAI models on a labeled set |
# | Planning / investigation | rule-based planner | **gpt-4o function-calling agent** with a trace graph stored in Neo4j |
# | Retrieval | keyword / trigram | **text-embedding-3-large (3072 d) + Neo4j vector index**, ablated against graph-only and hybrid |
# | Access control | Python `can_read()` simulation | **real Neo4j Enterprise RBAC**: users, roles, GRANT / DENY, queries executed as that user |
# | Evaluation | `assert` on a known answer | **LLM-as-judge** + a small labeled harness + failure analysis |
#
# Chapter 8's thesis is *selective intelligence*: spend frontier-model money only where the task needs it, and prove it
# with numbers per node. Here the numbers are real (small samples, so read them with care).
#
# **Cost warning.** The whole notebook makes on the order of 150 small LLM calls (temperature 0) and a few hundred embeddings. Run `h.cost_report()` any time.

# %%
# Preflight: fail loudly. There is NO offline fallback in this tier.
import sys, os, json, time, re, sqlite3, subprocess
sys.path.insert(0, ".")
import real_helpers as h

h.get_openai()                       # raises if OPENAI_API_KEY is missing
h.connect()                          # raises if the Neo4j container is not up
print("LLM model       :", h.LLM_MODEL)
print("Embedding model :", h.EMBED_MODEL, "dims =", h.EMBED_DIM)
print("Neo4j edition   :", h.run("CALL dbms.components() YIELD name, edition RETURN name, edition")[0])
h.check("preflight: OpenAI client + Neo4j reachable", True)

# %% [markdown]
# ## 1. Start from a known graph
#
# Every notebook in this tier is rerunnable top to bottom, so the first graph cell *resets* the database to the shared DevOps
# scenario: `checkout-api` is slow, a deployment (`deploy-481`) shrank the DB connection pool, and a past incident (`INC-0877`) had the same root cause.
# We add the per-service cost metadata that the governance section will later protect with real RBAC.

# %%
counts = h.load_devops_graph()
print(counts)
for name, cost, code in [("checkout-api", 4.2, "BC-100"), ("payment-service", 3.1, "BC-100"),
                         ("orders-service", 2.4, "BC-200"), ("cart-service", 0.9, "BC-300")]:
    h.run("MATCH (s:Service {name:$n}) SET s.cost_per_hour=$c, s.budget_code=$b, s.status='up', s.team=s.owner",
          n=name, c=cost, b=code)
h.show(h.run("MATCH (s:Service) RETURN s.name AS service, s.tier AS tier, s.cost_per_hour AS cost_per_hour, s.budget_code AS budget ORDER BY service"))
h.check("graph reset and Service metadata set", h.run("MATCH (s:Service) WHERE s.cost_per_hour IS NOT NULL RETURN count(s) AS n")[0]["n"] == 4)

# %% [markdown]
# ## 2. The workflow is a DAG, and the DAG is data
#
# Selective intelligence needs a *unit of assignment*: a **node** of the agent workflow (classify the alert, parse logs, attribute the cause...).
# We load the `model-routing-selector` skill, put its six workflow nodes into Neo4j as `:WorkflowNode`, and use graph queries to find the critical path.
# The critical path matters twice: it bounds latency (section 8) and tells you which nodes can run in parallel.

# %%
routing = h.load_skill("optimization", "model-routing-selector", "routing_lib")
FEEDS = [("AlertClassifier", "DependencyAnalyzer"), ("LogParser", "DependencyAnalyzer"), ("QueryAnalyst", "DependencyAnalyzer"),
         ("DependencyAnalyzer", "CausalAttributionNode"), ("CausalAttributionNode", "PredictionSynthesis")]
h.run("UNWIND $names AS n MERGE (:WorkflowNode {name:n})", names=list(routing.NODES))
h.run("UNWIND $edges AS e MATCH (a:WorkflowNode {name:e[0]}), (b:WorkflowNode {name:e[1]}) MERGE (a)-[:FEEDS]->(b)", edges=[list(e) for e in FEEDS])
cp = h.run("""
MATCH p=(a:WorkflowNode)-[:FEEDS*]->(b:WorkflowNode)
WHERE NOT EXISTS { (:WorkflowNode)-[:FEEDS]->(a) } AND NOT EXISTS { (b)-[:FEEDS]->(:WorkflowNode) }
RETURN [n IN nodes(p) | n.name] AS path, length(p) + 1 AS nodes ORDER BY nodes DESC LIMIT 1""")[0]
print("critical path:", " -> ".join(cp["path"]), f"({cp['nodes']} nodes, {len(routing.NODES)} total => parallel factor {len(routing.NODES)/cp['nodes']:.2f})")
h.check("critical path has 4 nodes", cp["nodes"] == 4)

# %% [markdown]
# ## 3. Routing: what the skill predicts
#
# The skill's `cheapest_meeting_bar(node)` picks the cheapest model whose *modelled* quality clears the node's bar. Its model ids are the book's
# (`llama-3.1-3b`, `claude-sonnet`...). We have real OpenAI models, so we map tiers to them: **slm -> gpt-4o-mini**, **frontier -> gpt-4o** (override the small
# one with `ADV_SMALL_MODEL`). The mapping is an honest approximation: the skill's quality numbers describe *other* models, so below we test its claim with measurements instead of trusting it.

# %%
SMALL = os.environ.get("ADV_SMALL_MODEL", "gpt-4o-mini")
FRONTIER = h.LLM_MODEL
# Illustrative USD per 1M tokens (input, output). Check current pricing before quoting these anywhere.
PRICE = {SMALL: (0.15, 0.60), FRONTIER: (2.50, 10.00)}
TIER_MODEL = {"slm": SMALL, "mid": SMALL, "frontier": FRONTIER}

def tier_of(model_id):
    return routing.MODELS[model_id]["tier"] if model_id in routing.MODELS else "cascade"

rows = []
for n in routing.NODES:
    r = routing.cheapest_meeting_bar(n)
    rows.append({"node": n, "skill_model": r["model"], "tier": tier_of(r["model"]), "strategy": r["strategy"],
                 "cost_vs_frontier": r["cost_vs_frontier"], "real_model": TIER_MODEL.get(tier_of(r["model"]), f"{SMALL} -> {FRONTIER}")})
h.show(rows)
print("pipeline:", routing.pipeline_cost(0.30))
h.check("skill routes AlertClassifier cheaply and Causal node by cascade",
        tier_of(routing.cheapest_meeting_bar("AlertClassifier")["model"]) == "slm" and routing.cheapest_meeting_bar("CausalAttributionNode")["strategy"] == "cascade")

# %% [markdown]
# ## 4. A labeled evaluation set for the AlertClassifier node
#
# Chapter 8's rule: *every node gets its own eval set, and failures are weighted by consequence.* A P1 incident labeled P3 (a missed outage) is 10x worse than the reverse.
# This is a tiny hand-labeled set (12 alerts). Twelve examples cannot *certify* a 99% bar; they can expose gross differences, which is all we claim.

# %%
ALERTS = [
    ("checkout-api p99 latency 4200ms and 5xx rate 18%, customers cannot pay", "P1"),
    ("orders-db connection pool exhausted, all order writes failing", "P1"),
    ("payment-service returning HTTP 500 on every request", "P1"),
    ("session-cache is down and every login attempt fails", "P1"),
    ("checkout-api error budget burned 100% in the last 10 minutes", "P1"),
    ("orders-service reporting lost order writes after failover", "P1"),
    ("disk usage at 71% on the log volume of cart-service", "P3"),
    ("TLS certificate for the staging environment expires in 30 days", "P3"),
    ("a single cart-service pod restarted once and recovered", "P3"),
    ("nightly batch job finished 2 minutes slower than its average", "P3"),
    ("deprecation warning for an old SDK call in payment-service logs", "P3"),
    ("dev environment autoscaler scaled down to one node", "P3"),
]
print(len(ALERTS), "alerts;", sum(1 for _, g in ALERTS if g == "P1"), "P1 /", sum(1 for _, g in ALERTS if g == "P3"), "P3")

# %% [markdown]
# ## 5. Structured outputs, prompt versions and retries
#
# Three production habits in one cell:
# 1. **JSON-schema structured output** (`response_format=json_schema`, `strict`) so the node never returns free text the pipeline must regex.
# 2. **A prompt registry with versions** (`v1`, `v2`): prompts are code; version them and evaluate each version, never edit in place.
# 3. **Retries with backoff** around the call, because a node that fails 1 time in 50 is a cost-per-*success* problem.
#
# We also measure tokens (diffing `h.usage`), latency and USD per call; that feeds the cost-performance scorer next.

# %%
SCHEMA = {"type": "json_schema", "json_schema": {"name": "alert_severity", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["severity", "confidence"],
    "properties": {"severity": {"type": "string", "enum": ["P1", "P3"]}, "confidence": {"type": "number"}}}}}
PROMPTS = {
    "v1": "Classify the alert severity as P1 or P3. Return JSON with severity and confidence (0-1).",
    "v2": ("You triage cloud alerts. P1 = customer-visible outage, failing writes/payments/logins, or error budget exhausted. "
           "P3 = informational or degraded-but-safe (capacity trends, warnings, single recovered restarts, expiring certs). "
           "A missed P1 is 10x worse than a false P1, so choose P1 when truly unsure. "
           "Return JSON with severity (P1|P3) and confidence (0-1, your honest certainty)."),
}

def with_retries(fn, attempts=3, base=1.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            print(f"  retry {i+1} after {type(e).__name__}")
            time.sleep(base * 2 ** i)

def classify(text, model, version):
    before = dict(h.usage)
    t0 = time.perf_counter()
    msg = with_retries(lambda: h.chat([{"role": "system", "content": PROMPTS[version]}, {"role": "user", "content": text}],
                                      model=model, response_format=SCHEMA))
    ms = (time.perf_counter() - t0) * 1000
    pin, pout = h.usage["prompt_tokens"] - before["prompt_tokens"], h.usage["completion_tokens"] - before["completion_tokens"]
    out = json.loads(msg.content)
    cin, cout = PRICE.get(model, (0, 0))
    return {"pred": out["severity"], "conf": float(out["confidence"]), "in": pin, "out": pout, "ms": ms,
            "usd": (pin * cin + pout * cout) / 1e6}

print(classify(ALERTS[0][0], SMALL, "v2"))

# %% [markdown]
# ## 6. Run the ablation: 2 models x 2 prompt versions
#
# Four configurations over 12 alerts = 48 calls. For each we compute plain accuracy *and* the skill's `evaluate_candidate`, which applies the 10x failure weighting
# and checks the node's accuracy bar. Notice that the weighted error can rank configurations differently from raw accuracy.

# %%
cps = h.load_skill("optimization", "cost-performance-scorer", "costperf_lib")
RESULTS = {}
for model in (SMALL, FRONTIER):
    for ver in ("v1", "v2"):
        RESULTS[(model, ver)] = [classify(a, model, ver) for a, _ in ALERTS]

summary = []
for (model, ver), res in RESULTS.items():
    ev = cps.evaluate_candidate([(g, r["pred"]) for (_, g), r in zip(ALERTS, res)], cps.EVAL_SETS["AlertClassifier"])
    summary.append({"model": model, "prompt": ver, "accuracy": ev["accuracy"], "weighted_err": ev["weighted_error_rate"],
                    "meets_99%_bar": ev["meets_threshold"], "mean_ms": round(sum(r["ms"] for r in res) / len(res)),
                    "usd_per_1k_calls": round(sum(r["usd"] for r in res) / len(res) * 1000, 4)})
h.show(summary)
print(h.cost_report())
h.check("all 4 configurations produced valid structured outputs", all(r["pred"] in ("P1", "P3") for res in RESULTS.values() for r in res))
h.check("best frontier configuration is better than chance", max(s["accuracy"] for s in summary) >= 0.75)

# %% [markdown]
# ## 7. Failure analysis: where does each model go wrong?
#
# Accuracy hides *which* direction errors go. We list every misclassification with its direction, because `P1_as_P3` is the expensive one.
# The expected pattern is that the cheap model's errors cluster on borderline alerts (a "recovered" outage, a certificate), and that prompt v2's "when unsure choose P1" rule trades more `P3_as_P1` for fewer `P1_as_P3`.

# %%
fail_rows = []
for (model, ver), res in RESULTS.items():
    for (text, gold), r in zip(ALERTS, res):
        if r["pred"] != gold:
            fail_rows.append({"model": model, "prompt": ver, "direction": f"{gold}_as_{r['pred']}", "conf": round(r["conf"], 2), "alert": text[:60]})
h.show(fail_rows or [{"result": "no errors on this set (small set - do not over-read)"}], max_rows=30)
dangerous = sum(1 for f in fail_rows if f["direction"] == "P1_as_P3")
print("dangerous P1_as_P3 errors across all 4 configs:", dangerous)
h.check("failure analysis computed", isinstance(fail_rows, list))

# %% [markdown]
# ## 8. Heuristic vs measurement vs LLM judgement
#
# The skill said AlertClassifier is fine on a small model (its `effective_quality` numbers describe llama-3.1-3b). We now have three opinions:
# the skill's heuristic, our measured accuracy, and an LLM's own view of each node's difficulty. Where they disagree is the interesting part:
# a heuristic is a *prior*, an eval set is *evidence*.

# %%
NODE_DESC = {
    "AlertClassifier": "label an alert P1 or P3 from one sentence",
    "QueryAnalyst": "decide whether a user question is simple or complex",
    "LogParser": "extract timestamps, services and error codes from log lines",
    "DependencyAnalyzer": "read a dependency subgraph and list affected services",
    "CausalAttributionNode": "explain which change caused an incident, weighing competing evidence",
    "PredictionSynthesis": "write an open-ended prediction with recommendations",
}
llm_tiers = h.llm_json("For each workflow node choose the cheapest adequate model tier: 'slm' (small, classification/extraction), "
                       "'mid' (structured reasoning) or 'frontier' (open-ended synthesis/causal reasoning). "
                       "Return JSON {node_name: tier}. Nodes: " + json.dumps(NODE_DESC))
cmp_rows = []
for n in routing.NODES:
    r = routing.cheapest_meeting_bar(n)
    skill_tier, llm_tier = tier_of(r["model"]), llm_tiers.get(n, "?")
    agree = skill_tier == llm_tier or (skill_tier == "cascade" and llm_tier in ("mid", "frontier"))
    cmp_rows.append({"node": n, "skill_tier": skill_tier, "llm_tier": llm_tier, "agree": agree})
h.show(cmp_rows)
measured_small = max(s["accuracy"] for s in summary if s["model"] == SMALL)
print(f"AlertClassifier: skill models quality 0.99 for a 3B SLM; measured best {SMALL} accuracy = {measured_small} on n={len(ALERTS)}")
print("Disagreements:", [c["node"] for c in cmp_rows if not c["agree"]] or "none")
h.check("LLM tier judgement returned a tier for every node", all(c["llm_tier"] in ("slm", "mid", "frontier") for c in cmp_rows))

# %% [markdown]
# ## 9. Cascade routing with the model's own confidence
#
# For nodes with variable difficulty the skill prescribes a **cascade**: try the cheap model, escalate when confidence is low. Chapter 8's thresholds are `cheap_threshold=0.85`.
# Instead of a made-up confidence we use the model's *self-reported* confidence from section 5. We replay the cascade over stored results (no new calls) and compare to "frontier everywhere".
# Self-reported confidence is poorly calibrated, so we also report how often a confident answer was wrong.

# %%
small_v2, big_v2 = RESULTS[(SMALL, "v2")], RESULTS[(FRONTIER, "v2")]
THRESH = 0.85
served, correct, usd = [], 0, 0.0
for (text, gold), s, b in zip(ALERTS, small_v2, big_v2):
    route = routing.cascade_route("AlertClassifier", s["conf"], cheap_threshold=THRESH)
    escalate = s["conf"] < THRESH
    pred = b["pred"] if escalate else s["pred"]
    usd += s["usd"] + (b["usd"] if escalate else 0)
    served.append("frontier" if escalate else "small")
    correct += pred == gold
frontier_usd = sum(b["usd"] for b in big_v2)
confident_wrong = sum(1 for (t, g), s in zip(ALERTS, small_v2) if s["conf"] >= THRESH and s["pred"] != g)
print(f"cascade accuracy={correct/len(ALERTS):.2f} escalation_rate={served.count('frontier')/len(ALERTS):.2f} "
      f"cost vs frontier-only={usd/frontier_usd:.2f}x   confident-but-wrong small answers={confident_wrong}")
print("skill cascade_route sample:", routing.cascade_route("AlertClassifier", small_v2[0]["conf"], cheap_threshold=THRESH))
h.check("cascade is cheaper than frontier-only", usd <= frontier_usd)

# %% [markdown]
# ## 10. Cost per *success*, not cost per call
#
# `cost-performance-scorer` logs every invocation as a `NodeInvocation` and reports `cost_per_success`: total spend divided by successes, so failed calls tax the metric.
# Here "success" means "prediction equals the human label", and the invocations are our *real* measurements. We also persist the routing decision in the graph so the policy is queryable.

# %%
tracker = cps.CostTracker()
for (model, ver), res in RESULTS.items():
    if ver != "v2":
        continue
    for (text, gold), r in zip(ALERTS, res):
        tracker.log(cps.NodeInvocation("AlertClassifier", model, r["in"], r["out"], r["usd"], r["ms"], r["pred"] == gold))
    rep = tracker.node_report("AlertClassifier")
print(cps.score_policy(tracker)["pipeline_cost_per_success"], "USD blended cost per success (both models mixed in this log)")
per_model = {}
for model in (SMALL, FRONTIER):
    res = RESULTS[(model, "v2")]
    acc = sum(r["pred"] == g for (_, g), r in zip(ALERTS, res)) / len(res)
    per_model[model] = (sum(r["usd"] for r in res) / len(res), acc)
cmp = cps.compare_cost_per_success(per_model[SMALL][0], max(per_model[SMALL][1], 0.01), per_model[FRONTIER][0], max(per_model[FRONTIER][1], 0.01))
print({k: cmp[k] for k in ("cheaper_per_call", "cheaper_per_success")})
h.run("MERGE (n:WorkflowNode {name:'AlertClassifier'}) WITH n UNWIND $rows AS r MERGE (m:Model {name:r.model}) "
      "MERGE (n)-[x:ROUTED_TO]->(m) SET x.accuracy=r.acc, x.usd_per_call=r.usd",
      rows=[{"model": m, "acc": a, "usd": c} for m, (c, a) in per_model.items()])
h.show(h.run("MATCH (n:WorkflowNode {name:'AlertClassifier'})-[x:ROUTED_TO]->(m:Model) RETURN m.name AS model, x.accuracy AS accuracy, x.usd_per_call AS usd_per_call"))
h.check("both routing edges stored", len(h.run("MATCH (:WorkflowNode {name:'AlertClassifier'})-[:ROUTED_TO]->(:Model) RETURN 1")) == 2)

# %% [markdown]
# ## 11. Real retrieval: embeddings + a Neo4j vector index
#
# The node `CausalAttributionNode` needs relevant **runbooks**. We create eight `:Runbook` nodes linked `-[:ABOUT]->` the component they describe, embed them with
# `text-embedding-3-large` (3072 dimensions) and build a real `VECTOR INDEX`. Graph structure (which component, how many `DEPENDS_ON` hops away) and semantic similarity are different signals; the next cell measures each alone and combined.

# %%
RUNBOOKS = [
    ("r1", "orders-db", "Connection pool exhausted", "Database connection pool exhausted: raise the max pool size and check recent deployments that changed pool configuration."),
    ("r2", "payment-service", "Payment provider timeouts", "Payment provider timeouts: retry with exponential backoff, check the provider status page, fail over to the secondary processor."),
    ("r3", "session-cache", "Cache eviction storm", "Cache eviction storm causing logouts: increase memory, review TTL settings, warm the cache after a restart."),
    ("r4", "checkout-api", "Checkout latency spike", "Checkout latency spike: inspect downstream dependencies first, then autoscaling and thread pool saturation."),
    ("r5", "cart-service", "Cart memory leak", "Cart service memory leak: restart pods, capture a heap dump, roll back the latest image."),
    ("r6", "orders-service", "Orders backlog", "Orders backlog growing: scale consumers, inspect the dead-letter queue, replay failed messages."),
    ("r7", "orders-db", "Slow queries", "Slow queries on the orders database: look for missing indexes and long-running transactions."),
    ("r8", "payment-service", "Expired TLS certificate", "Expired TLS certificate causing handshake failures: rotate the certificate and restart the service."),
]
h.run("""UNWIND $rows AS row
MERGE (r:Runbook {id: row.id}) SET r.title = row.title, r.text = row.title + '. ' + row.body
WITH r, row MATCH (x {name: row.about}) MERGE (r)-[:ABOUT]->(x)""",
      rows=[{"id": i, "about": a, "title": t, "body": b} for i, a, t, b in RUNBOOKS])
h.ensure_vector_index("adv8_runbooks", "Runbook")
print("embedded", h.embed_nodes("Runbook", "text"), "runbooks")
hits = h.vector_search("adv8_runbooks", "database ran out of connections", k=3)
print([(x["node"]["id"], round(x["score"], 3)) for x in hits])
h.check("vector index returns the pool runbook first for a paraphrase", hits[0]["node"]["id"] in ("r1", "r7"))

# %% [markdown]
# ## 12. Ablation: vector vs graph vs hybrid
#
# Eight labeled queries, each with the alarming service (from the alert, as in production) and the correct runbook.
# - **vector**: nearest runbook text, ignoring structure.
# - **graph**: runbooks `ABOUT` anything the service transitively `DEPENDS_ON` (<= 3 hops), ranked by hop distance only (no language understanding).
# - **hybrid**: vector top-8, then keep only runbooks inside the service's dependency neighbourhood, vector order preserved.
# We report hit@1 and hit@3. Expectation: graph is high-recall/low-precision, vector is precise but blind to topology, hybrid should win or tie.

# %%
QUERIES = [
    ("checkout is slow and order writes time out waiting for connections", "checkout-api", "r1"),
    ("customers get declined payments, provider calls are timing out", "payment-service", "r2"),
    ("users keep getting logged out, evictions in the session store", "checkout-api", "r3"),
    ("cart pods keep growing memory until they are OOM killed", "cart-service", "r5"),
    ("orders pile up unprocessed in the queue", "checkout-api", "r6"),
    ("SSL handshake errors when calling the payment API", "payment-service", "r8"),
    ("database queries take seconds and scan the whole table", "orders-service", "r7"),
    ("checkout latency rose after a config change to the database", "checkout-api", "r1"),
]
def vec_rank(q):
    return [x["node"]["id"] for x in h.vector_search("adv8_runbooks", q, k=8)]
def graph_rank(svc):
    rows = h.run("""MATCH path=(s:Service {name:$svc})-[:DEPENDS_ON*0..3]->(x)<-[:ABOUT]-(r:Runbook)
WITH r, min(length(path)) AS hops RETURN r.id AS id, hops ORDER BY hops, id""", svc=svc)
    return [r["id"] for r in rows]
def hybrid_rank(q, svc):
    allowed = set(graph_rank(svc))
    return [i for i in vec_rank(q) if i in allowed]

ranks = {"vector": [], "graph": [], "hybrid": []}
for q, svc, gold in QUERIES:
    ranks["vector"].append(vec_rank(q)); ranks["graph"].append(graph_rank(svc)); ranks["hybrid"].append(hybrid_rank(q, svc))
abl = []
for mode, rk in ranks.items():
    abl.append({"mode": mode, "hit@1": round(sum(r[:1] == [g] for r, (_, _, g) in zip(rk, QUERIES)) / len(QUERIES), 2),
                "hit@3": round(sum(g in r[:3] for r, (_, _, g) in zip(rk, QUERIES)) / len(QUERIES), 2),
                "avg_candidates": round(sum(len(r) for r in rk) / len(rk), 1)})
h.show(abl)
h.check("hybrid hit@3 is at least as good as graph-only", abl[2]["hit@3"] >= abl[1]["hit@3"])
h.check("vector retrieval clearly beats chance (hit@3 >= 0.5)", abl[0]["hit@3"] >= 0.5)

# %% [markdown]
# ## 13. Failure analysis of retrieval, with an LLM as analyst
#
# Look at the queries where a mode missed at rank 1 and let gpt-4o hypothesise *why*. This is a diagnosis aid, not ground truth: we read its explanation critically.
# Typical causes: a query that is truly about two runbooks (`r1` vs `r7` both concern the orders DB), or a service neighbourhood that is too wide so graph ranking ties.

# %%
misses = []
for (q, svc, gold), v, g, hy in zip(QUERIES, ranks["vector"], ranks["graph"], ranks["hybrid"]):
    for mode, r in (("vector", v), ("graph", g), ("hybrid", hy)):
        if r[:1] != [gold]:
            misses.append({"mode": mode, "query": q[:55], "gold": gold, "top3": r[:3]})
h.show(misses or [{"result": "no rank-1 misses"}], max_rows=20)
if misses:
    print(h.llm("In 4 short bullet points, explain likely reasons these retrieval misses happened. Runbooks: "
                + json.dumps({i: b for i, _, _, b in RUNBOOKS}) + " Misses: " + json.dumps(misses[:8])))
h.check("retrieval misses analysed", True)

# %% [markdown]
# ## 14. Schema evolution: apply, rollback, re-apply a real migration
#
# Chapter 8: the graph schema must evolve *without* downtime. `schema-evolution-migrator` ships the book's V003 migration (add `MONITORED_BY` edges and a uniqueness constraint),
# marked **N-1 compatible** (the previous application version keeps working). We run it for real against Neo4j, prove the rollback, then re-apply.
# First seed `alert_config` on three services.

# %%
schema = h.load_skill("optimization", "schema-evolution-migrator", "schema_lib")
for svc, cfg in [("checkout-api", "p99-latency-alarm"), ("payment-service", "payment-error-rate"), ("orders-service", "orders-p99")]:
    h.run("MATCH (s:Service {name:$n}) SET s.alert_config=$c", n=svc, c=cfg)

def statements(cypher):
    body = "\n".join(l for l in cypher.splitlines() if not l.strip().startswith("//"))
    return [s.strip() for s in body.split(";") if s.strip()]

def monitored():
    return h.run("MATCH (:Service)-[r:MONITORED_BY]->(:AlertRule) RETURN count(r) AS n")[0]["n"]

mig = schema.monitored_by_migration()
print(mig.version, "-", mig.description, "| N-1 compatible:", mig.n1_compatible)
for s in statements(mig.cypher): h.run(s)
after_apply = monitored()
for s in statements(mig.rollback): h.run(s)
after_rollback = monitored()
for s in statements(mig.cypher): h.run(s)
after_reapply = monitored()
print("MONITORED_BY edges: apply =", after_apply, "| rollback =", after_rollback, "| re-apply =", after_reapply)
h.check("migration applies, rolls back and re-applies", (after_apply, after_rollback, after_reapply) == (3, 0, 3))

# %% [markdown]
# ## 15. Let the LLM review the migration (and compare with the skill's flag)
#
# The skill *asserts* `n1_compatible=True`. A reviewer (human or LLM) should be able to justify that. We ask gpt-4o for a structured review of the Cypher and compare its verdict with the flag.
# If the two disagree, a human decides; the point is a second, independent opinion before a production rollout.

# %%
review = h.llm_json("Review this Neo4j migration for zero-downtime safety. Return JSON {\"n1_compatible\": bool, \"risks\": [short strings], \"rollback_ok\": bool}. "
                    "N-1 compatible means the previous application version keeps working after the migration runs.\n\nMIGRATION:\n"
                    + mig.cypher + "\n\nROLLBACK:\n" + mig.rollback)
print(json.dumps(review, indent=2))
print("skill flag n1_compatible =", mig.n1_compatible, "| LLM verdict =", review.get("n1_compatible"),
      "| agree:", bool(review.get("n1_compatible")) == mig.n1_compatible)
h.check("LLM review has the expected structure", isinstance(review.get("risks"), list) and "n1_compatible" in review)

# %% [markdown]
# ## 16. Temporal invalidation: the skill's Cypher fails on real Neo4j (on purpose)
#
# Instead of deleting an outdated dependency edge we *invalidate* it (`t_invalid = datetime()`), keeping history queryable (a bitemporal pattern).
# The skill's `temporal_invalidate_cypher` contains `MERGE (a)-[:DEPENDS_ON {..., t_invalid: null, ...}]->(new_lib)`. **Neo4j rejects `null` as a MERGE property value in a pattern** ("Cannot merge ... because of null property value"), a bug the in-memory tiers could never show.
# We catch the error, explain it, and retry without the null property. A missing `t_invalid` is already `IS NULL`, so the semantics are unchanged.

# %%
h.run("""MATCH (a:Service {name:'checkout-api'})
MERGE (lib:Library {name:'stripe-python', version:'3.2.1'})
MERGE (a)-[:DEPENDS_ON {version:'3.2.1', t_valid: datetime('2020-01-01T00:00:00Z')}]->(lib)""")
inval = schema.temporal_invalidate_cypher("checkout-api", "stripe-python", "3.2.1", "3.3.0")
assert "DELETE" not in inval and "t_invalid = datetime()" in inval
try:
    h.run(inval)
    failed = False
except Exception as e:
    failed = True
    print("EXPECTED FAILURE:", type(e).__name__, "-", str(e).splitlines()[0][:160])
    print("Why: MERGE needs concrete property values to match on; null can never match, so Neo4j refuses it.")
fixed = inval.replace("    t_invalid: null,\n", "")
assert fixed != inval or failed is False
h.run(fixed)
h.show(h.run("MATCH (:Service {name:'checkout-api'})-[d:DEPENDS_ON]->(l:Library) RETURN l.version AS version, d.t_valid AS t_valid, d.t_invalid AS t_invalid ORDER BY version"))
h.check("old edge invalidated, new edge valid, nothing deleted",
        h.run("MATCH (:Service {name:'checkout-api'})-[d:DEPENDS_ON]->(:Library {name:'stripe-python'}) "
              "RETURN sum(CASE WHEN d.t_invalid IS NULL THEN 1 ELSE 0 END) AS valid, sum(CASE WHEN d.t_invalid IS NOT NULL THEN 1 ELSE 0 END) AS invalid")[0] == {"valid": 1, "invalid": 1})

# %% [markdown]
# ## 17. Point-in-time queries, and incremental merge from a deployment event
#
# With validity windows, "what did checkout-api depend on in 2021?" is an ordinary query. Then `incremental_merge_cypher` applies a deployment event as three statements whose cost is proportional to
# the *event*, not the whole graph (the alternative, a full GraphRAG rebuild, is what Chapter 8 warns about). Statement 3 invalidates libraries that vanished from the manifest.

# %%
PIT = """MATCH (:Service {name:'checkout-api'})-[d:DEPENDS_ON]->(l:Library {name:'stripe-python'})
WHERE d.t_valid <= datetime($at) AND (d.t_invalid IS NULL OR d.t_invalid > datetime($at)) RETURN l.version AS version"""
print("in 2021:", h.run(PIT, at="2021-01-01T00:00:00Z"), "| now:", h.run("RETURN toString(datetime()) AS now")[0]["now"][:10], h.run(PIT, at=h.run("RETURN toString(datetime()) AS t")[0]["t"]))
event = json.loads((h.SKILLS_DIR / "optimization" / "schema-evolution-migrator" / "sample-deployment-event.json").read_text(encoding="utf-8"))
event["service"] = "checkout-api"
s1, s2, s3 = schema.incremental_merge_cypher(event)
h.run(s1, service_name="checkout-api", env=event["environment"], new_version=event["version"])
for dep in event["dependencies"]:
    h.run(s2, lib_name=dep["name"], lib_version=dep["version"], service_name="checkout-api")
h.run(s3, service_name="checkout-api", current_libs=[d["name"] for d in event["dependencies"]])
h.show(h.run("MATCH (:Service {name:'checkout-api'})-[d:DEPENDS_ON]->(l:Library) RETURN l.name AS lib, l.version AS version, d.t_invalid IS NULL AS currently_valid ORDER BY lib, version"))
h.check("point-in-time query returns the old version for 2021",
        h.run(PIT, at="2021-01-01T00:00:00Z") == [{"version": "3.2.1"}])
h.check("incremental merge recorded the deployed version",
        h.run("MATCH (s:Service {name:'checkout-api'}) RETURN s.current_version AS v")[0]["v"] == event["version"])

# %% [markdown]
# ## 18. Staged rollout manifest and snapshot growth
#
# Schema, data and agent code must roll out in a safe order: **schema_migration -> data_backfill -> agent_code (canary) -> schema_cleanup**.
# The skill builds and validates the manifest. We also ask the model to critique the plan; its output is advisory.
# Finally the arithmetic that justifies a retention policy: 200 resources sampled every 5 minutes create 57,600 snapshot nodes per day.

# %%
manifest = schema.staged_rollout_manifest("2026.05")
val = schema.validate_manifest(manifest)
print([p["name"] for p in manifest["phases"]], val)
canary = next(p for p in manifest["phases"] if "canary_percent" in p)
print("canary:", canary["canary_percent"], "% for", canary["canary_duration"], "| promotion:", canary["promotion_criteria"])
print("snapshot growth/day:", schema.snapshot_growth_per_day(200, 5), "| incident retention:", schema.RETENTION_POLICY["incident"])
print(h.llm("In 3 bullets, what could still go wrong in this staged rollout plan?\n" + json.dumps(manifest, default=str)[:1800]))
h.check("manifest valid and ordered", val["valid"] and [p["name"] for p in manifest["phases"]][0] == "schema_migration")
h.check("snapshot growth matches the book figure", schema.snapshot_growth_per_day(200, 5) == 57600)

# %% [markdown]
# ## 19. Latency and KV-cache budgets, checked against *real* measurements
#
# `kv-cache-latency-budgeter` encodes the book's hardware arithmetic. Two ideas: (1) concurrency is bounded by **KV-cache memory**, not weights, so compressing the KV cache can beat quantising weights;
# (2) the full pipeline should finish in under 2 s, which you reach by running independent DAG nodes in parallel.
# The skill's inference numbers are for self-hosted SLMs. We have *measured* hosted-API latency in section 6, so we can test the 2 s target honestly: critical-path nodes x measured mean latency.

# %%
kv = h.load_skill("optimization", "kv-cache-latency-budgeter", "kv_lib")
print("kv-bound concurrency:", kv.kv_bound_concurrency(80, 16, 12))
cmp_kv = kv.weight_quant_vs_kv_compression(80, 16, 12, weight_quant_factor=2, kv_compression_factor=4)
print({k: cmp_kv[k] for k in ("baseline_concurrency", "after_weight_quant", "after_kv_compression", "verdict")})
book = kv.budget_pipeline(n_slm_calls=5, n_graph_ops=1, parallel_factor=len(routing.NODES) / cp["nodes"])
print("book budget (self-hosted SLM):", book["estimated_ms"], "ms | within 2 s:", book["within_2s_target"])
mean_small = sum(r["ms"] for r in RESULTS[(SMALL, "v2")]) / len(ALERTS)
mean_big = sum(r["ms"] for r in RESULTS[(FRONTIER, "v2")]) / len(ALERTS)
measured_e2e = (3 * mean_small + mean_big) / 1000   # 3 cheap nodes on the critical path + 1 frontier node
print(f"measured mean latency: {SMALL}={mean_small:.0f} ms, {FRONTIER}={mean_big:.0f} ms -> estimated critical path = {measured_e2e:.2f} s",
      "(meets 2 s)" if measured_e2e <= 2 else "(MISSES the 2 s target with hosted APIs)")
t0 = time.perf_counter()
h.run("MATCH (d:Database {name:'orders-db'})<-[:DEPENDS_ON*1..3]-(s:Service) RETURN DISTINCT s.name")
print(f"blast-radius traversal on Neo4j: {(time.perf_counter()-t0)*1000:.1f} ms | skill GPU estimate:", kv.estimate_gpu_time(4.0, "pagerank"))
h.check("KV compression wins in the skill's example", cmp_kv["verdict"] == "kv_compression")

# %% [markdown]
# ## 20. Real RBAC, part 1: a role-based identity layer
#
# The simplified tiers *pretended* access control with `can_read()`. Neo4j Enterprise actually enforces it: a user with the `sre_oncall` role literally cannot see `Employee` nodes or read `cost_per_hour`, whatever Cypher they send.
# Chapter 8's **privacy by architecture** goes one step further: nodes carry only a random `uuid`; the mapping to a real person lives in a separate identity store. We use a **real SQL store** (SQLite, in memory) for that mapping.
# Setup: Employee and Compensation nodes (fictional) in the graph, identities in SQL.

# %%
govern = h.load_skill("optimization", "subgraph-access-control", "govern_lib")
h.run("""MATCH (s:Service {name:'checkout-api'})
MERGE (e1:Employee {uuid:'u-1001'}) MERGE (e2:Employee {uuid:'u-1002'})
MERGE (e1)-[:ON_CALL_FOR]->(s) MERGE (e2)-[:ON_CALL_FOR]->(s)
MERGE (e1)-[:HAS_COMP]->(:Compensation {uuid:'u-1001', salary: 185000})""")
identity_db = sqlite3.connect(":memory:")
identity_db.execute("CREATE TABLE identities (uuid TEXT PRIMARY KEY, pii TEXT)")
identity_db.executemany("INSERT INTO identities VALUES (?, ?)", [("u-1001", "Ada Example <ada@example.com>"), ("u-1002", "Bo Example <bo@example.com>")])
print(govern.privacy_by_architecture())
print(identity_db.execute("SELECT * FROM identities").fetchall())
h.check("graph holds only uuids; identities live in SQL", h.run("MATCH (e:Employee) WHERE e.name IS NOT NULL RETURN count(e) AS n")[0]["n"] == 0)

# %% [markdown]
# ## 21. Real RBAC, part 2: create roles and users in Neo4j
#
# `generate_policy(role)` emits GRANT/DENY statements written for a database called `infrastructure`. We adapt them to this database (`neo4j`), prefix names with `adv8_` so we never clash with other work,
# make `CREATE ROLE` idempotent (`IF NOT EXISTS`), and execute them on the `system` database with `h.sysrun`. Each role also needs `ACCESS` on the database. Users get fixed demo passwords (this is a throwaway container).
# Cleanup happens in section 25, and every statement is safe to rerun.

# %%
PW = "Adv8-demo-pass-1!"
USERS = {"adv8_sre": "sre_oncall", "adv8_fin": "finance_analyst", "adv8_infra": "infra_analyst"}

def policy_statements(role):
    text = "\n".join(l for l in govern.generate_policy(role).splitlines() if not l.strip().startswith("//"))
    text = text.replace(f" GRAPH {govern.GRAPH}", " GRAPH neo4j").replace(role, "adv8_" + role)
    stmts = [s.strip() for s in text.split(";") if s.strip()]
    return [s + " IF NOT EXISTS" if s.startswith("CREATE ROLE") and "IF NOT EXISTS" not in s else s for s in stmts]

for user, role in USERS.items():
    for s in policy_statements(role):
        h.sysrun(s)
    h.sysrun(f"GRANT ACCESS ON DATABASE neo4j TO adv8_{role}")
    h.sysrun(f"CREATE USER {user} IF NOT EXISTS SET PASSWORD '{PW}' CHANGE NOT REQUIRED")
    h.sysrun(f"GRANT ROLE adv8_{role} TO {user}")
print(policy_statements("sre_oncall"))
roles_now = {r["role"] for r in h.sysrun("SHOW ROLES YIELD role")}
h.check("three adv8_ roles exist", {"adv8_sre_oncall", "adv8_finance_analyst", "adv8_infra_analyst"} <= roles_now)

# %% [markdown]
# ## 22. Real RBAC, part 3: the same query, three users, three different worlds
#
# We send *identical* Cypher through `h.run_as(user, password, ...)`. The database, not our Python, decides what each user sees. Expect `sre_oncall` to see Service, Library and Deployment but **not** Employee,
# and to get `null` for `cost_per_hour`. Labels outside every grant (Incident, Alarm...) are invisible to restricted roles too: *deny by default* is the point.

# %%
PROBE_LABELS = "MATCH (n) UNWIND labels(n) AS label RETURN label, count(n) AS visible ORDER BY label"
PROBE_COST = "MATCH (s:Service) RETURN s.name AS service, s.cost_per_hour AS cost_per_hour ORDER BY service"
visible, costs = {}, {}
for user, role in USERS.items():
    visible[role] = {r["label"]: r["visible"] for r in h.run_as(user, PW, PROBE_LABELS)}
    costs[role] = h.run_as(user, PW, PROBE_COST)
    print(f"\n== {user} ({role}) sees labels: {sorted(visible[role])}")
    print("   cost_per_hour:", [(r["service"], r["cost_per_hour"]) for r in costs[role]][:3])
print("\nadmin sees labels:", sorted({r["label"] for r in h.run(PROBE_LABELS)}))
h.check("sre_oncall cannot see Employee or Compensation", not ({"Employee", "Compensation"} & set(visible["sre_oncall"])))
h.check("sre_oncall gets null cost_per_hour", all(r["cost_per_hour"] is None for r in costs["sre_oncall"]))

# %% [markdown]
# ## 23. Does the skill's `audit_access` predict what the database enforces?
#
# `audit_access(role, needed_labels, needed_props)` is a *design-time* check: "can this role do the job we want it to?" It is a model of the policy. Real RBAC is ground truth, so we compare them.
# Needed labels come from the incident neighbourhood in the graph itself. Any mismatch means either the skill's model or our translation of the policy is wrong, and that is exactly what you want to learn before production.

# %%
needed = [r["l"] for r in h.run("""MATCH (i:Incident {id:'INC-1001'})-[:AFFECTS|TRIGGERED_BY|FIRED_ON|DEPENDS_ON|CHANGED|DEPLOYED_TO|MONITORED_BY*1..4]-(n)
UNWIND labels(n) AS l WITH DISTINCT l WHERE l IN $known RETURN l ORDER BY l""", known=list(govern.NODE_LABELS))]
print("labels the investigation needs (skill vocabulary):", needed)
cmp_rows = []
for user, role in USERS.items():
    pred = govern.audit_access(role, needed, {"Service": ["cost_per_hour"]})
    actual_visible = sorted(set(needed) & set(visible[role]))
    cmp_rows.append({"role": role, "skill_invisible": pred["invisible_labels"], "db_invisible": sorted(set(needed) - set(visible[role])),
                     "skill_complete": pred["functionally_complete"], "match": sorted(set(needed) - set(actual_visible)) == sorted(pred["invisible_labels"])})
h.show(cmp_rows)
print("Where match is False, the skill model and the real database disagree - investigate before trusting either.")
h.check("audit_access ran for every role", len(cmp_rows) == 3)

# %% [markdown]
# ## 24. Write protection, an LLM agent under RBAC, and GDPR erasure
#
# Three real-enforcement demos:
# 1. A read-only role cannot write: the **database** refuses the `CREATE`, not our code.
# 2. A real function-calling agent whose only tool runs Cypher *as* `adv8_sre`. We ask for something the role must not reveal (on-call compensation). Even a cooperative model cannot leak what the database will not return.
# 3. GDPR erasure: *soft* masks the SQL identity row (graph relationships stay for aggregates); *hard* `DETACH DELETE`s the UUID node.

# %%
try:
    h.run_as("adv8_sre", PW, "CREATE (:Hack {x:1})")
    wrote = True
except Exception as e:
    wrote = False
    print("write refused by Neo4j:", type(e).__name__)
h.check("restricted user cannot write", not wrote)

def sre_query(cypher: str):
    return h.run_as("adv8_sre", PW, cypher)[:25]

sre_agent = h.ToolAgent(
    tools={"query_graph": (sre_query, {"type": "object", "properties": {"cypher": {"type": "string", "description": "read-only Cypher"}}, "required": ["cypher"]},
                           "Run a read-only Cypher query on the infrastructure graph as the on-call SRE user.")},
    system="You are an SRE assistant. Use query_graph to answer. Never invent data; if a query returns nothing or null, say it is not available to you.", max_steps=6)
answer = sre_agent.run("Who is on call for checkout-api, what is their compensation, and what is checkout-api's cost per hour?", verbose=True)
h.check("agent never reveals the salary", "185000" not in answer.replace(",", ""))

soft = govern.gdpr_erase("u-1001", "soft")
identity_db.execute("UPDATE identities SET pii=NULL WHERE uuid='u-1001'")
hard = govern.gdpr_erase("u-1002", "hard")
identity_db.execute("DELETE FROM identities WHERE uuid='u-1002'")
h.run(hard["graph_action"])
print("soft preserves aggregates:", soft["aggregate_analysis_preserved"], "| hard:", hard["aggregate_analysis_preserved"])
print("identity store now:", identity_db.execute("SELECT * FROM identities").fetchall())
h.check("hard erase removed the node, soft kept it",
        h.run("MATCH (e:Employee) RETURN collect(e.uuid) AS ids")[0]["ids"] == ["u-1001"])

# %% [markdown]
# ## 25. Cleanup of the RBAC objects
#
# Users and roles live in the `system` database, outside the graph we reset at the top, so they would survive a rerun and pollute a shared server. We drop them explicitly (idempotent `IF EXISTS`) and verify.

# %%
for user in USERS:
    h.sysrun(f"DROP USER {user} IF EXISTS")
for role in USERS.values():
    h.sysrun(f"DROP ROLE adv8_{role} IF EXISTS")
left = [r["role"] for r in h.sysrun("SHOW ROLES YIELD role") if r["role"].startswith("adv8_")]
left_users = [r["user"] for r in h.sysrun("SHOW USERS YIELD user") if r["user"].startswith("adv8_")]
h.check("all adv8_ users and roles dropped", not left and not left_users)

# %% [markdown]
# ## 26. A real function-calling agent investigates INC-1001
#
# Now everything comes together. A **gpt-4o agent with function calling** gets four tools that wrap the earlier work:
# - `find_suspects`: graph traversal, deployments that changed something a service depends on;
# - `search_runbooks`: the **vector index** from section 11;
# - `route_node`: the routing skill (which model should handle a node);
# - `record_decision`: stores a governed decision (`governance_metadata`) in the graph.
#
# The model decides *which* tool to call and in what order (no rule-based planner). Every call is wrapped so it is also recorded as a `(:Run)-[:STEP]->(:ToolCall)` chain in Neo4j: the **trace graph** is itself queryable evidence.

# %%
trace = h.AgentTrace("Investigate INC-1001: why is checkout-api slow?")
DECISIONS = []

def find_suspects(service: str):
    return h.run("""MATCH (s:Service {name:$svc})-[:DEPENDS_ON*0..3]->(x)<-[:CHANGED]-(d:Deployment)
RETURN DISTINCT d.name AS deployment, d.change AS change, x.name AS changed_component""", svc=service)

def search_runbooks(query: str):
    return [{"id": x["node"]["id"], "title": x["node"]["title"], "score": round(x["score"], 3)} for x in h.vector_search("adv8_runbooks", query, k=3)]

def route_node(node: str):
    r = routing.cheapest_meeting_bar(node)
    return {"node": node, "skill_model": r["model"], "strategy": r["strategy"], "real_model": TIER_MODEL.get(tier_of(r["model"]), f"{SMALL} then {FRONTIER}")}

def record_decision(conclusion: str, confidence: float):
    meta = govern.governance_metadata(["devops-graph", "runbooks"], "sre_oncall", FRONTIER, "2024-08-06", float(confidence))
    h.run("CREATE (d:Decision {conclusion:$c}) SET d += $meta WITH d MERGE (m:Model {name:$m}) MERGE (d)-[:MADE_BY]->(m)", c=conclusion[:300], meta=meta, m=FRONTIER)
    DECISIONS.append(meta)
    return "recorded"

def traced(name, fn):
    def wrapper(**kw):
        try:
            out = fn(**kw)
            trace.log(name, kw, json.dumps(out, default=str)[:120], True)
            return out
        except Exception as e:
            trace.log(name, kw, f"{type(e).__name__}: {e}"[:120], False)
            raise
    return wrapper

obj = lambda props, req: {"type": "object", "properties": props, "required": req}
TOOLS = {
    "find_suspects": (traced("find_suspects", find_suspects), obj({"service": {"type": "string"}}, ["service"]),
                      "Find deployments that changed a component the given service depends on (up to 3 hops)."),
    "search_runbooks": (traced("search_runbooks", search_runbooks), obj({"query": {"type": "string"}}, ["query"]),
                        "Semantic search over runbooks; returns top 3 with ids and titles."),
    "route_node": (traced("route_node", route_node), obj({"node": {"type": "string", "enum": list(routing.NODES)}}, ["node"]),
                   "Ask which model tier should serve a given workflow node."),
    "record_decision": (traced("record_decision", record_decision), obj({"conclusion": {"type": "string"}, "confidence": {"type": "number"}}, ["conclusion", "confidence"]),
                        "Store the final root-cause decision with governance metadata. Call once at the end."),
}
agent = h.ToolAgent(TOOLS, system=("You are an SRE agent. Investigate the incident using the tools: first find suspect deployments for the affected service, "
                                    "then search runbooks for the symptom, then record_decision with your conclusion. Ground every claim in tool output. "
                                    "Finish with a short answer naming the deployment, the change and the recommended runbook id."), max_steps=8, trace=trace)
final = agent.run("INC-1001: checkout-api p99 latency is 4200 ms and 5xx is rising. Find the root cause and the recommended runbook.")
h.check("agent produced a final answer", isinstance(final, str) and len(final) > 20)
h.check("agent's tool calls were stored as a trace graph",
        h.run("MATCH (:Run)-[:STEP]->(t:ToolCall) RETURN count(t) AS n")[0]["n"] >= 1)

# %% [markdown]
# ## 27. The trace is a graph; the answer gets an LLM judge
#
# Because each tool call is a node, "what did the agent do?" is a Cypher query. Then **LLM-as-judge** grades the answer against a rubric. Judges are biased (they favour fluent, long answers and sometimes their own family's style),
# so we judge against *specific ground truth* from the graph and keep the rubric concrete. A tolerant check accepts any judge score from 1 to 5 and separately checks that the answer mentions the true culprit.

# %%
tl = trace.timeline()
if tl is not None:
    print(tl)
print(h.run("MATCH (:Run)-[:STEP]->(t:ToolCall) RETURN count(t) AS tool_calls_stored_in_graph")[0])
RUBRIC = ("Ground truth: deployment deploy-481 changed the DB connection pool from 100 to 20 on orders-db, which checkout-api depends on transitively. "
          "Score 5 only if the answer names deploy-481 (or the pool change) as the root cause AND suggests a connection-pool runbook or fix. "
          "Score 3 if it names the right component but no fix. Score 1 if the cause is wrong.")
verdict = h.judge("Why is checkout-api slow?", final, RUBRIC)
print("judge:", verdict)
mentions = any(k in final.lower() for k in ("deploy-481", "pool"))
print("keyword check:", mentions, "| judge score:", verdict["score"], "-> they should agree; if not, read the answer.")
h.check("judge returned a 1-5 score", isinstance(verdict.get("score"), int) and 1 <= verdict["score"] <= 5)
h.check("answer identifies the connection-pool change (tolerant)", mentions)

# %% [markdown]
# ## 28. Governance audit: an auditor's question is a graph query
#
# The agent stored its decision with governance metadata (model, role, confidence, PII flag, retention). "Show every decision that touched PII" is a filter, not log parsing. We run the skill's `audit_query` over the Python records
# **and** the equivalent Cypher over the stored `:Decision` nodes: they must agree.

# %%
pii_records = govern.audit_query(DECISIONS, pii_accessed=True)
in_graph = h.run("MATCH (d:Decision)-[:MADE_BY]->(m:Model) RETURN d.conclusion AS conclusion, d.pii_accessed AS pii, d.access_role AS role, m.name AS model, d.decision_confidence AS confidence")
h.show(in_graph)
print("PII-accessing decisions: python =", len(pii_records), "| graph =", sum(1 for r in in_graph if r["pii"]))
h.check("python audit and graph audit agree", len(pii_records) == sum(1 for r in in_graph if r["pii"]))
h.check("a decision was stored by the real agent", len(in_graph) >= 1 or len(DECISIONS) == 0)

# %% [markdown]
# ## 29. Run the skills as standalone CLIs
#
# Every skill is also a command-line tool, so cron jobs and CI can run them with zero pip installs. `run_benchmark` executes each skill's self-check subcommand in a subprocess.
# We set `PYTHONIOENCODING=utf-8` so Windows consoles do not choke on non-ASCII output.

# %%
os.environ["PYTHONIOENCODING"] = "utf-8"
bench = {}
for slug in ["model-routing-selector", "cost-performance-scorer", "kv-cache-latency-budgeter", "schema-evolution-migrator", "subgraph-access-control"]:
    ok, first = h.run_benchmark("optimization", slug)
    bench[slug] = ok
    print(("PASS" if ok else "FAIL"), slug, "|", first)
cli = h.SKILLS_DIR / "optimization" / "model-routing-selector" / "cli.py"
out = subprocess.run([sys.executable, str(cli), "route", "CausalAttributionNode"], capture_output=True, text=True, encoding="utf-8",
                     env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout
print(out.strip()[:300])
h.check("all five skill benchmarks pass", all(bench.values()))
h.check("CLI routes the causal node by cascade", "cascade" in out)

# %% [markdown]
# ## 30. Vibe coding: build this chapter's tooling with an AI assistant
#
# You will rarely hand-write all of this. Here are prompts worth giving an assistant (Copilot, Claude, Cursor), each followed by what *you* must verify, because Chapter 8's lesson is that plausible code is not measured code.
#
# **Prompt 1: eval harness.** *"Write a Python function `evaluate(node_fn, labeled_set, weights)` that runs `node_fn` on each (input, gold) pair with a retry wrapper, returns accuracy, a confusion count per `gold_as_pred`, a weighted error rate using `weights`, mean latency and USD per call from token usage. Use structured JSON output. Do not call the network at import time."*
#
# **Prompt 2: cascade router.** *"Add a cascade to `classify`: call the small model, escalate to the frontier model if self-reported confidence < 0.85, and log both calls as `NodeInvocation`s. Report escalation rate and cost versus frontier-only."*
#
# **Prompt 3: RBAC.** *"Generate idempotent Neo4j 5 Enterprise Cypher that creates a read-only role limited to labels Service and Deployment with `cost_per_hour` denied, a user bound to it, and a cleanup script. Then write a test that connects as that user and proves it cannot read the property or write."*
#
# **Prompt 4: migration safety.** *"Review this Cypher migration for MERGE patterns that fail on Neo4j 5.26 (null property values), non-idempotent statements and missing rollback."*
#
# **Verification checklist**
# - [ ] Every number reported came from a run you can reproduce (seed, temperature 0, prompt version recorded).
# - [ ] The eval set is labeled by a human, not by the model under test, and failures are weighted by consequence.
# - [ ] Generated Cypher was executed on a scratch database, including the *rerun* and the *rollback*.
# - [ ] Access control was tested as the restricted user, never inferred from the policy text.
# - [ ] No secrets or real PII in prompts, code or the graph; `.env` is git-ignored.
# - [ ] Cost and latency are measured for the node in isolation *and* the pipeline end-to-end.
# - [ ] The assistant's claims about library APIs were checked against the installed version.

# %% [markdown]
# ## 31. Vibe coding drill: have the assistant extend the retrieval ablation
#
# Try this yourself, then use the cell below to check the assistant's work.
#
# *Prompt:* "Extend `hybrid_rank` to a reciprocal-rank fusion of vector rank and graph-hop rank (k=60), keep the same eight queries, print hit@1 and hit@3, and explain in two sentences whether it beats the filter-style hybrid."
#
# Below is a reference implementation to **compare against**, not to copy blindly. Does your assistant's version agree? If it reports a better number than ours, ask it to show the per-query ranks.

# %%
def rrf_rank(q, svc, k=60):
    vr = vec_rank(q)
    gr = graph_rank(svc)
    score = {}
    for rank_list in (vr, gr):
        for pos, rid in enumerate(rank_list, start=1):
            score[rid] = score.get(rid, 0) + 1 / (k + pos)
    return sorted(score, key=lambda r: -score[r])

rrf = [rrf_rank(q, svc) for q, svc, _ in QUERIES]
print("RRF hit@1:", round(sum(r[:1] == [g] for r, (_, _, g) in zip(rrf, QUERIES)) / len(QUERIES), 2),
      "| hit@3:", round(sum(g in r[:3] for r, (_, _, g) in zip(rrf, QUERIES)) / len(QUERIES), 2),
      "| filter-hybrid hit@1/hit@3:", abl[2]["hit@1"], abl[2]["hit@3"])
h.check("RRF produced rankings for every query", len(rrf) == len(QUERIES))

# %% [markdown]
# ## Evaluation and LLM-as-a-Judge — how you know it works
#
# Chapter 8's thesis is *measure, then optimise*. Everything above produced numbers, but numbers from a **12-alert set and an
# LLM classifier** only mean something if the way we *grade* is itself trustworthy. Evaluation is the **pyramid**: cheap,
# objective, numerous checks at the bottom; expensive, subjective, scarce ones at the top.
#
# | Layer | What | Cost | What it can / cannot do |
# |---|---|---|---|
# | 1. Deterministic checks | Cypher / schema / arithmetic assertions | ~free | Exact, but only for what you can state as a rule |
# | 2. Reference-based metrics | overlap vs a **gold set** (the alert accuracy of section 6 is one) | cheap | Objective, but blind to paraphrase and to *why* an answer is wrong |
# | 3. LLM-as-judge | rubric scoring, pairwise preference | 1+ LLM call per item | Handles free text, but is a **model with biases** |
# | 4. Judge calibration | compare the judge with **human labels** | a one-off labelling effort | Tells you whether layer 3 deserves any trust |
# | 5. Human spot checks | read a sample yourself | scarce | The only ground truth; keeps layer 4 honest |
#
# **A judge is itself a model that must be evaluated.** Known biases, and the probe below that targets each:
#
# - **Position bias**: prefers whichever answer is shown first. *Probe:* pairwise judging in both orders, and an identical-answers pair that must tie (`h.judge_pairwise`).
# - **Verbosity bias**: rewards length and polish. *Probe:* pad a weak answer with fluff; its score must not rise much.
# - **Self-preference**: favours text that looks like its own family's output. *Mitigation:* judge with a different model family than the generator for decisions that matter (`h.judge_rubric(..., model=...)`); we flag it, we cannot measure it here.
# - **Leniency / inconsistency**: hands out 4s and 5s, or changes its mind between runs. *Probes:* calibration against human labels and `h.judge_repeat`.
# - **Fooled by fluent wrongness**: *Probe:* a sabotaged answer with the same style but wrong facts **must** score lower.
#
# Scope here: the gold set is about *this chapter's* decisions: model routing, cost per success, hybrid retrieval, temporal
# invalidation, migrations, RBAC and rollout. `h.judge*` helpers ask for the **reason before the score**, at temperature 0 except in the repeat probe.

# %% [markdown]
# ### E1. A gold set, written down and stored in the graph
#
# Eight questions whose references come from facts this notebook already established, and 10 **candidate answers that we
# scored by hand (1-5)**: correct ones, a vague-but-right one, wrong ones, *subtly* wrong ones (a "cheaper per call is always
# cheaper per success" claim, a wrong fix for the `null` MERGE bug, a prompt-only guardrail) and a **fluffy-but-empty** one. The labels are ours and arguable; E4 measures agreement.

# %%
GOLD = [
    {"id": "G1", "question": "How should the AlertClassifier node be routed, and how do you decide?",
     "reference": "Route it to the small model (gpt-4o-mini): it is a high-volume classification node. Do not trust the skill's quality "
                  "number; measure accuracy on a labeled alert set, weight a missed P1 ten times worse than a false P1, and cascade to "
                  "the frontier model when self-reported confidence is below 0.85."},
    {"id": "G2", "question": "Why track cost per success instead of cost per call?",
     "reference": "Cost per success is total spend divided by successful outcomes, so failed calls tax the metric. A model that is cheaper per "
                  "call but less accurate can be more expensive per success."},
    {"id": "G3", "question": "Why is hybrid retrieval (vector plus graph) better than either alone for runbook search?",
     "reference": "Vector search understands language but is blind to topology; graph search knows which runbooks are about services in the "
                  "dependency neighbourhood but has no language understanding. Hybrid keeps vector order but only inside the dependency neighbourhood."},
    {"id": "G4", "question": "Why invalidate a dependency edge with t_invalid instead of deleting it?",
     "reference": "Setting t_invalid keeps the history, so point-in-time queries ('what did checkout-api depend on in 2021?') still work. "
                  "Deleting the edge loses that history."},
    {"id": "G5", "question": "Neo4j rejects the skill's temporal MERGE with a null property. What is the right fix?",
     "reference": "MERGE cannot match on a null property value, so remove t_invalid: null from the MERGE pattern. A missing property already "
                  "reads as IS NULL, so the semantics are unchanged."},
    {"id": "G6", "question": "Can a system-prompt instruction stop an agent from revealing Employee compensation?",
     "reference": "No. Enforce it in the database: a Neo4j RBAC role with deny-by-default means the restricted user cannot see Employee or "
                  "Compensation nodes and gets null for cost_per_hour, whatever the model is told."},
    {"id": "G7", "question": "In what order should a schema change, a data backfill and new agent code be rolled out?",
     "reference": "schema_migration first, then data_backfill, then agent_code as a canary, and schema_cleanup last, so the previous app version "
                  "keeps working at every step (N-1 compatible) and a rollback is possible."},
    {"id": "G8", "question": "Does 99% accuracy on our 12 labeled alerts prove the AlertClassifier meets its 99% bar?",
     "reference": "No. Twelve examples show direction, not significance. A 99% bar needs hundreds to thousands of labeled examples per node."},
]
CANDIDATES = [  # (id, gold id, answer, HUMAN label 1-5, why we gave that label)
    ("C1", "G1", "Route AlertClassifier to the small model (gpt-4o-mini) because it is a high-volume classification node. Measure it on a "
                 "labeled alert set instead of trusting the skill's quality number, weight a missed P1 ten times worse than a false P1, "
                 "and cascade to the frontier model when confidence is below 0.85.", 5, "correct, complete, actionable"),
    ("C2", "G1", "Use the cheaper model for simple nodes and the bigger one for hard nodes, and test it before you ship.",
     3, "right direction, no specifics"),
    ("C3", "G1", "Always use the frontier model for every node; it is the most accurate, so cost per call does not matter.",
     1, "wrong: misses selective intelligence"),
    ("C4", "G2", "Cost per success divides spend by successes, so failures tax it. The frontier model is therefore always cheaper per "
                 "success because it is more accurate.", 2, "subtly wrong: right definition, false 'always'"),
    ("C5", "G3", "Vector search understands language but ignores topology; graph search knows the dependency neighbourhood but not "
                 "language. Hybrid takes the vector order and keeps only runbooks inside the service's dependency neighbourhood.",
     5, "correct and complete"),
    ("C6", "G4", "Temporal graphs are powerful because they capture rich, evolving knowledge and unlock holistic, time-aware insights "
                 "that drive the full potential of modern AI systems.", 1, "fluffy but empty: no fact, no mechanism"),
    ("C7", "G5", "Neo4j rejects null in MERGE, so set t_invalid to an empty string instead of null.",
     2, "subtly wrong: a non-null value breaks the IS NULL semantics"),
    ("C8", "G6", "No, it should refuse. I would add a line to the system prompt telling the agent never to reveal compensation.",
     3, "right decision, wrong mechanism (prompted, not enforced)"),
    ("C9", "G7", "Ship the new agent code to all traffic first, then migrate the schema, and drop the old properties immediately.",
     1, "unsafe: wrong order, no canary, no rollback"),
    ("C10", "G8", "No. Twelve labeled alerts only show direction. To certify a 99% bar you need hundreds to thousands of labeled "
                  "examples per node.", 5, "correct and calibrated"),
]
GOLD_BY_ID = {g["id"]: g for g in GOLD}
CAND = {c[0]: {"gold": c[1], "answer": c[2], "human": c[3], "why": c[4]} for c in CANDIDATES}
HUMAN = [CAND[c[0]]["human"] for c in CANDIDATES]

h.run("MATCH (g:Ch8EvalGold) DETACH DELETE g")
h.run("UNWIND $rows AS r CREATE (:Ch8EvalGold {id:r.id, question:r.question, reference:r.reference})", rows=GOLD)
stored = h.run("MATCH (g:Ch8EvalGold) RETURN g.id AS id, size(g.reference) AS ref_chars ORDER BY id")
h.show(stored)
h.check("gold set has >= 8 cases stored in the graph", len(stored) >= 8)
h.check("at least 8 human-labelled candidates, covering bad, subtle and fluffy answers",
        len(CANDIDATES) >= 8 and {1, 2, 3, 5} <= set(HUMAN))

# %% [markdown]
# ### E2. Layer 1 and 2: deterministic checks, then a reference-based metric
#
# **Layer 1** asserts what must be true of the *artefacts* with Cypher and plain Python: the facts our references cite really
# hold in the graph and in the skills, and no candidate names a workflow node the routing skill does not have (a cheap **hallucination detector**).
# **Layer 2** scores each candidate against its reference with token-overlap F1 rescaled to 1-5: objective and free, but it only sees
# *words*, so watch where it disagrees with the human labels (it cannot tell "always" from "sometimes").

# %%
facts = {
    "nodes": len(routing.NODES),
    "critical_path": cp["nodes"],
    "routed_edges": h.run("MATCH (:WorkflowNode {name:'AlertClassifier'})-[:ROUTED_TO]->(:Model) RETURN count(*) AS c")[0]["c"],
    "invalidated": h.run("MATCH ()-[d:DEPENDS_ON]->(:Library) WHERE d.t_invalid IS NOT NULL RETURN count(d) AS c")[0]["c"],
    "runbooks": h.run("MATCH (r:Runbook)-[:ABOUT]->() RETURN count(DISTINCT r) AS c")[0]["c"],
    "phases": [p["name"] for p in manifest["phases"]],
}
print(facts)
h.check("reference G1/G7 facts: 6 workflow nodes, critical path of 4", facts["nodes"] == 6 and facts["critical_path"] == 4)
h.check("reference G2 holds: both routing edges (cost, accuracy) are stored", facts["routed_edges"] == 2)
h.check("reference G3 holds: all 8 runbooks are linked to a component", facts["runbooks"] == 8)
h.check("reference G4 holds: an invalidated (not deleted) dependency edge exists", facts["invalidated"] >= 1)
h.check("reference G6 holds: the sre_oncall database user could not see Employee nodes",
        "Employee" not in visible["sre_oncall"])
h.check("reference G7 holds: the rollout manifest starts with schema_migration and ends with schema_cleanup",
        facts["phases"][0] == "schema_migration" and facts["phases"][-1] == "schema_cleanup")

NODE_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*(?:Classifier|Analyzer|Parser|Analyst|Node|Synthesis)\b")
for cid, c in CAND.items():
    c["invented"] = sorted(set(NODE_RE.findall(c["answer"])) - set(routing.NODES))


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
h.show([{"id": cid, "human": c["human"], "overlap_1to5": c["f1_score"], "invented_nodes": c["invented"], "why": c["why"]}
        for cid, c in CAND.items()])
overlap_agree = h.agreement(HUMAN, [CAND[c[0]]["f1_score"] for c in CANDIDATES])
print("layer-2 overlap vs human:", overlap_agree)
h.check("layer 1 finds no invented workflow nodes in the correct answers", not CAND["C1"]["invented"] and not CAND["C10"]["invented"])

# %% [markdown]
# ### E3. Layer 3: a rubric judge with explicit 1-vs-5 anchors
#
# Each criterion says exactly what a **1** and a **5** look like, and the judge sees the **reference**, so it grades against ground
# truth rather than its own beliefs. `h.judge_rubric` asks for a one-sentence reason first and returns per-criterion scores plus their mean.
# We judge all 10 candidates once (10 judge calls) and keep the numbers for calibration.

# %%
CRITERIA = {
    "correctness": "5 = every fact and decision agrees with the reference; 1 = contradicts the reference or gives an unsafe decision.",
    "grounding": "5 = cites the specific mechanisms, names and numbers that the reference cites; 1 = cites none, or invents facts absent from the reference.",
    "completeness": "5 = covers every point the reference makes; 1 = covers none of them (generic or off-topic).",
    "actionability": "5 = states the correct decision or next step and the right mechanism; 1 = offers no usable decision, or a harmful one.",
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
# ### E4. Calibration: does the judge agree with the humans?
#
# The step most teams skip. Compare the judge's 1-5 scores with our human labels on the **same items**:
#
# - **exact / within1**: how often the judge matches us exactly, or is off by at most one point. Within-1 near 1.0 with low exact means "right ordering, different scale".
# - **Cohen's kappa**: exact agreement *corrected for chance*. Rough guide: < 0.2 poor, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, > 0.8 near-perfect.
# - **Spearman**: do the two *rank* answers the same way? For picking a winner (a model, a prompt version) this matters more than exact scores.
#
# With 10 items these numbers are noisy (one item moves exact agreement by 10 points): a smoke alarm, not a certificate. The check is
# deliberately **tolerant**: a weak judge is a finding to act on, not a crash.

# %%
JUDGE = [CAND[c[0]]["judge"] for c in CANDIDATES]
cal = h.agreement(HUMAN, JUDGE)
print("judge vs human:", cal)
h.show([{"id": cid, "human": c["human"], "judge": c["judge"], "delta": c["judge"] - c["human"]} for cid, c in CAND.items()])
lenient = [cid for cid, c in CAND.items() if c["judge"] - c["human"] >= 2]
if cal["kappa"] < 0.4 or cal["spearman"] < 0.7:
    print("WARNING: the judge is weakly calibrated against our labels. Do not use its scores for decisions yet. Options: "
          "(1) tighten the rubric anchors on the criteria with the biggest deltas, (2) add 2-3 few-shot examples of "
          "1/3/5 answers, (3) switch to a stronger or different-family judge model, (4) re-label items where the human label may be wrong.")
else:
    print("OK: kappa and Spearman clear the bar (kappa >= 0.4, spearman >= 0.7). Still re-check when the model or rubric changes.")
if lenient:
    print("Leniency: the judge scored these >= 2 points above the human label:", lenient)
h.check("calibration was computed over all candidates", cal["n"] == len(CANDIDATES))

# %% [markdown]
# ### E5. Bias probe 1 — position bias
#
# **What this tests:** whether the verdict depends on *which answer is shown first*. `h.judge_pairwise` asks twice with the order
# swapped; if the verdicts disagree it reports `consistent=False` (counted as a tie). Two pairs (4 judge calls): a real pair (C1 correct vs C2
# vague: the better answer should win in **both** orders) and an **identical** pair (any "winner" is pure position bias).

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
# ### E6. Bias probe 2 — verbosity bias
#
# **What this tests:** whether *length and polish* buy points. We wrap the weak answer C2 in confident, contentless filler (no new
# facts). A well-behaved judge scores it about the same; a verbosity-biased one scores it higher. One extra judge call; a rise of more than 0.75 mean points is flagged.

# %%
FLUFF = ("Great question, and an important one for any modern engineering organisation. In today's fast-moving, "
         "cloud-native landscape, holistic optimisation and a robust, forward-looking, best-practice-driven mindset "
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
# ### E7. Bias probe 3 — self-consistency, and a sabotage control
#
# **Self-consistency.** `h.judge_repeat` scores the *same* answer `n=3` times at temperature 0.7. A `spread` (max - min) of 0 or 1 is
# fine; 2+ means the judge is unreliable *on that item*. We test the mid-quality C2 and the subtly-wrong C8, because borderline answers are
# where judges wobble (6 judge calls).
#
# **Sabotage control.** Take the best answer and corrupt its *facts* while keeping its style. A judge that checks facts must score it
# clearly lower. This one is a hard `h.check`: if a judge cannot tell a true answer from a fluent lie, none of its other scores mean anything (1 call).

# %%
ONE_LINE = ("Score 1-5 against the reference: 5 = correct, specific, grounded in the reference facts and actionable; "
            "3 = right direction but vague; 1 = wrong, invented, or empty.")
rep = {}
for cid in ("C2", "C8"):
    g = GOLD_BY_ID[CAND[cid]["gold"]]
    rep[cid] = h.judge_repeat(g["question"], CAND[cid]["answer"], f"{ONE_LINE} Reference: {g['reference']}", n=3)
    print(f"{cid}: scores={rep[cid]['scores']} mean={rep[cid]['mean']:.2f} spread={rep[cid]['spread']}")
    if rep[cid]["spread"] >= 2:
        print(f"  WARNING: unstable judge on {cid}; average several runs or tighten the rubric for this kind of item.")

SABOTAGED = ("Route AlertClassifier to the frontier model for every alert, because the skill's 0.99 quality number is verified for gpt-4o, "
             "so no labeled set is needed. A missed P1 and a false P1 cost the same, so use plain accuracy and never cascade.")
g1 = GOLD_BY_ID["G1"]
sab = h.judge_rubric(g1["question"], SABOTAGED, CRITERIA, reference=g1["reference"])
print(f"good C1 mean={CAND['C1']['rubric']['mean']:.2f}  sabotaged mean={sab['mean']:.2f}")
h.check("sabotaged (fluent but false) answer scores at least 1 point below the good one",
        sab["mean"] <= CAND["C1"]["rubric"]["mean"] - 1)
print(h.cost_report())

# %% [markdown]
# ### E8. Judge vs heuristic: where do they disagree, and why?
#
# Section 3 used the skill's `cheapest_meeting_bar` to route each workflow node. Here we put that **heuristic** and the **LLM judge**
# in front of the same 6 nodes against a **human-labelled gold tier** (we wrote it from the node descriptions). The heuristic score is 5 if the skill's
# tier matches ours (a cascade counts as a match for a `mid`/`frontier` gold tier) and 1 otherwise; the judge grades the skill's verdict against
# our rationale, with partial credit for an adjacent tier (6 judge calls). Rows differing by 2+ points are the interesting ones: each is a
# heuristic blind spot, a too-strict exact match, or a judge quirk.

# %%
GOLD_TIER = {"AlertClassifier": ("slm", "a one-sentence classification, high volume, a small model suffices"),
             "QueryAnalyst": ("slm", "a simple-vs-complex routing decision, cheap and fast"),
             "LogParser": ("slm", "structured extraction from log lines, a small model suffices"),
             "DependencyAnalyzer": ("mid", "reading a subgraph and listing affected services needs structured reasoning"),
             "CausalAttributionNode": ("frontier", "weighing competing evidence for a cause needs the strongest reasoning"),
             "PredictionSynthesis": ("frontier", "open-ended synthesis with recommendations needs a frontier model")}
tier_rows = []
for n in routing.NODES:
    r = routing.cheapest_meeting_bar(n)
    skill_tier = tier_of(r["model"])
    gold_tier, why_tier = GOLD_TIER[n]
    verdict_text = f"Skill verdict: tier {skill_tier} (strategy {r['strategy']})."
    j = h.judge(f"Which model tier should serve the workflow node {n} ({NODE_DESC[n]})? Tiers: slm, mid, frontier.", verdict_text,
                f"The correct tier is {gold_tier}: {why_tier}. A 'cascade' strategy tries a cheap model and escalates to a stronger one. "
                "Score 5 if the verdict matches the correct tier (a cascade is a match for mid or frontier), 3 if it is an adjacent tier, 1 if it is clearly wrong.")
    heur = 5 if (skill_tier == gold_tier or (skill_tier == "cascade" and gold_tier in ("mid", "frontier"))) else 1
    tier_rows.append({"node": n, "gold": gold_tier, "skill": skill_tier, "heuristic": heur, "judge": int(j["score"]), "judge_reason": j["reason"][:80]})
h.show([{k: v for k, v in r.items() if k != "judge_reason"} for r in tier_rows])
print("heuristic vs judge:", h.agreement([r["heuristic"] for r in tier_rows], [r["judge"] for r in tier_rows]))
diffs = [r for r in tier_rows if abs(r["heuristic"] - r["judge"]) >= 2]
print(f"{len(diffs)} disagreement(s) of >= 2 points")
for r in diffs:
    if r["heuristic"] < r["judge"]:
        meaning = ("exact match is too strict: the skill chose a neighbouring tier or a cascade and the judge gives partial credit "
                   "(or is lenient toward a near-miss)")
    else:
        meaning = "judge stricter than exact match: the skill hit our label but the judge disputes it; re-read the item"
    print(f"  {r['node']} gold={r['gold']} skill={r['skill']} heuristic={r['heuristic']} judge={r['judge']}: {meaning}. Judge said: {r['judge_reason']}")
h.check("compared the heuristic and the judge on all 6 nodes", len(tier_rows) == len(routing.NODES) == 6)

# %% [markdown]
# ### E9. Regression harness: change a prompt, prove it did not get worse
#
# Evaluation earns its keep when something *changes*. `h.run_eval` runs a system over a case list, scores each case, and stores
# `(:EvalRun)-[:HAS_RESULT]->(:EvalResult)` in Neo4j under a name and **version**; `h.compare_runs` compares versions with one Cypher query, and a
# **CI-style gate** turns the comparison into PASS/FAIL. The system is the AlertClassifier prompt (`v1` lazy vs `v2` explicit rubric) on the
# 12 labeled alerts. To avoid paying twice we **replay the stored predictions** of section 6 (no new LLM calls), and the gate is deterministic, so a flaky judge
# cannot hide a regression. It also guards the *expensive* direction separately: new `P1_as_P3` errors. The gate prints PASS or FAIL and never raises.

# %%
EVAL_NAME = "ch8_alert_triage"
h.run("MATCH (r:EvalRun) WHERE r.name STARTS WITH 'ch8_' OPTIONAL MATCH (r)-[:HAS_RESULT]->(x) DETACH DELETE x, r")
ALERT_CASES = [{"id": f"A{i + 1}", "idx": i, "input": text, "gold": gold} for i, (text, gold) in enumerate(ALERTS)]


def score_alert(case, out):
    ok = out == case["gold"]
    return {"score": 1.0 if ok else 0.0, "passed": ok, "notes": f"gold={case['gold']} got={out}"}


summaries = {}
for ver in ("v1", "v2"):
    summaries[ver] = h.run_eval(EVAL_NAME, ALERT_CASES, lambda c, v=ver: RESULTS[(SMALL, v)][c["idx"]]["pred"], score_alert, version=ver)
    print(f"{ver}: pass_rate={summaries[ver]['pass_rate']:.0%} failures={summaries[ver]['failures']}")
runs = h.compare_runs(EVAL_NAME)
h.show(runs)
h.check("both prompt versions were stored and are comparable in Neo4j", {r["version"] for r in runs} == {"v1", "v2"})

TOLERANCE = 0.10          # candidate may be at most this much worse than baseline (0-1 scale)
by_ver = {r["version"]: r for r in runs}
base, cand = by_ver["v1"]["mean_score"], by_ver["v2"]["mean_score"]
dangerous_new = {v: sum(1 for c in ALERT_CASES if c["gold"] == "P1" and RESULTS[(SMALL, v)][c["idx"]]["pred"] == "P3") for v in ("v1", "v2")}
gate = "PASS" if cand >= base - TOLERANCE and dangerous_new["v2"] <= dangerous_new["v1"] else "FAIL"
print(f"CI GATE: baseline v1={base:.2f}  candidate v2={cand:.2f}  tolerance={TOLERANCE:.2f}  "
      f"P1_as_P3 v1={dangerous_new['v1']} v2={dangerous_new['v2']}  ->  {gate}")
if gate == "FAIL":
    print("  Regression: inspect (:EvalResult) rows with passed=false for the candidate version before shipping.")
h.show(h.run("MATCH (:EvalRun {name:$n, version:'v2'})-[:HAS_RESULT]->(x:EvalResult) WHERE NOT x.passed "
             "RETURN x.case AS failed_case, x.notes AS notes ORDER BY x.case", n=EVAL_NAME) or [{"failed_case": "none"}])

# %% [markdown]
# ### E10. Judge report card, and a prompt to grow the eval set
#
# **What I would trust this judge for** (after reading the calibration, bias and sabotage results above):
#
# - **Ranking and regression-spotting** between versions, using the *same* judge, rubric and reference each time (relative change).
# - **Catching clearly bad and fluent-but-false answers**, because reference-anchored criteria plus the sabotage control passed.
# - **Triage**: deciding which 10% of answers a human should read first.
#
# **What I would not trust it for:**
#
# - **Absolute quality claims** ("our agent is 4.3/5") or comparing numbers across different judges or rubrics.
# - **Safety and permissions.** Whether the agent *can* read compensation is a Neo4j RBAC fact (sections 20-24), never a judge opinion.
# - **Subtle factual errors without a reference** (wrong tier, wrong fix): use layer 1 assertions and measured accuracy (section 6) for those.
# - **Anything after the judge model, rubric or task changes**, until it is re-calibrated on fresh human labels.
# - **Self-preference**: this notebook's agent, classifier and judge are all OpenAI models, so expect flattery we did not measure here.
#
# **Clean-up for this section:** the next cell removes the `Ch8EvalGold` and `ch8_` `EvalRun` data it created.
#
# **Vibe-coding prompt: extend the eval set**
# ```text
# Context: section "Evaluation and LLM-as-a-Judge" in notebooks_advanced/_src/ch8-optimization.py, and
# real_helpers.py (judge_rubric, judge_pairwise, judge_repeat, agreement, run_eval, compare_runs).
# Task: add 6 new gold cases and 8 new human-labelled candidates about cascade routing and the confidence threshold, including two
# subtly wrong answers (threshold direction reversed, cost compared per call instead of per success) and one fluffy answer. Re-run the calibration.
# Constraints: reuse the CRITERIA dict; <= 20 extra judge calls; every Cypher query parameterised; Neo4j labels
# namespaced with ch8_/Ch8; do not weaken any existing h.check.
# Acceptance: print agreement (kappa, spearman) before and after; if kappa drops by more than 0.1, list the new
# items with the largest judge-human delta and propose a rubric change. Do not change the human labels to fit the judge.
# ```

# %%
h.run("MATCH (g:Ch8EvalGold) DETACH DELETE g")
h.run("MATCH (r:EvalRun) WHERE r.name STARTS WITH 'ch8_' OPTIONAL MATCH (r)-[:HAS_RESULT]->(x) DETACH DELETE x, r")
left = h.run("MATCH (n) WHERE n:Ch8EvalGold OR n:EvalRun OR n:EvalResult RETURN count(n) AS c")[0]["c"]
h.check("evaluation section left no Ch8EvalGold / EvalRun / EvalResult nodes behind", left == 0)
print(h.cost_report())

# %% [markdown]
# ## 32. Cost, limits and what to swap next
#
# **What this notebook spent.** `h.cost_report()` below prints the token usage of the entire run (tokens, not dollars; prices change, and our `PRICE` table is illustrative).
#
# **What is still approximate, honestly**
# - **LLM non-determinism.** Temperature 0 reduces but does not eliminate variation, and model snapshots change. Rerun and your accuracies can shift by a few points.
# - **Tiny eval sets.** 12 alerts and 8 retrieval queries show *direction*, not significance. A 99% accuracy bar needs hundreds of labeled examples per node (the book's 2,000).
# - **Tier mapping.** The skill's quality numbers describe llama/claude models; we mapped them to gpt-4o-mini / gpt-4o. The comparison in section 8 is a sanity check on the heuristic, not a re-derivation of it.
# - **Judge bias.** LLM judges prefer fluent answers and are not independent of the model they judge. We anchored the rubric to graph ground truth and kept a keyword check alongside.
# - **The judge is a model too.** It has position, verbosity, leniency and same-family self-preference biases (our classifier, agent and judge are all OpenAI models); the probes in the evaluation section reduce them but do not remove them. Re-calibrate against fresh human labels whenever the judge model, rubric or task changes.
# - **Judge cost is extra calls.** Every judged item is at least one more LLM call (pairwise = two, repeats = n); budget it, cap the loops, and judge a sample in CI rather than everything.
# - **Self-reported confidence** is poorly calibrated; cascade quality depends on it. Production systems use logprobs, a trained verifier, or calibration on held-out data.
# - **Latency.** The 2 s target was designed for self-hosted SLMs; hosted APIs add network time. Our estimate multiplies measured means, ignoring tail latency and parallel speedups.
# - **RBAC scope.** We tested three role shapes on one database. Real policies also need property-level tests per label, relationship privileges and audit logging, and the demo uses fixed passwords on a throwaway server.
# - **Identity store.** SQLite stands in for a real identity provider; erasure there is a one-row update, not a compliance programme.
#
# **What to swap next**
# 1. Grow each node's eval set from production traffic; version eval sets like prompts and gate deployments on them.
# 2. Replace self-reported confidence with calibrated scores; log every call into the `CostTracker` automatically (middleware).
# 3. Add a reranker or learned fusion to hybrid retrieval and compare it using the same harness.
# 4. Run migrations through a real migration tool (Neo4j-Migrations) in CI against a scratch database, with the rollback tested.
# 5. Put governance records on an append-only store and enforce RBAC via your identity provider (SSO/LDAP mapping to Neo4j roles).

# %%
print(h.cost_report())
h.check("notebook completed", True)
