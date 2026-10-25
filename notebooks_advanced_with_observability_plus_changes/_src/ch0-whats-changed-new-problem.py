# %% [markdown]
# # Chapter 0 — What changed? A new kind of data and a *different* problem
# > **Tier: advanced + changes.** Needs `OPENAI_API_KEY` in `notebooks_advanced_with_observability_plus_changes/.env`, the Neo4j
# > container from `docker-compose/`, and (optionally) Phoenix or DeepEval for observability. Start here, then do ch1-ch9 in order.
#
# > **Estimated OpenAI cost of one run (measured 2026-10-24 at `gpt-4o` rates): about $0.09.** See the README for all ten notebooks.
#
# ## The story so far
# Every earlier notebook investigated **one** problem on **one** small graph: **INC-1001, "checkout-api is slow"**.
# The root cause was a deployment (`deploy-481`) that shrank a database connection pool from 100 to 20. That is a
# *capacity* problem and the graph only needed services, a database, a deployment, an alarm and two incidents.
#
# ## What this folder adds
# Real systems do not stand still. A few hours after INC-1001 was fixed, the company shipped more changes, and a
# **completely different** problem appeared: **INC-2001 — customers in the EU are being charged twice**.
# That is a *correctness* problem (money!), not a latency problem, and **the old root cause is innocent**.
#
# To investigate it, the graph needs new *kinds* of data. This notebook:
# 1. recaps the old graph and its old problem,
# 2. loads **11 new kinds of data** into the same Neo4j (nothing is deleted),
# 3. shows you the new problem and investigates it **with plain Cypher first** (no AI),
# 4. then adds embeddings, an LLM investigator, and **evaluation / LLM-as-a-Judge** to see how much the AI really helps.
#
# Chapters 1-9 each end with a **Part 2** section that re-solves INC-2001 using *that chapter's* technique.
#
# ### How to read this notebook as a beginner
# * Every code cell is preceded by a markdown cell explaining *why* we run it and *what to look for*.
# * `h` = `real_helpers.py` (LLM, embeddings, Neo4j, evaluation helpers). `scn` = `incident2.py` (the new data and the
#   gold answers). Open both files; they are short.
# * The word **"investigate"** here means: gather evidence, form hypotheses, try to **eliminate** them, and only then
#   commit to a root cause. Eliminating wrong ideas is as important as finding the right one.

# %% [markdown]
# ## 0. Preflight — touch every dependency once
# We call the chat model, the embedding model and Neo4j once so a missing key or a stopped container fails **now**
# with a clear message instead of halfway through.

# %%
import sys
sys.path.insert(0, ".")
import json
import os

import real_helpers as h
import incident2 as scn

h.obs_start("ch0-whats-changed-new-problem")   # observability: reads OBSERVABILITY_BACKEND from .env (phoenix | deepeval | both | none)

client = h.get_openai()
print("chat model     :", h.LLM_MODEL)
print("embedding model:", h.EMBED_MODEL, f"({h.EMBED_DIM} dims)")
print("LLM round trip :", h.llm("Reply with the single word: ready").strip())
h.connect()
info = h.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions[0] AS version, edition")
h.show(info)
h.check("Neo4j is Enterprise edition", info[0]["edition"].lower() == "enterprise")

# %% [markdown]
# ## 1. Recap: the OLD graph and the OLD problem (INC-1001)
# `h.load_devops_graph()` wipes the database and loads the original story. It is small on purpose: 10 nodes.
# Remember it, because in section 2 we will add about 90 more nodes **around** it.

# %%
base = h.load_devops_graph()
print("original graph:", base)
h.show(h.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS nodes ORDER BY nodes DESC"))
old = h.run("""
    MATCH (i:Incident {id:'INC-1001'})-[:AFFECTS]->(s:Service)
    MATCH (d:Deployment)-[:CHANGED]->(db:Database)
    RETURN i.title AS incident, s.name AS affected, d.id AS suspect_deploy, d.change AS what_changed, db.name AS database
""")
h.show(old)
h.check("the original problem is still the pool shrink", "100 -> 20" in old[0]["what_changed"])

# %% [markdown]
# **Reading the result.** INC-1001 = checkout is slow; the suspect is `deploy-481`, which changed `orders-db`'s pool from
# 100 to 20. One hop, one cause. A single Cypher query solved it, which is exactly why it was a good *first* problem.
#
# ## 2. Load the changes: 11 new kinds of data
#
# | Wave | New node labels | Why they exist (what question they answer) |
# |---|---|---|
# | **1. Topology** | `Region`, `Vendor`, `Queue`, `Engineer`, plus two more `Service`s | *Where does it run? Who do we call? Who is on call?* |
# | **2. Change events** | `ConfigChange`, `FeatureFlag`, `SecretRotation` (+ more `Deployment`s) | *What changed right before the problem?* Most outages are caused by a change. |
# | **3. Signals** | `VendorStatus`, `MetricSample` (hourly time series), `CostRecord` (USD per hour) | *What did the world and the numbers do?* |
# | **4. Documents** | `Ticket` (customer complaints), `LogEvent` (log lines), `Runbook`, a past `Incident` | *What do people and machines **say**?* Free text, so we will embed it. |
#
# Three design choices worth noticing:
# * **Idempotent loading.** `scn.load_changes()` uses `MERGE`, so running it twice does not duplicate anything.
# * **Time is a property.** Every event node has an `at` timestamp (ISO strings sort correctly as text), so one query can
#   build a timeline across *different* node types.
# * **The answer is NOT stored.** The graph has no `ROOT_CAUSE` edge for INC-2001. The ground truth lives only in
#   Python (`scn.GROUND_TRUTH`) so we can grade ourselves later.

# %%
result = scn.load_changes()
print(f"nodes {result['nodes_before']} -> {result['nodes_after']}, relationships {result['rels_before']} -> {result['rels_after']}")
print("new labels :", result["new_labels"])
h.check("changes added many new nodes", result["nodes_after"] > result["nodes_before"] + 80)
again = scn.load_changes()
h.check("loading twice is idempotent (MERGE)", again["nodes_after"] == result["nodes_after"] and again["rels_after"] == result["rels_after"])

# %% [markdown]
# ### Look at one sample of each new kind
# Never trust data you have not looked at. We print one row per new label with its properties, so you see the *shape*
# of each kind: change events have `at`/`change`, metrics have `metric`/`value`, documents have `text`.

# %%
for label in ["FeatureFlag", "ConfigChange", "SecretRotation", "VendorStatus", "MetricSample", "CostRecord", "Ticket", "LogEvent", "Runbook"]:
    row = h.run(f"MATCH (n:{label}) RETURN properties(n) AS p LIMIT 1")[0]["p"]
    print(f"{label:<15}", {k: (v[:60] + '...' if isinstance(v, str) and len(v) > 60 else v) for k, v in row.items()})

# %% [markdown]
# ### The new schema
# `db.schema.visualization()` returns the label-to-label relationship pattern actually present in the database. In Neo4j
# Browser (http://localhost:7474, or 7475 / 7476 on the observability stacks) run `CALL db.schema.visualization()` to see it
# as a picture. Here we print the relationship *types* and how often each appears. Notice the **verbs**: `CALLS`,
# `GATES`, `ROTATED_FOR`, `PUBLISHES_TO`, `COST_OF`. Good relationship names make Cypher read like English.

# %%
h.show(h.run("MATCH ()-[r]->() RETURN type(r) AS relationship, count(*) AS n ORDER BY n DESC"), max_rows=40)

# %% [markdown]
# ## 3. The new problem: INC-2001
# Three alarms fired in the last half hour and a sev1 incident was opened. Read the alarms: they are about **duplicate
# charges**, a **dead-letter queue**, and **vendor spend**, not latency.

# %%
h.show(h.run("""
    MATCH (a:Alarm) WHERE a.at >= '2026-10-24T16:00:00Z'
    OPTIONAL MATCH (a)-[:FIRED_ON]->(t)
    RETURN a.at AS at, a.id AS alarm, a.metric AS metric, a.value AS value, a.threshold AS threshold, coalesce(t.name, t.id) AS fired_on
    ORDER BY at
"""))
h.show(h.run("MATCH (i:Incident {id:'INC-2001'}) RETURN i.id AS id, i.title AS title, i.severity AS severity, i.status AS status"))
h.show(scn.q("tickets", inc="INC-2001"))

# %% [markdown]
# **Why is the old explanation tempting but wrong?** An engineer who just closed INC-1001 will think "database pool again".
# The graph lets us *test* that instead of arguing. A hypothesis is only useful if some evidence could refute it.
#
# ## 4. Investigate with plain Cypher first (no AI at all)
# ### 4a. The timeline — the single most useful query in incident response
# Everything that happened since 08:00, oldest first, across **six node types** in one result. Ask of every row: "could
# this *cause* the symptom?" A cause must come **before** its effect.

# %%
tl = scn.q("timeline", since="2026-10-24T08:00:00Z")
h.show(tl, max_rows=30)
h.check("timeline spans several node types", len({r["kind"] for r in tl}) >= 5)
h.check("deploy-490 precedes the first duplicate-charge alarm",
        next(r["at"] for r in tl if r["id"] == "deploy-490") < next(r["at"] for r in tl if r["id"] == "alarm-91"))

# %% [markdown]
# ### 4b. The numbers: when did each symptom *start*?
# We print the four hourly series for `payment-service` side by side. Look for **the first hour that is not flat**.
# Everything before 15:00 is quiet. Latency, retries and duplicates all rise together.

# %%
import collections
series = collections.defaultdict(dict)
for m in scn.METRICS:
    for r in scn.q("metric", metric=m):
        series[r["at"][11:16]][m] = r["value"]
h.show([{"hour": hr, **vals} for hr, vals in sorted(series.items())])
spend = scn.q("cost", cat="paygate-api-calls")
print("PayGate spend USD/hour:", [int(r["usd"]) for r in spend])
h.check("duplicates were zero before 15:00 and non-zero after", series["14:00"]["duplicate_charges"] == 0 and series["16:00"]["duplicate_charges"] > 0)
h.check("vendor spend more than tripled", spend[-1]["usd"] > 3 * spend[0]["usd"])

# %% [markdown]
# **Money is a signal.** The vendor bills per call. Spend jumped from ~$41/h to ~$152/h with no traffic increase, so each
# customer action is now causing **several** vendor calls. That points at *retries* before we have read a single log line.
#
# ### 4c. Hypotheses and how to eliminate each one
# Good incident response keeps a list of suspects and tries to kill them. Each check below is a **query that could have
# come out the other way**. If a suspect survives, it stays on the list.

# %%
HYP = {}

# H1: the database pool again? It was restored by deploy-482 hours earlier, and the db logs look healthy.
pool = h.run("MATCH (d:Deployment {id:'deploy-482'})-[:CHANGED]->(db:Database) RETURN d.at AS at, d.change AS change, db.max_connections AS max_conn")[0]
quiet = series["12:00"]["p99_latency_ms"] <= 800 and series["14:00"]["p99_latency_ms"] <= 800
HYP["H1 db pool (INC-1001 again)"] = ("ruled out", f"{pool['change']} at {pool['at']}; p99 was normal until 15:00")
h.check("H1: pool was restored and latency was normal for hours afterwards", quiet)

# H2: the PayGate API-key rotation (sec-22) at 15:00 is suspiciously close. Would a bad key show up as 401/403 errors?
auth_errors = h.run("MATCH (l:LogEvent) WHERE l.level IN ['WARN','ERROR'] AND (l.text CONTAINS '401' OR l.text CONTAINS '403') RETURN count(l) AS c")[0]["c"]
HYP["H2 secret rotation sec-22"] = ("ruled out", f"auth check returned 200 and there are {auth_errors} 401/403 warnings or errors")
h.check("H2: no authentication errors anywhere", auth_errors == 0)

# H3: the new checkout UI (deploy-489, 15:10). Does it touch payments, and did it log anything wrong?
ui_logs = h.run("MATCH (l:LogEvent)-[:LOGGED_BY]->(s:Service {name:'cart-service'}) RETURN l.text AS t")
retry_services = {r["s"] for r in h.run("MATCH (l:LogEvent)-[:LOGGED_BY]->(s:Service) WHERE l.text CONTAINS 'retrying charge' RETURN DISTINCT s.name AS s")}
HYP["H3 new-checkout-ui (deploy-489)"] = ("ruled out", f"only cart-service logs: {ui_logs[0]['t']}; retries come from {retry_services}")
h.check("H3: the retrying charges are logged by payment-service, not cart-service", retry_services == {"payment-service"})

# H4: PayGate is degraded in eu-west-1. That is real, but is it enough on its own?
vendor = scn.q("vendor", vendor="PayGate")
late_success = h.run("MATCH (l:LogEvent) WHERE l.text CONTAINS 'late success' RETURN count(l) AS c")[0]["c"]
HYP["H4 PayGate degradation (vs-1)"] = ("contributing, not sufficient",
                                        f"p95 {vendor[-1]['p95_ms']}ms in {vendor[-1]['region']}; PayGate still answered 200 ({late_success} 'late success' log)")
h.check("H4: PayGate has one healthy and one degraded region", {v['state'] for v in vendor} == {"healthy", "degraded"})

# H5: retries WITHOUT an idempotency key + timeout (3000ms) BELOW the vendor p95 (4200ms)
flag = h.run("MATCH (f:FeatureFlag {id:'payment-auto-retry'}) RETURN f.max_retries AS retries, f.idempotency_key AS idem")[0]
cfg = h.run("MATCH (c:ConfigChange {id:'cfg-310'}) RETURN c.old AS old, c.new AS new")[0]
mechanism = flag["retries"] > 1 and not flag["idem"] and int(cfg["new"]) < vendor[-1]["p95_ms"]
HYP["H5 retries + no idempotency key + timeout < vendor p95"] = ("ROOT CAUSE" if mechanism else "unproven", f"retries={flag['retries']}, idempotency_key={flag['idem']}, timeout {cfg['old']}->{cfg['new']}ms < p95 {vendor[-1]['p95_ms']}ms")
h.check("H5: all three conditions of the mechanism hold", mechanism)
h.show([{"hypothesis": k, "verdict": v[0], "evidence": v[1]} for k, v in HYP.items()])

# %% [markdown]
# ### The mechanism in one paragraph (read this twice)
# 1. `cfg-310` cut the client timeout from 30 s to 3 s (well meant: "reduce tail latency after INC-1001").
# 2. `deploy-490` switched on automatic retries (5 attempts) **without an `Idempotency-Key` header**. The header is what
#    lets a payment processor recognise "this is the same charge again" and ignore it.
# 3. At 15:25 PayGate in eu-west-1 slowed to 4.2 s. It still **succeeds**, but the client gave up after 3 s,
#    retried, and PayGate happily charged the customer a second (and third) time. Each retry is also another billable call
#    and another job in the retry queue, which explains the vendor spend and the dead-letter queue.
#
# **No single change is the bug**; the *combination* is. A keyword search for "timeout" or for "retry" finds half the
# story. Only following relationships (`ConfigChange -> Service <- Deployment -> FeatureFlag`, `Vendor -> VendorStatus`)
# assembles the whole. That is the argument for a graph.

# %% [markdown]
# ## 5. Meaning, not keywords: embeddings over the new documents
# Tickets, log lines, runbooks and past incidents are **free text**. Customers write "paying twice", "double charge",
# "3 pending payments" and never the word "duplicate". `h.embed_nodes` stores a 3072-number vector on each node and a
# **Neo4j vector index** finds the nearest ones by meaning. About 40 short texts: a fraction of a cent.

# %%
done = scn.embed_new_documents()
print("embedded nodes per label:", done)
hits = h.vector_search("ticket_text_idx", "customers paying twice for one purchase", k=6)
for r in hits:
    print(f"{r['score']:.3f}  {r['node']['id']}  {r['node']['text'][:70]}")
dup_tickets = {r["node"]["id"] for r in hits[:5]}
h.check("the top-5 semantic hits are duplicate-charge tickets, not the password or invoice ones", dup_tickets <= {"T-9001", "T-9002", "T-9003", "T-9004", "T-9005", "T-9009"})

# %% [markdown]
# ### "Have we seen this before?" and "what does the runbook say?"
# Two more vector searches, over `Incident.text` and `Runbook.text`. Memory of past incidents is one of the biggest wins
# of an agentic system: INC-0650 (March) had the **same** root cause, and RB-12 already contains the mitigation.

# %%
past = h.vector_search("incident_text_idx", "duplicate charges after enabling retries on payment calls", k=2)
for r in past:
    print(f"{r['score']:.3f}  {r['node']['id']}  {r['node']['title']}")
book = h.vector_search("runbook_text_idx", "what to do when customers are charged twice", k=2)
for r in book:
    print(f"{r['score']:.3f}  {r['node']['id']}  {r['node']['title']}")
h.check("the most similar past incident is INC-0650", past[0]["node"]["id"] == scn.GROUND_TRUTH["similar_incident"])
h.check("the best runbook is RB-12", book[0]["node"]["id"] == "RB-12")

# %% [markdown]
# ## 6. Now ask an LLM to investigate, and **grade** it
# We give `gpt-4o` the same evidence we just gathered (a compact dossier from `scn.facts_text()`, built **only** from the
# graph) and force a structured JSON answer. Then we grade it two ways, because one grader is never enough:
# * **Deterministic checks**: does the answer name `deploy-490` / `payment-auto-retry`, `cfg-310` or the timeout, and
#   mention idempotency? Does it rule out the red herrings?
# * **LLM-as-a-Judge**: a second call scores correctness, use of evidence, red-herring handling and actionability
#   against the reference answer from `scn.GROUND_TRUTH`, giving the reason *before* the score.

# %%
dossier = scn.facts_text()
print(dossier[:1200], "\n...")
SCHEMA = ('{"root_cause": "<2-3 sentences>", "key_evidence": ["<fact with id>", ...], "ruled_out": ["<suspect id: why>", ...], '
          '"mitigation_now": "<one action>", "permanent_fix": ["<action>", ...]}')
inv = h.llm_json(f"You are an SRE. Investigate incident INC-2001 from this dossier only. Cite ids.\n\n{dossier}\n\nReply as JSON: {SCHEMA}",
                 system="You are a careful site reliability engineer. Reply with JSON only. Never invent ids.")
print(json.dumps(inv, indent=2)[:1500])

# %%
text = json.dumps(inv).lower()
det = {
    "names the retry change": "deploy-490" in text or "payment-auto-retry" in text,
    "names the timeout change": "cfg-310" in text or "3000" in text or "3s" in text or "3 s" in text,
    "mentions idempotency": "idempotency" in text,
    "rules out the secret rotation": "sec-22" in json.dumps(inv.get("ruled_out", [])).lower(),
    "mitigation is to disable retries": any(w in inv.get("mitigation_now", "").lower() for w in ("disable", "turn off", "switch off", "roll back")),
}
for k, v in det.items():
    h.check(k, v)
CRIT = {"correctness": "5 = identifies retries without idempotency + the 3s timeout + the vendor slowdown as one mechanism; 1 = blames the database or the secret",
        "evidence_use": "5 = every claim cites an id/log/metric from the dossier; 1 = no citations or invented ids",
        "red_herrings": "5 = explicitly and correctly rules out sec-22, deploy-489 and the db pool; 1 = chases one of them",
        "actionability": "5 = immediate mitigation AND permanent fix are specific and safe; 1 = vague"}
verdict = h.judge_rubric("What is the root cause of INC-2001 and what should we do?", json.dumps(inv), CRIT, reference=scn.GROUND_TRUTH["summary"])
h.show([{"criterion": k, "score": v, "reason": verdict["reasons"].get(k, "")[:110]} for k, v in verdict["scores"].items()])
print("judge mean:", round(verdict["mean"], 2))
h.check("judge mean >= 3.5 for the investigation", verdict["mean"] >= 3.5)

# %% [markdown]
# ## 7. How much does the *graph* help? An evaluation with a control group
# Claim to test: **an LLM that is given the graph evidence answers INC-2001 questions better than an LLM that is not**.
# We use the 8 gold questions in `scn.GOLD_QA`. Three **versions** of a system answer the same questions:
# * `no-context` — the bare model (it has never heard of your incident, so it can only guess),
# * `graph-context` — the model plus a dossier built from the graph and the top vector hits (first a naive dossier, then a richer one).
#
# `h.run_eval` runs each system over every case, scores it with a deterministic keyword grader, **stores the run in Neo4j**
# (`EvalRun` / `EvalResult` nodes), and `h.compare_runs` compares versions with one Cypher query. This is the loop you will
# repeat for every change you make: *measure before, change one thing, measure after*.

# %%
def ask_bare(case):
    return h.llm(case["input"] + " (Context: a payments platform incident called INC-2001.) Answer in 2 sentences.")

def make_graph_system(doss):
    def ask(case):
        extra = h.vector_search("runbook_text_idx", case["input"], k=1) + h.vector_search("incident_text_idx", case["input"], k=1)
        docs = "\n".join(f"- {r['node'].get('id')}: {r['node'].get('text', '')[:300]}" for r in extra)
        return h.llm(f"Answer using ONLY this evidence. Cite ids. If something is ruled out by the evidence say 'No' first. 2-3 sentences.\n\n"
                     f"EVIDENCE\n{doss}\n\nRELATED DOCUMENTS\n{docs}\n\nQUESTION: {case['input']}")
    return ask

def grade(case, out):
    return scn.keyword_score(out, case["must"])

dossier_rich = scn.facts_text(rich=True)
print(f"naive dossier {len(dossier)} chars, rich dossier {len(dossier_rich)} chars")
r_bare = h.run_eval("inc2001-qa", scn.GOLD_QA, ask_bare, grade, version="no-context")
r_graph = h.run_eval("inc2001-qa", scn.GOLD_QA, make_graph_system(dossier), grade, version="graph-context-v1")
r_rich = h.run_eval("inc2001-qa", scn.GOLD_QA, make_graph_system(dossier_rich), grade, version="graph-context-v2-rich")
h.show(h.compare_runs("inc2001-qa"))
h.show([{"case": a["case"], "no-context": round(a["score"], 2), "v1": round(b["score"], 2), "v2-rich": round(c["score"], 2)}
        for a, b, c in zip(r_bare["results"], r_graph["results"], r_rich["results"])])
h.check("graph context beats the bare model", r_graph["mean_score"] > r_bare["mean_score"])
h.check("the richer dossier is at least as good as the naive one", r_rich["mean_score"] >= r_graph["mean_score"])
h.check("rich graph-context answers score a mean of at least 0.7 on the gold set", r_rich["mean_score"] >= 0.7)

# %% [markdown]
# ### Reading the comparison (this is the lesson of the cell above)
# * The bare model scores near zero on identifiers (`deploy-490`, `cfg-310`, `INC-0650`) because **it cannot know
#   them**. It will often answer confidently anyway. That is the hallucination risk the graph removes.
# * **v1 (naive dossier) still misses questions.** Typical misses: *"Which region is affected?"* (the naive dossier has no
#   region information, so the model can only say "the EU"), and *"is orders-db implicated?"* (the evidence that the pool
#   was restored is missing from the payment-service logs). Those are **retrieval gaps, not model weaknesses**.
# * **v2 (rich dossier)** adds vendor health per region and every service's logs with their region. We changed **one thing**
#   (what we retrieve), re-ran the same 8 cases, and `compare_runs` shows the delta. That is the engineering loop of the whole repo.
# * A keyword grader is strict and cheap. If a case still fails, **read the output** before blaming the model; sometimes the
#   right answer is phrased in a way the keywords do not cover. Fix the retrieval, the prompt, *or* the grader, one at a time.
# * Deterministic grader + LLM judge **together** is the pattern for every later chapter.

# %%
bad = [x for x in r_rich["results"] if not x["passed"]]
print(f"{len(bad)} rich graph-context case(s) failed; and v1 failed: {r_graph['failures']}")
for x in bad:
    print("-", x["case"], x["notes"], "|", x["output"][:160].replace("\n", " "))
j = h.judge_rubric(scn.GOLD_QA[2]["input"], r_rich["results"][2]["output"],
                   {"correctness": "5 = names cfg-310, the new value 3000ms and who made it; 1 = wrong", "grounding": "5 = only facts present in the evidence"},
                   reference=scn.GOLD_QA[2]["reference"])
print("judge on q3:", j["scores"], "| keyword grader:", r_rich["results"][2]["score"])
h.check("judge produced scores for q3", all(1 <= v <= 5 for v in j["scores"].values()))

# %% [markdown]
# ## 8. What you learned, and what comes next
# | You saw | Where it returns |
# |---|---|
# | New data kinds make *new questions* answerable (changes, flags, vendors, money, text) | ch3 models them properly |
# | A timeline across node types finds the suspect list fast | ch4 stores time as memory |
# | Hypotheses are tested, not believed | ch5 plans and verifies them |
# | Tickets/logs are found by meaning, structure by Cypher | ch2 routes between them |
# | Tools and permissions around the investigator | ch6 |
# | Measure, change, re-measure | ch7 closes the loop, ch8 makes it cheaper |
#
# **Try it yourself (vibe-coding prompts):**
# ```text
# Prompt 1: Add a Cypher query to scn.EVIDENCE named 'blast_radius' that, given a Service, returns every Region it runs in and
#           every Queue it publishes to. Acceptance: for payment-service it returns eu-west-1, us-east-1, payments-dlq, payments-retry-queue.
# Prompt 2: Add a GOLD_QA case about alarm-92 and make sure both systems are re-evaluated. Acceptance: compare_runs shows 9 cases.
# Prompt 3: Write a hypothesis 'H6: notification-service cfg-311' and a Cypher check that eliminates it with evidence.
# ```
#
# **Honest limits.** The data is synthetic and tiny; the keyword grader is crude; the dossier is built with hand-written
# queries (later chapters let the agent choose its own queries). The point is the *method*, not the numbers.

# %%
print(h.cost_report())

# %% [markdown]
# ## Observability — what you can look at now
# Everything above ran with the backend chosen by `OBSERVABILITY_BACKEND` in `.env`.
# **Phoenix** (http://localhost:6006): project *"ch0-whats-changed-new-problem"*. Look at the two `eval_case:inc2001-qa:*` span groups:
# the `no-context` spans have tiny prompts and wrong answers, the `graph-context` spans carry the whole dossier. Compare token counts
# and latency to see what the graph context costs. **DeepEval** (http://localhost:8765): re-scores a sample of the judged answers.

# %%
obs_result = h.obs_finish()
print(h.cost_report())
