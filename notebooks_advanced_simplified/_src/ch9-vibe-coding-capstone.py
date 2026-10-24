# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > Beginner (tier 1/3, no notebook for this topic)  →  Intermediate (tier 2/3, no notebook for this topic)  →  **Advanced (tier 3/3) ← you are here**

# %% [markdown]
# # Chapter 9 (Advanced, Simplified) — Vibe-Coding Capstone
#
# **Who this is for:** you finished the chapter notebooks (Ch1-Ch8) and now want to *use* the repo
# the way people really build with AI coding assistants today.
#
# **What you will do**
# 1. Learn **vibe coding**: directing an AI assistant with *precise intent* and *fast verification*.
# 2. Turn the repo's 51 skills into a **graph in Neo4j** and ask "which skills help with X?".
# 3. **Run every skill's benchmark** and store PASS/FAIL on the graph (honestly — failures are shown).
# 4. Build an **end-to-end incident agent** that uses at least one skill from **each of the 8 chapters**.
# 5. **Build a brand-new skill the repo way** (SKILL.md + lib.py + cli.py + benchmark) from a prompt.
# 6. Take away a **prompt cookbook** and a **review checklist** for AI-generated code.
#
# No LLM and no API key is needed. *You* play the AI assistant's director; the "generated" code is
# shown so you can practise the part that matters most: **verifying** it.
#
# > Needs Neo4j: `docker compose -f docker-compose/docker-compose.yml up -d`.
# > This notebook wipes the database at the start of Section 2.

# %% [markdown]
# ## 1. What is vibe coding?
#
# **Vibe coding** = you describe *what you want* in natural language, an AI assistant writes the code,
# and you steer by running and checking the result instead of typing every line yourself.
#
# The danger is the word "vibe": if you only *feel* that the code is right, you ship bugs.
# The craft is a tight loop:
#
# ```
#  intent  ->  prompt  ->  generate  ->  VERIFY  ->  iterate
#    ^                                               |
#    +-----------------------------------------------+
# ```
#
# * **Intent** – one sentence of what "done" looks like.
# * **Prompt** – the intent made precise enough that a stranger could implement it.
# * **Generate** – the assistant writes code (you do not trust it yet).
# * **Verify** – run something that can say *no*: a benchmark, a test, a linter, a Cypher count.
# * **Iterate** – paste the failure back, tighten the prompt, repeat.
#
# ### Anatomy of a good prompt: four parts
# | Part | Question it answers | Example |
# |---|---|---|
# | **Context** | Where am I, what exists already? | "Repo uses one folder per skill with SKILL.md, lib.py, cli.py" |
# | **Constraint** | What must/must not happen? | "Python stdlib only; no network; ASCII output" |
# | **Example** | What does a right answer look like? | "tier=critical, 3x latency -> SEV1" |
# | **Acceptance test** | How will I know it works? | "`python cli.py benchmark` exits 0" |
#
# The cell below builds prompts from those four parts and compares a vague prompt with a precise one.

# %%
import adv_helpers as a

PROMPT_PARTS = ("context", "constraint", "example", "acceptance")


def compose_prompt(goal, context, constraint, example, acceptance):
    """Assemble the four-part prompt; refuses to build one with an empty part."""
    parts = dict(context=context, constraint=constraint, example=example, acceptance=acceptance)
    missing = [k for k, v in parts.items() if not v.strip()]
    if missing:
        raise ValueError(f"prompt is missing: {missing}")
    return (f"GOAL: {goal}\nCONTEXT: {context}\nCONSTRAINTS: {constraint}\n"
            f"EXAMPLE: {example}\nACCEPTANCE TEST: {acceptance}")


def prompt_score(text: str) -> int:
    """Crude lint: how many of the four anatomy parts does a prompt mention? (0-4)"""
    t = text.lower()
    markers = {"context": ("context", "repo", "existing"), "constraint": ("constraint", "must", "only", "never", "no "),
               "example": ("example", "e.g.", "->"), "acceptance": ("acceptance", "exits 0", "passes", "assert")}
    return sum(any(m in t for m in ms) for ms in markers.values())


BAD = "make something that tells me how bad an incident is"
GOOD = compose_prompt(
    goal="Add a skill that maps an incident's signals to a severity SEV1..SEV4.",
    context="Repo skills live in skills/<chapter>/<slug>/ with SKILL.md, lib.py, cli.py.",
    constraint="Python 3.10 stdlib only; no network; raise ValueError on bad input; ASCII output.",
    example="tier=critical, latency 3x threshold, 2% errors, 30% users -> SEV1 and page on-call.",
    acceptance="`python cli.py --help` exits 0 and `python cli.py benchmark` exits 0 with a negative case.",
)
print("BAD  prompt:", BAD, "\n  anatomy score:", prompt_score(BAD), "/ 4\n")
print("GOOD prompt:\n" + GOOD, "\n  anatomy score:", prompt_score(GOOD), "/ 4")
a.check("a vague prompt scores below a precise one", prompt_score(BAD) < prompt_score(GOOD) == 4)
try:
    compose_prompt("x", "", "c", "e", "a")
except ValueError as exc:
    a.check("compose_prompt refuses an empty part", "context" in str(exc))

# %% [markdown]
# ### Why this repo is "vibe-friendly"
# An AI assistant is only as good as the *feedback it can get*. Every skill here ships three things that
# make verification cheap:
#
# 1. **`SKILL.md` – the contract.** When to use it, when *not* to, process, red flags. The assistant reads
#    it instead of guessing.
# 2. **`lib.py` – stdlib-only code.** No installs, no network -> deterministic, runs anywhere.
# 3. **`cli.py benchmark` – a self-check.** One command that answers "does this still work?" with an exit code.
#
# Because the "verify" step is one command, the vibe-coding loop stays fast. That is the whole trick.

# %% [markdown]
# ## 2. Skill discovery: the skills themselves as a graph
#
# The repo has 51 skills. Finding the right one by browsing folders does not scale — but a *graph* does:
#
# ```
# (:Chapter {number, folder}) -[:HAS_SKILL]-> (:Skill {slug, description})
# (:Skill) -[:SIMILAR {score}]- (:Skill)      <- computed from shared keywords
# ```
#
# `a.skill_catalog()` reads every `SKILL.md` frontmatter. We load it into Neo4j (this is the
# "skills as a graph" idea: an agent can *query* its own capabilities).

# %%
CHAPTER_NUMBER = {"crisis": 1, "architecture": 2, "knowledge-representation": 3, "memory": 4,
                  "reasoning-planning": 5, "tool-orchestration": 6, "self-evolution": 7, "optimization": 8}

catalog = a.skill_catalog()
print("skills found:", len(catalog))
a.check("catalog lists 51 skills", len(catalog) == 51)
a.check("all 8 chapters present", {c["chapter"] for c in catalog} == set(CHAPTER_NUMBER))

a.clear_graph()
for folder, num in CHAPTER_NUMBER.items():
    a.run("MERGE (:Chapter {folder:$f, number:$n})", f=folder, n=num)
for row in catalog:
    a.run("MATCH (c:Chapter {folder:$f}) "
          "MERGE (s:Skill {slug:$slug}) SET s.description=$d, s.chapter=$n "
          "MERGE (c)-[:HAS_SKILL]->(s)", f=row["chapter"], slug=row["slug"], d=row["description"],
          n=CHAPTER_NUMBER[row["chapter"]])
a.show(a.run("MATCH (c:Chapter)-[:HAS_SKILL]->(s) RETURN c.number AS chapter, c.folder AS folder, "
             "count(s) AS skills ORDER BY chapter"))

# %% [markdown]
# ### Computing `SIMILAR` edges in Python, storing them in the graph
# Two skills are *similar* if their descriptions share distinctive words. We drop very common words
# (appearing in >15% of skills), compute **Jaccard similarity** (shared / total distinct words), and keep
# each skill's top 3 neighbours above a threshold. Python computes; Neo4j stores and later answers questions.

# %%
import re
from collections import Counter

STOP = set("""about above after agent agents against also among and are because been before being between both but
can cannot chapter does each for from given has have how into its more most not only other over per same should
skill skills that the their then these this those through use used uses using what when where which while with
without would your you""".split())


def tokens(row):
    text = row["slug"].replace("-", " ") + " " + row["description"]
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP}


tok = {r["slug"]: tokens(r) for r in catalog}
df = Counter(w for ws in tok.values() for w in ws)
common = {w for w, n in df.items() if n > 0.15 * len(catalog)}
tok = {s: ws - common for s, ws in tok.items()}

pairs = {}
for s1, w1 in tok.items():
    scored = []
    for s2, w2 in tok.items():
        if s1 != s2 and w1 and w2:
            j = len(w1 & w2) / len(w1 | w2)
            if j >= 0.08:
                scored.append((j, s2))
    for j, s2 in sorted(scored, reverse=True)[:3]:
        pairs[tuple(sorted((s1, s2)))] = round(j, 3)

for (s1, s2), j in pairs.items():
    a.run("MATCH (x:Skill {slug:$a}), (y:Skill {slug:$b}) MERGE (x)-[r:SIMILAR]->(y) SET r.score=$j",
          a=s1, b=s2, j=j)
print("dropped generic words:", sorted(common)[:12], "...")
print("SIMILAR edges stored:", a.run("MATCH ()-[r:SIMILAR]->() RETURN count(r) AS c")[0]["c"])
a.show(a.run("MATCH (x:Skill)-[r:SIMILAR]->(y:Skill) RETURN x.slug AS skill_a, y.slug AS skill_b, "
             "r.score AS score ORDER BY score DESC LIMIT 8"))

# %% [markdown]
# ### "Which skills help with X?" — a keyword-retrieval Cypher query
# We lowercase the question, keep meaningful words, and let Cypher count how many appear in each
# skill's slug + description. More hits = better match. This is plain keyword retrieval — no embeddings —
# exactly the kind of tiny, checkable tool an agent can call.

# %%
WHICH_SKILLS = """
MATCH (c:Chapter)-[:HAS_SKILL]->(s:Skill)
WITH c, s, toLower(s.slug + ' ' + s.description) AS text
WITH c, s, [k IN $kws WHERE text CONTAINS k] AS hits
WHERE size(hits) > 0
RETURN c.number AS ch, s.slug AS skill, size(hits) AS hits, hits AS matched
ORDER BY hits DESC, ch, skill LIMIT $limit
"""


def which_skills(question: str, limit: int = 5) -> list[dict]:
    kws = sorted({w for w in re.findall(r"[a-z]{3,}", question.lower()) if w not in STOP})
    return a.run(WHICH_SKILLS, kws=kws, limit=limit)


for q in ["which skills help with memory consistency", "cost of routing models", "tool selection for an agent"]:
    print("\nQ:", q)
    a.show(which_skills(q, 4))

top = which_skills("how do I choose a memory consistency model")
a.check("memory question finds the memory-consistency skill",
        any(r["skill"] == "memory-consistency-model-selector" for r in top))
a.check("nonsense question returns nothing", which_skills("zzzqqq xxyyzz") == [])

# %% [markdown]
# ## 3. Verify everything: run all 51 benchmarks
#
# Remember the loop: *verify* is the step that separates vibe coding from vibes.
# `a.run_benchmark(chapter, slug)` runs each skill's own `cli.py benchmark` in a subprocess
# (one skill, `mcp-gateway-two-meta-tools`, names it `prompt-budget`). We store the result on each
# `:Skill` node so the graph now knows which of its capabilities are healthy.
# **Failures are reported, not hidden.**

# %%
import time

t0 = time.time()
results = {}
for row in catalog:
    ok, line = a.run_benchmark(row["chapter"], row["slug"])
    results[row["slug"]] = (ok, line)
    a.run("MATCH (s:Skill {slug:$slug}) SET s.bench_pass=$ok, s.bench_line=$line",
          slug=row["slug"], ok=ok, line=line)
passed = sum(ok for ok, _ in results.values())
print(f"{passed}/{len(results)} benchmarks passed in {time.time() - t0:.1f}s")

# %%
table = a.run("""
MATCH (c:Chapter)-[:HAS_SKILL]->(s:Skill)
RETURN c.number AS chapter, c.folder AS folder, count(s) AS skills,
       sum(CASE WHEN s.bench_pass THEN 1 ELSE 0 END) AS passed,
       sum(CASE WHEN s.bench_pass THEN 0 ELSE 1 END) AS failed
ORDER BY chapter""")
a.show(table)
failures = a.run("MATCH (s:Skill) WHERE NOT s.bench_pass RETURN s.slug AS skill, s.bench_line AS first_line")
print("\nFAILING SKILLS:" if failures else "\nNo failing skills.")
a.show(failures)
a.check("every skill has a stored benchmark verdict",
        a.run("MATCH (s:Skill) WHERE s.bench_pass IS NULL RETURN count(s) AS c")[0]["c"] == 0)
a.check("per-chapter table sums to 51", sum(r["skills"] for r in table) == 51)
BENCH_PASSED = passed

# %% [markdown]
# ## 4. The end-to-end agent: one skill from every chapter
#
# The incident: **"checkout API is slow"**. `a.MiniAgent` is a ~30-line loop: *plan -> act -> observe -> log*.
# Our "tools" are real repo skills (plus Cypher against the DevOps graph). The plan is a fixed list here;
# in production an LLM would produce it — the tools and the trace would be identical.
#
# | Step | Chapter | Skill used as a tool | Question it answers |
# |---|---|---|---|
# | 1 | Ch1 | `context-failure-classifier`, `vector-vs-graph-retrieval-selector` | What failure mode are we fighting? Vector, graph or hybrid? |
# | 2 | Ch2 | `dual-graph-router` | Vertical knowledge graph or horizontal workflow graph? |
# | 3 | Ch3 | `three-graph-router` | Which graph does a new fact belong in? |
# | 4 | Ch4 | `memory-consistency-model-selector` | How consistent must shared incident memory be? |
# | 5 | Ch5 | `investigation-dag-planner` | Which hypotheses to test, in what order? |
# | 6 | Ch6 | `rag-mcp-tool-selection` | Which operational tool fits the fix? |
# | 7 | Ch7 | `four-layer-eval-cascade` | Is the agent's answer actually good? |
# | 8 | Ch8 | `model-routing-selector` | What model tier is cheap enough per step? |
#
# First we add the DevOps graph **next to** the skill graph (`reset=False`, so Section 2-3 data survives).

# %%
counts = a.load_devops_graph(reset=False)
print("graph now holds skills + incident data:", counts)
a.show(a.run("MATCH (i:Incident)-[:AFFECTS]->(s:Service) RETURN i.id AS incident, i.title AS title, "
             "i.status AS status, s.name AS service"))

# %% [markdown]
# ### Register the tools
# Each `@agent.tool` wraps one skill. Notice tools are small and return small, loggable dicts — that is what
# makes the agent's behaviour *verifiable* later.

# %%
import json

agent = a.MiniAgent("INC-1001: checkout API is slow")
CH = {}  # tool name -> chapter number it exercises (used by the verification battery)

clf = a.load_skill("crisis", "context-failure-classifier")
vgr = a.load_skill("crisis", "vector-vs-graph-retrieval-selector")
dgr = a.load_skill("architecture", "dual-graph-router")
tgr = a.load_skill("knowledge-representation", "three-graph-router")
mcm = a.load_skill("memory", "memory-consistency-model-selector")
dag = a.load_skill("reasoning-planning", "investigation-dag-planner")
rag = a.load_skill("tool-orchestration", "rag-mcp-tool-selection")
cas = a.load_skill("self-evolution", "four-layer-eval-cascade")
mrs = a.load_skill("optimization", "model-routing-selector")


@agent.tool("classify_failure")
def classify_failure(symptom):                                   # Ch1
    r = clf.classify(symptom)["primary"]
    return {"mode": r["failure_mode"], "root_flaw": r["root_flaw"], "cure": r["cure"]}


@agent.tool("choose_retrieval")
def choose_retrieval():                                          # Ch1
    r = vgr.recommend(query_scope="local", multi_hop=True, temporal=True, agentic=True, name="checkout-rca")
    return {"strategy": r["recommendation"]}


@agent.tool("route_graph")
def route_graph(request):                                        # Ch2
    d = dgr.route(request)
    return {"target": d.target, "hint": d.node_hint[:70]}


@agent.tool("place_fact")
def place_fact(origin):                                          # Ch3
    rec = tgr.Record({"fact": "deploy-481 changed pool"}, origin=origin, entity_resolved=True,
                     has_provenance=True, confidence=0.95)
    return {"graph": tgr.route(rec)["graph"]}


@agent.tool("pick_memory_model")
def pick_memory_model():                                         # Ch4
    r = mcm.recommend_model(mcm.Operation(shared_authoritative_state=2, conflict_intolerance=2))
    return {"model": r["recommended"]}


def graph_test(task):
    """Hypothesis test backed by the REAL graph: returns (confirmed, evidence)."""
    if task.id == "h_db_pool":
        rows = a.run("MATCH (d:Deployment)-[:CHANGED]->(db:Database) WHERE toLower(d.change) CONTAINS 'pool' "
                     "RETURN d.id AS dep, d.change AS change, db.name AS db")
        return (bool(rows), f"{rows[0]['dep']}: {rows[0]['change']} on {rows[0]['db']}" if rows else "none")
    if task.id == "h_cache":
        rows = a.run("MATCH (d:Deployment)-[:CHANGED]->(:Cache) RETURN d.id AS dep")
        return (bool(rows), "no deployment touched session-cache")
    return (False, "context gathered")


@agent.tool("plan_investigation")
def plan_investigation():                                        # Ch5
    tasks = dag.tasks_from_dicts([
        {"id": "collect_alarm", "depends_on": [], "duration_s": 2},
        {"id": "h_cache", "depends_on": ["collect_alarm"], "duration_s": 3},
        {"id": "h_db_pool", "depends_on": ["collect_alarm"], "duration_s": 3},
        {"id": "h_network", "depends_on": ["h_cache", "h_db_pool"], "duration_s": 5},
    ])
    plan = dag.build_investigation_dag(tasks)
    out = dag.execute_with_early_termination(plan, tasks, graph_test)
    return {"phases": len(plan.phases), "confirmed": out["confirmed"], "skipped": out["phases_skipped"]}


REGISTRY = a.SKILLS_DIR / "tool-orchestration" / "rag-mcp-tool-selection" / "sample-aws-tools.json"


@agent.tool("select_tool")
def select_tool(query):                                          # Ch6
    r = rag.select(query, REGISTRY, top_k=3)
    return {"picked": [t["name"] for t in r["selected"]], "token_reduction_pct": r["reduction_pct"]}


@agent.tool("evaluate_answer")
def evaluate_answer(answer, premise):                            # Ch7
    rep = cas.run_cascade({"execution_id": "inc-1001", "query": "why is checkout slow?",
                           "answer": answer, "context_premise": premise})
    return {"verdict": rep["overall_verdict"], "stopped_at_layer": rep["stopped_at_layer"],
            "intervention": rep["recommended_intervention"]}


@agent.tool("optimize_cost")
def optimize_cost():                                             # Ch8
    r = mrs.cascade_route("AlertClassifier", confidence=0.92)
    p = mrs.pipeline_cost()
    return {"tier": r["tier"], "pipeline_saving_pct": p["reduction_pct"]}


@agent.tool("lookup_past_fix")
def lookup_past_fix():                                           # graph memory (Ch4 flavour)
    r = a.run("MATCH (i:Incident {status:'resolved'})-[:AFFECTS]->(:Service {name:'checkout-api'}) "
              "RETURN i.id AS id, i.root_cause AS cause, i.fix AS fix")[0]
    return r


@agent.tool("find_skills")
def find_skills(question):                                       # the skill graph from Section 2
    return [r["skill"] for r in which_skills(question, 3)]


CH.update({"classify_failure": 1, "choose_retrieval": 1, "route_graph": 2, "place_fact": 3,
           "pick_memory_model": 4, "plan_investigation": 5, "select_tool": 6, "evaluate_answer": 7,
           "optimize_cost": 8})
print("registered tools:", sorted(agent.tools))

# %% [markdown]
# ### Plan -> act -> observe
# The plan below is the agent's "thinking". Each `agent.act(...)` calls a tool, stores the observation in
# `agent.memory`, and logs the call to Neo4j via `AgentTrace`. Later steps *use* earlier observations
# (e.g. the fix is chosen from the confirmed root cause).

# %%
symptom = "Every investigation starts from scratch; the agent forgot last week's checkout incident."
print("1.", agent.act("classify_failure", symptom=symptom))
print("2.", agent.act("choose_retrieval"))
print("3.", agent.act("route_graph", request="which services depend on orders-db and what changed recently"))
print("4.", agent.act("place_fact", origin="structured"))
print("5.", agent.act("pick_memory_model"))
print("6.", agent.act("plan_investigation"))
root = agent.memory["plan_investigation"]["confirmed"]
past = agent.act("lookup_past_fix")
print("7.", past)
print("8.", agent.act("select_tool", query="roll back ECS service deployment to previous task definition"))
answer = (f"Root cause is {root}: deployment deploy-481 reduced the orders-db connection pool from 100 to 20, "
          f"as in INC-0877. Fix: raise the pool size back to 100.")
premise = ("deploy-481 changed orders-db connection pool 100 -> 20. INC-0877 root cause: db connection pool "
           "too small, fix raise pool size to 100. checkout-api depends on orders-service which depends on orders-db.")
print("9.", agent.act("evaluate_answer", answer=answer, premise=premise))
print("10.", agent.act("optimize_cost"))
print("11.", agent.act("find_skills", question="rollback a risky change safely"))
print("\nFINAL ANSWER:", answer)

# %% [markdown]
# ### The trace is a graph too
# `AgentTrace` stored `(:Run)-[:STEP]->(:ToolCall)-[:NEXT]->(:ToolCall)`. Printing the timeline *from Neo4j*
# (not from Python memory) proves the run is auditable after the fact — the same idea as Ch7's execution graph.

# %%
timeline = agent.trace.timeline()
a.show(timeline)
chain = a.run("MATCH p=(:ToolCall {id:$first})-[:NEXT*]->(last:ToolCall) RETURN length(p) AS hops",
              first=f"{agent.trace.run_id}-1")
print("\nNEXT-chain length from step 1:", max(r["hops"] for r in chain))

a.check("trace holds every act() call", len(timeline) == 11 and all(r["ok"] for r in timeline))
a.check("NEXT chain links all steps", max(r["hops"] for r in chain) == len(timeline) - 1)
a.check("agent used a skill from each of the 8 chapters", set(CH.values()) == set(range(1, 9)))
a.check("every chapter tool appears in the persisted trace",
        set(CH) <= {r["tool"] for r in timeline})
a.check("root cause found via graph evidence", root == "h_db_pool")
a.check("cascade accepts the grounded answer", agent.memory["evaluate_answer"]["verdict"] == "PASS")

# %% [markdown]
# ## 5. Build a NEW skill live, the repo way
#
# Now the vibe-coding exercise. Task: *"the agent should triage how severe an incident is."*
# Following `CONTRIBUTING.md`, a skill needs: a 7-section **SKILL.md** (+ Security Posture and Source
# Attribution), a **stdlib-only `lib.py`**, a **`cli.py`** whose `--help` prints the SKILL.md description, and a
# **`benchmark`** subcommand with at least one negative case.
#
# **Step 1 — the prompt you would give your AI assistant.** Look at how every sentence maps to the four parts.

# %%
SKILL_PROMPT = compose_prompt(
    goal="Create a new skill `incident-severity-triage` that maps incident signals (service tier, latency vs "
         "threshold, error rate %, users affected %, workaround available) to SEV1..SEV4 and a page-on-call flag.",
    context="Repo layout: skills/<chapter>/<slug>/{SKILL.md,lib.py,cli.py}. Mirror skills/crisis/context-failure-classifier. "
            "SKILL.md needs frontmatter (name, description, chapter-source), Overview, When to Use, When NOT to Use, "
            "Process, Rationalizations, Red Flags, Non-Negotiable Verification, Security Posture, Source Attribution.",
    constraint="lib.py: Python 3.10 standard library ONLY, no network, deterministic, raise ValueError on invalid input "
               "(unknown tier, negative numbers, percentages above 100). cli.py: argparse, ASCII-only output, "
               "`--help` prints the SKILL.md description, subcommands `triage` and `benchmark`.",
    example="tier=critical, latency 2400ms vs threshold 800ms, 2% errors, 30% users, no workaround -> SEV1, page on-call. "
            "tier=low, latency at threshold, 0% errors, 0% users -> SEV4, no page.",
    acceptance="`python cli.py --help` exits 0; `python cli.py benchmark` exits 0 and includes negative cases "
               "(invalid tier raises, a workaround never raises severity); lib.py imports only stdlib modules.",
)
print(SKILL_PROMPT)
a.check("the skill prompt has all four anatomy parts", prompt_score(SKILL_PROMPT) == 4)

# %% [markdown]
# **Step 2 — the "generated" result.** Below is what a good assistant returns for that prompt. We write it to a
# **scratch folder inside `notebooks_advanced_simplified/`** (never into `skills/`) and delete it at the end.
# Read the scoring rules in `lib.py`: the example in the prompt (score 10 -> SEV1) is one of the benchmark cases.

# %%
import shutil
from pathlib import Path

SCRATCH = Path(a.HERE) / "_scratch_ch9"
shutil.rmtree(SCRATCH, ignore_errors=True)
NEW = SCRATCH / "incident-severity-triage"
NEW.mkdir(parents=True)

SKILL_MD = r'''---
name: incident-severity-triage
description: |
  Map an incident's observable signals (service tier, latency versus its
  alarm threshold, error rate, share of users affected, whether a workaround
  exists) to a severity SEV1..SEV4 and a page-on-call decision. Use at the
  start of an investigation to decide how loudly to react.
  NOT for root-cause analysis, NOT for postmortem writing.
chapter-source: "Agentic GraphRAG (O'Reilly) Ch9 vibe-coding capstone - demo skill written live, not part of the book"
---

# Incident Severity Triage

## Overview

A fixed, explainable scoring rule: points for service tier, latency ratio,
error rate and users affected, minus one point if a workaround exists. The
score maps to SEV1 (score >= 9), SEV2 (>= 6), SEV3 (>= 3), SEV4 (otherwise).
SEV1 and SEV2 page the on-call engineer. Every decision lists its reasons.

## When to Use

- First minutes of an incident, before the root cause is known
- Deciding whether an alarm should page a human
- Phrases: "how bad is this", "should we page", "what severity"

## When NOT to Use

- Finding the root cause (use an investigation planner)
- Legal or customer-contract severity definitions - this is an engineering heuristic

## Process

| Step | Input | Action | Output | Verification |
|------|-------|--------|--------|--------------|
| 1 | tier, latency, threshold, error %, users % | `lib.triage(...)` | severity, score, page flag, reasons | reasons add up to the score |
| 2 | bad input | `lib.triage(...)` | ValueError | invalid input is rejected, never guessed |

## Rationalizations

| Agent rationalization | Documented rebuttal |
|-----------------------|--------------------|
| "Latency is high, so it must be SEV1." | Severity combines tier, errors and reach; a slow low-tier service stays low. |
| "A workaround exists, so downgrade to SEV4." | A workaround subtracts one point only; it never erases the other signals. |

## Red Flags

- Every incident comes back SEV1: thresholds passed in are probably wrong.
- Severity changed between two runs on the same inputs: the skill must be deterministic.

## Non-Negotiable Verification

1. `python cli.py benchmark` must exit 0 (includes negative cases).
2. `python cli.py --help` must exit 0 and print the description above.

## Security Posture

- **Prompt injection.** Inputs are typed numbers and a fixed tier vocabulary; free text is never interpreted.
- **Data exfiltration.** No network, no file writes.
- **Privilege escalation.** No shell, no eval, no dynamic import; the page flag is advice for a human.

## Source Attribution

Demo skill created in the Ch9 vibe-coding capstone notebook of the companion repo for
*Agentic GraphRAG* (O'Reilly) by Anthony Alcaraz and Sam Julien. Not a book chapter primitive.
'''

LIB_PY = r'''"""incident-severity-triage: explainable SEV1..SEV4 scoring (stdlib only)."""
from __future__ import annotations

TIER_POINTS = {"critical": 3, "high": 2, "medium": 1, "low": 0}


def _band(value: float, cuts: list) -> tuple:
    """Return (points, threshold_hit) for the first cut with value >= cut."""
    for cut, points in cuts:
        if value >= cut:
            return points, cut
    return 0, None


def triage(tier: str, latency_ms: float, threshold_ms: float, error_rate_pct: float,
           users_affected_pct: float, has_workaround: bool = False) -> dict:
    """Score an incident. Raises ValueError on invalid input."""
    if tier not in TIER_POINTS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(TIER_POINTS)}")
    if threshold_ms <= 0:
        raise ValueError("threshold_ms must be > 0")
    for name, v in (("latency_ms", latency_ms), ("error_rate_pct", error_rate_pct),
                    ("users_affected_pct", users_affected_pct)):
        if v < 0:
            raise ValueError(f"{name} must be >= 0")
    if error_rate_pct > 100 or users_affected_pct > 100:
        raise ValueError("percentages must be <= 100")

    ratio = latency_ms / threshold_ms
    reasons, score = [], TIER_POINTS[tier]
    reasons.append(f"tier {tier}: +{score}")
    for label, value, cuts in (
        ("latency ratio", ratio, [(3.0, 3), (1.5, 2), (1.0000001, 1)]),
        ("error rate %", error_rate_pct, [(5.0, 3), (1.0, 2), (0.0000001, 1)]),
        ("users affected %", users_affected_pct, [(50.0, 3), (10.0, 2), (0.0000001, 1)]),
    ):
        pts, _ = _band(value, cuts)
        score += pts
        reasons.append(f"{label} {value:.2f}: +{pts}")
    if has_workaround:
        score = max(0, score - 1)
        reasons.append("workaround available: -1")

    severity = "SEV1" if score >= 9 else "SEV2" if score >= 6 else "SEV3" if score >= 3 else "SEV4"
    return {"severity": severity, "score": score, "page_oncall": severity in ("SEV1", "SEV2"),
            "reasons": reasons}
'''

CLI_PY = r'''#!/usr/bin/env python3
"""incident-severity-triage CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib import triage  # noqa: E402

SKILL_MD = HERE / "SKILL.md"


def _skill_description() -> str:
    lines, in_desc, fences = [], False, 0
    for line in SKILL_MD.read_text(encoding="utf-8").splitlines():
        if line.strip() == "---":
            fences += 1
            if fences == 2:
                break
            continue
        if line.startswith("description:"):
            in_desc = True
            continue
        if in_desc:
            if line and not line[0].isspace():
                break
            lines.append(line.strip())
    return " ".join(x for x in lines if x)


def cmd_triage(args):
    out = triage(args.tier, args.latency_ms, args.threshold_ms, args.error_rate,
                 args.users_pct, args.workaround)
    print(json.dumps(out, indent=2))


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def cmd_benchmark(args):
    failures = []

    def expect(name, ok):
        if not ok:
            failures.append(name)

    r = triage("critical", 2400, 800, 2.0, 30, False)
    expect("checkout example is SEV1 and pages", r["severity"] == "SEV1" and r["page_oncall"])
    r = triage("low", 800, 800, 0, 0)
    expect("quiet low-tier service is SEV4, no page", r["severity"] == "SEV4" and not r["page_oncall"])
    expect("score 6 is SEV2", triage("high", 1200, 800, 1.0, 0)["severity"] == "SEV2")
    expect("score 3 is SEV3", triage("medium", 900, 800, 0.5, 0)["severity"] == "SEV3")
    base = triage("critical", 2400, 800, 2.0, 30, False)["score"]
    expect("workaround never raises the score", triage("critical", 2400, 800, 2.0, 30, True)["score"] <= base)
    expect("more errors never lower the score",
           triage("high", 900, 800, 6.0, 5)["score"] >= triage("high", 900, 800, 0.5, 5)["score"])
    expect("deterministic", triage("high", 900, 800, 1, 5) == triage("high", 900, 800, 1, 5))
    expect("reasons are listed", len(triage("high", 900, 800, 1, 5)["reasons"]) >= 4)
    expect("NEGATIVE: unknown tier raises", _raises(lambda: triage("urgent", 1, 1, 0, 0)))
    expect("NEGATIVE: negative latency raises", _raises(lambda: triage("low", -1, 1, 0, 0)))
    expect("NEGATIVE: users pct above 100 raises", _raises(lambda: triage("low", 1, 1, 0, 101)))
    expect("NEGATIVE: zero threshold raises", _raises(lambda: triage("low", 1, 0, 0, 0)))

    total = 12
    print("=" * 60)
    print(f"incident-severity-triage benchmark - {total - len(failures)}/{total} passed")
    print("=" * 60)
    for f in failures:
        print("  - FAIL:", f)
    sys.exit(1 if failures else 0)


def main():
    p = argparse.ArgumentParser(description=_skill_description())
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triage", help="Score one incident")
    t.add_argument("--tier", required=True)
    t.add_argument("--latency-ms", type=float, required=True)
    t.add_argument("--threshold-ms", type=float, required=True)
    t.add_argument("--error-rate", type=float, default=0.0)
    t.add_argument("--users-pct", type=float, default=0.0)
    t.add_argument("--workaround", action="store_true")
    t.set_defaults(func=cmd_triage)
    b = sub.add_parser("benchmark", help="Verification gate battery")
    b.set_defaults(func=cmd_benchmark)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
'''

(NEW / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
(NEW / "lib.py").write_text(LIB_PY, encoding="utf-8")
(NEW / "cli.py").write_text(CLI_PY, encoding="utf-8")
print("wrote:", sorted(p.name for p in NEW.iterdir()), "to", NEW.name, "(scratch, not skills/)")

# %% [markdown]
# **Step 3 — the acceptance test.** Never trust generated code on sight. We run exactly what the prompt
# promised, *plus* a repo-style lint (required SKILL.md sections, stdlib-only imports in `lib.py`).
# Each failing result would go straight back into the next prompt — that is the **iterate** step.

# %%
import ast
import subprocess
import sys

REQUIRED_SECTIONS = ["## Overview", "## When to Use", "## When NOT to Use", "## Process", "## Rationalizations",
                     "## Red Flags", "## Non-Negotiable Verification", "## Security Posture",
                     "## Source Attribution"]


def py(skill_dir, *argv):
    return subprocess.run([sys.executable, str(skill_dir / "cli.py"), *argv], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=60)


def lint_skill(skill_dir: Path) -> list[str]:
    """Return a list of problems (empty = repo-style clean)."""
    problems = []
    md = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    for key in ("name:", "description:", "chapter-source:"):
        if key not in md.split("---")[1]:
            problems.append(f"frontmatter missing {key}")
    problems += [f"missing section {s}" for s in REQUIRED_SECTIONS if s not in md]
    tree = ast.parse((skill_dir / "lib.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        mods = [x.name for x in node.names] if isinstance(node, ast.Import) else \
               [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
        for m in mods:
            if m.split(".")[0] not in sys.stdlib_module_names:
                problems.append(f"non-stdlib import {m}")
    for path in skill_dir.glob("*.py"):
        if any(ord(ch) > 127 for ch in path.read_text(encoding="utf-8")):
            problems.append(f"{path.name} contains non-ASCII characters")
    return problems


help_run = py(NEW, "--help")
bench_run = py(NEW, "benchmark")
print(bench_run.stdout)
a.check("--help exits 0", help_run.returncode == 0)
a.check("--help prints the SKILL.md description", "severity SEV1..SEV4" in help_run.stdout.replace("\n", " "))
a.check("benchmark exits 0", bench_run.returncode == 0)
a.check("benchmark has negative cases", bench_run.stdout.count("passed") == 1 and "12/12" in bench_run.stdout)
a.check("repo-style lint is clean", lint_skill(NEW) == [])
demo = py(NEW, "triage", "--tier", "critical", "--latency-ms", "2400", "--threshold-ms", "800",
          "--error-rate", "2", "--users-pct", "30")
print(demo.stdout)
a.check("the prompt's example yields SEV1", json.loads(demo.stdout)["severity"] == "SEV1")

# %% [markdown]
# ### The iterate step: what if the assistant got it wrong?
# Verification only matters if it can *fail*. We simulate a sloppy generation: a `lib.py` that imports a
# third-party package and a SKILL.md missing a section. The same lint catches both, and the message is
# exactly what you would paste back to the assistant.

# %%
BROKEN = SCRATCH / "broken-skill"
BROKEN.mkdir()
(BROKEN / "SKILL.md").write_text(SKILL_MD.replace("## Security Posture", "## Notes"), encoding="utf-8")
(BROKEN / "lib.py").write_text("import requests\n" + LIB_PY, encoding="utf-8")
problems = lint_skill(BROKEN)
print("lint findings to paste back into the next prompt:")
for p_ in problems:
    print("  -", p_)
a.check("lint catches a third-party import", any("requests" in p_ for p_ in problems))
a.check("lint catches a missing SKILL.md section", any("Security Posture" in p_ for p_ in problems))

# %% [markdown]
# Is the lint fair to the real repo? Run the *same* lint over all 51 shipped skills. This is informational:
# any finding is a real inconsistency in the repo (or a too-strict lint rule) — worth a prompt of its own.

# %%
repo_findings = {}
for row in catalog:
    found = lint_skill(a.SKILLS_DIR / row["chapter"] / row["slug"])
    if found:
        repo_findings[row["slug"]] = found
print(f"{len(catalog) - len(repo_findings)}/{len(catalog)} shipped skills pass the notebook lint")
for slug, found in list(repo_findings.items())[:8]:
    print(" ", slug, "->", found[:2])
a.check("lint ran over all 51 shipped skills", len(catalog) == 51)

# %% [markdown]
# ## 6. Prompt cookbook
# Ten ready-to-paste prompts. Each is stored as the four anatomy parts, so you can read *why* it works.
# The last cell of this section lints every prompt (all four parts present) — a cheap guard for your own
# prompt library.

# %%
COOKBOOK = [
    dict(title="Add a Cypher query",
         goal="Add a Cypher query to the notebook that lists services whose owner has more than one critical service.",
         context="Neo4j DevOps graph: (:Service {name,tier,owner}) from graph_helpers.load_devops_graph().",
         constraint="Read-only Cypher; use parameters, not string formatting; return named columns.",
         example="Expected: team-payments with services checkout-api, payment-service.",
         acceptance="Run via g.run(); assert the result has exactly the owner 'team-payments' and 2 services."),
    dict(title="Add a benchmark case",
         goal="Add one benchmark case to skills/<chapter>/<skill>/cli.py covering an input the skill currently mishandles.",
         context="cli.py benchmark collects failures in a list and exits 1 if any; read lib.py first.",
         constraint="Do not change lib.py behaviour; one new case only; keep ASCII output.",
         example="expect('empty string is unclassified', not classify('')['classified'])",
         acceptance="`python cli.py benchmark` exits 0 and the pass count increases by exactly one."),
    dict(title="Debug a failing skill",
         goal="Find and fix why `python cli.py benchmark` fails for this skill.",
         context="Paste the FAILURES block from the benchmark output and the relevant lib.py function.",
         constraint="Smallest possible diff; do not edit the benchmark to make it pass; explain the root cause first.",
         example="FAIL: top cure should be evolving_memory, got ['temporal_evolution'] -> check tie-breaking.",
         acceptance="Benchmark exits 0, and `git diff` touches only lib.py."),
    dict(title="Refactor safely",
         goal="Extract the duplicated scoring loop in lib.py into one helper function.",
         context="Behaviour is pinned by `python cli.py benchmark` (all cases currently pass).",
         constraint="No behaviour change; no new imports; keep function signatures that cli.py uses.",
         example="Before: two similar for-loops. After: one `_score(items, weights)` used twice.",
         acceptance="Benchmark output is identical before and after; lib.py is shorter."),
    dict(title="Explain a skill",
         goal="Explain what this skill does to a beginner in 5 sentences and name one situation to NOT use it.",
         context="Paste the skill's SKILL.md and the signature list of lib.py.",
         constraint="Only claim what the files say; quote the 'When NOT to Use' section; no invented benchmarks.",
         example="'It scores X using Y; do not use it for Z.'",
         acceptance="Every claim can be matched to a line in SKILL.md; reviewer spot-checks two claims."),
    dict(title="Write a notebook cell",
         goal="Write one notebook cell that calls the skill on the DevOps incident and prints a table.",
         context="Notebooks use `import adv_helpers as a`; skills load via a.load_skill(chapter, slug).",
         constraint="No new dependencies, no API keys, deterministic output, under 25 lines.",
         example="a.show([{'service': 'checkout-api', 'severity': 'SEV1'}])",
         acceptance="Cell runs top-to-bottom after a fresh restart with no errors and ends with an a.check(...)."),
    dict(title="Security and secrets review",
         goal="Review this diff for hard-coded secrets, network calls, shell injection and unsafe eval.",
         context="Skills must be stdlib-only, offline and deterministic (see CONTRIBUTING.md).",
         constraint="Report findings with file and line; do not rewrite code; say 'none found' explicitly if clean.",
         example="lib.py:42 uses subprocess with shell=True on user input.",
         acceptance="Findings list is checked against a regex scan for keys/passwords and urls."),
    dict(title="Add input validation",
         goal="Make lib.py reject invalid inputs instead of guessing.",
         context="Current functions accept any value and return a plausible-looking result.",
         constraint="Raise ValueError with a message that names the bad field; stdlib only.",
         example="triage('urgent', ...) -> ValueError: unknown tier 'urgent'",
         acceptance="Add one negative benchmark case per validation rule; benchmark exits 0."),
    dict(title="Add a SIMILAR-style graph relationship",
         goal="Store a relationship between skills that share at least 3 keywords.",
         context="Neo4j has (:Skill {slug, description}); computing similarity in Python is allowed.",
         constraint="Compute in Python, write with MERGE so reruns do not duplicate; parameterised Cypher.",
         example="(:Skill {slug:'dual-graph-router'})-[:SIMILAR {score:0.2}]->(:Skill {slug:'three-graph-router'})",
         acceptance="Run twice; `MATCH ()-[r:SIMILAR]->() RETURN count(r)` is identical both times."),
    dict(title="Turn a failure into a regression test",
         goal="Convert this bug report into a failing benchmark case first, then fix it.",
         context="Paste the symptom, the input that triggered it and the wrong output.",
         constraint="Write the test first and show it failing; then the smallest fix.",
         example="Input 'forgot everything' returned unclassified; expected memory_fragmentation.",
         acceptance="Benchmark fails before the fix and passes after; both runs pasted."),
]
for i, item in enumerate(COOKBOOK, 1):
    text = compose_prompt(item["goal"], item["context"], item["constraint"], item["example"], item["acceptance"])
    print(f"--- {i}. {item['title']} ---\n{text}\n")
a.check("cookbook has 8-10 prompts", 8 <= len(COOKBOOK) <= 10)
a.check("every prompt has all four anatomy parts",
        all(all(it[k].strip() for k in PROMPT_PARTS) for it in COOKBOOK))
a.check("every prompt scores 4/4 on the anatomy lint",
        all(prompt_score(compose_prompt(**{k: v for k, v in it.items() if k != "title"})) == 4 for it in COOKBOOK))

# %% [markdown]
# ## 7. Review checklist for AI-generated code
#
# Run this **every time**, in this order (cheapest check first):
#
# 1. **Verify** – does the acceptance test from your prompt pass? (If you had none, write one first.)
# 2. **Run the benchmark** – `python cli.py benchmark` for the skill, plus `--help` exits 0.
# 3. **Read the diff** – `git diff`: only the files you expected? Any deleted tests or loosened assertions?
# 4. **No secrets** – no keys, tokens, passwords, internal URLs. Credentials come from environment variables.
# 5. **Stdlib-only** – `lib.py` imports nothing outside the standard library (the lint above checks this).
# 6. **Negative cases** – at least one test where the right answer is "no" or an exception.
# 7. **Explain it back** – can *you* say what each function does? If not, ask the assistant to explain, then re-verify.
#
# Items 4 and 5 are automatable. Here is a small secret scanner and a demonstration that it catches a planted
# key while passing our generated skill.

# %%
SECRET_PATTERNS = {
    "aws access key": r"AKIA[0-9A-Z]{16}",
    "hard-coded password": r"(?i)password\s*=\s*['\"][^'\"]{4,}['\"]",
    "api key assignment": r"(?i)(api[_-]?key|secret|token)\s*=\s*['\"][A-Za-z0-9_\-]{12,}['\"]",
    "private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}


def scan_secrets(skill_dir: Path) -> list[str]:
    hits = []
    for path in skill_dir.glob("*"):
        if path.suffix in {".py", ".md", ".json"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for name, pat in SECRET_PATTERNS.items():
                if re.search(pat, text):
                    hits.append(f"{path.name}: {name}")
    return hits


PLANTED = SCRATCH / "planted-secret"
PLANTED.mkdir()
(PLANTED / "lib.py").write_text('API_KEY = "sk_live_abcdef1234567890"\nDB = "x"\nPASSWORD = "hunter2hunter2"\n',
                                encoding="utf-8")
print("generated skill findings:", scan_secrets(NEW) or "none found")
print("planted file findings   :", scan_secrets(PLANTED))
a.check("generated skill has no secrets", scan_secrets(NEW) == [])
a.check("scanner catches planted secrets", len(scan_secrets(PLANTED)) >= 2)

shipped_hits = {r["slug"]: scan_secrets(a.SKILLS_DIR / r["chapter"] / r["slug"]) for r in catalog}
shipped_hits = {k: v for k, v in shipped_hits.items() if v}
print("shipped skills with scanner hits:", shipped_hits or "none")

# %% [markdown]
# ### Clean up and final verification battery
# We delete the scratch folder, confirm `skills/` was not touched (still exactly 51 skills, no
# `incident-severity-triage`), and run the checks that tie the whole notebook together.

# %%
shutil.rmtree(SCRATCH, ignore_errors=True)
print("== Verification battery ==")
a.check("scratch dir removed", not SCRATCH.exists())
a.check("skills/ unchanged: still 51 skills", len(list(a.SKILLS_DIR.glob("*/*/SKILL.md"))) == 51)
a.check("new demo skill was never written into skills/", not list(a.SKILLS_DIR.glob("*/incident-severity-triage")))
a.check("graph has 8 chapters and 51 skills with HAS_SKILL edges",
        a.run("MATCH (:Chapter)-[r:HAS_SKILL]->(:Skill) RETURN count(r) AS c")[0]["c"] == 51)
a.check("SIMILAR edges exist", a.run("MATCH ()-[r:SIMILAR]->() RETURN count(r) AS c")[0]["c"] > 0)
a.check("benchmark verdict stored for all 51 skills", len(results) == 51)
a.check("DevOps graph coexists with the skill graph",
        a.run("MATCH (s:Service) RETURN count(s) AS c")[0]["c"] == 4)
a.check("agent run persisted in Neo4j",
        a.run("MATCH (r:Run {id:$id})-[:STEP]->(c) RETURN count(c) AS c", id=agent.trace.run_id)[0]["c"] == 11)
print(f"\nBenchmarks: {BENCH_PASSED}/51 passed.  Notebook complete.")

# %% [markdown]
# ## Next steps
#
# * **Use it:** `python skills/<chapter>/<skill>/cli.py --help`, then `benchmark`. Start from the skill graph
#   query in Section 2 to find the right skill.
# * **Contribute a skill:** copy the prompt in Section 5, change the goal, keep the acceptance test, and open a PR
#   following `CONTRIBUTING.md` (CI re-runs `--help` and `benchmark` for every skill with zero pip installs).
# * **Make the agent real:** replace the fixed plan in Section 4 with an LLM that picks tools from the skill graph.
#   Keep the tools and the Neo4j trace — they are what make the agent verifiable.
# * **Remember the loop:** intent -> prompt -> generate -> **verify** -> iterate. The assistant writes; *you* decide
#   what "done" means and prove it.
#
# This is a companion to the book *Agentic GraphRAG* (O'Reilly, Anthony Alcaraz and Sam Julien): read the book for
# the theory, run these skills for the practice.
