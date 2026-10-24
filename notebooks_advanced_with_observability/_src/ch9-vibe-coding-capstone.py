# %% [markdown]
# # Chapter 9 — Vibe-Coding Capstone (advanced, real LLM + real Neo4j)
# > **Tier: advanced (real).** Needs OPENAI_API_KEY in notebooks_advanced/.env and the Neo4j container from docker-compose/.
# > **Evaluation focus:** see the section "Evaluation and LLM-as-a-Judge" near the end — every notebook in this tier treats eval and judging as first-class.
#
# > **Estimated OpenAI cost per full run (measured 2026-10-24 at `gpt-4o` rates): about $0.21.** 43 LLM calls (27 of them LLM-as-a-Judge), 30,809 input / 13,251 output tokens, plus a few thousand `text-embedding-3-large` tokens (<$0.01). Re-run costs about the same; `h.cost_report()` prints your own numbers. See [README](README.md#estimated-openai-cost-measured-24-october-2026).
#
# > **Observability (this folder):** set `OBSERVABILITY_BACKEND=phoenix`, `deepeval`, `both` or `none` in `.env`. Phoenix traces every LLM call, judge and agent step at http://localhost:6006; DeepEval re-scores a sample of the judged answers in a local container (dashboard http://localhost:8765). See the final section of this notebook and the README.
#
# **Who this is for:** you finished every chapter notebook and the two lower tiers. You already know the
# primitives; this notebook shows how you *drive* them (and extend them) with an AI assistant, when nothing
# is faked.
#
# **What you will build, end to end**
# 1. A **vibe-coding method**: precise intent, fast verification, with real prompts you can reuse.
# 2. **GraphRAG over this repo itself**: all 51 `SKILL.md` descriptions become `:Skill` nodes with real
#    `text-embedding-3-large` vectors, a real Neo4j **vector index**, `SIMILAR` edges, and three retrievers
#    (vector / graph / hybrid).
# 3. A real **function-calling agent** (`h.ToolAgent`, gpt-4o) that investigates "checkout-api is slow": it
#    *discovers* skills by retrieval, *runs* them through their CLIs, queries the graph through a **real
#    least-privilege Neo4j user**, and stores its whole trace as a graph.
# 4. An LLM that **generates a brand-new skill** (`SKILL.md` + `lib.py` + `cli.py`) into a scratch folder, validated by
#    actually running it in a subprocess, with an automatic **repair loop** (max 3 tries).
# 5. An **LLM-as-judge** review, and a final **eval table**: vector vs graph vs hybrid skill retrieval.
#
# > Run top to bottom. The notebook wipes the Neo4j database in Section 2. It makes a bounded number of
# > LLM calls (temperature 0) and prints a token report at the end. There is **no offline fallback**.

# %% [markdown]
# ## What is real here (and was not in the simplified tier)
#
# | Piece | Simplified tier | This notebook |
# |---|---|---|
# | Reasoning / planning | fixed list of steps | gpt-4o chooses tools via **function calling** (`h.ToolAgent`) |
# | Skill discovery | keyword `CONTAINS` in Cypher | **3072-dim embeddings + Neo4j vector index**, graph expansion, hybrid RRF |
# | "Generated" skill | pasted by the author | **written by the LLM**, executed, repaired, judged |
# | Access control | not shown | **real Neo4j RBAC**: agent runs as a user that cannot write or read vectors |
# | Verification | self-written checks | subprocess benchmark + **held-out acceptance tests** the model never saw |
#
# What stays approximate is spelled out in the last cell.

# %%
import sys; sys.path.insert(0, ".")
import real_helpers as h

h.obs_start("ch9-vibe-coding-capstone")                       # observability: reads OBSERVABILITY_BACKEND from .env (phoenix | deepeval | both | none)
import json, os, re, shutil, subprocess, ast, time
from pathlib import Path
from collections import Counter

# Preflight: every failure here is loud on purpose; there is no offline fallback.
h.get_openai()                       # raises with instructions if OPENAI_API_KEY is missing
h.connect()                          # waits for Neo4j, raises if the container is not up
comp = h.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions[0] AS version, edition")[0]
print("Neo4j:", comp)
print("LLM model      :", h.LLM_MODEL)
print("Embedding model:", h.EMBED_MODEL, "| dimensions:", h.EMBED_DIM)

pong = h.llm("Reply with exactly the word: ready", max_tokens=5)
vec = h.embed(["preflight"])[0]
h.check("OpenAI chat answers", "ready" in pong.lower())
h.check("embedding has the configured dimension", len(vec) == h.EMBED_DIM)
h.check("Neo4j is Enterprise (needed for real RBAC)", comp["edition"].lower() == "enterprise")
h.check("repo has 51 skills on disk", len(list(h.SKILLS_DIR.glob("*/*/SKILL.md"))) == 51)


def soft(name, ok):
    """Tolerant check for LLM-dependent claims: reports WARN instead of aborting the notebook."""
    print(f"  [{'PASS' if ok else 'WARN'}] {name}")
    return bool(ok)

# %% [markdown]
# ## 1. The vibe-coding method
#
# **Vibe coding** = you state *what you want* in natural language, an AI assistant writes the code, and you steer by
# running and checking results instead of typing every line. The risk is hidden in the word "vibe": if the
# only evidence is that the code *looks* right, you ship bugs. The craft is a tight loop:
#
# ```
#  intent -> prompt -> generate -> VERIFY -> iterate
#    ^                                          |
#    +------------------------------------------+
# ```
#
# **Precise intent.** One sentence describing "done" in observable terms. "Make a severity tool" is a wish;
# "`cli.py triage` prints JSON with `severity=SEV1` for this input" is an intent.
#
# **Fast verification.** The loop speed is set by the *cheapest check that can say no*. Order your checks from cheapest to
# most expensive and stop at the first failure:
#
# | Rung | Check | Cost | Catches |
# |---|---|---|---|
# | 1 | syntax / `ast.parse` | ms | broken code |
# | 2 | static gates (banned imports, non-stdlib, secrets) | ms | unsafe or non-portable code, **before** it runs |
# | 3 | `--help` exits 0 | 100 ms | wiring errors |
# | 4 | the skill's own `benchmark` | ~1 s | the model's own claims |
# | 5 | **held-out acceptance tests** you wrote | ~1 s | the model grading its own homework |
# | 6 | LLM-as-judge | one call | style, completeness, hidden risk (least trustworthy rung) |
#
# **Prompt anatomy.** Context (what exists), Constraint (what must/must not happen), Example (what a right answer
# looks like), Acceptance test (how you will know). The next cell builds prompts from those parts and lets a real LLM
# critique a vague prompt versus a precise one.

# %%
def compose_prompt(goal, context, constraint, example, acceptance):
    """Assemble the four-part prompt; refuses to build one with an empty part."""
    parts = dict(context=context, constraint=constraint, example=example, acceptance=acceptance)
    missing = [k for k, v in parts.items() if not v.strip()]
    if missing:
        raise ValueError(f"prompt is missing: {missing}")
    return (f"GOAL: {goal}\nCONTEXT: {context}\nCONSTRAINTS: {constraint}\n"
            f"EXAMPLE: {example}\nACCEPTANCE TEST: {acceptance}")


def prompt_score(text: str) -> int:
    """Cheap lint: how many of the four anatomy parts does the prompt mention? (0-4)"""
    t = text.lower()
    markers = {"context": ("context",), "constraint": ("constraint", "must", "never", "only"),
               "example": ("example", "e.g."), "acceptance": ("acceptance", "exits 0", "passes")}
    return sum(any(m in t for m in ms) for ms in markers.values())


def critique_prompt(text: str) -> dict:
    """Ask a real LLM to grade a prompt the way a strict reviewer would."""
    return h.llm_json(
        "You review prompts written for an AI coding assistant. Grade the prompt below on four parts: "
        "context, constraints, a concrete example, and a runnable acceptance test. "
        'Return JSON {"score": <int 0-4, one point per part that is genuinely present>, "missing": [<part names>]}.\n\n'
        f"PROMPT:\n{text}")


BAD = "make something that tells me how bad an incident is"
GOOD = compose_prompt(
    goal="Add a skill that maps an incident's signals to a severity SEV1..SEV4.",
    context="Repo skills live in skills/<chapter>/<slug>/ with SKILL.md, lib.py, cli.py.",
    constraint="Python 3.10 stdlib only; no network; raise ValueError on bad input; ASCII output.",
    example="tier=critical, latency 3x threshold, 2% errors, 30% users -> SEV1 and page on-call.",
    acceptance="`python cli.py --help` exits 0 and `python cli.py benchmark` exits 0 with a negative case.",
)
bad_c, good_c = critique_prompt(BAD), critique_prompt(GOOD)
print("BAD :", bad_c)
print("GOOD:", good_c)
h.check("regex lint: precise prompt scores 4/4, vague prompt lower", prompt_score(BAD) < prompt_score(GOOD) == 4)
soft("LLM reviewer also scores the precise prompt higher", good_c["score"] > bad_c["score"])
try:
    compose_prompt("x", "", "c", "e", "a")
except ValueError as exc:
    h.check("compose_prompt refuses an empty part", "context" in str(exc))

# %% [markdown]
# ### The prompts behind this notebook
# Everything below was built by directing an assistant with prompts of this shape. They are listed so you
# can reuse them for your own repo. Each one ends in a *runnable acceptance test*, which is why the
# corresponding cells can verify themselves.

# %%
PROMPTS = {
    "index the repo": compose_prompt(
        goal="Load every skills/*/*/SKILL.md description into Neo4j as (:Skill) nodes and make them searchable by meaning.",
        context="Neo4j 5.26 Enterprise at bolt://localhost:7687; OpenAI text-embedding-3-large (3072 dims); helpers in real_helpers.py.",
        constraint="Parameterised Cypher only; MERGE so reruns do not duplicate; cosine vector index; no fallback if the API key is missing.",
        example="Query 'limit what each persona can see' -> subgraph-access-control ranks first.",
        acceptance="51 nodes with embedding of length 3072; an index query returns the expected slug in the top 3.",
    ),
    "build the agent": compose_prompt(
        goal="Write a function-calling agent that investigates 'checkout-api is slow' by finding and running repo skills.",
        context="h.ToolAgent takes {name: (callable, json_schema, description)}; skills expose cli.py with --help.",
        constraint="Tools must never raise to the model; graph access only through a read-only Neo4j user; max 10 steps; temperature 0.",
        example="find_skills('classify why agent fails') -> context-failure-classifier; then skill_cli(slug, ['--help']).",
        acceptance="Every tool call is persisted as (:Run)-[:STEP]->(:ToolCall); the count in Neo4j equals the count in Python.",
    ),
    "generate a skill": compose_prompt(
        goal="Generate skill incident-severity-triage (SKILL.md, lib.py, cli.py) into a scratch folder.",
        context="Mirror the nearest existing skill retrieved from the skill graph; repo CI runs --help and benchmark.",
        constraint="stdlib only, no subprocess/os/network, ASCII only, raise ValueError on invalid input.",
        example="critical tier, 2400ms vs 800ms threshold, 2% errors, 30% users -> SEV1, page on-call.",
        acceptance="static gates clean, benchmark exits 0, AND held-out acceptance tests the model never saw all pass.",
    ),
    "review with a judge": compose_prompt(
        goal="Review the generated skill on correctness, safety, verifiability and repo fit with evidence.",
        context="The judge receives the files plus the real validation results (not just the code).",
        constraint="Scores 1-5 per dimension; cite a concrete line for any score below 4; be strict, no flattery.",
        example='{"safety": 5, "issues": ["lib.py line 12 accepts negative latency"]}',
        acceptance="A deliberately broken variant of the skill must score lower than the repaired one.",
    ),
    "measure retrieval": compose_prompt(
        goal="Compare vector, graph and hybrid skill retrieval on a labeled query set.",
        context="18 natural-language needs, each with the slug(s) a human considers correct.",
        constraint="Report hit@1, hit@3 and MRR; do not tune retrievers on the test queries; state n explicitly.",
        example="'which model for cheap nodes' -> model-routing-selector or cost-performance-scorer.",
        acceptance="Every expected slug exists in the repo; each method returns results for every query.",
    ),
}
for name, p in PROMPTS.items():
    print(f"--- {name} ---\n{p}\n")
h.check("every notebook prompt has all four anatomy parts", all(prompt_score(p) == 4 for p in PROMPTS.values()))

# %% [markdown]
# ## 2. GraphRAG over the repo itself: a skill graph
#
# An agent that can *query its own capabilities* needs the skills as data. Model:
#
# ```
# (:Chapter {number, folder}) -[:HAS_SKILL]-> (:Skill {slug, description, text, embedding})
# (:Skill) -[:SIMILAR {score}]-> (:Skill)      <- from real embedding cosine, not word overlap
# ```
#
# The `description` in each `SKILL.md` is already written for routing ("Use when... NOT for..."), so it is the right
# text to embed. We read the **full** description (the helper `h.skill_catalog()` truncates at 200 chars, good for
# a table, too lossy for embeddings).

# %%
CHAPTER_NUMBER = {"crisis": 1, "architecture": 2, "knowledge-representation": 3, "memory": 4,
                  "reasoning-planning": 5, "tool-orchestration": 6, "self-evolution": 7, "optimization": 8}


def full_catalog():
    rows = []
    for md in sorted(h.SKILLS_DIR.glob("*/*/SKILL.md")):
        text = md.read_text(encoding="utf-8")
        m = re.search(r"^description:\s*\|?\s*\n?(.*?)^\w[\w-]*:", text, re.S | re.M)
        desc = " ".join((m.group(1) if m else "").split())
        rows.append({"chapter": md.parent.parent.name, "slug": md.parent.name, "description": desc})
    return rows


CATALOG = full_catalog()
SKILL_DIR = {r["slug"]: h.SKILLS_DIR / r["chapter"] / r["slug"] for r in CATALOG}
SLUGS = {r["slug"] for r in CATALOG}
h.check("51 skills across 8 chapters", len(CATALOG) == 51 and {r["chapter"] for r in CATALOG} == set(CHAPTER_NUMBER))
h.check("every description was parsed", all(len(r["description"]) > 80 for r in CATALOG))

h.clear_graph()
h.run("CREATE CONSTRAINT ch9_skill_slug IF NOT EXISTS FOR (s:Skill) REQUIRE s.slug IS UNIQUE")
for folder, num in CHAPTER_NUMBER.items():
    h.run("MERGE (:Chapter {folder:$f, number:$n})", f=folder, n=num)
h.run("UNWIND $rows AS r MATCH (c:Chapter {folder:r.chapter}) "
      "MERGE (s:Skill {slug:r.slug}) SET s.description=r.description, s.chapter=c.number, "
      "s.text=r.slug_words + ': ' + r.description MERGE (c)-[:HAS_SKILL]->(s)",
      rows=[{**r, "slug_words": r["slug"].replace("-", " ")} for r in CATALOG])
h.show(h.run("MATCH (c:Chapter)-[:HAS_SKILL]->(s) RETURN c.number AS chapter, c.folder AS folder, "
             "count(s) AS skills ORDER BY chapter"))
h.check("graph holds 51 :Skill nodes", h.run("MATCH (s:Skill) RETURN count(s) AS c")[0]["c"] == 51)

# %% [markdown]
# ### Real embeddings + a real vector index
# `text-embedding-3-large` returns 3072 floats per text. Neo4j stores them with
# `db.create.setNodeVectorProperty` (compact vector storage) and `CREATE VECTOR INDEX` builds an approximate
# nearest-neighbour index using **cosine** similarity (direction matters, not length). Embedding 51 short texts is one
# or two API calls; the index lets any later query find neighbours in milliseconds.

# %%
INDEX = "ch9_skill_vec"
h.ensure_vector_index(INDEX, "Skill", "embedding")
n_new = h.embed_nodes("Skill", "text", "embedding")
stats = h.run("MATCH (s:Skill) RETURN count(s.embedding) AS embedded, min(size(s.embedding)) AS dim")[0]
print("embedded now:", n_new, "| stats:", stats)
idx = h.run("SHOW VECTOR INDEXES YIELD name, state, options RETURN name, state")
print([r for r in idx if r["name"] == INDEX])
h.check("all 51 skills have a 3072-dim embedding", stats["embedded"] == 51 and stats["dim"] == h.EMBED_DIM)
h.check("vector index is ONLINE", any(r["name"] == INDEX and r["state"] == "ONLINE" for r in idx))

# %% [markdown]
# ### Semantic search: meaning, not keywords
# We embed the *question* with the same model and ask the index for its nearest `:Skill` nodes. The queries below
# deliberately share few words with the target descriptions.

# %%
_qcache = {}


def qvec(q):
    if q not in _qcache:
        _qcache[q] = h.embed([q])[0]
    return _qcache[q]


def vec_search(q, k=8):
    return h.run("CALL db.index.vector.queryNodes($i, $k, $v) YIELD node, score "
                 "RETURN node.slug AS slug, node.chapter AS ch, round(score, 4) AS score ORDER BY score DESC",
                 i=INDEX, k=k, v=qvec(q))


for q in ["limit what each agent persona is allowed to see in the graph",
          "my agent keeps forgetting what happened in earlier investigations",
          "stop the agent from doing something it cannot undo"]:
    print("\nQ:", q)
    h.show(vec_search(q, 3))

top = vec_search("limit what each agent persona is allowed to see in the graph", 3)
h.check("access-control question finds subgraph-access-control in top 3", "subgraph-access-control" in [r["slug"] for r in top])
top = vec_search("stop the agent from doing something it cannot undo", 3)
h.check("irreversibility question finds irreversible-action-gate in top 3", "irreversible-action-gate" in [r["slug"] for r in top])

# %% [markdown]
# ### `SIMILAR` edges from embedding cosine
# For each skill we keep its 3 nearest neighbours as edges. Neo4j computes the cosine itself with
# `vector.similarity.cosine` (Neo4j >= 5.18), so the graph stores *meaning-based* adjacency. These edges are the "graph" half of
# graph + vector: they let retrieval walk from one good hit to its siblings.

# %%
h.run("""
MATCH (a:Skill), (b:Skill) WHERE a <> b
WITH a, b, vector.similarity.cosine(a.embedding, b.embedding) AS s
ORDER BY s DESC
WITH a, collect({b: b, s: s})[..3] AS top
UNWIND top AS t
WITH a, t.b AS b, t.s AS s
MERGE (a)-[r:SIMILAR]->(b) SET r.score = round(s, 4)
""")
n_sim = h.run("MATCH ()-[r:SIMILAR]->() RETURN count(r) AS c")[0]["c"]
print("SIMILAR edges:", n_sim)
h.show(h.run("MATCH (x:Skill)-[r:SIMILAR]->(y:Skill) RETURN x.slug AS skill, y.slug AS neighbour, r.score AS score "
             "ORDER BY score DESC LIMIT 8"))
h.check("each skill got exactly 3 neighbours", n_sim == 51 * 3)

# %% [markdown]
# ## 3. Three retrievers: vector, graph, hybrid
#
# * **Vector**: nearest embeddings to the query (good at paraphrase, blind to exact terms).
# * **Graph**: keyword hits in slug + description (exact terms), then a 1-hop walk along `SIMILAR` edges from the best
#   seeds (good when the query shares vocabulary with the skill, and for pulling in siblings).
# * **Hybrid**: **Reciprocal Rank Fusion** of the two ranked lists, `score = sum 1/(60 + rank)`. It is the same idea as the repo's
#   `rrf-hybrid-retrieval` skill: no score calibration needed, because only ranks are combined.
#
# We will measure them honestly on a labeled query set at the end; do not assume hybrid always wins.

# %%
STOP = set("""about above after agent agents also and are because been before between both but can cannot does each for from
give has have how into its more most not only other over per same should skill skills that the their then these this
those through use used using what when where which while with without would your you make need needs want pick choose
decide my our any all one way own""".split())


def vec_rank(q, k=10):
    return [r["slug"] for r in vec_search(q, k)]


def kw_scores(q):
    kws = sorted({w for w in re.findall(r"[a-z]{3,}", q.lower()) if w not in STOP})
    if not kws:
        return []
    return h.run("MATCH (s:Skill) WITH s, toLower(s.slug + ' ' + s.description) AS t "
                 "WITH s, [k IN $kws WHERE t CONTAINS k] AS hits WHERE size(hits) > 0 "
                 "RETURN s.slug AS slug, size(hits) AS hits ORDER BY hits DESC, slug", kws=kws)


def neighbours(slug):
    return h.run("MATCH (a:Skill {slug:$s})-[r:SIMILAR]-(b:Skill) RETURN b.slug AS slug, r.score AS score", s=slug)


def graph_rank(q, k=10):
    base = {r["slug"]: float(r["hits"]) for r in kw_scores(q)}
    if not base:
        return []
    total = Counter(base)
    for slug, sc in sorted(base.items(), key=lambda x: (-x[1], x[0]))[:3]:
        for n in neighbours(slug):
            total[n["slug"]] += 0.5 * sc * n["score"]
    return [s for s, _ in sorted(total.items(), key=lambda x: (-x[1], x[0]))[:k]]


def rrf(lists, k=60):
    sc = Counter()
    for lst in lists:
        for i, s in enumerate(lst, 1):
            sc[s] += 1.0 / (k + i)
    return [s for s, _ in sorted(sc.items(), key=lambda x: (-x[1], x[0]))]


def hybrid_rank(q, k=10):
    return rrf([vec_rank(q, 10), graph_rank(q, 10)])[:k]


METHODS = {"vector": vec_rank, "graph": graph_rank, "hybrid": hybrid_rank}
print("rrf demo:", rrf([["a", "b", "c"], ["c", "a", "d"]]))
h.check("RRF rewards agreement between lists", rrf([["a", "b", "c"], ["c", "a", "d"]])[:2] == ["a", "c"])

# %%
demo_qs = ["how do I choose a memory consistency model", "cheapest model that still meets the quality bar",
           "agent drowns in 40 MCP tools"]
rows = []
for q in demo_qs:
    for name, fn in METHODS.items():
        rows.append({"query": q[:40], "method": name, "top3": ", ".join(fn(q)[:3])})
h.show(rows)
h.check("hybrid returns results for every demo query", all(hybrid_rank(q) for q in demo_qs))
h.check("keyword-only retrieval finds the exact-term match", "memory-consistency-model-selector" in graph_rank(demo_qs[0], 5))

# %% [markdown]
# ## 4. Real Neo4j RBAC for the agent
#
# The agent will run Cypher that **a model wrote**. Never give that Cypher your admin connection. We create a role and user that
# can only *read*:
#
# * `GRANT MATCH {*}` on nodes and relationships: read-only access to graph data (no `CREATE`/`SET`/`DELETE`).
# * `DENY READ {embedding}`: the 3072-float vectors are useless to the model and would flood its context.
# * `DENY TRAVERSE` on `:Run` / `:ToolCall`: the agent cannot read or edit its own audit trail.
#
# This is enforced by the database for that user. A clever prompt cannot talk its way past it. The user and role are removed in the final
# cleanup cell. First the incident data goes into the same graph (`reset=False` keeps the skill graph).

# %%
print(h.load_devops_graph(reset=False))
AGENT_USER, AGENT_PW, ROLE = "ch9_agent", "Ch9-agent-pw-1", "ch9_reader"


def rbac_teardown():
    h.sysrun(f"DROP USER {AGENT_USER} IF EXISTS")
    h.sysrun(f"DROP ROLE {ROLE} IF EXISTS")


rbac_teardown()
for stmt in [f"CREATE ROLE {ROLE}",
             f"GRANT ACCESS ON DATABASE neo4j TO {ROLE}",
             f"GRANT MATCH {{*}} ON GRAPH neo4j NODES * TO {ROLE}",
             f"GRANT MATCH {{*}} ON GRAPH neo4j RELATIONSHIPS * TO {ROLE}",
             f"DENY READ {{embedding}} ON GRAPH neo4j NODES * TO {ROLE}",
             f"DENY TRAVERSE ON GRAPH neo4j NODES Run, ToolCall TO {ROLE}",
             f"CREATE USER {AGENT_USER} SET PASSWORD '{AGENT_PW}' CHANGE NOT REQUIRED",
             f"GRANT ROLE {ROLE} TO {AGENT_USER}"]:
    h.sysrun(stmt)
print("privileges:")
h.show([{"command": r["command"]} for r in h.sysrun(f"SHOW ROLE {ROLE} PRIVILEGES AS COMMANDS")])

# %%
seen = h.run_as(AGENT_USER, AGENT_PW, "MATCH (s:Skill) RETURN count(s) AS skills, count(s.embedding) AS visible_vectors")[0]
print("as the agent user:", seen)
h.check("agent user can read the 51 skills", seen["skills"] == 51)
h.check("embedding property is hidden from the agent", seen["visible_vectors"] == 0)
inc = h.run_as(AGENT_USER, AGENT_PW, "MATCH (i:Incident) RETURN count(i) AS n")[0]["n"]
h.check("agent can read incidents", inc == 2)
try:
    h.run_as(AGENT_USER, AGENT_PW, "CREATE (:Evil {x:1})")
    denied = False
except Exception as exc:
    denied = True
    print("write attempt rejected by Neo4j:", type(exc).__name__)
h.check("write attempt as agent user is denied by the database", denied)
h.check("no :Evil node exists (admin view)", h.run("MATCH (n:Evil) RETURN count(n) AS c")[0]["c"] == 0)

# %% [markdown]
# ## 5. The incident agent: gpt-4o picks and runs skills
#
# **Why function calling and not a fixed plan?** The simplified tier hard-coded the sequence. A real agent sees only a *tool menu* and
# must decide: query the graph for facts, retrieve relevant skills, read their CLI help, run one, and conclude. We expose three tools:
#
# | Tool | What it does | Guardrail |
# |---|---|---|
# | `find_skills(query)` | hybrid skill retrieval from the skill graph | read-only |
# | `skill_cli(slug, argv)` | runs `skills/.../cli.py` in a subprocess | known slug only, no path-like args, timeout, scrubbed env |
# | `graph_query(cypher)` | Cypher against the DevOps graph | executes as the **read-only Neo4j user** |
#
# Tools return error *strings* instead of raising, so the model can read the error and recover. Every call is wrapped so it is
# logged into Neo4j through `h.AgentTrace`, and calls that name a skill get a `USED_SKILL` edge to the skill node.

# %%
SAFE_ENV = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "COMSPEC")}
SAFE_ENV["PYTHONIOENCODING"] = "utf-8"   # secrets such as OPENAI_API_KEY are NOT passed to child processes


def run_cli(skill_dir: Path, *argv, timeout=60):
    """Run a skill CLI in a subprocess: scrubbed env, timeout, no shell."""
    try:
        return subprocess.run([sys.executable, str(Path(skill_dir) / "cli.py"), *map(str, argv)],
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, env=SAFE_ENV, cwd=str(skill_dir))
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, 124, "", f"TIMEOUT after {timeout}s")


def find_skills(query, k=5):
    desc = {r["slug"]: r["description"] for r in CATALOG}
    return [{"slug": s, "chapter": s and next(r["chapter"] for r in CATALOG if r["slug"] == s),
             "summary": desc[s][:150]} for s in hybrid_rank(query, int(k))]


def skill_cli(slug, argv=None):
    argv = [str(a) for a in (argv or [])]
    if slug not in SKILL_DIR:
        return f"ERROR: unknown skill {slug!r}. Use find_skills first."
    if len(argv) > 12 or any(".." in a or a.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", a) for a in argv):
        return "ERROR: at most 12 args and no path-like arguments."
    p = run_cli(SKILL_DIR[slug], *argv, timeout=30)
    return f"exit={p.returncode}\n{p.stdout}{p.stderr}"[:2500]


def graph_query(cypher):
    return h.run_as(AGENT_USER, AGENT_PW, cypher)[:30]


SCHEMA = ("Graph schema: (Service {name,tier,owner}), (Database {name,engine,max_connections}), (Cache {name}), "
          "(Deployment {id,version,at,change}), (Alarm {id,metric,value,threshold}), "
          "(Incident {id,title,status,root_cause,fix}); relationships: DEPENDS_ON, CHANGED, DEPLOYED_TO, FIRED_ON, "
          "AFFECTS, TRIGGERED_BY, ROOT_CAUSE. Skill graph: (Skill {slug,description}) and SIMILAR.")
SYSTEM = ("You are an SRE agent. Investigate the incident using tools; do not guess. " + SCHEMA +
          " Rules: (1) query the graph for facts about the affected service, its dependencies, recent deployments and past "
          "incidents; (2) use find_skills to discover repo skills relevant to the investigation, ALWAYS run "
          "skill_cli(slug, ['--help']) before using a skill, then run at least one skill subcommand with real arguments; "
          "(3) finish with at most 120 words: root cause with evidence, recommended fix, and which skills you used.")
INCIDENT = "INC-1001: checkout-api p99 latency is 2400ms against an 800ms threshold. Find the root cause and the fix."

TRACE = h.AgentTrace(INCIDENT)
CALLS = []


def traced(name, fn):
    def wrapper(**kw):
        try:
            out, ok = fn(**kw), True
        except Exception as exc:          # fed back to the model as text
            out, ok = f"ERROR {type(exc).__name__}: {exc}", False
        text = out if isinstance(out, str) else json.dumps(out, default=str)
        if text.startswith("ERROR"):
            ok = False
        TRACE.log(name, kw, text, ok=ok)
        CALLS.append(name)
        slug = kw.get("slug")
        if slug in SLUGS:
            h.run("MATCH (c:ToolCall {id:$cid}), (s:Skill {slug:$slug}) MERGE (c)-[:USED_SKILL]->(s)",
                  cid=f"{TRACE.run_id}-{TRACE.n}", slug=slug)
        return text
    return wrapper


TOOLS = {
    "find_skills": (traced("find_skills", find_skills),
                    {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                     "required": ["query"]},
                    "Semantic+graph search over the repo's 51 skills. Returns slug, chapter, summary."),
    "skill_cli": (traced("skill_cli", skill_cli),
                  {"type": "object", "properties": {"slug": {"type": "string"},
                                                    "argv": {"type": "array", "items": {"type": "string"}}},
                   "required": ["slug"]},
                  "Run a skill's cli.py with argv (e.g. ['--help'] or ['classify','--symptom','...']). Returns exit code and output."),
    "graph_query": (traced("graph_query", graph_query),
                    {"type": "object", "properties": {"cypher": {"type": "string"}}, "required": ["cypher"]},
                    "Run a READ-ONLY Cypher query on the DevOps/skill graph (max 30 rows)."),
}
print("tools:", list(TOOLS), "| trace run id:", TRACE.run_id)

# %%
agent = h.ToolAgent(TOOLS, SYSTEM, max_steps=12, trace=TRACE)
ANSWER = agent.run(INCIDENT)
h.run("MATCH (r:Run {id:$id}) SET r.answer=$a, r.finished=timestamp()", id=TRACE.run_id, a=ANSWER[:1500])
print("\ntool calls:", Counter(CALLS))

# %% [markdown]
# ### The trace is a graph
# `(:Run)-[:STEP]->(:ToolCall)-[:NEXT]->(:ToolCall)` plus `(:ToolCall)-[:USED_SKILL]->(:Skill)`. We read it back *from Neo4j* (as admin;
# the agent user is not allowed to see it) and compare with what Python remembers. Cross-checking two sources of truth
# is a verification habit worth keeping: it catches logging bugs.

# %%
timeline = TRACE.timeline()
h.show(timeline)
used = h.run("MATCH (:Run {id:$id})-[:STEP]->(c)-[:USED_SKILL]->(s:Skill) "
             "RETURN s.slug AS skill, s.chapter AS chapter, count(c) AS calls ORDER BY calls DESC", id=TRACE.run_id)
print("\nskills the agent actually used (from the graph):")
h.show(used)
chain = h.run("MATCH p=(:ToolCall {id:$f})-[:NEXT*]->(:ToolCall) RETURN max(length(p)) AS hops", f=f"{TRACE.run_id}-1")[0]["hops"]
as_agent = h.run_as(AGENT_USER, AGENT_PW, "MATCH (r:Run) RETURN count(r) AS c")[0]["c"]

h.check("every tool call was persisted (Neo4j count == Python count)", len(timeline) == len(CALLS) > 0)
h.check("NEXT chain links all steps", (chain or 0) == len(timeline) - 1)
h.check("agent cannot see its own audit trail (DENY TRAVERSE)", as_agent == 0)
soft("agent queried the graph for facts", "graph_query" in CALLS)
soft("agent discovered skills by retrieval", "find_skills" in CALLS)
soft("agent ran at least one skill CLI successfully", any(r["tool"] == "skill_cli" and r["ok"] for r in timeline))
soft("answer names the real root cause (deploy-481 / connection pool)", "deploy-481" in ANSWER or "pool" in ANSWER.lower())

# %% [markdown]
# ### Grade the answer, then distrust the grade
# `h.judge` returns `{"score": 1-5, "reason": ...}`. We give the judge a rubric grounded in facts we *know* from the seeded graph
# (deployment `deploy-481` cut the `orders-db` pool from 100 to 20; the past incident INC-0877 had the same fix). A judge with a
# rubric is more reproducible than a judge with a vibe, but it is still an LLM; see the limits at the end.

# %%
RUBRIC = ("5 = names deploy-481 reducing the orders-db connection pool (100 -> 20) as root cause, cites graph evidence "
          "(and ideally the past incident INC-0877), recommends restoring pool size or rolling back, and names the skills used. "
          "3 = plausible cause but missing evidence or fix. 1 = wrong or unsupported cause.")
verdict = h.judge(INCIDENT, ANSWER, RUBRIC)
print(verdict)
h.check("judge returned a 1-5 score", isinstance(verdict.get("score"), int) and 1 <= verdict["score"] <= 5)
soft("judge rates the grounded answer >= 4", verdict["score"] >= 4)

# %% [markdown]
# ## 6. The LLM writes a brand-new skill
#
# Task: *"the agent should triage how severe an incident is."* No existing skill does it (we will *check* that with retrieval first).
# Three design decisions make this safe and honest:
#
# 1. **Two layers of tests.** The model writes its own `benchmark`, which is useful but circular: a model can write tests its code passes.
#    So *we* write **held-out acceptance tests** (they run the CLI contract from the prompt) that the model never sees.
# 2. **Static gates before execution.** Generated code is parsed with `ast` and rejected for banned imports (`subprocess`, `os`, `socket`...),
#    `eval`/`exec`/`open`, or non-stdlib imports **before anything runs**. Only code that passes is executed, in a subprocess with a timeout,
#    a scrubbed environment (no API key) and a scratch working directory, never from `skills/`.
# 3. **A bounded repair loop.** Failures are fed back verbatim, at most 3 validation rounds in total.
#
# First the specification: an intent precise enough to test.

# %%
SLUG = "incident-severity-triage"
SPEC = compose_prompt(
    goal=f"Create a new skill `{SLUG}` that maps incident signals (service tier, latency vs threshold, error rate %, users "
         "affected %, workaround available) to SEV1..SEV4 and a page-on-call flag, with an explainable score.",
    context="Repo layout: skills/<chapter>/<slug>/{SKILL.md,lib.py,cli.py}. SKILL.md has frontmatter (name, description as a "
            "`description: |` block, chapter-source) and sections: Overview, When to Use, When NOT to Use, Process, Rationalizations, "
            "Red Flags, Non-Negotiable Verification, Security Posture, Source Attribution.",
    constraint="lib.py and cli.py: Python 3.10 standard library ONLY; never import os, subprocess, socket, urllib, shutil or "
               "importlib; no eval/exec/open; ASCII characters only; deterministic; raise ValueError on invalid input (unknown tier, "
               "negative numbers, percentages above 100, threshold <= 0).",
    example="tier=critical, latency 2400ms vs threshold 800ms, 2% errors, 30% users, no workaround -> SEV1, page_oncall true. "
            "tier=low, latency 800 vs 800, 0% errors, 0% users -> SEV4, page_oncall false.",
    acceptance="CLI contract: `python cli.py triage --tier T --latency-ms N --threshold-ms N --error-rate N --users-pct N "
               "[--workaround]` prints ONE JSON object with keys severity, score, page_oncall, reasons; invalid input exits non-zero. "
               "`python cli.py --help` exits 0 and prints the SKILL.md description. `python cli.py benchmark` exits 0 and includes "
               "negative cases (it may sys.exit(1) on failure).",
)
print(SPEC)

SEV_ORDER = {"SEV1": 1, "SEV2": 2, "SEV3": 3, "SEV4": 4}


def held_out_acceptance(d: Path) -> list[str]:
    """Tests WE wrote; the model never sees them. They only rely on the CLI contract in the prompt."""
    probs = []

    def tri(*a):
        p = run_cli(d, "triage", *a, timeout=30)
        try:
            return p.returncode, json.loads(p.stdout)
        except Exception:
            return p.returncode, None

    base = ["--tier", "critical", "--latency-ms", "2400", "--threshold-ms", "800", "--error-rate", "2", "--users-pct", "30"]
    rc, o = tri(*base)
    if rc != 0 or not o or o.get("severity") != "SEV1" or o.get("page_oncall") is not True:
        probs.append(f"held-out: prompt example 1 should give SEV1 + page_oncall true, got rc={rc} {o and {k: o.get(k) for k in ('severity', 'page_oncall')}}")
    rc, o = tri("--tier", "low", "--latency-ms", "800", "--threshold-ms", "800", "--error-rate", "0", "--users-pct", "0")
    if rc != 0 or not o or o.get("severity") != "SEV4" or o.get("page_oncall") is not False:
        probs.append(f"held-out: prompt example 2 should give SEV4 + page_oncall false, got rc={rc} {o and {k: o.get(k) for k in ('severity', 'page_oncall')}}")
    rc, w = tri(*base, "--workaround")
    if rc == 0 and o is not None and w and SEV_ORDER.get(w.get("severity"), 0) < SEV_ORDER["SEV1"]:
        probs.append("held-out: a workaround must never raise severity")
    rc, bad = tri("--tier", "urgent", "--latency-ms", "1", "--threshold-ms", "1")
    if rc == 0:
        probs.append("held-out: unknown tier must exit non-zero")
    rc, bad = tri("--tier", "low", "--latency-ms", "1", "--threshold-ms", "1", "--users-pct", "101")
    if rc == 0:
        probs.append("held-out: users-pct 101 must exit non-zero")
    lo = tri("--tier", "high", "--latency-ms", "900", "--threshold-ms", "800", "--error-rate", "0.5", "--users-pct", "1")[1]
    hi = tri("--tier", "high", "--latency-ms", "5000", "--threshold-ms", "800", "--error-rate", "9", "--users-pct", "60")[1]
    if lo and hi and SEV_ORDER.get(hi.get("severity"), 9) > SEV_ORDER.get(lo.get("severity"), 0):
        probs.append("held-out: worse signals must not give a LOWER severity")
    return probs


h.check("spec has all four anatomy parts", prompt_score(SPEC) == 4)
h.check("the new slug does not collide with an existing skill", SLUG not in SLUGS)

# %% [markdown]
# ### The validation harness
# Order matters: files exist -> `SKILL.md` lint -> **static safety scan** -> (only now) run `--help` -> run `benchmark` -> held-out acceptance.
# Each problem is a plain sentence, which is exactly what we paste back to the model in the repair loop.

# %%
REQUIRED_SECTIONS = ["## Overview", "## When to Use", "## When NOT to Use", "## Process", "## Rationalizations",
                     "## Red Flags", "## Non-Negotiable Verification", "## Security Posture", "## Source Attribution"]
BANNED_IMPORTS = {"os", "subprocess", "socket", "urllib", "http", "requests", "ctypes", "multiprocessing",
                  "importlib", "pickle", "shutil"}
BANNED_CALLS = {"eval", "exec", "compile", "__import__", "open", "system", "popen", "write_text", "write_bytes",
                "unlink", "rmdir", "mkdir"}


def skill_description(md: str) -> str:
    lines, in_desc, fences = [], False, 0
    for line in md.splitlines():
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


def safety_scan(src: str, fname: str) -> list[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"{fname}: syntax error line {e.lineno}: {e.msg}"]
    out = []
    for node in ast.walk(tree):
        mods = [n.name for n in node.names] if isinstance(node, ast.Import) else \
               [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
        for m in mods:
            root = m.split(".")[0]
            if root in BANNED_IMPORTS:
                out.append(f"{fname}: banned import {m}")
            elif root not in sys.stdlib_module_names and root != "lib":
                out.append(f"{fname}: non-stdlib import {m}")
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if name in BANNED_CALLS:
                out.append(f"{fname}: banned call {name}() at line {node.lineno}")
    return out


def lint_md(md: str) -> list[str]:
    parts = md.split("---")
    front = parts[1] if len(parts) > 2 else ""
    probs = [f"SKILL.md frontmatter missing {k}" for k in ("name:", "description:", "chapter-source:") if k not in front]
    return probs + [f"SKILL.md missing section {s}" for s in REQUIRED_SECTIONS if s not in md]


def validate_skill(d: Path, acceptance=None) -> dict:
    """Return {problems, bench_tail}. Code is executed only if every static gate passed."""
    problems = [f"missing file {f}" for f in ("SKILL.md", "lib.py", "cli.py") if not (d / f).exists()]
    if problems:
        return {"problems": problems, "bench_tail": ""}
    md = (d / "SKILL.md").read_text(encoding="utf-8")
    problems += lint_md(md)
    for f in ("lib.py", "cli.py"):
        src = (d / f).read_text(encoding="utf-8")
        problems += safety_scan(src, f)
        if any(ord(ch) > 127 for ch in src):
            problems.append(f"{f} contains non-ASCII characters")
    if "raise ValueError" not in (d / "lib.py").read_text(encoding="utf-8"):
        problems.append("lib.py never raises ValueError")
    tail = ""
    if not problems:
        hp = run_cli(d, "--help", timeout=30)
        want = " ".join(skill_description(md).split()[:5])
        if hp.returncode != 0:
            problems.append(f"`--help` exited {hp.returncode}: {hp.stderr[-300:]}")
        elif want not in " ".join(hp.stdout.split()):
            problems.append(f"`--help` does not print the SKILL.md description (expected to see: {want!r})")
        bp = run_cli(d, "benchmark", timeout=60)
        tail = (bp.stdout + bp.stderr)[-1200:]
        if bp.returncode != 0:
            problems.append(f"`benchmark` exited {bp.returncode}. Output tail:\n{tail}")
        if not re.search(r"negative|raises", (d / "cli.py").read_text(encoding="utf-8"), re.I):
            problems.append("benchmark has no negative case")
        if acceptance and not problems:
            problems += acceptance(d)
    return {"problems": problems, "bench_tail": tail}


SCRATCH = Path(h.HERE) / "_scratch_ch9"
shutil.rmtree(SCRATCH, ignore_errors=True)
SCRATCH.mkdir()
h.check("scratch dir is outside skills/", h.SKILLS_DIR not in SCRATCH.parents)

# sanity: the harness must pass on a REAL shipped skill (copied) and must fail on a broken one
real = SCRATCH / "real-copy"
shutil.copytree(SKILL_DIR["context-failure-classifier"], real, ignore=shutil.ignore_patterns("__pycache__"))
rv = validate_skill(real)
print("harness on a shipped skill:", rv["problems"][:3] or "clean")
soft("harness accepts a shipped skill (repo style) with at most the strictness gaps shown above", len(rv["problems"]) <= 3)
(real / "lib.py").write_text("import subprocess\n" + (real / "lib.py").read_text(encoding="utf-8"), encoding="utf-8")
h.check("harness rejects a skill with a banned import before running it",
        any("banned import subprocess" in p for p in validate_skill(real)["problems"]))
shutil.rmtree(real, ignore_errors=True)

# %% [markdown]
# ### Use the skill graph to inform generation (gap analysis + exemplar)
# Before asking a model to write a skill, ask the *graph*: **does one already exist?** and **which existing skills are the best style
# examples?** This is GraphRAG applied to the repo's own development. We embed the intent, take the nearest skills, let the LLM
# judge coverage from their descriptions, and feed the closest skill's `SKILL.md` (not our hand-written template) to the generator as
# a style reference.

# %%
intent = ("Triage how severe an incident is from its signals (service tier, latency, error rate, users affected) "
          "and decide SEV1..SEV4 and whether to page on-call.")
near = vec_search(intent, 5)
h.show(near)
desc_by_slug = {r["slug"]: r["description"] for r in CATALOG}
gap = h.llm_json("Existing skills (slug: description):\n" +
                 "\n".join(f"- {r['slug']}: {desc_by_slug[r['slug']][:300]}" for r in near) +
                 f"\n\nNeed: {intent}\n"
                 'Does ANY listed skill already fully provide this need? Return JSON {"covered": <bool>, "closest": "<slug>", "reason": "<one sentence>"}.')
print(gap)
EXEMPLAR = hybrid_rank(intent, 1)[0]
print("style exemplar:", EXEMPLAR)
h.check("gap analysis returned a verdict", isinstance(gap.get("covered"), bool))
soft("no existing skill already covers severity triage", gap["covered"] is False)

# %% [markdown]
# ### Generate
# The generator gets: the specification, the exemplar skill's `SKILL.md` and `cli.py` (truncated), and a **known-good snippet** for the
# one fiddly requirement (a `--help` that prints the `SKILL.md` description). It must return JSON with the three files in full. Temperature 0,
# one call, generous `max_tokens` so the files are not truncated.

# %%
DESC_SNIPPET = '''def _skill_description() -> str:
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
# usage: argparse.ArgumentParser(description=_skill_description()); SKILL_MD = Path(__file__).resolve().parent / "SKILL.md"'''

GEN_SYSTEM = ("You are a senior Python engineer extending a repository of agent skills. Reply with JSON only: an object with exactly the keys "
              '"SKILL.md", "lib.py", "cli.py", each a string holding the COMPLETE file. ASCII characters only (no em dashes, no smart quotes).')
FILES = ("SKILL.md", "lib.py", "cli.py")


def gen_prompt():
    ex = SKILL_DIR[EXEMPLAR]
    return (f"{SPEC}\n\nSTYLE REFERENCE (an existing skill, mirror its structure, do not copy its content):\n"
            f"--- {EXEMPLAR}/SKILL.md ---\n{(ex / 'SKILL.md').read_text(encoding='utf-8')[:4500]}\n"
            f"--- {EXEMPLAR}/cli.py (truncated) ---\n{(ex / 'cli.py').read_text(encoding='utf-8')[:3000]}\n\n"
            f"KNOWN-GOOD helper to reuse in cli.py for --help:\n{DESC_SNIPPET}\n\n"
            f"cli.py must expose subcommands `triage` and `benchmark`; benchmark prints a one-line summary like "
            f"'{SLUG} benchmark - N/M passed' and exits 1 if any case fails. lib.py exposes triage(tier, latency_ms, threshold_ms, "
            f"error_rate_pct, users_affected_pct, has_workaround=False) -> dict. Score rule suggestion: points for tier, latency ratio, "
            f"error rate and users affected, minus one for a workaround; SEV1 >= 9, SEV2 >= 6, SEV3 >= 3, else SEV4; the prompt's example 1 "
            f"must land in SEV1. Include at least 10 benchmark cases, 4 of them negative (expected ValueError).")


def coerce(obj: dict, fallback: dict | None = None) -> dict:
    return {k: str(obj.get(k) or (fallback or {}).get(k) or "") for k in FILES}


def write_files(d: Path, files: dict):
    d.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        (d / name).write_text(files[name], encoding="utf-8", newline="\n")


gen_raw = h.llm_json(gen_prompt(), system=GEN_SYSTEM, max_tokens=6000)
GENERATED = coerce(gen_raw)
print({k: f"{len(v)} chars" for k, v in GENERATED.items()})
h.check("generator returned all three files", all(len(GENERATED[k]) > 200 for k in FILES))
h.check("generator did not need the filesystem (nothing written yet)", not (SCRATCH / SLUG).exists())

# %% [markdown]
# ### The repair loop
# `repair_loop` writes the files to the scratch folder, validates, and on failure sends the model exactly what you would type to an
# assistant by hand: *"here are the files, here are the failures, fix only what is needed, return the full files"*. It stops at the first clean
# validation or after 3 rounds, and keeps a per-attempt history so that we can store it in the graph. Bounded loops matter: an unbounded
# "fix it until it passes" loop is how you burn money and, worse, how models learn to weaken tests.

# %%
def repair_prompt(files, problems, bench_tail):
    body = "\n\n".join(f"--- {k} ---\n{files[k]}" for k in FILES)
    return (f"{SPEC}\n\nThe files below FAILED validation.\nFAILURES:\n" + "\n".join(f"- {p}" for p in problems) +
            f"\n\nBENCHMARK OUTPUT TAIL:\n{bench_tail}\n\nCURRENT FILES:\n{body}\n\n"
            "Fix only what is needed to resolve the failures, keep the CLI contract, do NOT weaken or delete tests to make them pass. "
            "Return JSON with all three files in full.")


def repair_loop(dirname: str, files: dict, max_tries: int = 3):
    d = SCRATCH / dirname
    history = []
    for n in range(1, max_tries + 1):
        write_files(d, files)
        res = validate_skill(d, acceptance=held_out_acceptance)
        history.append({"try": n, "problems": len(res["problems"]), "first_problem": (res["problems"] or ["-"])[0][:110]})
        if not res["problems"]:
            return files, history, True, d
        if n == max_tries:
            break
        files = coerce(h.llm_json(repair_prompt(files, res["problems"], res["bench_tail"]), system=GEN_SYSTEM, max_tokens=6000), files)
    return files, history, False, d


FINAL, HIST, OK, GEN_DIR = repair_loop(SLUG, GENERATED)
h.show(HIST)
print("converged:", OK)
h.check("repair loop stayed within 3 tries", len(HIST) <= 3)
soft("generated skill passes every gate (static, benchmark, held-out)", OK)

# %% [markdown]
# ### Store the generation as a graph, and look at the outputs
# `(:GenRun)-[:ATTEMPT]->(:Attempt {n, problems, first_problem})`: the same audit idea as the agent trace. Later you can query things such as
# "which prompts needed repair, and for what?", which is how a team improves its prompt library with evidence.

# %%
def persist_history(run_id, label, history, ok):
    h.run("MERGE (g:GenRun {id:$id}) SET g.label=$label, g.converged=$ok, g.slug=$slug", id=run_id, label=label, ok=ok, slug=SLUG)
    for e in history:
        h.run("MATCH (g:GenRun {id:$id}) MERGE (a:Attempt {id:$aid}) "
              "SET a.n=$n, a.problems=$p, a.first_problem=$fp MERGE (g)-[:ATTEMPT]->(a)",
              id=run_id, aid=f"{run_id}-{e['try']}", n=e["try"], p=e["problems"], fp=e["first_problem"])


persist_history("gen-main", "generated from the spec", HIST, OK)
h.show(h.run("MATCH (g:GenRun {id:'gen-main'})-[:ATTEMPT]->(a) RETURN a.n AS try, a.problems AS problems, a.first_problem AS first ORDER BY try"))
if OK:
    demo = run_cli(GEN_DIR, "triage", "--tier", "critical", "--latency-ms", "2400", "--threshold-ms", "800",
                   "--error-rate", "2", "--users-pct", "30")
    print(demo.stdout)
    print(run_cli(GEN_DIR, "benchmark").stdout.strip()[:400])
h.check("attempts persisted in Neo4j",
        h.run("MATCH (:GenRun {id:'gen-main'})-[:ATTEMPT]->(a) RETURN count(a) AS c")[0]["c"] == len(HIST))

# %% [markdown]
# ### Fault injection: prove the repair loop can actually repair
# If the first generation happens to be clean, the loop above never exercised its repair branch. So we **sabotage a copy** the way a sloppy
# assistant could (a banned import, a deleted `Security Posture` section, a broken early return) and let the same loop fix it with a real LLM call.
# A verifier you have never seen fail is a verifier you do not yet trust.

# %%
def sabotage(files: dict) -> dict:
    bad = dict(files)
    bad["lib.py"] = "import subprocess\n" + bad["lib.py"]
    bad["SKILL.md"] = bad["SKILL.md"].replace("## Security Posture", "## Notes")
    return bad


SABOTAGED = sabotage(FINAL)
write_files(SCRATCH / "sabotaged-initial", SABOTAGED)
pre = validate_skill(SCRATCH / "sabotaged-initial", acceptance=held_out_acceptance)
print("findings before repair:")
for p in pre["problems"]:
    print("  -", p[:140])
h.check("static gates catch the planted banned import",
        any("banned import subprocess" in p for p in pre["problems"]))
h.check("lint catches the removed Security Posture section", any("Security Posture" in p for p in pre["problems"]))

FIXED, HIST2, OK2, FIX_DIR = repair_loop("sabotaged-repair", SABOTAGED)
h.show(HIST2)
persist_history("gen-sabotage", "fault injection", HIST2, OK2)
soft("the repair loop fixed the sabotaged copy within 3 tries", OK2)
h.check("final repaired code no longer imports subprocess", "import subprocess" not in FIXED["lib.py"] or not OK2)

# %% [markdown]
# ## 7. LLM-as-judge: review the generated skill
#
# Tests say *whether it works*; a reviewer says whether it is *good*: complete docs, honest red flags, sensible security notes, consistency with the
# repo. We ask the model to score four dimensions 1-5 with evidence. Three safeguards against the usual judge failure modes:
#
# * **Ground it in facts.** The judge receives the *actual* validation outcome, so it cannot praise code that failed.
# * **Control sample.** We also judge the sabotaged version. A judge that cannot rank a broken skill below a repaired one is useless.
# * **Consistency check.** If it scores verifiability >= 4 for something whose benchmark failed, we flag the judge, not the code.

# %%
JUDGE_SYSTEM = ("You are a strict reviewer of agent skills in a Python repo. Be critical; do not flatter. "
                "Score each dimension 1-5 and cite a concrete line or section for any score below 4.")


def judge_skill(files: dict, problems: list[str]) -> dict:
    body = "\n\n".join(f"--- {k} ---\n{files[k][:6000]}" for k in FILES)
    return h.llm_json(
        f"SPEC:\n{SPEC}\n\nREAL VALIDATION RESULT: {'CLEAN, all gates and held-out tests passed' if not problems else 'FAILED: ' + '; '.join(p[:160] for p in problems[:6])}\n\n"
        f"FILES:\n{body}\n\n"
        'Return JSON {"correctness": int, "safety": int, "verifiability": int, "repo_fit": int, "overall": int, "issues": [<short strings>]}.',
        system=JUDGE_SYSTEM)


final_check = validate_skill(GEN_DIR, acceptance=held_out_acceptance)
j_good = judge_skill(FINAL, final_check["problems"])
j_bad = judge_skill(SABOTAGED, pre["problems"])
h.show([{"variant": "generated (validated)", **{k: j_good.get(k) for k in ("correctness", "safety", "verifiability", "repo_fit", "overall")}},
        {"variant": "sabotaged (before repair)", **{k: j_bad.get(k) for k in ("correctness", "safety", "verifiability", "repo_fit", "overall")}}])
print("good issues:", j_good.get("issues"))
print("bad  issues:", j_bad.get("issues"))
dims = ("correctness", "safety", "verifiability", "repo_fit", "overall")
h.check("judge returned integer scores for both variants",
        all(isinstance(j.get(k), int) for j in (j_good, j_bad) for k in dims))
soft("control sample: judge ranks the sabotaged skill below the validated one", j_bad["overall"] < j_good["overall"])
soft("judge flags the sabotaged variant's safety problem", j_bad["safety"] < j_good["safety"] or j_bad["safety"] <= 3)
soft("judge consistency: it does not give verifiability >= 4 to a failed variant", not (pre["problems"] and j_bad["verifiability"] >= 4))

# %% [markdown]
# ## 8. Close the loop: the new skill becomes retrievable
#
# A skill nobody can *find* does not exist for an agent. We (a) run a **novelty check** (the nearest existing skill should not be a near-duplicate:
# cosine ~ 1.0), (b) add the generated skill to the graph as a `:Skill {generated:true}` node with its own embedding, (c) confirm that the retrieval
# stack now returns it for the original need, and (d) **remove it again** so the 51-skill graph used by the eval stays pristine. Nothing is ever written
# to `skills/`; promoting the skill means a human opening a PR after review.

# %%
GEN_TEXT = f"{SLUG.replace('-', ' ')}: {skill_description(FINAL['SKILL.md'])}"
nearest = h.run("CALL db.index.vector.queryNodes($i, 3, $v) YIELD node, score RETURN node.slug AS slug, round(score, 4) AS score",
                i=INDEX, v=h.embed([GEN_TEXT])[0])
print("nearest existing skills to the generated one:")
h.show(nearest)
soft("novelty: no existing skill is a near-duplicate (cosine < 0.92)", nearest[0]["score"] < 0.92)

h.run("MERGE (s:Skill {slug:$slug}) SET s.description=$d, s.text=$t, s.chapter=9, s.generated=true",
      slug=SLUG, d=skill_description(FINAL["SKILL.md"]), t=GEN_TEXT)
h.embed_nodes("Skill", "text", "embedding")
after = vec_search("how severe is this incident and should we page the on-call engineer", 3)
h.show(after)
soft("the generated skill is now the top semantic hit for its own need", after[0]["slug"] == SLUG)

h.run("MATCH (s:Skill {generated:true}) DETACH DELETE s")
h.check("generated skill node removed again; graph back to 51 skills",
        h.run("MATCH (s:Skill) RETURN count(s) AS c")[0]["c"] == 51)

# %% [markdown]
# ## 9. Final scoreboard: vector vs graph vs hybrid skill retrieval
#
# The claim "hybrid is better" is a hypothesis. We test it on **18 labeled needs** written in everyday language (some close to the skill's vocabulary, some paraphrased). For each,
# a human-assigned set of acceptable slugs. Metrics: **hit@1**, **hit@3**, and **MRR** (mean of 1/rank of the first correct hit, 0 if outside the top 8). We
# did not tune any retriever on these queries. With n=18 a difference of one query is 5.6 percentage points, so treat small gaps as noise.

# %%
LABELED = [
    ("should this request go to the knowledge graph or to a workflow graph", {"dual-graph-router"}),
    ("pick vector, graph or hybrid retrieval for my query workload", {"vector-vs-graph-retrieval-selector", "rrf-hybrid-retrieval"}),
    ("limit what each agent persona can see in Neo4j with GRANT and DENY", {"subgraph-access-control"}),
    ("stop an agent from running a delete that cannot be undone", {"irreversible-action-gate"}),
    ("too many MCP tools are bloating my prompt, select just a few", {"rag-mcp-tool-selection", "mcp-gateway-two-meta-tools"}),
    ("use a cheaper model for simple nodes to cut the bill", {"model-routing-selector", "cost-performance-scorer"}),
    ("which node actually caused the failure and who should get the feedback", {"semantic-backprop-attributor"}),
    ("record every tool call and decision as a queryable graph", {"execution-graph"}),
    ("strong or eventual consistency for memory shared between agents", {"memory-consistency-model-selector"}),
    ("decide when two records refer to the same real-world entity", {"entity-resolution-strategy-selector"}),
    ("track when a fact was true versus when the system learned it", {"bi-temporal-edge"}),
    ("plan the investigation hypotheses as a DAG and stop early when one is confirmed", {"investigation-dag-planner"}),
    ("check a generated plan against domain constraints before executing it", {"constraint-guided-plan-validator"}),
    ("is this tool's description trustworthy or just marketing", {"draft-tool-trust-verifier"}),
    ("labeled property graph or RDF or hypergraph for my domain", {"graph-model-selector"}),
    ("migrate the graph schema safely while the system is live", {"schema-evolution-migrator"}),
    ("merge the results of parallel agent branches without conflicts", {"parallel-reconcile-merge"}),
    ("how ready is my agent platform across the eight pillars", {"eight-pillar-readiness-map"}),
]
h.check("all expected slugs exist in the repo", all(exp <= SLUGS for _, exp in LABELED))
h.check("n = 18 labeled queries", len(LABELED) == 18)
missing = [q for q, _e in LABELED if q not in _qcache]
if missing:
    for q, v in zip(missing, h.embed(missing)):      # one batched embedding call for all queries
        _qcache[q] = v
print("queries embedded and cached:", len(_qcache))

# %%
def evaluate(fn):
    hit1 = hit3 = 0
    rr = 0.0
    misses = []
    for q, exp in LABELED:
        ranked = fn(q, 8)
        pos = next((i for i, s in enumerate(ranked, 1) if s in exp), None)
        hit1 += pos == 1
        hit3 += bool(pos and pos <= 3)
        rr += 1.0 / pos if pos else 0.0
        if not pos or pos > 3:
            misses.append(q[:45])
    n = len(LABELED)
    return {"hit@1": round(hit1 / n, 2), "hit@3": round(hit3 / n, 2), "MRR": round(rr / n, 3), "misses_top3": len(misses)}, misses


SCORES, MISSES = {}, {}
for name, fn in METHODS.items():
    SCORES[name], MISSES[name] = evaluate(fn)
h.show([{"method": m, **s} for m, s in SCORES.items()])
for m in METHODS:
    print(f"{m:7s} missed from top 3:", MISSES[m] or "none")
h.check("each retriever returned results for every query", all(METHODS[m](q, 8) or m == "graph" for m in METHODS for q, _ in LABELED))
h.check("vector retrieval clearly beats chance (hit@3 >= 0.6)", SCORES["vector"]["hit@3"] >= 0.6)
soft("hybrid MRR >= graph-only MRR", SCORES["hybrid"]["MRR"] >= SCORES["graph"]["MRR"])
soft("hybrid MRR within 0.05 of the best single retriever or better",
     SCORES["hybrid"]["MRR"] >= max(SCORES["vector"]["MRR"], SCORES["graph"]["MRR"]) - 0.05)

# %% [markdown]
# ## 10. Verification checklist for AI-generated code
#
# Run it every time, cheapest check first. Items marked (auto) were automated in this notebook.
#
# 1. **State the acceptance test before you generate.** (auto: `SPEC` ends in a runnable contract)
# 2. **Static gates before execution**: syntax, banned imports/calls, stdlib-only, no secrets. (auto: `safety_scan`)
# 3. **Run in a box**: subprocess, timeout, scrubbed environment, scratch directory, never from `skills/`. (auto: `run_cli`)
# 4. **The model's own tests** (`benchmark`) must pass *and* contain negative cases. (auto)
# 5. **Held-out tests the model never saw.** The only defence against self-graded homework. (auto: `held_out_acceptance`)
# 6. **Bound the loop**: at most 3 repair rounds; never let the model edit the verifier. (auto)
# 7. **Read the diff**: only expected files, no deleted tests, no loosened assertions. (human)
# 8. **Judge as a second opinion**, with real results as evidence and a control sample. (auto, but advisory)
# 9. **Least privilege for runtime agents**: a read-only database role, a tool allow-list, audit trail. (auto)
# 10. **Promote by PR, not by script**: a human reviews generated code before it enters `skills/`. (human)
#
# The next cell runs the notebook-wide verification battery and removes everything this notebook created outside Neo4j's graph data.

# %%
shutil.rmtree(SCRATCH, ignore_errors=True)
rbac_teardown()
print("== Final verification battery ==")
h.check("scratch folder removed", not SCRATCH.exists())
h.check("skills/ unchanged: still 51 skills", len(list(h.SKILLS_DIR.glob("*/*/SKILL.md"))) == 51)
h.check("generated skill was never written into skills/", not list(h.SKILLS_DIR.glob(f"*/{SLUG}")))
users = {r["user"] for r in h.sysrun("SHOW USERS")}
roles = {r["role"] for r in h.sysrun("SHOW ROLES")}
h.check("RBAC user and role removed", AGENT_USER not in users and ROLE not in roles)
h.check("skill graph intact: 51 skills, 8 chapters, 153 SIMILAR edges",
        h.run("MATCH (s:Skill) RETURN count(s) AS c")[0]["c"] == 51 and
        h.run("MATCH (c:Chapter) RETURN count(c) AS c")[0]["c"] == 8 and
        h.run("MATCH ()-[r:SIMILAR]->() RETURN count(r) AS c")[0]["c"] == 153)
h.check("agent run persisted with its steps",
        h.run("MATCH (:Run {id:$id})-[:STEP]->(c) RETURN count(c) AS c", id=TRACE.run_id)[0]["c"] == len(CALLS))
h.check("generation attempts persisted", h.run("MATCH (:GenRun)-[:ATTEMPT]->(a) RETURN count(a) AS c")[0]["c"] == len(HIST) + len(HIST2))
h.check("no :Evil node (denied write really was denied)", h.run("MATCH (n:Evil) RETURN count(n) AS c")[0]["c"] == 0)
print("\nRetrieval scoreboard:", SCORES)
print("Judge (generated vs sabotaged):", j_good.get("overall"), "vs", j_bad.get("overall"))
print("Repair rounds: main", len(HIST), "| fault-injection", len(HIST2))

# %%
print(h.cost_report())
print("(token counts only: multiply by your current model prices; they change)")

# %% [markdown]
# ## Evaluation and LLM-as-a-Judge — how you know it works
#
# This notebook built three things that can each be wrong: a **skill retriever**, an **agent run**, and an **LLM-generated skill**. "It looked fine" is not
# evidence. Each needs a different kind of check, ordered cheapest and most trustworthy first:
#
# | Layer | Subject here | Check | Trust |
# |---|---|---|---|
# | 1. Deterministic | retrieval | hit@k / MRR on labelled queries (Section 9) | exact, repeatable |
# | 1. Deterministic | agent run | trace assertions: calls persisted, `NEXT` chain, `USED_SKILL` edge, a skill CLI exited 0 | exact |
# | 1. Deterministic | generated skill | *execution*: static gates, benchmark, held-out acceptance (Sections 6-7) | exact |
# | 2. LLM judge | agent answer, skill review | rubric score (Sections 5 and 7 already did this once) | useful, **biased** |
# | 3. Human | the judge itself | a small labelled set to **calibrate** the judge | the reference |
#
# We have used a judge twice already (the `verdict` cell in Section 5 and `judge_skill` in Section 7). Here we ask the question those cells skip:
# **how far can the judge be trusted?** Known judge biases, each with the probe we run below:
# * **Position bias**: in a pairwise comparison the first answer wins more often. *Probe:* swap the order (`h.judge_pairwise` does both orders).
# * **Verbosity bias**: longer sounds better. *Probe:* pad a weak answer with fluff and see whether the score rises.
# * **Self-preference**: gpt-4o likes gpt-4o prose (also true of the skill it generated). *Probe:* a sabotaged-answer control that is fluent but false.
# * **Instability**: the same input gets different scores. *Probe:* repeat the call (`h.judge_repeat`) and look at the spread.
# * **Leniency / rubric sensitivity**: compare with human labels (`h.agreement`).

# %% [markdown]
# ### 11.1 A gold set and human labels
# Eight questions about the things this notebook built, each with a short reference, plus 12 candidate answers **I wrote and labelled 1-5 by hand**
# (5 = fully correct and specific, 3 = partly right or vague, 1 = false or off-topic). They are stored as `:Ch9EvalGold` nodes (namespaced, removed at the end).

# %%
GOLD = [
    {"id": "G1", "q": "In incident INC-1001 (checkout-api p99 2400ms), what is the root cause and fix?",
     "ref": "deploy-481 cut the orders-db connection pool from 100 to 20, exhausting connections; fix: restore the pool to 100 or roll back deploy-481."},
    {"id": "G2", "q": "How was the earlier incident INC-0877 resolved?",
     "ref": "INC-0877 was fixed by raising the orders-db connection pool back to 100."},
    {"id": "G3", "q": "Why is hybrid retrieval of skills not guaranteed to beat vector retrieval, and what weakens the graph/vector comparison?",
     "ref": "The graph retriever walks SIMILAR edges derived from the same embeddings as the vector retriever, so the two are not independent evidence; "
            "hybrid fuses both rankings with reciprocal rank fusion and only helps when the graph adds signal."},
    {"id": "G4", "q": "What must an LLM-generated skill pass before it may be promoted into skills/?",
     "ref": "Static AST safety scan, the skill's own benchmark, held-out acceptance tests, and a human review; it is generated in a scratch directory and never written to skills/."},
    {"id": "G5", "q": "How is the agent's Neo4j access restricted?",
     "ref": "A read-only user with MATCH access, READ on embedding properties denied and TRAVERSE on Run and ToolCall denied, so it cannot see its own audit trail."},
    {"id": "G6", "q": "Why are the acceptance tests held out from the generator?",
     "ref": "They encode the contract the author defined independently, so the model cannot tailor its code and tests to pass its own checks."},
    {"id": "G7", "q": "Which checks of an agent run need no LLM judge?",
     "ref": "Persisted ToolCall count equals the Python count, the NEXT chain links all steps, a USED_SKILL edge exists, and a skill CLI call exited 0; the judge is only needed for the quality of the conclusion."},
    {"id": "G8", "q": "What is the shape of the skill graph?",
     "ref": "51 Skill nodes across 8 Chapter nodes with 3 SIMILAR edges per skill, 153 in total."},
]
CANDIDATES = [  # (cand id, gold id, answer, HUMAN label 1-5, why)
    ("c1", "G1", "deploy-481 reduced the orders-db connection pool from 100 to 20, so checkout-api waited for connections. Restore the pool to 100 or roll back deploy-481.", 5, "correct and specific"),
    ("c2", "G1", "It is probably a database problem. Check the connections and maybe restart something.", 2, "vague, no evidence"),
    ("c3", "G1", "A recent deployment, deploy-481, changed orders-db and is the likely cause, so roll it back.", 4, "right cause and fix, no pool numbers"),
    ("c4", "G1", "deploy-481 raised the orders-db pool from 20 to 100, which overloaded the cache. Scale the Redis cache up.", 1, "fluent but false"),
    ("c5", "G2", "INC-0877 had the same symptom and was resolved by raising the orders-db connection pool to 100.", 5, "correct"),
    ("c6", "G2", "It was fixed by some configuration change on the database.", 3, "right area, no detail"),
    ("c7", "G3", "The graph retriever uses SIMILAR edges built from the same embeddings, so it is not independent of vector search; RRF fusion helps only if the graph adds signal.", 5, "correct"),
    ("c8", "G3", "Graph and vector are fully independent evidence, so hybrid always wins.", 1, "wrong"),
    ("c9", "G4", "It must pass a static safety scan, its benchmark and held-out acceptance tests, then a human reviews it; it never goes into skills/ automatically.", 5, "correct"),
    ("c10", "G4", "Once its own benchmark passes it can be copied straight into skills/.", 1, "unsafe and wrong"),
    ("c11", "G5", "It uses a read-only role, and some data is hidden from it.", 3, "right idea, vague"),
    ("c12", "G6", "So that the model cannot write its own tests to match its code; the author's tests define done.", 5, "correct"),
    ("c13", "G7", "Compare the number of persisted tool calls with what Python logged, check the NEXT chain and the USED_SKILL edge; judge only the final conclusion.", 4, "correct, misses CLI exit check"),
    ("c14", "G8", "There are around fifty skill nodes in some chapters.", 2, "imprecise"),
]
GOLD_BY_ID = {g["id"]: g for g in GOLD}
CAND = {c[0]: {"gold": c[1], "answer": c[2], "why": c[4]} for c in CANDIDATES}
HUMAN = {c[0]: c[3] for c in CANDIDATES}

h.run("MATCH (g:Ch9EvalGold) DETACH DELETE g")
h.run("UNWIND $rows AS x CREATE (:Ch9EvalGold {id:x.id, q:x.q, ref:x.ref})", rows=GOLD)
labels = sorted(set(HUMAN.values()))
print("gold cases:", len(GOLD), "| candidates:", len(CANDIDATES), "| human label values:", labels)
h.check("gold set has >= 8 cases stored in Neo4j", h.run("MATCH (g:Ch9EvalGold) RETURN count(g) AS c")[0]["c"] >= 8)
h.check(">= 8 human-labelled candidates", len(CANDIDATES) >= 8)
h.check("labels span the scale, including the middle (2 and 3)", {1, 2, 3, 5} <= set(labels))

# %% [markdown]
# ### 11.2 Layer 1: deterministic checks on all three subjects
# No judge here. **Retrieval** is already scored exactly in Section 9 (`SCORES`); we add a paired per-query view, because averages hide who wins where.
# The **agent run** is checked from its trace. The **generated skill** is checked by *executing* it (`OK` came from the repair loop's real subprocess runs).
# Finally a crude text heuristic (token-overlap F1 against the reference, mapped to 1-5) gives the judge something cheap to be compared with.

# %%
def rr_of(fn, q, exp):
    ranked = fn(q, 8)
    pos = next((i for i, s in enumerate(ranked, 1) if s in exp), None)
    return 1.0 / pos if pos else 0.0


wins = {"hybrid>vector": 0, "vector>hybrid": 0, "tie": 0}
for q, exp in LABELED:
    a, b = rr_of(METHODS["hybrid"], q, exp), rr_of(METHODS["vector"], q, exp)
    wins["hybrid>vector" if a > b else "vector>hybrid" if b > a else "tie"] += 1
print("retrieval (see Section 9 table for hit@k/MRR); paired per-query:", wins)

agent_rows = TRACE.timeline()
agent_checks = {
    "trace persisted every call": len(agent_rows) == len(CALLS) > 0,
    "queried the graph": "graph_query" in CALLS,
    "retrieved skills": "find_skills" in CALLS,
    "a skill CLI exited ok": any(r["tool"] == "skill_cli" and r["ok"] for r in agent_rows),
    "answer names deploy-481 or the pool": "deploy-481" in ANSWER or "pool" in ANSWER.lower(),
}
for k, v in agent_checks.items():
    print(f"  agent  {'ok ' if v else 'NO '} {k}")
print("  skill  executed OK (static + benchmark + held-out):", OK, "| files:", sorted(FINAL))


def tokens(s):
    return set(re.findall(r"[a-z0-9\-]+", s.lower()))


def f1_to_5(cand_id):
    a, r = tokens(CAND[cand_id]["answer"]), tokens(GOLD_BY_ID[CAND[cand_id]["gold"]]["ref"])
    inter = len(a & r)
    if not inter:
        return 1
    p, rc = inter / len(a), inter / len(r)
    f1 = 2 * p * rc / (p + rc)
    return 1 + round(4 * min(1.0, f1 / 0.6))


HEUR = {cid: f1_to_5(cid) for cid in CAND}
ag_h = h.agreement([HUMAN[c] for c in CAND], [HEUR[c] for c in CAND])
print("heuristic vs human:", ag_h)
h.check("deterministic retrieval scores exist for all three methods", set(SCORES) == set(METHODS))
h.check("trace persisted and chain intact (Layer 1 on the agent run)", agent_checks["trace persisted every call"])
soft("agent run passes all deterministic trace checks", all(agent_checks.values()))
h.check("heuristic produced a 1-5 score for every candidate", all(1 <= v <= 5 for v in HEUR.values()))

# %% [markdown]
# ### 11.3 Layer 2: rubric judging with explicit anchors
# A criterion is only as good as its anchors: say what a **1** looks like and what a **5** looks like. `h.judge_rubric` scores each criterion separately, with the
# reference supplied so the judge is checking facts rather than style. The rubric for the *agent answer* in Section 5 and the *skill review* in Section 7
# are the same idea with different criteria; here we judge the 14 labelled answers.

# %%
CRITERIA = {
    "correctness": "1 = contradicts the reference or invents facts (wrong cause, wrong control, wrong numbers); "
                   "5 = every claim agrees with the reference.",
    "specificity": "1 = generic advice that could apply to any system; "
                   "5 = names the concrete entities and values from the reference (deployment, pool sizes, checks, counts).",
    "completeness": "1 = misses the main point of the question; 5 = covers all the parts of the question that the reference covers.",
}


def to_5(mean):
    return max(1, min(5, int(round(mean))))


JUDGE = {}
for cid, c in CAND.items():
    g = GOLD_BY_ID[c["gold"]]
    r = h.judge_rubric(g["q"], c["answer"], CRITERIA, reference=g["ref"])
    JUDGE[cid] = {"mean": r["mean"], "score": to_5(r["mean"])}
h.show([{"cand": cid, "human": HUMAN[cid], "judge": JUDGE[cid]["score"], "heur": HEUR[cid], "why": CAND[cid]["why"]} for cid in CAND])
h.check("every candidate got a judge score in 1-5", all(1 <= v["score"] <= 5 for v in JUDGE.values()))

# %% [markdown]
# ### 11.4 Judge the judge: calibration against the human labels
# `h.agreement` reports exact match, within-one, Cohen's kappa (agreement beyond chance) and Spearman (rank order). A judge that ranks well but is
# systematically lenient shows high Spearman and a middling kappa. We print a **WARNING** rather than fail: a low number is a finding about this judge,
# not a bug in the notebook.

# %%
cal = h.agreement([HUMAN[c] for c in CAND], [JUDGE[c]["score"] for c in CAND])
print("judge vs human:", cal)
if cal["kappa"] < 0.4 or cal["spearman"] < 0.7:
    print(f"WARNING: judge agreement is low (kappa={cal['kappa']}, spearman={cal['spearman']}); do not gate releases on this judge without improving the rubric.")
else:
    print(f"OK: judge agrees well with the human labels (kappa={cal['kappa']}, spearman={cal['spearman']}).")
lenient = [c for c in CAND if JUDGE[c]["score"] - HUMAN[c] >= 2]
harsh = [c for c in CAND if HUMAN[c] - JUDGE[c]["score"] >= 2]
print("judge >= 2 points more lenient than the human on:", lenient or "none", "| harsher on:", harsh or "none")
h.check("calibration covered every labelled candidate", cal["n"] == len(CANDIDATES))

# %% [markdown]
# ### 11.5 Bias probe: position
# `h.judge_pairwise` asks twice with the answers swapped. If the winner changes with the order, the verdict was position, not quality (`consistent=False`).
# We test a real pair (good vs vague) and an **identical** pair, which should tie.

# %%
g1 = GOLD_BY_ID["G1"]
pw = h.judge_pairwise(g1["q"], CAND["c1"]["answer"], CAND["c2"]["answer"], "correctness and specificity versus the reference", reference=g1["ref"])
pw_same = h.judge_pairwise(g1["q"], CAND["c1"]["answer"], CAND["c1"]["answer"], "correctness and specificity versus the reference", reference=g1["ref"])
print("good (a) vs vague (b):", {k: pw[k] for k in ("winner", "consistent")}, "| identical pair:", {k: pw_same[k] for k in ("winner", "consistent")})
print("PASS: picked the good answer in both orders" if pw["winner"] == "a" and pw["consistent"] else "WARNING: the order of the answers changed or blurred the verdict")
print("PASS: identical answers tie" if pw_same["winner"] == "tie" else "WARNING: identical answers did not tie (position or noise effect)")
h.check("pairwise verdicts are valid", pw["winner"] in ("a", "b", "tie") and pw_same["winner"] in ("a", "b", "tie"))

# %% [markdown]
# ### 11.6 Bias probe: verbosity
# We take the weakest G1 answer (`c2`, human 2) and pad it with confident filler that adds **no facts**. If the score rises, the judge is paying for length.

# %%
FLUFF = (" In today's complex distributed environments, reliability engineering demands a holistic, data-driven and systematic approach. "
         "It is important to carefully consider the full range of contributing factors, to collaborate across teams, to follow established best practices, "
         "and to continuously monitor the system so that stakeholders can be kept informed throughout the investigation.")
base_r = h.judge_rubric(g1["q"], CAND["c2"]["answer"], CRITERIA, reference=g1["ref"])
pad_r = h.judge_rubric(g1["q"], CAND["c2"]["answer"] + FLUFF, CRITERIA, reference=g1["ref"])
rise = pad_r["mean"] - base_r["mean"]
print(f"weak answer mean={base_r['mean']:.2f}; padded mean={pad_r['mean']:.2f}; rise={rise:+.2f}")
print("WARNING: padding raised the score; verbosity bias" if rise > 0.75 else "OK: padding did not buy a meaningful score")
h.check("verbosity probe produced valid scores", 1 <= base_r["mean"] <= 5 and 1 <= pad_r["mean"] <= 5)

# %% [markdown]
# ### 11.7 Bias probes: self-consistency and a sabotaged control
# `h.judge_repeat` calls the judge three times at temperature 0.7 on the same input; a large **spread** means a single score is not reliable. Then the control:
# a *fluent but false* answer (`c4`) must score clearly below the correct one (`c1`). This is the same logic as the sabotaged skill in Section 6: a grader that
# cannot tell the sabotaged sample from the good one is not measuring anything.

# %%
RUBRIC_SHORT = "5 = agrees with the reference and names the concrete entities; 3 = partly right or vague; 1 = false or off-topic. Reference: "
rep = {cid: h.judge_repeat(GOLD_BY_ID[CAND[cid]["gold"]]["q"], CAND[cid]["answer"], RUBRIC_SHORT + GOLD_BY_ID[CAND[cid]["gold"]]["ref"], n=3)
       for cid in ("c3", "c11")}
for cid, r in rep.items():
    print(f"{cid} (human {HUMAN[cid]}): scores={r['scores']} mean={r['mean']:.2f} spread={r['spread']}",
          "WARNING: unstable" if r["spread"] >= 2 else "OK: stable")
good_s, bad_s = JUDGE["c1"]["mean"], JUDGE["c4"]["mean"]
print(f"control: correct answer {good_s:.2f} vs fluent-but-false {bad_s:.2f}")
h.check("judge scores the fluent-but-false answer at least 1 point below the correct one", good_s - bad_s >= 1.0)

# %% [markdown]
# ### 11.8 Judge versus the cheap heuristic
# Where the judge and the token-overlap heuristic differ by two or more points, one of them is wrong; read those rows. Typically the heuristic is fooled by
# paraphrase (right idea, different words) and the judge by confident prose.

# %%
disagree = [{"cand": c, "human": HUMAN[c], "judge": JUDGE[c]["score"], "heur": HEUR[c], "answer": CAND[c]["answer"][:70]}
            for c in CAND if abs(JUDGE[c]["score"] - HEUR[c]) >= 2]
print(f"rows where judge and heuristic differ by >= 2: {len(disagree)} of {len(CAND)}")
h.show(disagree)
j_err = sum(abs(JUDGE[c]["score"] - HUMAN[c]) for c in CAND) / len(CAND)
h_err = sum(abs(HEUR[c] - HUMAN[c]) for c in CAND) / len(CAND)
print(f"mean absolute error vs human: judge={j_err:.2f}, heuristic={h_err:.2f}")
print("the judge earns its cost here" if j_err < h_err else "WARNING: the free heuristic is at least as close to the humans; the judge is not paying for itself on this set")

# %% [markdown]
# ### 11.9 Regression harness and a CI-style gate
# Evaluation matters when something *changes*. We treat the retriever as the system under test: **baseline** = vector only, **candidate** = hybrid. Each of the
# 18 labelled queries is one case (score = reciprocal rank, pass = hit@3); `h.run_eval` stores an `(:EvalRun)` per version, `h.compare_runs` compares them with Cypher,
# and a gate turns the comparison into PASS/FAIL without raising, so the notebook finishes. In CI you would exit non-zero on FAIL.

# %%
EVAL_NAME = "ch9_skill_retrieval"
h.run("MATCH (r:EvalRun) WHERE r.name STARTS WITH 'ch9_' OPTIONAL MATCH (r)-[:HAS_RESULT]->(x) DETACH DELETE x, r")
RET_CASES = [{"id": f"q{i}", "input": q, "exp": exp} for i, (q, exp) in enumerate(LABELED, 1)]


def retr_score(case, ranked):
    pos = next((i for i, s in enumerate(ranked, 1) if s in case["exp"]), None)
    return {"score": 1.0 / pos if pos else 0.0, "passed": bool(pos and pos <= 3), "notes": f"pos={pos}"}


summaries = {}
for ver, name in (("v1-vector", "vector"), ("v2-hybrid", "hybrid")):
    summaries[ver] = h.run_eval(EVAL_NAME, RET_CASES, lambda c, n=name: METHODS[n](c["input"], 8), retr_score, version=ver)
runs = h.compare_runs(EVAL_NAME)
h.show(runs)
TOLERANCE = 0.10
base, cand = summaries["v1-vector"]["pass_rate"], summaries["v2-hybrid"]["pass_rate"]
gate = "PASS" if cand >= base - TOLERANCE else "FAIL"
print(f"CI GATE: baseline vector={base:.2f}  candidate hybrid={cand:.2f}  tolerance={TOLERANCE:.2f}  ->  {gate}")
h.check("both versions stored and comparable via Cypher", len(runs) == 2 and all(r["n"] == len(LABELED) for r in runs))

# %% [markdown]
# ### 11.10 Report card, a vibe-coding prompt, and cleanup
# **Trust:** hit@k / MRR and trace assertions (exact); the generated skill's *execution* results (benchmark, held-out tests); the judge's ordering of clearly
# good versus clearly bad answers (the sabotage control). **Do not trust without checking:** absolute judge scores, the judge's reading of the skill it
# co-authored (same model family), single-sample scores, and 18 queries / 14 labels as a benchmark: they are a method demo.
#
# A prompt to give your coding assistant to add this to your own project:
#
# ```text
# Goal: add an evaluation harness for my agent, my retriever and my generated code.
# Context: Python 3.11, Neo4j, an OpenAI-compatible LLM. Helpers exist: judge_rubric(q, answer, criteria, reference), judge_pairwise, judge_repeat, agreement, run_eval, compare_runs.
# Constraints: deterministic checks first (hit@k/MRR, trace assertions, execution of generated code); judge criteria need explicit 1-vs-5 anchors; supply a reference answer.
# Calibrate the judge on >= 8 human-labelled answers (print kappa and spearman, WARN not FAIL when low); probe position, verbosity and self-consistency; include a fluent-but-false control.
# Store every EvalRun in Neo4j with a version and add a gate that fails when the candidate is worse than the baseline by more than a tolerance.
# Acceptance: the control scores >= 1 point below the correct answer; the gate prints PASS/FAIL; all test data is namespaced and cleaned up.
# ```

# %%
h.run("MATCH (g:Ch9EvalGold) DETACH DELETE g")
h.run("MATCH (r:EvalRun) WHERE r.name STARTS WITH 'ch9_' OPTIONAL MATCH (r)-[:HAS_RESULT]->(x) DETACH DELETE x, r")
left = h.run("MATCH (n) WHERE n:Ch9EvalGold OR n:EvalResult OR (n:EvalRun AND n.name STARTS WITH 'ch9_') RETURN count(n) AS c")[0]["c"]
h.check("evaluation artefacts cleaned up", left == 0)
h.check("eval section left the skill graph intact (51 skills)", h.run("MATCH (s:Skill) RETURN count(s) AS c")[0]["c"] == 51)
print(h.cost_report())

# %% [markdown]
# ## Cost, limits and what to swap next
#
# **Cost.** One run makes roughly 25-35 gpt-4o calls (agent steps with growing context, one generation, up to two repairs, three reviews/gap checks)
# and a handful of embedding calls (51 skills + ~25 queries). The cell above prints exact token counts; multiply them by current prices. Rerunning the
# notebook repeats all of it, so do not put it in a loop.
#
# **What is still approximate, honestly**
# * **LLM non-determinism.** `temperature=0` reduces variance but does not remove it (model updates, batching, tool-call ordering). The agent may
#   take a different tool path next run; that is why LLM-dependent claims use tolerant `soft` checks, while data and safety invariants use hard `h.check`.
# * **Judge bias.** gpt-4o grades text written by gpt-4o: self-preference, verbosity bias and rubric sensitivity are real. The control sample and the grounding
#   in real validation results *reduce* the problem; they do not make the score a measurement. Use another model family or human spot checks for decisions.
# * **The judge is a model too.** Section 11 calibrates it against 14 human labels and probes position, verbosity and self-consistency; a low kappa means the judge
#   is a screening tool, not a gate. Re-calibrate whenever the judge model, rubric or task changes.
# * **Judge cost is extra calls.** Every judged item is another gpt-4o call (the calibration alone is 14 rubric calls plus the probes); budget for it and judge only what execution cannot check.
# * **Gold labels are ours.** The 14 labelled answers and 8 references were written by the notebook author; they calibrate the method, they are not a benchmark.
# * **Small eval.** 18 hand-labeled queries by the notebook author; the "acceptable slugs" are a judgement call, and the `SIMILAR` edges are derived from the same
#   embeddings the vector retriever uses, so graph and vector are not independent. Treat the table as a demo of the method, not a benchmark result.
# * **Generated-code safety.** The static gates are an AST *denylist*; they stop obvious mistakes, not a determined adversary (for example, obfuscated attribute access).
#   Defence in depth here is: run only in a subprocess with a timeout, a scrubbed environment and a scratch directory, and **never** execute generated code from `skills/`
#   or promote it without human review. For untrusted models use a container or VM sandbox with no network.
# * **Held-out tests are only as good as their author.** They check the contract we thought of, not every behaviour.
# * **RBAC scope.** The demo user is read-only for this database; production should add per-persona roles (see the `subgraph-access-control` skill), query timeouts and result limits.
#
# **What to swap next**
# * Replace the `SIMILAR`-based graph retriever with real multi-hop traversal over typed relations you extract (e.g. `PRECEDES`, `COMPOSES_WITH`) or Cypher generated by an LLM under the same read-only role.
# * Add a **reranker** (cross-encoder or LLM) after hybrid retrieval, and measure it on a larger, independently labeled set.
# * Move the generation harness into a container sandbox and open a **pull request** automatically, so CI (`--help` + `benchmark` for every skill) and a human review gate promotion.
# * Persist the agent traces in Neo4j across incidents and feed repeated successful paths back as examples (the Ch7 self-evolution loop).
#
# The remaining discipline is the one from Section 1: **intent -> prompt -> generate -> verify -> iterate**. The assistant writes; you define "done" and prove it.
# This notebook is a companion to *Agentic GraphRAG* (O'Reilly, Anthony Alcaraz and Sam Julien): read the book for the theory, run these skills for the practice.

# %% [markdown]
# ## Observability — what you can look at now
#
# Everything above ran with the backend chosen by `OBSERVABILITY_BACKEND` in `.env`:
#
# - **`phoenix`** (Arize Phoenix, http://localhost:6006): open project *"ch9-vibe-coding-capstone"*. Every chat/embedding call is a span
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
