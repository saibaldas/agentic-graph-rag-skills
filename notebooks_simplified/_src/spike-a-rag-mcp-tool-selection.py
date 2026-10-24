# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > **Beginner (tier 1/3) ← you are here**  →  Intermediate (tier 2/3, no notebook for this topic)  →  Advanced (tier 3/3, no notebook for this topic)

# %% [markdown]
# # Spike A (Simplified) — Picking the right tool with RAG-MCP, backed by Neo4j
#
# **Who this is for:** you are new to GraphRAG and agents. Every step is explained in plain language.
#
# **What is a "spike"?** A spike is a *small experiment* that answers one risky question
# before you build the real thing. This spike asks:
# *"If an agent has 30 tools, can we show it only the 3–5 relevant ones and still pick well?"*
#
# **What you will learn**
# 1. Why dumping *every* tool description into an LLM prompt wastes context (tokens).
# 2. How the **RAG-MCP** skill retrieves only the relevant tools for a question.
# 3. How to store tools as **nodes** in Neo4j and record which ones were selected.
# 4. How the agent then *acts* on the top tool — here with a real Cypher query over log events stored in the graph.
# 5. A known **failure mode** of simple word-matching, and where embeddings would fix it.
#
# **Story:** the checkout API is slow (the running example of this repo). The agent must choose
# which of 30 AWS tools to use for the investigation.
#
# **Jargon, defined once**
# * **Token** – a chunk of text (roughly ¾ of a word). LLMs have a limited token budget per prompt.
# * **MCP tool** – a function an agent may call (e.g. "query CloudWatch logs"), described in text.
# * **Embedding** – a list of numbers that captures the *meaning* of text. This spike uses a
#   simpler stand-in (word overlap); the skill marks where embeddings would plug in.
# * **Node / relationship / Cypher** – a graph database stores *nodes* (things) and *relationships*
#   (links between them); *Cypher* is its query language, e.g. `(a)-[:SELECTED]->(b)`.
#
# > **Before running:** start Neo4j with `docker compose -f docker-compose/docker-compose.yml up -d`.

# %% [markdown]
# ## 1. Load the skill and connect to Neo4j
#
# `g.load_skill(chapter_folder, skill_name)` imports the skill's `lib.py`. The skill ships a
# registry (a JSON list) of 30 tools. We load it into Python first, then put it in the graph.

# %%
import graph_helpers as g

lib = g.load_skill("tool-orchestration", "rag-mcp-tool-selection")
SKILL_DIR = g.SKILLS_DIR / "tool-orchestration" / "rag-mcp-tool-selection"
REGISTRY_PATH = SKILL_DIR / "sample-aws-tools.json"

registry = lib.load_registry(REGISTRY_PATH)
print(f"Registry has {len(registry)} AWS service tools.")
print("First tool:", registry[0]["name"])

# %% [markdown]
# ## 2. Store the tools in Neo4j
#
# We create one `Tool` node per tool and one `Topic` node per keyword (like `latency`, `logs`),
# linked by `HAS_TOPIC`. Tools sharing a topic become neighbours in the graph — something a
# flat list can't show you.
#
# `UNWIND $tools AS t` means "run the following once for every item in the list I pass in".

# %%
g.clear_graph()
g.run("""
    UNWIND $tools AS t
    CREATE (tool:Tool {name: t.name, description: t.description, boto3_method: t.boto3_method})
    WITH tool, t
    UNWIND t.key_topics AS topic_name
    MERGE (topic:Topic {name: topic_name})
    MERGE (tool)-[:HAS_TOPIC]->(topic)
""", tools=registry)

g.show(g.run("""
    MATCH (t:Tool) WITH count(t) AS tools
    MATCH (p:Topic) RETURN tools, count(p) AS topics
"""))

# %% [markdown]
# **Which topics are shared by the most tools?** (Reading it: `tools_with_topic` is how many tools
# point at that topic.) Popular topics like `latency` mean several tools are plausible candidates —
# exactly why a selector is needed.

# %%
g.show(g.run("""
    MATCH (t:Tool)-[:HAS_TOPIC]->(p:Topic)
    RETURN p.name AS topic, count(t) AS tools_with_topic,
           collect(t.name)[..3] AS example_tools
    ORDER BY tools_with_topic DESC, topic LIMIT 6
"""))

# %% [markdown]
# ## 3. The question and the baseline (no filtering)
#
# The on-call engineer asks:

# %%
QUERY = "why is the checkout api slow during the 3pm latency spike"
print(QUERY)

# %% [markdown]
# **Baseline** = what a standard MCP `tools/list` does: paste *all* 30 tool descriptions into the
# prompt. At 30 tools that is already big; real companies have thousands.

# %%
baseline = lib.baseline_prompt(registry, QUERY)
baseline_tokens = lib.approximate_token_count(baseline)
print(f"Baseline prompt: {baseline_tokens} approx. tokens, {len(baseline)} characters")
print("First 300 characters:")
print(baseline[:300] + "...")

# %% [markdown]
# ## 4. RAG-MCP: Retrieve → Validate → Invoke
#
# 1. **Retrieve** – score every tool by word overlap with the question, keep the top-k.
# 2. **Validate** – drop tools that share no domain word with the question.
# 3. **Invoke** – build a prompt containing only the survivors.
#
# `top_k=5` means "at most five tools". The `score` is a similarity number (higher = better match).
# `reduction_pct` compares the filtered prompt with the baseline.

# %%
result = lib.select(QUERY, REGISTRY_PATH, top_k=5)
print(f"Registry size: {result['registry_size']} tools, selected: {len(result['selected'])}\n")
for s in result["selected"]:
    print(f"  [{s['score']:.3f}] {s['name']}")
print()
print(f"Baseline tokens: {result['baseline_tokens']}")
print(f"Filtered tokens: {result['filtered_tokens']}")
print(f"Reduction:       {result['reduction_pct']}%")

# %% [markdown]
# **Verification gate (the spike's pass/fail):** the book claims a 50–70% token reduction.
# With only 30 tools we expect to be *above* 50%. If not, the retriever is broken.

# %%
assert result["reduction_pct"] > 50.0, (
    f"Token reduction {result['reduction_pct']}% below the 50% floor."
)
print(f"PASS: {result['reduction_pct']}% reduction (book floor: 50%)")

# %% [markdown]
# ## 5. Remember the decision in the graph
#
# We save the question as a `Query` node and connect it to each selected tool with a `SELECTED`
# relationship that carries the `score` and `rank`. Now "which tools did the agent consider?" is a
# graph query, and over time you could learn which tools actually helped.

# %%
g.run("""
    CREATE (q:Query {text: $text, baseline_tokens: $bt, filtered_tokens: $ft, reduction_pct: $red})
    WITH q
    UNWIND range(0, size($selected) - 1) AS i
    WITH q, $selected[i] AS s, i
    MATCH (t:Tool {name: s.name})
    CREATE (q)-[:SELECTED {score: s.score, rank: i + 1}]->(t)
""", text=QUERY, bt=result["baseline_tokens"], ft=result["filtered_tokens"],
      red=result["reduction_pct"], selected=result["selected"])

g.show(g.run("""
    MATCH (q:Query)-[r:SELECTED]->(t:Tool)
    RETURN r.rank AS rank, t.name AS tool, r.score AS score, t.boto3_method AS boto3_method
    ORDER BY rank
"""))

# %% [markdown]
# ## 6. The agent acts on the top tool
#
# The top-ranked tool is `cloudwatch_logs_insights_query` ("search service logs").
# Real CloudWatch isn't needed here: we store five log lines from `checkout-api` as `LogEvent`
# nodes **in Neo4j**, and the equivalent of the Logs Insights filter
# `filter status=504 or duration_ms > 1000` becomes a Cypher `WHERE`.

# %%
top_tool = result["selected"][0]["name"]
print("Top tool chosen:", top_tool)
assert top_tool == "cloudwatch_logs_insights_query"

g.run("""
    MATCH (t:Tool {name: $tool})
    MERGE (svc:Service {name: 'checkout-api'})
    WITH svc
    UNWIND $events AS e
    CREATE (l:LogEvent {request_id: e.id, level: e.level, duration_ms: e.duration,
                        status: e.status, db_pool_exhausted: e.pool})
    CREATE (l)-[:EMITTED_BY]->(svc)
""", tool=top_tool, events=[
    {"id": "req-001", "level": "INFO",  "duration": 42,   "status": 200, "pool": False},
    {"id": "req-007", "level": "WARN",  "duration": 2814, "status": 200, "pool": False},
    {"id": "req-013", "level": "ERROR", "duration": 5012, "status": 504, "pool": True},
    {"id": "req-019", "level": "ERROR", "duration": 4998, "status": 504, "pool": True},
    {"id": "req-024", "level": "WARN",  "duration": 3110, "status": 200, "pool": False},
])

rows = g.run("""
    MATCH (l:LogEvent)-[:EMITTED_BY]->(:Service {name:'checkout-api'})
    WHERE l.status = 504 OR l.duration_ms > 1000
    RETURN l.request_id AS request, l.level AS level, l.duration_ms AS duration_ms,
           l.status AS status, l.db_pool_exhausted AS pool_exhausted
    ORDER BY l.duration_ms DESC
""")
g.show(rows)

# %% [markdown]
# **How to read it:** four of five requests are slow, and the two `504` errors say
# `pool_exhausted = True` — the database connection pool ran out. That matches the story's
# root cause (a deploy shrank the pool from 100 to 20).

# %%
assert len(rows) == 4
assert sum(1 for r in rows if r["pool_exhausted"]) == 2
print("PASS: the selected tool's query returned the slow/failing requests from Neo4j.")

# %% [markdown]
# ## 7. Benchmark battery: does it work beyond one question?
#
# One good result could be luck. We run eight different DevOps questions and average the token
# reduction. We also store each result as a `Query` node so we can aggregate with Cypher.

# %%
scenarios = [
    "why is my checkout api slow",
    "show me 5xx errors in the last hour",
    "which downstream service is slow during the latency spike",
    "audit who changed the production database",
    "find the latest backup in s3",
    "ssh into the host without ssh",
    "page the on-call engineer about the queue backlog",
    "why is this getting access denied",
]
reductions = []
for q in scenarios:
    r = lib.select(q, REGISTRY_PATH, top_k=5)
    top = r["selected"][0]["name"] if r["selected"] else "-"
    reductions.append(r["reduction_pct"])
    g.run("""
        CREATE (x:Query {text: $text, reduction_pct: $red, benchmark: true})
        WITH x
        OPTIONAL MATCH (t:Tool {name: $top})
        FOREACH (_ IN CASE WHEN t IS NULL THEN [] ELSE [1] END |
                 CREATE (x)-[:SELECTED {rank: 1}]->(t))
    """, text=q, red=r["reduction_pct"], top=top)
    print(f"{r['reduction_pct']:>6}%  {top:<32} {q}")

avg = sum(reductions) / len(reductions)
print(f"\nAverage reduction across {len(scenarios)} scenarios: {avg:.1f}%")
assert avg > 50.0, "Average reduction below 50%"
print(f"PASS: average {avg:.1f}% > 50% floor.")

# %% [markdown]
# Ask the graph itself: which tools get picked most often?

# %%
g.show(g.run("""
    MATCH (q:Query {benchmark:true})-[:SELECTED]->(t:Tool)
    RETURN t.name AS tool, count(q) AS times_top_pick
    ORDER BY times_top_pick DESC, tool
"""))

# %% [markdown]
# ## 8. A known failure mode (honest limits)
#
# Word overlap rewards *keyword density*, not meaning. For *"audit who changed the production
# database"* the right tool is `cloudtrail_lookup_events` (an audit log), yet a simple scorer may
# pick another tool such as `dynamodb_scan`. Let's see what it picked:

# %%
r = lib.select("audit who changed the production database", REGISTRY_PATH, top_k=5)
for s in r["selected"]:
    print(f"  [{s['score']:.3f}] {s['name']}")

# %% [markdown]
# That is why the skill marks `score_tool` as the **seam**: swap in embeddings (which understand
# that "who changed" ≈ "audit trail") without changing the rest of the pipeline.
#
# ## Recap
# * Dumping all tools into the prompt is expensive; RAG-MCP keeps only the top-k.
# * Token reduction passed the 50% gate on the headline question **and** on the 8-question battery.
# * Tools, topics, queries and selections all live in Neo4j, so tool choice is queryable.
# * The agent acted on the top tool via a real Cypher query over `LogEvent` nodes — the pool-exhaustion clue.
# * Simple word matching has a known failure mode; embeddings are the planned upgrade.
