# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > [Beginner (tier 1/3)](../notebooks_simplified/ch3-knowledge-representation.ipynb)  →  **Intermediate (tier 2/3) ← you are here**  →  [Advanced (tier 3/3)](../notebooks_advanced_simplified/ch3-knowledge-representation.ipynb)

# %% [markdown]
# # Chapter 3 (Intermediate) — Knowledge Representation: understand the why, then do it yourself
#
# **Tier 2 of 3.** The beginner notebook showed *what* each skill does. Here we open the box:
#
# * see the **inputs and outputs** of six skills from `skills/knowledge-representation/` ("anatomy of the output"),
# * feed them from **Cypher on a real Neo4j graph** and **write results back**,
# * chain skills in a plain-Python **pipeline** on *checkout-api is slow*,
# * do two **Your turn** exercises, and learn how to **trust** a skill (CLI + benchmark).
#
# Scenario (fictional AWS account `123456789012`). Book mapping: *Agentic GraphRAG*, Ch. 3.
# Tier 3 uses all 8 skills, an agent, and more experiments.

# %%
import int_helpers as h
import json
import os
import subprocess
import sys

print(h.load_devops_graph())
CH = "knowledge-representation"
gms = h.load_skill(CH, "graph-model-selector")
tgr = h.load_skill(CH, "three-graph-router")
ers = h.load_skill(CH, "entity-resolution-strategy-selector")
sps = h.load_skill(CH, "schema-pattern-selector")
hms = h.load_skill(CH, "homoiconic-meta-schema")
cag = h.load_skill(CH, "capability-authorization-gate")
h.show([{"skill": r["slug"], "what it does": r["description"][:80]} for r in h.skill_catalog(CH)])

# %% [markdown]
# ## 1. Seed extra data
#
# `MERGE` = create-if-missing (idempotent, safe to re-run); `UNWIND` turns a list into rows so one statement creates
# many nodes. We add three callers of `checkout-api` and two log lines.

# %%
h.run("""
MATCH (c:Service {name:'checkout-api'})
UNWIND ['web-frontend','mobile-bff','partner-api'] AS n
MERGE (s:Service {name:n}) SET s.tier='medium', s.owner='team-edge'
MERGE (s)-[:DEPENDS_ON]->(c)
""")
h.run("""
UNWIND [{id:'log-0', text:'checkout-api p99 latency spiked at 8:41am'},
        {id:'log-1', text:'orders-db connection pool exhausted: 20/20 in use'}] AS l
MERGE (:LogLine {id:l.id, text:l.text})
""")
n = h.run("MATCH (s:Service)-[:DEPENDS_ON]->(:Service {name:'checkout-api'}) RETURN count(s) AS n")[0]["n"]
h.check("checkout-api has 4 direct dependents", n == 4)

# %% [markdown]
# ## 2. `graph-model-selector` — pick the model from the reasoning you need
#
# **Why it matters:** choosing a graph model by looking at your data leads to regret; choose by *what the agent must
# reason about*. Trade-off: hypergraphs express n-ary facts natively, but property graphs have the tooling.
#
# **Cypher #1 — variable-length path.** `-[:DEPENDS_ON*1..3]->` follows 1 to 3 hops; `count(DISTINCT d)` counts each
# node once even if several paths reach it. This is the *blast radius*: who is hurt if `orders-db` fails.

# %%
blast = h.run("""
MATCH (d:Service)-[:DEPENDS_ON*1..3]->(t {name:$t})
RETURN t.name AS target, count(DISTINCT d) AS blast_radius
""", t="orders-db")
h.show(blast)

# %% [markdown]
# **Anatomy of the output.**
#
# * `Requirements(formal_reasoning, n_ary_relations, performance, tool_ecosystem, constraint_expr)` — each 0–3,
#   how much the workload needs that capability.
# * `recommend_model(reqs)` → dict: `recommended` (winner), `scores` (per model), `rationale`,
#   `hybrid_recommended` (top two nearly tied?) and `hybrid` (which pair).
# * `HyperEdge(type, nodes, attributes)` + `representation_cost(edge)` → `arity`, and element counts for each model.

# %%
ctx = h.run("""
MATCH (dep:Deployment {id:'deploy-481'})-[:DEPLOYED_TO]->(svc), (dep)-[:CHANGED]->(res)
RETURN dep.id AS dep, svc.name AS svc, res.name AS res
""")[0]
edge = gms.HyperEdge(type="DEPLOYMENT", nodes={"deployment": ctx["dep"], "service": ctx["svc"],
                                              "resource": ctx["res"]}, attributes={"version": "v2.14.0"})
cost = gms.representation_cost(edge)
reqs = gms.Requirements(formal_reasoning=1, n_ary_relations=2, performance=3, tool_ecosystem=3, constraint_expr=1)
rec = gms.recommend_model(reqs)
print("cost:", cost)
print("recommended:", rec["recommended"], "| scores:", rec["scores"], "| hybrid?", rec["hybrid_recommended"])

# Write-back: apply the reification the skill describes (a Fact node linking each participant).
reif = gms.reify_as_property_graph(edge)
h.run("""
MERGE (f:Reified:Fact {id:$id}) SET f.type=$type, f.version=$ver
WITH f
UNWIND $parts AS p
MATCH (n) WHERE n.name = p OR n.id = p
MERGE (f)-[:PARTICIPANT]->(n)
""", id="fact-deploy-481", type=edge.type, ver="v2.14.0", parts=list(edge.nodes.values()))
n_part = h.run("MATCH (:Fact {id:'fact-deploy-481'})-[:PARTICIPANT]->(n) RETURN count(n) AS n")[0]["n"]
print("read back: participants =", n_part, "| skill said auxiliary_edges =", reif["auxiliary_edges"])
h.check("property graph wins for ops workloads", rec["recommended"] == "property_graph")
h.check("hypergraph 1 element vs property graph 1+arity",
        cost["hypergraph_elements"] == 1 and cost["property_graph_elements"] == 1 + cost["arity"])
h.check("reified fact has one edge per participant", n_part == reif["auxiliary_edges"])

# %% [markdown]
# ## 3. `three-graph-router` — keep facts, text and guesses apart
#
# **Why it matters:** if LLM guesses (Subject graph) sit beside trusted records (Domain graph), a hallucination can
# masquerade as a fact. The router *refuses* records that break the rules. Trade-off: three layers mean more links to
# maintain, but every claim keeps its provenance.
#
# **Anatomy.** `Record(payload, origin, entity_resolved, has_provenance)` → `route(record)` returns `graph`
# (`domain` / `lexical` / `subject`) and raises `ValueError` if a rule is broken (e.g. raw text without provenance).
# `link_subject_to_domain(mention, {id: name}, threshold)` → a `Correspondence` with `domain_id`, `linked`, score.

# %%
for r in h.run("MATCH (s:Service) RETURN s.name AS name, s.tier AS tier"):
    out = tgr.route(tgr.Record(payload=r, origin="structured", entity_resolved=True))
    assert out["graph"] == "domain"
    h.run("MATCH (s:Service {name:$n}) SET s:Domain", n=r["name"])
for r in h.run("MATCH (l:LogLine) RETURN l.id AS id, l.text AS text"):
    out = tgr.route(tgr.Record(payload=r, origin="raw_text", has_provenance=True))
    assert out["graph"] == "lexical"
    h.run("MATCH (l:LogLine {id:$i}) SET l:Chunk:Lexical", i=r["id"])

mention = "the checkout service"
names = {r["id"]: r["id"] for r in h.run("MATCH (s:Service) RETURN s.name AS id")}
corr = tgr.link_subject_to_domain(mention, names, threshold=0.5)
print(corr)
h.run("""
MERGE (f:Finding:Subject {id:'subj-rootcause'}) SET f.confidence=0.82, f.mention=$m
WITH f
MATCH (l:Lexical {id:'log-0'}) MERGE (f)-[:EXTRACTED_FROM]->(l)
WITH f
MATCH (s:Service {name:$d}) MERGE (f)-[:CORRESPONDS_TO]->(s)
""", m=mention, d=corr.domain_id)

rejected = False
try:
    tgr.route(tgr.Record(payload={"text": "x"}, origin="raw_text", has_provenance=False))
except ValueError:
    rejected = True
h.check("raw text without provenance is rejected", rejected)
h.check("mention links to checkout-api", corr.domain_id == "checkout-api" and corr.linked)

# %% [markdown]
# **Cypher #2 — multi-hop provenance + aggregation.** From a subject-graph claim, hop to the trusted entity *and* to
# its source text. `OPTIONAL MATCH` is a left join (the row survives when evidence is missing, which exposes
# unsupported claims); `collect()` aggregates matches into a list.

# %%
prov = h.run("""
MATCH (f:Subject)-[:CORRESPONDS_TO]->(s:Domain)
OPTIONAL MATCH (f)-[:EXTRACTED_FROM]->(c:Lexical)
RETURN f.id AS claim, f.confidence AS conf, s.name AS entity, collect(c.text) AS evidence
""")
h.show(prov)
h.check("claim has evidence and a domain entity", prov[0]["entity"] == "checkout-api" and len(prov[0]["evidence"]) == 1)

# %% [markdown]
# ## 4. `entity-resolution-strategy-selector` — evidence, not guesses
#
# **Entity resolution** decides if two records are the same real thing. `resolve_match(a, b)` compares features
# (name, address…) and returns `confidence`, `edge_type` and an `evidence` list explaining each feature's score.
# Edge types encode confidence: `RESOLVED` (≥ 0.85), `POSSIBLY_RELATED` (≥ 0.5), else none — so a downstream query
# can choose to trust only `RESOLVED`. **Trade-off:** evidence-based is explainable and deterministic; learned
# matchers are more flexible but opaque.

# %%
prof = ers.StrategyProfile(high_stakes=3, adversarial=1, needs_explainability=3, needs_determinism=3,
                           cultural_variation=0, has_training_examples=0)
print("strategy:", ers.recommend_strategy(prof)["recommended"])

h.run("MERGE (al:Alias {name:'Checkout API (prod)'}) SET al.address='us-east-1', al.source='cmdb'")
h.run("MATCH (s:Service {name:'checkout-api'}) SET s.address='us-east-1'")
pairs = h.run("""
MATCH (s:Service), (al:Alias) WHERE s.tier = 'critical'
RETURN s.name AS a_name, s.address AS a_addr, al.name AS b_name, al.address AS b_addr
""")
for p in pairs:
    res = ers.resolve_match({"name": p["a_name"], "address": p["a_addr"]},
                            {"name": p["b_name"], "address": p["b_addr"]})
    print(p["a_name"], "<->", p["b_name"], "| confidence", res["confidence"], "|", res["edge_type"])
    print("  evidence:", json.dumps(res["evidence"])[:200])
    if res["edge_type"] in ("RESOLVED", "POSSIBLY_RELATED"):
        h.run(f"""
        MATCH (s:Service {{name:$a}}), (al:Alias {{name:$b}})
        MERGE (al)-[r:{res['edge_type']}]->(s) SET r.confidence=$c
        """, a=p["a_name"], b=p["b_name"], c=res["confidence"])
links = h.run("MATCH (:Alias)-[r]->(:Service {name:'checkout-api'}) RETURN type(r) AS t, r.confidence AS c")
h.show(links)
h.check("evidence-based strategy for high-stakes ops", ers.recommend_strategy(prof)["recommended"] == "evidence_based")
h.check("classify_edge thresholds", ers.classify_edge(0.9)["edge_type"] == "RESOLVED"
        and ers.classify_edge(0.6)["edge_type"] == "POSSIBLY_RELATED" and ers.classify_edge(0.2)["edge_type"] == "NO_MATCH")
h.check("alias linked into the domain graph", len(links) == 1)

# %% [markdown]
# ## 5. `schema-pattern-selector` — reusable shapes that catch bad data
#
# A **schema pattern** is a reusable shape of knowledge. `validate_instance(pattern, instance)` returns `valid` and
# `errors`; the *event-centric* pattern demands a time link, so a deployment without one is rejected.
# `recommend_pattern(text)` picks a pattern from a description. Instances are built from Neo4j. We also store two
# clocks on the deployment (**bi-temporal**: `valid_at` = true in the world, `recorded_at` = when we learned it).

# %%
dep = h.run("""
MATCH (d:Deployment {id:'deploy-481'})-[:DEPLOYED_TO]->(s), (d)-[:CHANGED]->(r)
RETURN d.at AS at, collect(DISTINCT s.name) + collect(DISTINCT r.name) AS parts
""")[0]
h.run("MATCH (d:Deployment {id:'deploy-481'}) SET d.valid_at=d.at, d.recorded_at='2026-10-24T08:45:00Z'")
good = {"relationships": {"hasParticipant": dep["parts"], "hasStartTime": [dep["at"]]}}
no_time = {"relationships": {"hasParticipant": dep["parts"]}}
for name, inst in [("good deploy", good), ("no-time deploy", no_time)]:
    v = sps.validate_instance("event_centric", inst)
    print(f"{name:15s} valid={v['valid']} errors={v['errors'][:1]}")
rp = sps.recommend_pattern("deployment event with a timestamp and git commit, affecting services, "
                           "preceded by a prior deploy in the deployment history")
print("recommended pattern:", rp["recommended"])
h.check("event without time rejected", not sps.validate_instance("event_centric", no_time)["valid"])
h.check("good event accepted", sps.validate_instance("event_centric", good)["valid"])
h.check("deployment text -> event_centric", rp["recommended"] == "event_centric")

# %% [markdown]
# ## 6. `homoiconic-meta-schema` — the rule lives in the graph
#
# **Homoiconic** = schema and rules are stored in the same representation as the data, so they can be validated and
# edited like data. `evaluate_rule(rule, facts)` takes a rule (`condition` / `action` text) plus facts
# (`dependent_count`) and returns the matching `criticality`.
#
# **Cypher #3 — aggregation.** `OPTIONAL MATCH` + `count(d)` gives 0 for leaf services; `WITH` passes the aggregate
# to the next part. We feed each count into the rule and **write `criticality` back**.

# %%
rule = {"name": "DetermineServiceCriticality", "description": "tier by dependents",
        "condition": "MATCH (s:Service)<-[:DEPENDS_ON]-(d) RETURN s, count(d) AS dependent_count",
        "action": "WHEN dependent_count > 5 THEN SET s.criticality = 'Critical' "
                  "WHEN dependent_count > 2 THEN SET s.criticality = 'Important' "
                  "ELSE SET s.criticality = 'Standard'"}
h.check("rule is valid", hms.validate_rule(rule)["valid"])
h.run("MERGE (r:Rule {name:$n}) SET r.action=$a, r.condition=$c", n=rule["name"], a=rule["action"], c=rule["condition"])

for r in h.run("""
MATCH (s:Service)
OPTIONAL MATCH (d:Service)-[:DEPENDS_ON]->(s)
WITH s, count(d) AS dependent_count
RETURN s.name AS name, dependent_count
"""):
    crit = hms.evaluate_rule(rule, {"dependent_count": r["dependent_count"]})["criticality"]
    h.run("MATCH (s:Service {name:$n}) SET s.criticality=$c, s.dependents=$k", n=r["name"], c=crit, k=r["dependent_count"])
crit_rows = h.run("MATCH (s:Service) RETURN s.name AS service, s.dependents AS dependents, "
                  "s.criticality AS criticality ORDER BY dependents DESC")
h.show(crit_rows)
by_name = {r["service"]: r["criticality"] for r in crit_rows}
h.check("checkout-api (4 dependents) is Important", by_name["checkout-api"] == "Important")
h.check("8 dependents would be Critical", hms.evaluate_rule(rule, {"dependent_count": 8})["criticality"] == "Critical")

# %% [markdown]
# ## 7. `capability-authorization-gate` — what may the agent do?
#
# A **capability model** keeps permissions in the knowledge: each capability has a required level, required grants and
# an optional amount limit. `authorize(agent, capability, amount)` returns `decision` (`allow` / `escalate` /
# `deny`), `reasons`, `required_level`, `granted_level`. Undeclared capability → `deny`. We log every decision as an
# **audit trail** in the graph. *Trade-off:* the gate is a planning-time check, not a replacement for IAM.

# %%
KR = h.SKILLS_DIR / CH
spec = json.loads((KR / "capability-authorization-gate" / "sample_devops_agent.json").read_text(encoding="utf-8"))
agent = cag.agent_from_spec(spec)
h.run("MERGE (ag:AgentProfile {id:$i}) SET ag.level=$l", i=spec["id"], l=spec["granted_level"])
for cap, amt in [("read_metrics", None), ("query_logs", None), ("restart_instance", None),
                 ("scale_autoscaling_group", 25), ("delete_database", None)]:
    d = cag.authorize(agent, cap, amt)
    h.run("""
    MATCH (ag:AgentProfile {id:$i}) MERGE (c:Capability {name:$cap})
    CREATE (ag)-[:ATTEMPTED {decision:$d, amount:$amt, reasons:$why}]->(c)
    """, i=spec["id"], cap=cap, d=d["decision"], amt=amt or 0, why="; ".join(d["reasons"])[:150])
audit = h.run("""
MATCH (:AgentProfile {id:$i})-[r:ATTEMPTED]->(c:Capability)
RETURN r.decision AS decision, count(*) AS n, collect(c.name) AS capabilities ORDER BY n DESC
""", i=spec["id"])
h.show(audit)
dec = {r["decision"]: set(r["capabilities"]) for r in audit}
h.check("reads allowed", {"read_metrics", "query_logs"} <= dec["allow"])
h.check("restart + big scale escalate", {"restart_instance", "scale_autoscaling_group"} <= dec["escalate"])
h.check("undeclared capability denied", "delete_database" in dec["deny"])

# %% [markdown]
# ## 8. Pipeline: chain skills on *checkout-api is slow*
#
# No framework — just a function. Each step takes what the previous step wrote to the graph. (Tier 3 replaces this
# with an agent that logs its own trace.)

# %%
def investigate(service: str) -> dict:
    blast_n = h.run("MATCH (d:Service)-[:DEPENDS_ON*1..3]->(t {name:$s}) RETURN count(DISTINCT d) AS n", s=service)[0]["n"]
    crit = h.run("MATCH (s:Service {name:$s}) RETURN s.criticality AS c", s=service)[0]["c"]
    cause = h.run("""
        MATCH (i:Incident)-[:AFFECTS]->(:Service {name:$s}), (i)-[:ROOT_CAUSE]->(x)
        RETURN i.id AS incident, x.name AS component""", s=service)
    fix = cag.authorize(agent, "restart_instance")
    return {"service": service, "blast_radius": blast_n, "criticality": crit,
            "past_cause": cause[0] if cause else None, "restart_decision": fix["decision"]}

report = investigate("checkout-api")
print(f"DECISION REPORT for {report['service']}")
print(f"  criticality     : {report['criticality']} (blast radius {report['blast_radius']})")
print(f"  related incident: {report['past_cause']}")
print(f"  restart allowed?: {report['restart_decision']} -> "
      f"{'ask a supervisor' if report['restart_decision'] == 'escalate' else 'proceed'}")
h.check("report: Important + restart escalates", report["criticality"] == "Important"
        and report["restart_decision"] == "escalate")

# %% [markdown]
# ## 9. ✍️ Your turn
#
# Try each yourself first; the reference solution runs below and is checked against the expected value.

# %%
def ex1():
    return h.run("""
    MATCH (s:Service)-[:DEPENDS_ON]->(:Service {name:'checkout-api'})
    WHERE s.owner = 'team-edge' RETURN count(s) AS n""")[0]["n"]

h.your_turn("Cypher: how many services owned by 'team-edge' depend directly on checkout-api?",
            "MATCH the DEPENDS_ON pattern, filter with WHERE on s.owner, then count(s).", ex1, 3)

# %%
def ex2():
    boosted = dict(spec, granted_level="Supervisor", granted_resources=spec["granted_resources"] + ["ec2:write"])
    return cag.authorize(cag.agent_from_spec(boosted), "restart_instance")["decision"]

h.your_turn("Skill inputs: change the agent spec so that restart_instance is allowed",
            "Look at `reasons` for restart_instance: it needs level Supervisor and the grant ec2:write.", ex2, "allow")

# %% [markdown]
# ## 10. Trusting a skill: CLI and benchmark
#
# Every skill is a stdlib-only `lib.py` plus a `cli.py`. `--help` prints its description; `benchmark` runs the
# skill's own self-checks (exit code 0 = pass). Tier 3 benchmarks all skills.

# %%
cli = KR / "capability-authorization-gate" / "cli.py"
env = dict(os.environ, PYTHONIOENCODING="utf-8")
p = subprocess.run([sys.executable, str(cli), "scenario", "devops-latency"], capture_output=True, text=True,
                   encoding="utf-8", env=env)
print((p.stdout or "")[:500])
h.check("CLI scenario exits 0", p.returncode == 0 and len(p.stdout) > 20)
ok, first = h.run_benchmark(CH, "graph-model-selector")
print("benchmark:", "PASS" if ok else "FAIL", "|", first[:80])
h.check("graph-model-selector benchmark passes", ok)

# %% [markdown]
# ## 11. 🎯 Vibe-coding tip
#
# Paste this into your AI assistant (Claude Code, Cursor, Copilot…):
#
# > *"Read `skills/knowledge-representation/schema-pattern-selector/SKILL.md` and `lib.py`. Write a script that
# > queries Neo4j (bolt://localhost:7687) for all `Deployment` nodes, builds an event-centric instance for each (their
# > participants and `at` timestamp), calls `validate_instance`, and prints the ids of invalid deployments. Do not
# > modify `lib.py`."*
#
# **Acceptance check:** run `python skills/knowledge-representation/schema-pattern-selector/cli.py benchmark` (must
# still pass, proving `lib.py` is untouched) and confirm the script flags a deployment you deliberately create
# without `at`.

# %% [markdown]
# ## 12. Why this matters & common mistakes
#
# * **Model by reasoning, not by data** — the wrong graph model is expensive to undo.
# * **Separate layers** buy provenance at the cost of extra links; **evidence-based matching** buys explainability at
#   the cost of flexibility; **rules-as-data** buys editability at the cost of an interpreter.
#
# Common mistakes:
# 1. Hard-coding skill inputs instead of deriving them from the graph (results go stale).
# 2. Using `CREATE` where `MERGE` was needed — re-running duplicates nodes.
# 3. Forgetting `OPTIONAL MATCH`, so nodes with zero matches silently vanish from aggregates.
# 4. Treating `POSSIBLY_RELATED` links as `RESOLVED`.
# 5. Mistaking the capability gate for real IAM enforcement.

# %%
h.check("graph has 4 criticality-labelled hubs or more", len(crit_rows) >= 4)
h.check("subject claim traceable to evidence", len(prov) == 1)
h.check("audit trail has 5 attempts", sum(r["n"] for r in audit) == 5)
h.check("ontology of skills: 8 in chapter", len(h.skill_catalog(CH)) == 8)

# %% [markdown]
# ## Recap
#
# You used six skills, derived their inputs with variable-length, optional-match and aggregation Cypher, wrote results
# back (reified fact, alias links, criticality, audit trail), chained them in a pipeline, and verified a skill via its
# CLI and benchmark.
#
# **Next:** `../notebooks_advanced_simplified/ch3-knowledge-representation.ipynb` — all 8 skills, an agent with a
# stored trace, every benchmark, vibe-coding corners and experiments.
