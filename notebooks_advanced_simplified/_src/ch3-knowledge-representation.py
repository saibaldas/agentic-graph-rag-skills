# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > [Beginner (tier 1/3)](../notebooks_simplified/ch3-knowledge-representation.ipynb)  →  [Intermediate (tier 2/3)](../notebooks_intermediate_simplified/ch3-knowledge-representation.ipynb)  →  **Advanced (tier 3/3) ← you are here**

# %% [markdown]
# # Chapter 3 (Advanced) — Knowledge Representation, end to end on a real graph
#
# **Tier 3 of 3 (advanced).** The beginner notebook showed each skill once. Here we:
#
# * use **all 8** skills of `skills/knowledge-representation/`,
# * derive every skill input **from Cypher** (no hard-coded dicts) and **write results back** to Neo4j,
# * run a small **agent** (plan → act → observe) whose trace is itself stored in the graph,
# * benchmark every skill, call one through its CLI, and finish with vibe-coding prompts and experiments.
#
# Scenario (fictional AWS account `123456789012`): **"checkout-api is slow"**.
#
# Book mapping: *Agentic GraphRAG*, Ch. 3 — choosing a graph model, three-graph architecture, entity resolution,
# extraction, schema patterns, homoiconic schemas, the knowledge-organization spectrum, and capability models.

# %%
import adv_helpers as a
import json
import subprocess
import sys

print(a.load_devops_graph())
CH = "knowledge-representation"
SLUGS = ["graph-model-selector", "three-graph-router", "entity-resolution-strategy-selector",
         "kg-extraction-approach-selector", "schema-pattern-selector", "homoiconic-meta-schema",
         "knowledge-organization-classifier", "capability-authorization-gate"]
gms, tgr, ers, kea, sps, hms, koc, cag = [a.load_skill(CH, s) for s in SLUGS]
KR = a.SKILLS_DIR / CH
a.show([{"skill": r["slug"], "what it does": r["description"][:90]} for r in a.skill_catalog(CH)])

# %% [markdown]
# ## 1. Seed the chapter data as a graph
#
# The base graph has only one dependent of `checkout-api`. We add three more callers and two log lines so the later
# queries have something to chew on.
#
# Cypher recap — `MATCH` finds a pattern, `MERGE` creates it only if missing (idempotent), `SET` writes properties,
# `UNWIND` turns a list into rows so one statement can create many nodes.

# %%
a.run("""
MATCH (c:Service {name:'checkout-api'})
UNWIND ['web-frontend','mobile-bff','partner-api'] AS n
MERGE (s:Service {name:n}) SET s.tier='medium', s.owner='team-edge'
MERGE (s)-[:DEPENDS_ON]->(c)
""")
a.run("""
UNWIND [{id:'log-0', text:'checkout-api p99 latency spiked at 8:41am'},
        {id:'log-1', text:'orders-db connection pool exhausted: 20/20 in use'}] AS l
MERGE (:LogLine {id:l.id, text:l.text})
""")
rows = a.run("MATCH (s:Service)-[:DEPENDS_ON]->(:Service {name:'checkout-api'}) RETURN count(s) AS n")
a.check("checkout-api has 4 direct dependents", rows[0]["n"] == 4)

# %% [markdown]
# ## 2. `graph-model-selector` — property graph vs RDF vs hypergraph
#
# Models differ in how they store a **multi-party fact**. A *hypergraph* edge can join N nodes in one element; a
# *property graph* must **reify** it (turn the fact into an intermediate node plus N edges); RDF needs even more
# triples.
#
# **Blast radius** = everything that would be hurt if one component fails. We measure it with a
# **variable-length path**: `-[:DEPENDS_ON*1..3]->` means "follow DEPENDS_ON between 1 and 3 hops".
# `count(DISTINCT x)` counts each node once even if several paths reach it.

# %%
blast = a.run("""
MATCH (d:Service)-[:DEPENDS_ON*1..3]->(t {name:$t})
RETURN t.name AS target, count(DISTINCT d) AS blast_radius
""", t="orders-db")
a.show(blast)

# The deployment is a 3-party fact: version + service + resource changed.
ctx = a.run("""
MATCH (dep:Deployment {id:'deploy-481'})-[:DEPLOYED_TO]->(svc), (dep)-[:CHANGED]->(res)
RETURN dep.id AS dep, svc.name AS svc, res.name AS res
""")[0]
edge = gms.HyperEdge(type="DEPLOYMENT", nodes={"deployment": ctx["dep"], "service": ctx["svc"],
                                              "resource": ctx["res"]}, attributes={"version": "v2.14.0"})
cost = gms.representation_cost(edge)
print(cost)

reqs = gms.Requirements(formal_reasoning=1, n_ary_relations=2, performance=3, tool_ecosystem=3, constraint_expr=1)
rec = gms.recommend_model(reqs)
print(rec["recommended"], "| hybrid:", rec["hybrid_recommended"], "|", rec["hybrid"])
a.check("property graph wins for ops workloads", rec["recommended"] == "property_graph")
a.check("hypergraph needs 1 element, property graph 1+arity", cost["hypergraph_elements"] == 1
        and cost["property_graph_elements"] == 1 + cost["arity"])

# %% [markdown]
# **Write-back.** We apply the reification the skill describes: a `:Reified:Fact` node that links its participants.
# Trade-off: one extra node per n-ary fact buys us tool support (Cypher, drivers) at a small storage cost.

# %%
reif = gms.reify_as_property_graph(edge)
a.run("""
MERGE (f:Reified:Fact {id:$id}) SET f.type=$type, f.version=$ver
WITH f
UNWIND $parts AS p
MATCH (n) WHERE n.name = p OR n.id = p
MERGE (f)-[:PARTICIPANT]->(n)
""", id="fact-deploy-481", type=edge.type, ver="v2.14.0", parts=list(edge.nodes.values()))
n_part = a.run("MATCH (:Fact {id:'fact-deploy-481'})-[:PARTICIPANT]->(n) RETURN count(n) AS n")[0]["n"]
print("participants linked:", n_part, "| skill said auxiliary edges =", reif["auxiliary_edges"])
a.check("reified fact has one edge per participant", n_part == reif["auxiliary_edges"])

# %% [markdown]
# ## 3. `three-graph-router` — Domain, Lexical, Subject graphs
#
# The chapter's architecture separates (1) **Domain** graph: trusted structured entities, (2) **Lexical** graph: raw
# text chunks with provenance, (3) **Subject** graph: LLM-extracted claims with confidence. They are linked by
# `EXTRACTED_FROM` (claim → chunk) and `CORRESPONDS_TO` (claim → trusted entity). Mixing them would let a
# hallucination masquerade as a fact, so the router refuses invalid records.
#
# We route real rows: each `Service` is a structured record, each `LogLine` is raw text.

# %%
svc_rows = a.run("MATCH (s:Service) RETURN s.name AS name, s.tier AS tier")
for r in svc_rows:
    out = tgr.route(tgr.Record(payload=r, origin="structured", entity_resolved=True))
    a.run("MATCH (s:Service {name:$n}) SET s:Domain", n=r["name"])
    assert out["graph"] == "domain"
for r in a.run("MATCH (l:LogLine) RETURN l.id AS id, l.text AS text"):
    out = tgr.route(tgr.Record(payload=r, origin="raw_text", has_provenance=True))
    a.run("MATCH (l:LogLine {id:$i}) SET l:Chunk:Lexical", i=r["id"])
    assert out["graph"] == "lexical"

mention = "the checkout service"
domain_names = {r["id"]: r["id"] for r in a.run("MATCH (s:Service) RETURN s.name AS id")}
corr = tgr.link_subject_to_domain(mention, domain_names, threshold=0.5)
print(corr)
a.run("""
MERGE (f:Finding:Subject {id:'subj-rootcause'}) SET f.confidence=0.82, f.mention=$m
WITH f
MATCH (l:Lexical {id:'log-0'}) MERGE (f)-[:EXTRACTED_FROM]->(l)
WITH f
MATCH (s:Service {name:$d}) MERGE (f)-[:CORRESPONDS_TO]->(s)
""", m=mention, d=corr.domain_id)

bad = False
try:
    tgr.route(tgr.Record(payload={"text": "x"}, origin="raw_text", has_provenance=False))
except ValueError:
    bad = True
a.check("raw text without provenance is rejected", bad)
a.check("mention links to checkout-api", corr.domain_id == "checkout-api" and corr.linked)

# %% [markdown]
# **Query 1 — multi-hop provenance.** Starting from a subject-graph claim, hop to its source text *and* to the
# trusted entity, using the path the skill recommends (`cross_graph_query_path`). `OPTIONAL MATCH` is a left join:
# rows survive even when the optional part is missing, which is how we spot claims with no evidence.
# `collect()` aggregates the matches into a list.

# %%
print("skill's cross-graph path:", tgr.cross_graph_query_path("domain", "lexical"))
prov = a.run("""
MATCH (f:Subject)-[:CORRESPONDS_TO]->(s:Domain)
OPTIONAL MATCH (f)-[:EXTRACTED_FROM]->(c:Lexical)
RETURN f.id AS claim, f.confidence AS conf, s.name AS entity, collect(c.text) AS evidence
""")
a.show(prov)
a.check("claim has evidence text and a domain entity", prov and prov[0]["entity"] == "checkout-api"
        and len(prov[0]["evidence"]) == 1)

# %% [markdown]
# ## 4. `entity-resolution-strategy-selector` — evidence, not guesses
#
# **Entity resolution** decides whether two records are the same real-world thing. Two families: *evidence-based*
# (weighted features, explainable, deterministic) and *generalization-AI* (learned, flexible, opaque). For
# high-stakes or adversarial data, the chapter prefers evidence-based with **separate edge types** per confidence:
# `RESOLVED` (≥ 0.85), `POSSIBLY_RELATED` (≥ 0.5), otherwise no link. *Blocking* (cheaply grouping candidates
# before comparing pairs) keeps this from being O(n²); here we block by tier.

# %%
prof = ers.StrategyProfile(high_stakes=3, adversarial=1, needs_explainability=3, needs_determinism=3,
                           cultural_variation=0, has_training_examples=0)
srec = ers.recommend_strategy(prof)
print(srec["recommended"], srec["scores"])
a.check("high-stakes, explainable ops data -> evidence_based", srec["recommended"] == "evidence_based")

# Alias records: the same service as named by different tools. Seed an alias, then pull candidate pairs by Cypher.
a.run("""
MERGE (al:Alias {name:'Checkout API (prod)'}) SET al.address='us-east-1', al.source='cmdb'
""")
a.run("MATCH (s:Service {name:'checkout-api'}) SET s.address='us-east-1'")
pairs = a.run("""
MATCH (s:Service), (al:Alias)
WHERE s.tier = 'critical'
RETURN s.name AS a_name, s.address AS a_addr, al.name AS b_name, al.address AS b_addr
""")
a.show(pairs)

# %%
resolved_log = []
for p in pairs:
    res = ers.resolve_match({"name": p["a_name"], "address": p["a_addr"]},
                            {"name": p["b_name"], "address": p["b_addr"]})
    resolved_log.append((p["a_name"], p["b_name"], res["confidence"], res["edge_type"]))
    if res["edge_type"] in ("RESOLVED", "POSSIBLY_RELATED"):
        a.run(f"""
        MATCH (s:Service {{name:$a}}), (al:Alias {{name:$b}})
        MERGE (al)-[r:{res['edge_type']}]->(s) SET r.confidence=$c
        """, a=p["a_name"], b=p["b_name"], c=res["confidence"])
for row in resolved_log:
    print(row)
print(json.dumps(ers.resolve_match({"name": "checkout-api"}, {"name": "Checkout API (prod)"})["evidence"], indent=1))

# Edge cases from the chapter: names differ but same entity; names match but different entity.
w = ers.flag_edge_cases({"name": "John R Smith", "address": "1 Main St"},
                        {"name": "John E Smith", "address": "1 Main St"})
print("flags:", w)
a.check("classify_edge thresholds", ers.classify_edge(0.9)["edge_type"] == "RESOLVED"
        and ers.classify_edge(0.6)["edge_type"] == "POSSIBLY_RELATED"
        and ers.classify_edge(0.2)["edge_type"] == "NO_MATCH")
links = a.run("MATCH (:Alias)-[r]->(:Service {name:'checkout-api'}) RETURN type(r) AS t, r.confidence AS c")
a.show(links)
a.check("alias linked into the domain graph", len(links) == 1)

# %% [markdown]
# ## 5. `kg-extraction-approach-selector` — how to build the graph from sources
#
# Structured sources (tables, CMDB) → direct mapping. Unstructured text → an LLM-style extraction; if documents
# arrive continuously, an **incremental** approach (iText2KG) avoids reprocessing everything; if each document
# needs surrounding context, a retrieval-augmented one (RAKG). We derive the profile from the graph itself.

# %%
n_logs = a.run("MATCH (l:Lexical) RETURN count(l) AS n")[0]["n"]
structured = a.run("MATCH (s:Domain) RETURN count(s) AS n")[0]["n"]
p_text = kea.Profile(source_type="unstructured", incremental=True, document_context_needed=False)
p_struct = kea.Profile(source_type="structured")
r1, r2 = kea.recommend_approach(p_text), kea.recommend_approach(p_struct)
print(f"{n_logs} log chunks ->", r1["recommended"], "| {structured} domain rows ->".format(structured=structured),
      r2["recommended"])
inc = kea.incremental_cost(new_docs=n_logs, total_docs=1000, approach=r1["recommended"])
full = kea.incremental_cost(new_docs=n_logs, total_docs=1000, approach="llm_extraction")
print("incremental:", inc["docs_processed"], "docs vs rebuild:", full["docs_processed"])
a.run("MATCH (f:Subject {id:'subj-rootcause'}) SET f.extraction=$a", a=r1["recommended"])
a.check("structured -> structured_db", r2["recommended"] == "structured_db")
a.check("incremental processes fewer docs than rebuild", inc["docs_processed"] < full["docs_processed"])

# %% [markdown]
# ## 6. `schema-pattern-selector` — event-centric, multi-perspective, contextual boundary, capability
#
# A **schema pattern** is a reusable shape of knowledge. The event-centric one demands a time link; this
# validator catches deployments recorded without one. The instance is built from Neo4j. We also add
# *bi-temporal* data: **bi-temporal** means two clocks — when something was true in the world (`valid_at`) and when
# the system learned it (`recorded_at`) — so late-arriving facts don't overwrite history.

# %%
dep = a.run("""
MATCH (d:Deployment {id:'deploy-481'})-[:DEPLOYED_TO]->(s), (d)-[:CHANGED]->(r)
RETURN d.at AS at, collect(DISTINCT s.name) + collect(DISTINCT r.name) AS parts
""")[0]
a.run("MATCH (d:Deployment {id:'deploy-481'}) SET d.valid_at=d.at, d.recorded_at='2026-10-24T08:45:00Z'")
good = {"relationships": {"hasParticipant": dep["parts"], "hasStartTime": [dep["at"]]}}
bad = {"relationships": {"hasParticipant": dep["parts"]}}
drift = {"relationships": {"according-to": ["terraform_state", "aws_api"]},
         "perspectives": [{"source": "terraform_state", "value": 2, "confidence": 1.0},
                          {"source": "aws_api", "value": 3, "confidence": 1.0}]}
for name, patt, inst in [("good deploy", "event_centric", good), ("no-time deploy", "event_centric", bad),
                         ("config drift", "multi_perspective", drift)]:
    v = sps.validate_instance(patt, inst)
    print(f"{name:15s} valid={v['valid']} {v['errors'][:1]}")
a.check("event without time rejected", not sps.validate_instance("event_centric", bad)["valid"])
a.check("good event and drift accepted", sps.validate_instance("event_centric", good)["valid"]
        and sps.validate_instance("multi_perspective", drift)["valid"])
rp = sps.recommend_pattern("deployment event with a timestamp and git commit, affecting services, "
                           "preceded by a prior deploy in the deployment history")
a.check("deployment text -> event_centric", rp["recommended"] == "event_centric")

# %% [markdown]
# ## 7. `homoiconic-meta-schema` — the schema lives in the graph
#
# **Homoiconic** = the schema is stored in the same representation as the data, so one validator checks both and
# rules can be edited as data. We store an `EntityType` and a `Rule` node, then *run* the rule: count each
# service's dependents with `OPTIONAL MATCH` + `count`, feed it to `evaluate_rule`, and write `criticality` back.
# `WITH` passes aggregated values to the next query part; counting a nullable optional match gives 0 for leaves.

# %%
service_type = {"name": "Service", "description": "A deployable service",
                "properties": [{"name": "name", "type": "string", "required": True},
                               {"name": "tier", "type": "string"}]}
rule = {"name": "DetermineServiceCriticality", "description": "tier by dependents",
        "condition": "MATCH (s:Service)<-[:DEPENDS_ON]-(d) RETURN s, count(d) AS dependent_count",
        "action": "WHEN dependent_count > 5 THEN SET s.criticality = 'Critical' "
                  "WHEN dependent_count > 2 THEN SET s.criticality = 'Important' "
                  "ELSE SET s.criticality = 'Standard'"}
a.check("entity type valid", hms.validate_entity_type(service_type)["valid"])
a.check("rule valid", hms.validate_rule(rule)["valid"])
a.run("MERGE (t:EntityType {name:'Service'}) SET t.definition=$d", d=json.dumps(service_type))
a.run("MERGE (r:Rule {name:$n}) SET r.action=$a, r.condition=$c", n=rule["name"], a=rule["action"],
      c=rule["condition"])

# Data rows are validated by the same machinery.
inst_rows = a.run("MATCH (s:Service) RETURN s{.name, .tier} AS i")
bad_inst = hms.validate_data_against_type(service_type, {"tier": 5})
a.check("all Service nodes satisfy the stored type",
        all(hms.validate_data_against_type(service_type, r["i"])["valid"] for r in inst_rows))
a.check("bad instance caught", not bad_inst["valid"])

# %%
for r in a.run("""
MATCH (s:Service)
OPTIONAL MATCH (d:Service)-[:DEPENDS_ON]->(s)
WITH s, count(d) AS dependent_count
RETURN s.name AS name, dependent_count
"""):
    crit = hms.evaluate_rule(rule, {"dependent_count": r["dependent_count"]})["criticality"]
    a.run("MATCH (s:Service {name:$n}) SET s.criticality=$c, s.dependents=$k",
          n=r["name"], c=crit, k=r["dependent_count"])
crit_rows = a.run("MATCH (s:Service) RETURN s.name AS service, s.dependents AS dependents, "
                  "s.criticality AS criticality ORDER BY dependents DESC")
a.show(crit_rows)
by_name = {r["service"]: r["criticality"] for r in crit_rows}
a.check("checkout-api (4 dependents) is Important", by_name["checkout-api"] == "Important")
a.check("8 dependents would be Critical", hms.evaluate_rule(rule, {"dependent_count": 8})["criticality"] == "Critical")

# %% [markdown]
# ## 8. `knowledge-organization-classifier` — the vocabulary spectrum
#
# Spectrum from simplest to richest: **pick list → taxonomy → thesaurus → ontology**. Richer means more expressive
# but costlier to maintain. `migration_cost` shows what happens when a *change event* hits a tier that cannot
# express it. An **ontology** has five parts: classes, subclasses, individuals, axioms, relationships. We assemble
# one from Neo4j: individuals are the real nodes, classes are their labels.

# %%
tiers = [r["t"] for r in a.run("MATCH (s:Service) RETURN DISTINCT s.tier AS t")]
print("pick list from graph:", koc.classify({"values": tiers})["classification"])
print("with hierarchy:", koc.classify({"values": tiers, "has_hierarchy": True})["classification"])
print("next step:", koc.recommend_upgrade({"values": tiers, "has_hierarchy": True})["next"])
print(koc.migration_cost("taxonomy", "new_synonym"))
print(koc.migration_cost("pick_list", "new_relationship_type"))
a.check("taxonomy cannot express synonyms",
        koc.migration_cost("taxonomy", "new_synonym")["cost"] == "NOT_EXPRESSIBLE")

# %%
labels = a.run("MATCH (n) WHERE n:Service OR n:Database OR n:Cache RETURN DISTINCT labels(n)[0] AS l")
individuals = [{"name": r["name"]} for r in a.run("MATCH (n) WHERE n:Service OR n:Database OR n:Cache "
                                                    "RETURN n.name AS name")]
ont = {"classes": [{"name": r["l"]} for r in labels] + [{"name": "Resource"}],
       "subclasses": [{"name": "Service", "parent": "Resource"}, {"name": "Database", "parent": "Resource"}],
       "individuals": individuals,
       "axioms": [{"name": "critical-depends-on-critical-or-data"}],
       "relationships": [{"name": "DEPENDS_ON", "from": "Service", "to": "Resource"}]}
v = koc.validate_ontology_components(ont)
print(len(individuals), "individuals ->", "valid" if v["valid"] else v)
a.run("MERGE (o:Ontology {name:'ops-ontology'}) SET o.classes=$c", c=[c["name"] for c in ont["classes"]])
ghost = dict(ont, relationships=[{"name": "RUNS_ON", "from": "Service", "to": "Ghost"}])
a.check("ontology with 5 components valid", v["valid"])
a.check("dangling relationship endpoint rejected", not koc.validate_ontology_components(ghost)["valid"])

# %% [markdown]
# ## 9. `capability-authorization-gate` — what may the agent do?
#
# A **capability model** puts permissions in the knowledge itself: each capability has a required level, resource
# scope and optional amount limit. The gate answers `allow`, `escalate` (ask a supervisor) or `deny` (undeclared).
# We store capabilities as graph nodes and write each decision to an **audit trail**.

# %%
spec = json.loads((KR / "capability-authorization-gate" / "sample_devops_agent.json").read_text(encoding="utf-8"))
agent = cag.agent_from_spec(spec)
a.run("MERGE (ag:AgentProfile {id:$i}) SET ag.level=$l", i=spec["id"], l=spec.get("granted_level", "User"))
attempts = [("read_metrics", None), ("query_logs", None), ("restart_instance", None),
            ("scale_autoscaling_group", 25), ("delete_database", None)]
for cap, amt in attempts:
    d = cag.authorize(agent, cap, amt)
    a.run("""
    MATCH (ag:AgentProfile {id:$i})
    MERGE (c:Capability {name:$cap})
    CREATE (ag)-[:ATTEMPTED {decision:$d, amount:$amt, reasons:$why}]->(c)
    """, i=spec["id"], cap=cap, d=d["decision"], amt=amt or 0, why="; ".join(d["reasons"])[:150])
    print(f"{cap:26s} -> {d['decision']}")

# %% [markdown]
# **Query 2 — aggregation over the audit trail.** `count(*)` per group and `collect()` of capability names per
# decision answer "how often did the gate say no, and to what?".

# %%
audit = a.run("""
MATCH (:AgentProfile {id:$i})-[r:ATTEMPTED]->(c:Capability)
RETURN r.decision AS decision, count(*) AS n, collect(c.name) AS capabilities
ORDER BY n DESC
""", i=spec["id"])
a.show(audit)
dec = {r["decision"]: set(r["capabilities"]) for r in audit}
a.check("reads allowed", {"read_metrics", "query_logs"} <= dec["allow"])
a.check("restart + big scale escalate", {"restart_instance", "scale_autoscaling_group"} <= dec["escalate"])
a.check("undeclared capability denied", "delete_database" in dec["deny"])

# %% [markdown]
# ## 10. The agent: plan → act → observe on *checkout-api is slow*
#
# A **DAG** (directed acyclic graph) of steps is the plan: each step depends only on earlier ones. `MiniAgent`
# registers our skills and Cypher as **tools**; every call is logged by `AgentTrace` as
# `(:Run)-[:STEP]->(:ToolCall)-[:NEXT]->(:ToolCall)`. The planner here is a fixed list (no LLM); in production an
# LLM would choose the next tool from observations.

# %%
agent_run = a.MiniAgent("Investigate INC-1001: checkout-api is slow")

@agent_run.tool("graph_blast_radius")
def t_blast(service: str):
    r = a.run("MATCH (d:Service)-[:DEPENDS_ON*1..3]->(t {name:$s}) RETURN count(DISTINCT d) AS n", s=service)
    return r[0]["n"]

@agent_run.tool("cypher_recent_cause")
def t_cause(service: str):
    r = a.run("""
    MATCH (i:Incident)-[:AFFECTS]->(:Service {name:$s}), (i)-[:ROOT_CAUSE]->(x)
    RETURN i.id AS incident, i.root_cause AS cause, x.name AS component
    """, s=service)
    return r[0] if r else None

@agent_run.tool("check_capability")
def t_cap(capability: str, amount=None):
    return cag.authorize(agent, capability, amount)["decision"]

@agent_run.tool("recommend_pattern")
def t_pat(text: str):
    return sps.recommend_pattern(text)["recommended"]

plan = [("graph_blast_radius", {"service": "orders-db"}),
        ("cypher_recent_cause", {"service": "checkout-api"}),
        ("recommend_pattern", {"text": "deployment event with a timestamp and git commit, affecting services, "
                                        "preceded by a prior deploy in the deployment history"}),
        ("check_capability", {"capability": "describe_infrastructure"}),
        ("check_capability", {"capability": "restart_instance"})]
for tool_name, args in plan:
    obs = agent_run.act(tool_name, **args)
    print(f"ACT {tool_name:22s} OBSERVE {obs}")
conclusion = ("Likely cause: pool shrink in deploy-481; matches INC-0877; "
              "agent may diagnose but must escalate any restart.")
print(conclusion)
a.check("agent saw prior incident on orders-db", agent_run.memory["cypher_recent_cause"]["component"] == "orders-db")
a.check("restart escalates", agent_run.memory["check_capability"] == "escalate")

# %% [markdown]
# **Query 3 — read the trace back.** A variable-length path `(first)-[:NEXT*1..]->(last)` finds every later call
# reachable from step 1 (a path variable `p = ...` names the whole path so `length(p)` counts hops): useful for "what had already been tried before the restart request?".

# %%
a.show(agent_run.trace.timeline())
before = a.run("""
MATCH (:Run {id:$rid})-[:STEP]->(first:ToolCall {seq:1}),
      p = (first)-[:NEXT*1..]->(last:ToolCall {tool:'check_capability'})
RETURN last.seq AS seq, last.args AS args, length(p) AS hops
ORDER BY seq
""", rid=agent_run.trace.run_id)
a.show(before)
a.check("trace persisted all steps", len(agent_run.trace.timeline()) == len(plan))
a.check("variable-length path reaches last step", before[-1]["hops"] == len(plan) - 1)

# %% [markdown]
# ## 11. Skill CLI & benchmark
#
# Every skill is also a command-line tool (`cli.py`), so cron jobs, CI, or other agents can use it without Python
# imports. `--help` prints the SKILL.md description; `benchmark` runs the skill's built-in self-checks.

# %%
results = {}
for slug in SLUGS:
    ok, first = a.run_benchmark(CH, slug)
    results[slug] = ok
    print(f"{'PASS' if ok else 'FAIL'}  {slug:38s} {first[:60]}")
a.check("all 8 skill benchmarks pass", all(results.values()) and len(results) == 8)

cli = KR / "capability-authorization-gate" / "cli.py"
import os
env = dict(os.environ, PYTHONIOENCODING="utf-8")
helped = subprocess.run([sys.executable, str(cli), "--help"], capture_output=True, text=True, encoding="utf-8",
                        env=env)
print((helped.stdout or "")[:400])
a.check("--help exits 0 and prints a description", helped.returncode == 0 and len(helped.stdout or "") > 50)

# %% [markdown]
# ## 12. 🎯 Vibe-coding corner
#
# **A good prompt has four parts:** *context* (repo path, which skill/graph), *task* (one verb, one output),
# *constraints* (stdlib only, no hard-coded data, no graph wipes), and an *acceptance check* (a command that must
# pass). Copy-paste these into Copilot / Claude Code:
#
# **Prompt A — new Cypher query.**
# > In `notebooks_advanced_simplified/ch3-knowledge-representation.ipynb` the graph has `(:Service)-[:DEPENDS_ON]->`
# > edges. Add a read-only Cypher query returning, for every Service, the length of its longest dependency chain
# > (variable-length path) and the names on that chain via `collect`. Don't modify data. Acceptance: checkout-api
# > returns depth 2 and the query runs with `a.run`. Show the query and explain each clause.
#
# **Prompt B — new skill.**
# > Write a new skill `skills/knowledge-representation/schema-drift-detector/` following the anatomy of
# > `skills/knowledge-representation/graph-model-selector/`: SKILL.md with its 7 sections + Source Attribution,
# > stdlib-only `lib.py`, and `cli.py` whose `--help` prints the SKILL.md description and has a `benchmark`
# > subcommand. It compares two entity-type dicts and reports added/removed/retyped properties. Acceptance:
# > `python cli.py --help` exits 0 and `python cli.py benchmark` prints PASS.
#
# **Verification you should always run:** the acceptance command yourself, then read the diff. AI output that
# *looks* right is not a test.

# %% [markdown]
# **Worked example (as if vibe-coded).** Prompt: *"Write `tier_histogram(rows)` that takes a list of dicts with a
# `tier` key and returns `{tier: count}`; ignore rows without a tier; no imports."* Below is the result and the test
# that checks it, fed from live Neo4j data.

# %%
def tier_histogram(rows):
    out = {}
    for r in rows:
        t = r.get("tier")
        if t is not None:
            out[t] = out.get(t, 0) + 1
    return out

hist = tier_histogram(a.run("MATCH (s:Service) RETURN s.tier AS tier"))
print(hist)
a.check("histogram handles missing tier", tier_histogram([{"tier": "x"}, {}]) == {"x": 1})
a.check("histogram matches Cypher aggregation",
        hist == {r["tier"]: r["n"] for r in a.run("MATCH (s:Service) RETURN s.tier AS tier, count(*) AS n")})

# %% [markdown]
# ## 13. Experiments — change a parameter, predict first
#
# 1. In section 3 change `threshold=0.5` to `0.95`. **Predict:** does `the checkout service` still link? (Think:
#    similarity is token-based and below 0.95.) Check `corr.linked`.
# 2. In section 7 add a 5th dependent of `checkout-api` and re-run the criticality loop. **Predict:** which tier
#    appears at 3, 6 dependents? (Thresholds are `> 2` and `> 5`, so boundaries are exclusive.)
# 3. In section 4 give the alias a different `address`. **Predict:** does `RESOLVED` drop to `POSSIBLY_RELATED`?
#    Compute by hand with weights name 0.4 / address 0.3 renormalised.

# %%
# Experiment 1, run for you: raising the threshold flips linking off.
strict = tgr.link_subject_to_domain("the checkout service", domain_names, threshold=0.95)
loose = tgr.link_subject_to_domain("the checkout service", domain_names, threshold=0.5)
print("strict linked:", strict.linked, "| loose linked:", loose.linked)
a.check("threshold controls linking", (not strict.linked) and loose.linked)

# %% [markdown]
# ## 14. Common pitfalls
#
# * **Hard-coding skill inputs** — then the notebook proves nothing about your data. Derive from Cypher.
# * **`CREATE` instead of `MERGE`** — re-running duplicates nodes. Use `MERGE` for idempotence.
# * **Counting with `MATCH` only** — leaves with zero dependents vanish; use `OPTIONAL MATCH` + `count`.
# * **Unbounded variable-length paths** (`*`) on big graphs explode; always cap (`*1..3`).
# * **Mixing graphs** — putting LLM claims straight into the domain graph destroys trust; route + link.
# * **Treating confidence as truth** — keep `POSSIBLY_RELATED` separate from `RESOLVED`.
# * **Wiping a shared DB** — only `load_devops_graph()` at the start; never ad-hoc deletes.

# %% [markdown]
# ## 15. Verification battery

# %%
counts = {r["l"]: r["n"] for r in a.run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS n")}
a.check("domain layer has >=7 services", counts.get("Domain", 0) >= 7)
a.check("lexical layer has 2 chunks", counts.get("Lexical", 0) == 2)
a.check("one subject claim", counts.get("Subject", 0) == 1)
a.check("meta-schema nodes stored", counts.get("EntityType", 0) == 1 and counts.get("Rule", 0) == 1)
a.check("ontology stored", counts.get("Ontology", 0) == 1)
a.check("audit trail has 5 attempts",
        a.run("MATCH (:AgentProfile)-[r:ATTEMPTED]->() RETURN count(r) AS n")[0]["n"] == 5)
a.check("every skill covered", len(SLUGS) == 8 and all(results.values()))

# %% [markdown]
# ## Recap
#
# All 8 Ch. 3 skills now run against real graph data: model choice and reification, three-graph routing,
# evidence-based entity resolution, extraction approach, schema patterns, executable rules, the vocabulary
# spectrum and capability gating — each reading from and writing to Neo4j, orchestrated by a traced agent.
#
# **Next:** `ch4-memory` — how an agent remembers across sessions, building on these graphs.
