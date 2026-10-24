# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > **Beginner (tier 1/3) ← you are here**  →  [Intermediate (tier 2/3)](../notebooks_intermediate_simplified/ch3-knowledge-representation.ipynb)  →  [Advanced (tier 3/3)](../notebooks_advanced_simplified/ch3-knowledge-representation.ipynb)

# %% [markdown]
# # Chapter 3 (Simplified) — Knowledge Representation, with a real Neo4j graph
#
# **Who this is for:** you are new to GraphRAG. Every step is explained in plain language.
#
# **What you will learn**
# 1. How to *choose* a graph data model (property graph vs RDF vs hypergraph) from what your agent must reason about.
# 2. The **three-graph architecture**: keep trusted facts, raw text and LLM guesses in separate layers.
# 3. How to build a **digital twin** of your infrastructure in Neo4j.
# 4. **Schema patterns** (event, multi-perspective, boundary, capability) that make a graph good at reasoning.
# 5. **Homoiconicity**: storing the schema and business rules *as data* the agent can read.
# 6. Where your vocabularies sit on the spectrum **pick list → taxonomy → thesaurus → ontology**.
# 7. An **authorization gate** so the agent knows what it may and may not do.
#
# **The thesis of the chapter:** *an agent is only as smart as the knowledge structures you give it.*
# If the graph cannot show time, the agent cannot see cause and effect. If it cannot hold
# disagreeing sources, it guesses. If it cannot describe its own abilities, it tries forbidden actions.
#
# **Story (same as the whole repo):** the checkout API is slow. We design the agent's "brain"
# (its knowledge representation) and store it in a **real Neo4j database**.
#
# | Step | Idea | Skill used |
# |---|---|---|
# | 1 | Pick a graph model | `graph-model-selector` |
# | 2 | Split knowledge into 3 graphs | `three-graph-router` |
# | 3 | Build the digital twin in Neo4j | (Cypher) |
# | 4 | Schema design patterns | `schema-pattern-selector` |
# | 5 | Schema + rules as data | `homoiconic-meta-schema` |
# | 6 | Vocabulary spectrum | `knowledge-organization-classifier` |
# | 7 | What may the agent do? | `capability-authorization-gate` |
#
# > **Before running:** `docker compose -f docker-compose/docker-compose.yml up -d`, then run the cells top to bottom.

# %% [markdown]
# ## 0. Setup: connect to Neo4j and load the incident graph
#
# A **graph database** stores **nodes** (things: a service, a database) and **relationships**
# (links between things: `checkout-api -[:DEPENDS_ON]-> orders-service`). You query it with
# **Cypher**, where `(a)-[:DEPENDS_ON]->(b)` reads "a depends on b".
#
# A **skill** is a small Python file (`lib.py`) from this repo. `g.load_skill(folder, name)` imports it
# so we can call its functions.

# %%
import json
import subprocess
import sys

import graph_helpers as g

counts = g.load_devops_graph()
print("Loaded into Neo4j:", counts)

CH = "knowledge-representation"
gms = g.load_skill(CH, "graph-model-selector")
tgr = g.load_skill(CH, "three-graph-router")
sps = g.load_skill(CH, "schema-pattern-selector")
hms = g.load_skill(CH, "homoiconic-meta-schema")
koc = g.load_skill(CH, "knowledge-organization-classifier")
cag = g.load_skill(CH, "capability-authorization-gate")
KR = g.SKILLS_DIR / CH


def run_cli(skill, *args):
    """Run a skill's command-line tool — the way a cron job or another agent harness would."""
    return subprocess.run([sys.executable, str(KR / skill / "cli.py"), *args],
                          capture_output=True, text=True, cwd=str(g.PROJECT_ROOT))


print("Loaded 6 skills.")

# %% [markdown]
# ## 1. Choose the graph data model (`graph-model-selector`)
#
# > *Start with your reasoning requirements, not your data.*
#
# There are three popular ways to model a graph:
# * **Property graph** (Neo4j): nodes and relationships can both carry key/value properties. Fast traversals, great tools.
# * **RDF** (triples of subject-predicate-object): standards-based, strong formal reasoning.
# * **Hypergraph**: one "edge" can connect *many* nodes at once (an **n-ary** relationship).
#
# The skill scores each model on five features, weighted 0–3 by how much *you* need them.
# First let's measure from the real graph how traversal-heavy our agent's questions are:
# how many hops long are our dependency chains?

# %%
rows = g.run("""
    MATCH p = (s:Service)-[:DEPENDS_ON*]->(end)
    WHERE NOT (end)-[:DEPENDS_ON]->()
    RETURN s.name AS service, end.name AS ends_at, length(p) AS hops
    ORDER BY hops DESC LIMIT 5
""")
g.show(rows)
max_hops = rows[0]["hops"]
print(f"\nLongest dependency chain: {max_hops} hops -> impact analysis is mostly *graph walking*.")

# %% [markdown]
# Walking many hops is the hot path for our agent, so **performance** gets the top weight (3).
# Neo4j is mature, so **tool ecosystem** is 3 too. A deployment connects many things at once
# (**n-ary**, weight 2). A little **formal reasoning** and **constraint expression** (weight 1) is nice to have.
#
# *How to read the output:* `scores` per model; `recommended` is the winner;
# `hybrid_recommended` says the runner-up is close enough that layering two models is sensible.

# %%
devops_reqs = gms.Requirements(
    formal_reasoning=1,
    n_ary_relations=2,
    performance=3 if max_hops >= 2 else 2,   # derived from the graph we just measured
    tool_ecosystem=3,
    constraint_expr=1,
)
model_rec = gms.recommend_model(devops_reqs)
print(json.dumps(model_rec, indent=2))

assert model_rec["recommended"] == "property_graph", model_rec["recommended"]
assert model_rec["hybrid_recommended"], "close perf/n-ary race should surface a hybrid"
print("\nPASS: property_graph leads; hypergraph is flagged as the n-ary layer ->", model_rec["hybrid"])

# %% [markdown]
# ### The n-ary cost, made concrete
#
# A **deployment** is *one fact* with many participants (release, database changed, service
# deployed to, time, actor...). A hyperedge holds it in **1 element**. A property graph must
# **reify** it: create an extra node for the event plus one edge per participant.
# We read the real deployment `deploy-481` from Neo4j and feed its participants to the skill.

# %%
dep = g.run("""
    MATCH (d:Deployment {id:'deploy-481'})
    OPTIONAL MATCH (d)-[:CHANGED]->(changed)
    OPTIONAL MATCH (d)-[:DEPLOYED_TO]->(target)
    RETURN d.id AS deployment, d.version AS version, d.at AS at,
           changed.name AS changed_resource, target.name AS deployed_to
""")[0]
g.show([dep])

deploy_event = gms.HyperEdge(
    type="DeploymentEvent",
    nodes={
        "deployment": dep["deployment"],
        "version": dep["version"],
        "changed_resource": dep["changed_resource"],
        "deployed_to": dep["deployed_to"],
        "timestamp": dep["at"],
        "actor": "ci-pipeline",
    },
    attributes={"change": "db connection pool 100 -> 20"},
)
cost = gms.representation_cost(deploy_event)
reified = gms.reify_as_property_graph(deploy_event)
print(json.dumps({"arity": cost["arity"],
                  "hypergraph_elements": cost["hypergraph_elements"],
                  "property_graph_elements": cost["property_graph_elements"],
                  "reification_intermediate_nodes": reified["intermediate_nodes"],
                  "reification_auxiliary_edges": reified["auxiliary_edges"]}, indent=2))

assert cost["hypergraph_elements"] == 1
assert cost["property_graph_elements"] == 1 + deploy_event.arity()
assert reified["intermediate_nodes"] == 1 and reified["auxiliary_edges"] == deploy_event.arity()
print(f"\nPASS: the same {deploy_event.arity()}-participant event = 1 hyperedge vs "
      f"{cost['property_graph_elements']} property-graph elements.")

# %% [markdown]
# *In Neo4j we use the reified form:* `Deployment` is a node, and `CHANGED` / `DEPLOYED_TO` are its edges.
# Version and time are simply properties on that event node. That is the idea behind the *event-centric* pattern in step 4.

# %% [markdown]
# ## 2. Route knowledge across the three-graph architecture (`three-graph-router`)
#
# A real agent ingests very different material. Mixing it all into one graph pollutes the truth.
# So we keep **three graphs**, separated by *origin and certainty*:
#
# | Graph | Holds | Rule |
# |---|---|---|
# | **domain** | trusted, deduplicated facts from structured APIs | single source of truth |
# | **lexical** | original raw text, verbatim | must have **provenance** (where it came from) |
# | **subject** | claims extracted by an LLM | must have a **confidence** score |
#
# The router decides which graph a record belongs in, and **refuses** records that break the rules.

# %%
domain_rec = tgr.Record(payload={"name": "checkout-api", "tier": "critical"},
                        origin="structured", entity_resolved=True)
lexical_rec = tgr.Record(payload={"text": "checkout-api p99 latency spiked at 8:41am"},
                         origin="raw_text", has_provenance=True)
subject_rec = tgr.Record(payload={"mention": "the checkout service"},
                         origin="extraction", confidence=0.82)

for label, r in [("structured describe", domain_rec),
                 ("raw incident text ", lexical_rec),
                 ("extracted mention  ", subject_rec)]:
    print(f"{label} -> {tgr.route(r)['graph']}")

assert tgr.route(domain_rec)["graph"] == "domain"
assert tgr.route(lexical_rec)["graph"] == "lexical"
assert tgr.route(subject_rec)["graph"] == "subject"

raised = False
try:
    tgr.route(tgr.Record(payload={"text": "x"}, origin="raw_text", has_provenance=False))
except ValueError:
    raised = True
assert raised, "raw_text without provenance should raise, not silently insert"
print("\nPASS: routing correct; provenance-less raw text is rejected.")

# %% [markdown]
# ### Linking subject back to domain (entity resolution)
#
# The LLM wrote *"the checkout service"*, but the trusted graph calls it `checkout-api`.
# **Entity resolution** decides they are the same thing. Only if the text similarity clears a
# **threshold** do we create a `CORRESPONDS_TO` link. We feed the skill the real service names from Neo4j.

# %%
services = g.run("MATCH (s:Service) RETURN s.name AS name")
candidates = {r["name"]: r["name"].replace("-", " ") for r in services}
print("Candidates from Neo4j:", candidates)

corr = tgr.link_subject_to_domain("the checkout service", candidates, threshold=0.5)
print("CORRESPONDS_TO:", json.dumps(corr.__dict__, indent=2))
path = tgr.cross_graph_query_path("domain", "lexical")
print("domain -> lexical traversal:", path)

assert corr.domain_id == "checkout-api" and corr.linked
assert path == ["CORRESPONDS_TO", "EXTRACTED_FROM"]
print("\nPASS: mention resolved to checkout-api; provenance path established.")

# %% [markdown]
# ## 3. Build the digital twin in Neo4j
#
# A **digital twin** is a queryable copy of your infrastructure in the graph. We already have the
# trusted services/databases. Now we add the three layers using **labels** (a label is a node's type tag):
# * mark existing facts `:Domain`,
# * add raw log lines as `:Chunk:Lexical` nodes (with a `source`),
# * add the LLM finding as `:Finding:Subject` with a confidence, linked to the domain service and the logs.
#
# We also add three more services that call `checkout-api` so "how critical is it?" gets interesting later.

# %%
g.run("""
    MATCH (n) WHERE n:Service OR n:Database OR n:Cache
    SET n:Domain, n.environment = 'production'
""")
g.run("""
    MATCH (checkout:Service {name:'checkout-api'})
    UNWIND ['web-frontend', 'mobile-bff', 'partner-api'] AS name
    CREATE (s:Service:Domain {name:name, tier:'medium', environment:'production'})
    CREATE (s)-[:DEPENDS_ON]->(checkout)
""")
g.run("""
    MATCH (checkout:Service {name:'checkout-api'})
    CREATE (c0:Chunk:Lexical {id:'log-0', source:'/aws/ecs/checkout-api',
        text:'ERROR checkout request_id=req-013 duration_ms=5012 status=504 db_pool_exhausted=true'})
    CREATE (c1:Chunk:Lexical {id:'log-1', source:'/aws/ecs/checkout-api',
        text:'ERROR checkout request_id=req-019 duration_ms=4998 status=504 db_pool_exhausted=true'})
    CREATE (f:Finding:Subject {id:'subj-rootcause',
        claim:'database connection pool exhausted on checkout-api', confidence:0.82})
    CREATE (f)-[:EXTRACTED_FROM]->(c0)
    CREATE (f)-[:EXTRACTED_FROM]->(c1)
    CREATE (f)-[:CORRESPONDS_TO]->(checkout)
""")
g.show(g.run("""
    MATCH (n)
    WITH CASE WHEN n:Domain THEN 'domain' WHEN n:Lexical THEN 'lexical'
              WHEN n:Subject THEN 'subject' ELSE 'unlabelled' END AS graph_layer
    RETURN graph_layer, count(*) AS nodes ORDER BY graph_layer
"""))

# %% [markdown]
# *Reading the table:* each node belongs to exactly one layer. `unlabelled` should be the incident,
# alarm and deployment events (the operational history we loaded at the start).
#
# **The payoff — a cross-graph provenance query.** "Which raw log lines back the claim about checkout-api?"
# We walk subject → domain and subject → lexical in one query, and then assert the shape.

# %%
prov = g.run("""
    MATCH (s:Service:Domain {name:'checkout-api'})<-[:CORRESPONDS_TO]-(f:Finding)-[:EXTRACTED_FROM]->(c:Chunk)
    RETURN s.name AS service, f.claim AS claim, f.confidence AS confidence, c.id AS log_id, c.text AS evidence
    ORDER BY log_id
""")
g.show(prov)
assert len(prov) == 2 and all("db_pool_exhausted" in r["evidence"] for r in prov)
domain_n = g.run("MATCH (n:Domain) RETURN count(n) AS c")[0]["c"]
lex_n = g.run("MATCH (n:Lexical) RETURN count(n) AS c")[0]["c"]
subj_n = g.run("MATCH (n:Subject) RETURN count(n) AS c")[0]["c"]
assert (domain_n, lex_n, subj_n) == (9, 2, 1)
print(f"\nPASS: twin has {domain_n} domain / {lex_n} lexical / {subj_n} subject nodes; every claim traces to raw logs.")

# %% [markdown]
# ## 4. Shape the knowledge with schema design patterns (`schema-pattern-selector`)
#
# Loading data is not enough — the structure must support reasoning. The chapter names four patterns. Each
# **reifies** something (turns a plain edge or property into its own node) so that metadata can hang off it:
#
# | Need | Pattern |
# |---|---|
# | history of deployments | **event-centric** (an Event node with a time) |
# | two sources disagree (Terraform vs AWS) | **multi-perspective** (each view + its source + confidence) |
# | production vs staging must not mix | **contextual-boundary** |
# | what the agent may do | **capability-model** |
#
# `recommend_pattern(description)` maps a plain-English description of a knowledge shape to a pattern.

# %%
shapes = {
    "deployment history": "a deployment event with a timestamp and git commit, affecting "
                          "services, preceded by a prior deploy in the deployment history",
    "config drift":       "conflicting configuration: Terraform state perspective disagrees "
                          "with the AWS api source, configuration drift, each according_to a "
                          "source with confidence",
    "environment scope":  "restrict knowledge to the production environment scope, applies_to "
                          "one team, valid_during a window, prevent context_mixing with staging",
    "agent limits":       "agent authorization to restart an instance with an operational "
                          "limit, escalate above authority, permission required",
}
expected = {"deployment history": "event_centric", "config drift": "multi_perspective",
            "environment scope": "contextual_boundary", "agent limits": "capability_model"}
for name, desc in shapes.items():
    prec = sps.recommend_pattern(desc)
    print(f"{name:20s} -> {prec['recommended']}")
    assert prec["recommended"] == expected[name], (name, prec["recommended"])
print("\nPASS: all four shapes mapped to their patterns.")

# %% [markdown]
# **Validating an instance.** The event-centric pattern *requires* a time link. We build the "good" deployment
# instance from the real `deploy-481` node in Neo4j, and a "broken" one with the time removed.
# The drift example has two perspectives (Terraform says 2 ingress rules, AWS says 3).

# %%
good_deploy = {"relationships": {"hasParticipant": [dep["deployed_to"], dep["changed_resource"]],
                                 "hasStartTime": [dep["at"]]}}
bad_deploy = {"relationships": {"hasParticipant": [dep["deployed_to"]]}}
drift = {"relationships": {"according-to": ["terraform_state", "aws_api"]},
         "perspectives": [{"source": "terraform_state", "value": 2, "confidence": 1.0},
                          {"source": "aws_api", "value": 3, "confidence": 1.0}]}

good_res = sps.validate_instance("event_centric", good_deploy)
bad_res = sps.validate_instance("event_centric", bad_deploy)
drift_res = sps.validate_instance("multi_perspective", drift)
print("valid deployment event :", good_res["valid"])
print("broken deployment event:", bad_res["valid"], "->", bad_res["errors"][0])
print("config-drift instance  :", drift_res["valid"])

assert good_res["valid"] and not bad_res["valid"] and drift_res["valid"]
print("\nPASS: validator accepts well-formed patterns and rejects the temporally-unlinked event.")

# %% [markdown]
# ## 5. Schema and rules as data (`homoiconic-meta-schema`)
#
# **Homoiconicity** (big word, simple idea): the schema itself is stored *as data in the same graph*,
# so the agent can look at — and even change — its own rules. Two parts:
# 1. **Meta-knowledge:** a description of "what an entity type is". We describe `Service` that way, and one
#    validator checks both the type *and* a data instance of it.
# 2. **Executable knowledge:** a business rule stored with a `condition` and an `action`.
#
# First, the instance we validate is `checkout-api`, read from Neo4j.

# %%
service_type = {
    "name": "Service",
    "description": "A logical service in the DevOps ontology",
    "properties": [
        {"name": "name", "type": "string", "required": True},
        {"name": "tier", "type": "string"},
        {"name": "environment", "type": "string"},
    ],
}
node = g.run("""MATCH (s:Service {name:'checkout-api'})
                RETURN s.name AS name, s.tier AS tier, s.environment AS environment""")[0]
type_res = hms.validate_entity_type(service_type)
inst_res = hms.validate_data_against_type(service_type, node)
print("Service type valid against metaschema:", type_res["valid"])
print("checkout-api (from Neo4j) valid against Service type:", inst_res["valid"])
assert type_res["valid"] and inst_res["valid"]
print("PASS: schema and data validated by the same machinery.")

# %% [markdown]
# **Storing the schema in the graph.** Because it is just data, we can put the type definition in Neo4j
# as a `:EntityType` node linked to its properties, and query our own schema with Cypher.

# %%
g.run("""
    CREATE (t:EntityType {name:$name, description:$desc})
    WITH t UNWIND $props AS p
    CREATE (t)-[:HAS_PROPERTY]->(:PropertyDef {name:p.name, type:p.type, required:coalesce(p.required, false)})
""", name=service_type["name"], desc=service_type["description"], props=service_type["properties"])
g.show(g.run("""
    MATCH (t:EntityType)-[:HAS_PROPERTY]->(p:PropertyDef)
    RETURN t.name AS entity_type, p.name AS property, p.type AS type, p.required AS required
"""))

# %% [markdown]
# **Executable rule.** `DetermineServiceCriticality` says: count a service's dependents; more than 5 = Critical,
# more than 2 = Important, otherwise Standard. We count dependents **from Neo4j** (direct `DEPENDS_ON`
# arrows pointing at the service) and feed each count to the rule. We store the verdict back on the node.

# %%
criticality_rule = {
    "name": "DetermineServiceCriticality",
    "description": "Assign a service criticality tier by number of dependents",
    "condition": "MATCH (s:Service)<-[:DEPENDS_ON]-(d) WITH s, COUNT(d) AS dependent_count "
                 "RETURN s, dependent_count",
    "action": "WHEN dependent_count > 5 THEN SET s.criticality = 'Critical' "
              "WHEN dependent_count > 2 THEN SET s.criticality = 'Important' "
              "ELSE SET s.criticality = 'Standard'",
}
rule_res = hms.validate_rule(criticality_rule)
assert rule_res["valid"], rule_res["errors"]

counts = g.run("""
    MATCH (s:Service)
    OPTIONAL MATCH (s)<-[:DEPENDS_ON]-(d)
    RETURN s.name AS service, count(d) AS dependent_count
    ORDER BY dependent_count DESC, service
""")
for row in counts:
    row["criticality"] = hms.evaluate_rule(criticality_rule, {"dependent_count": row["dependent_count"]})["criticality"]
    g.run("MATCH (s:Service {name:$n}) SET s.criticality = $c", n=row["service"], c=row["criticality"])
g.show(counts)

by_name = {r["service"]: r for r in counts}
# checkout-api: cart-service + web-frontend + mobile-bff + partner-api = 4 dependents -> Important.
assert by_name["checkout-api"]["dependent_count"] == 4
assert by_name["checkout-api"]["criticality"] == "Important"
assert hms.evaluate_rule(criticality_rule, {"dependent_count": 8}) == {"criticality": "Critical"}
print("\nPASS: checkout-api has 4 dependents -> 'Important'; a hub with 8 would be 'Critical'.")

# %% [markdown]
# ## 6. Locate vocabularies on the organization spectrum (`knowledge-organization-classifier`)
#
# Companies already have lists of terms. They range from simple to rich:
# **pick list** (just allowed values) → **taxonomy** (a hierarchy) → **thesaurus** (adds synonyms/related terms)
# → **ontology** (classes, rules and relationships a machine can reason over).
# The classifier places a vocabulary on this spectrum without over-claiming.

# %%
print("region codes       ->", koc.classify({"values": ["us-east-1", "eu-west-1"]})["classification"])
print("resource hierarchy ->", koc.classify({"has_hierarchy": True})["classification"])
devops_features = {"has_classes": True, "has_properties": True, "has_inference": True}
print("DevOps ontology    ->", koc.classify(devops_features)["classification"])
print("upgrade pick_list  ->", koc.recommend_upgrade({"values": ["a", "b"]})["next"])

assert koc.classify({"values": ["us-east-1"]})["classification"] == "pick_list"
assert koc.classify({"has_hierarchy": True})["classification"] == "taxonomy"
assert koc.classify(devops_features)["classification"] == "ontology"
print("\nPASS: region codes are a pick_list; the DevOps ontology is a full ontology.")

# %% [markdown]
# **The five parts of an ontology:** *classes* (kinds of things), *subclasses* (specialisations),
# *individuals* (actual instances), *axioms* (rules that must hold) and *relationships*.
# Our individuals come straight from the Neo4j domain graph. A deliberately broken copy, with a
# relationship pointing at a class `Ghost` that does not exist, must be rejected.

# %%
individuals = g.run("""
    MATCH (n:Domain)
    RETURN n.name AS name, [l IN labels(n) WHERE l <> 'Domain'][0] AS type ORDER BY name
""")
g.show(individuals)

devops_ontology = {
    "classes": [{"name": n} for n in
                ("Compute", "Service", "Database", "Cache", "VPC", "IAMRole", "LoadBalancer", "DeploymentEvent")],
    "subclasses": [{"name": "FargateService", "parent": "Service"},
                   {"name": "RDSDatabase", "parent": "Database"}],
    "individuals": individuals,
    "axioms": ["A production Service must be BOUND_BY at least one SLA",
               "Compute ASSUMES_ROLE exactly one IAMRole"],
    "relationships": [{"name": "DEPENDS_ON", "from": "Service", "to": "Service"},
                      {"name": "ROUTES_TO", "from": "LoadBalancer", "to": "Service"},
                      {"name": "ASSUMES_ROLE", "from": "Compute", "to": "IAMRole"},
                      {"name": "PART_OF", "from": "Compute", "to": "VPC"}],
}
res = koc.validate_ontology_components(devops_ontology)
print("\npresent:", res["present"])
print("missing:", res["missing"], "| valid:", res["valid"])
assert res["valid"], res

broken = dict(devops_ontology)
broken["relationships"] = devops_ontology["relationships"] + \
    [{"name": "TALKS_TO", "from": "Service", "to": "Ghost"}]
broken_res = koc.validate_ontology_components(broken)
assert not broken_res["valid"] and any("Ghost" in e for e in broken_res["errors"])
print("broken ontology rejected:", broken_res["errors"][-1])
print("\nPASS: 5-component ontology validates; dangling relationship endpoint caught.")

# %% [markdown]
# ## 7. Gate what the agent may do (`capability-authorization-gate`)
#
# The **capability model** makes the agent's boundaries something it can *check before acting*.
# While investigating, the agent should freely **read** (metrics, logs) but must **escalate**
# (ask a human) before it **remediates** (restart an instance, scale beyond a limit).
# `authorize(agent, capability, amount)` returns `allow`, `escalate` or `deny`.
#
# We also store the agent's capabilities in Neo4j so they are queryable like everything else.

# %%
spec = json.loads((KR / "capability-authorization-gate" / "sample_devops_agent.json").read_text(encoding="utf-8"))
agent = cag.agent_from_spec(spec)
print("Agent spec keys:", list(spec.keys()))

checks = [
    ("read_metrics", None, "allow"),
    ("query_logs", None, "allow"),
    ("restart_instance", None, "escalate"),
    ("scale_autoscaling_group", 25, "escalate"),
]
g.run("MATCH (a:Agent) DETACH DELETE a")
g.run("CREATE (:Agent {name:'devops-agent', level:'User'})")
for cap, amount, expected_decision in checks:
    d = cag.authorize(agent, cap, amount=amount)
    print(f"{cap:24s} amount={str(amount):4s} -> {d['decision']}")
    assert d["decision"] == expected_decision, (cap, d["decision"])
    g.run("""MATCH (a:Agent {name:'devops-agent'})
             CREATE (a)-[:ATTEMPTED {capability:$cap, amount:$amount, decision:$decision}]->(:Decision)""",
          cap=cap, amount=amount, decision=d["decision"])
print("\nPASS: the agent reads freely but must escalate to remediate.")

g.show(g.run("""
    MATCH (:Agent)-[r:ATTEMPTED]->(:Decision)
    RETURN r.capability AS capability, r.amount AS amount, r.decision AS decision ORDER BY decision, capability
"""))

# %% [markdown]
# *Reading the table:* the audit trail of decisions now lives in the graph. The same logic is also available as a
# command-line tool, which is how other harnesses (CI, cron) would call it:

# %%
proc = run_cli("capability-authorization-gate", "scenario", "devops-latency")
assert proc.returncode == 0, proc.stderr
print("CLI scenario (devops-latency) tail:")
print("\n".join(proc.stdout.strip().splitlines()[-6:]))

# %% [markdown]
# ## 8. Verdict — what the DevOps agent now knows
#
# We summarise by *asking Neo4j*, then run every skill's own `benchmark` self-check.

# %%
root = g.run("MATCH (f:Subject) RETURN f.claim AS claim, f.confidence AS confidence")[0]
knows = {
    "graph_model": model_rec["recommended"] + " (hybrid: " + model_rec["hybrid"] + ")",
    "nodes_in_neo4j": g.run("MATCH (n) RETURN count(n) AS c")[0]["c"],
    "relationships_in_neo4j": g.run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"],
    "root_cause_claim": root["claim"],
    "checkout_api_criticality": by_name["checkout-api"]["criticality"],
    "ontology_valid": res["valid"],
    "agent_boundary": "reads metrics+logs; escalates restart/scale",
}
print(json.dumps(knows, indent=2))

skill_names = ["graph-model-selector", "three-graph-router", "schema-pattern-selector",
               "homoiconic-meta-schema", "knowledge-organization-classifier",
               "capability-authorization-gate"]
results = {}
for s in skill_names:
    proc = run_cli(s, "benchmark")
    results[s] = proc.returncode
    print(f"  [{'PASS' if proc.returncode == 0 else 'FAIL'}] {s}")
assert all(rc == 0 for rc in results.values()), results
print("\nAll six knowledge-representation skills pass their benchmarks.")

# %% [markdown]
# ## Recap
# * Pick the graph model from **reasoning needs**; for DevOps that is a property graph (Neo4j), with n-ary events reified.
# * Keep **domain / lexical / subject** knowledge in separate layers; link them with entity resolution and `CORRESPONDS_TO`.
# * **Schema patterns** (event, multi-perspective, boundary, capability) give the graph time, disagreement, scope and limits.
# * **Homoiconic** schemas and rules are data in the graph, so the agent can inspect and apply them (e.g. criticality from real dependent counts).
# * Know where your vocabularies sit (pick list → ontology) and validate ontologies' five components.
# * A **capability gate** keeps the agent inside its authority: read freely, escalate to remediate.
#
# **Next:** `ch4-memory.ipynb`.
