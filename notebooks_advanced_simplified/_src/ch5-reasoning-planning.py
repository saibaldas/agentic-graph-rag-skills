# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > [Beginner (tier 1/3)](../notebooks_simplified/ch5-reasoning-planning.ipynb)  →  [Intermediate (tier 2/3)](../notebooks_intermediate_simplified/ch5-reasoning-planning.ipynb)  →  **Advanced (tier 3/3) ← you are here**

# %% [markdown]
# # Chapter 5 (Advanced, Explained) — Reasoning & Planning on a real Neo4j graph
#
# **Tier 3 of 3 (advanced).** You finished `notebooks_simplified/ch5-...` (beginner tier). Here we go deeper, still with
# plain-language explanations. What is new:
#
# * **All six** Chapter 5 skills (the beginner tier skipped `structured-output-contract-designer`).
# * Skill **inputs come from Cypher** and skill **outputs are written back** as nodes/edges.
# * Non-trivial Cypher: multi-hop, aggregation with `collect`/`count`, `OPTIONAL MATCH`, variable-length paths.
# * A **mini agent** (plan -> act -> observe) whose tools are skills + Cypher, with its trace stored in Neo4j.
# * Skill CLIs + benchmarks, a **vibe-coding corner**, experiments and pitfalls.
#
# **Story:** 08:42, `checkout-api` is slow (p99 200 ms -> 2.4 s). Chapter motto:
# *"The graph plans. Events execute. Results reshape the graph."*
#
# **Chapter map** (book -> skill): hybrid pipeline architectures -> `pipeline-architecture-selector`;
# DAG-planned investigation -> `investigation-dag-planner`; tree pipelines + error aggregation ->
# `parallel-reconcile-merge`; loop pipelines -> `loop-pipeline-router`; constraint-guided planning ->
# `constraint-guided-plan-validator`; structured generation -> `structured-output-contract-designer`.
#
# > Needs Neo4j at `bolt://localhost:7687`. No API keys, no LLM. This notebook **wipes the database** at start.

# %%
import adv_helpers as a
import json

counts = a.load_devops_graph()
print("Loaded base DevOps graph:", counts)

# %% [markdown]
# ## 0. Setup: extend the graph with history
#
# The base graph only has one past incident. A planner needs *history* to rank hypotheses, so we add three more
# resolved incidents. Each is a node with a `ROOT_CAUSE` arrow to the component that really failed.
#
# **Cypher clauses used here** (explained once):
# * `MATCH` finds existing nodes; `CREATE` adds new nodes/edges.
# * `UNWIND list AS row` turns a list into one row per element, so one query can create many nodes.
# * `{...}` is a *property map*; `(n:Label {k:v})` is a node with label and properties.

# %%
a.run("""
UNWIND [
  {id:'INC-0612', title:'Orders timeouts',        target:'orders-db',       cause:'connection pool exhausted'},
  {id:'INC-0540', title:'Payment batch stalls',   target:'orders-db',       cause:'connection pool exhausted'},
  {id:'INC-0701', title:'Payment OOM restarts',   target:'payment-service', cause:'heap leak'}
] AS h
MATCH (c {name:h.target})
CREATE (:Incident {id:h.id, title:h.title, status:'resolved', root_cause:h.cause})-[:ROOT_CAUSE]->(c)
""")
a.show(a.run("""
    MATCH (i:Incident)-[:ROOT_CAUSE]->(c)
    RETURN c.name AS component, count(i) AS past_incidents, collect(i.id) AS ids
    ORDER BY past_incidents DESC
"""))

# %% [markdown]
# `count(i)` and `collect(i.id)` are **aggregations**: rows are grouped by the non-aggregated column
# (`c.name`), then counted / gathered into a list. `orders-db` has 3 past root causes, `payment-service` 1.

# %% [markdown]
# ## 1. Load all six skills
#
# Skill anatomy in this repo: `SKILL.md` (7 sections, source attribution), `lib.py` (stdlib only),
# `cli.py` (with a `benchmark`). `a.load_skill(chapter, slug)` imports each `lib.py` under a unique name.

# %%
CH = "reasoning-planning"
SLUGS = ["pipeline-architecture-selector", "investigation-dag-planner", "parallel-reconcile-merge",
         "loop-pipeline-router", "constraint-guided-plan-validator", "structured-output-contract-designer"]
selector, dagplanner, merger, looprouter, validator, contracts = [a.load_skill(CH, s) for s in SLUGS]
a.show([{"skill": r["slug"], "description": r["description"][:70] + "..."}
        for r in a.skill_catalog(CH)])
a.check("all six Ch5 skills discovered", len(a.skill_catalog(CH)) == 6)

# %% [markdown]
# ## 2. Choose the pipeline shape — with graph-derived complexity
#
# *Skill: `pipeline-architecture-selector`.* Shapes: **sequential** (chain), **loop** (refine), **tree**
# (parallel branches). The skill scores a query on **complexity** and **uncertainty** (both 0..1).
#
# Text heuristics alone can't see how big the system is. The graph can: the **blast radius** = how many
# components could be involved if the alert's service misbehaves. Cypher clauses:
# * `-[:DEPENDS_ON*1..3]->` is a **variable-length path**: follow the arrow 1 to 3 times.
# * `count(DISTINCT d)` counts each reachable component once, even if several paths lead to it.

# %%
blast = a.run("""
    MATCH (inc:Incident {id:'INC-1001'})-[:AFFECTS]->(s:Service)-[:DEPENDS_ON*1..3]->(d)
    RETURN s.name AS service, count(DISTINCT d) AS blast_radius, collect(DISTINCT d.name) AS components
""")
a.show(blast)
BLAST = blast[0]["blast_radius"]

ALERT_QUERY = "Why did checkout latency spike from 200ms to 2.5s? Investigate root cause across services."
FLAT_QUERY = "What is the checkout service error rate right now?"


def graph_complexity(q):
    """Text complexity, raised to at least blast_radius/8 (the graph says the problem is wide)."""
    return max(selector.assess_task_complexity(q), min(1.0, BLAST / 8))


flat = selector.route_query(FLAT_QUERY)
alert = selector.route_query(ALERT_QUERY, complexity_fn=graph_complexity)
starved = selector.route_query(ALERT_QUERY, complexity_fn=graph_complexity, available_memory_mb=128)
for label, d in [("flat lookup", flat), ("latency alert", alert), ("alert on 128MB host", starved)]:
    print(f"{label:22} complexity={d.complexity:.2f} uncertainty={d.uncertainty:.2f} "
          f"ideal={d.architecture:10} final={d.final:20} degraded={d.degraded}")

# %% [markdown]
# **Latency trade-off.** `estimate_latency` prices each shape. *Tree* = slowest branch + merge cost;
# *sequential* = sum; *loop* = one pass x expected passes (a **truncated geometric** expectation
# `(1-p^n)/(1-p)`: with retry probability `p` and cap `n`, the average number of passes). Node times
# come from the graph (hypothesis steps are created in the next section, so we use planned durations
# here and persist the decision as a `Decision` node).

# %%
planned = [8.0, 6.0, 12.0]  # seconds: three hypothesis tests
lat = {arch: selector.estimate_latency(arch, planned, merge_seconds=1.0, retry_probability=0.3, max_iterations=3)
       for arch in selector.ARCHITECTURES}
a.show([{"arch": k, "expected_s": v["expected_seconds"], "detail": v["detail"]} for k, v in lat.items()])

a.run("""
MATCH (inc:Incident {id:'INC-1001'})
CREATE (d:Decision {architecture:$arch, final:$final, complexity:$cx, uncertainty:$un, blast_radius:$b})
CREATE (d)-[:FOR]->(inc)
""", arch=alert.architecture, final=alert.final, cx=alert.complexity, un=alert.uncertainty, b=BLAST)

a.check("flat lookup -> sequential", flat.final == "sequential")
a.check("alert -> tree", alert.architecture == "tree" and alert.final == "tree")
a.check("starved host degrades tree -> sequential_fallback",
        starved.architecture == "tree" and starved.final == "sequential_fallback" and starved.degraded)
a.check("tree is fastest for 3 independent hypotheses",
        lat["tree"]["expected_seconds"] < lat["sequential"]["expected_seconds"])

# %% [markdown]
# **Why this design:** committing to one architecture wastes work on easy questions (a tree for a lookup) and
# under-serves hard ones. The resource-aware wrapper makes failure *graceful*: tree -> sequential rather than OOM.

# %% [markdown]
# ## 3. Plan the investigation as a DAG stored in Neo4j
#
# *Skill: `investigation-dag-planner`.* A **DAG** (directed acyclic graph) is tasks with "must come after"
# arrows and **no cycles** (a cycle = deadlock). Tasks that don't depend on each other fall in the same
# **phase** and can run in parallel; a phase lasts as long as its *slowest* task.
#
# **Step 1 — rank hypotheses from the graph.** Plausibility = structural (how many dependency paths lead from the
# alerting service to a component) + historical (past incidents with that root cause).
# New clauses: `OPTIONAL MATCH` (keep the row even when no match, like a left join), `WITH` (pipe results
# into the next part of the query), `count(p)` over path variables.

# %%
rank_rows = a.run("""
    MATCH (:Incident {id:'INC-1001'})-[:AFFECTS]->(root:Service)
    MATCH p=(root)-[:DEPENDS_ON*1..3]->(c)
    WITH c, count(p) AS paths
    OPTIONAL MATCH (past:Incident)-[:ROOT_CAUSE]->(c)
    WITH c, paths, count(DISTINCT past) AS past_hits
    RETURN c.name AS component, paths, past_hits, paths + 2 * past_hits AS score
    ORDER BY score DESC
""")
a.show(rank_rows)
top = max(r["score"] for r in rank_rows)
priority = {r["component"]: round(r["score"] / top, 3) for r in rank_rows}
print("normalised priority:", priority)

# %% [markdown]
# `orders-db` is reachable by two paths (via orders and via payment) **and** has 3 past incidents -> top rank.
# That ranking is data, not a hard-coded guess. **Step 2 — write the plan as `Step` nodes.**
# `DEPENDS_ON` between `Step`s means "needs first". `TESTS` links a step to the component it probes.

# %%
a.run("""
MATCH (inc:Incident {id:'INC-1001'})
MATCH (db:Database {name:'orders-db'}), (pay:Service {name:'payment-service'})
CREATE (h1:Step {id:'h1_db_pool_exhaustion', kind:'hypothesis', status:'pending', duration_s:8.0, priority:$p_db,
        description:'Database connection pool exhaustion'})
CREATE (h2:Step {id:'h2_payment_memory_pressure', kind:'hypothesis', status:'pending', duration_s:6.0, priority:$p_pay,
        description:'Payment service memory pressure'})
CREATE (h3:Step {id:'h3_network_partition', kind:'hypothesis', status:'pending', duration_s:12.0, priority:0.1,
        description:'Network partition payment<->db (only after ruling out h1)'})
CREATE (c:Step {id:'confirm_root_cause', kind:'confirmation', status:'pending', duration_s:4.0, priority:0.0,
        description:'Gather corroborating evidence'})
CREATE (h3)-[:DEPENDS_ON]->(h1)
CREATE (c)-[:DEPENDS_ON]->(h1)
CREATE (c)-[:DEPENDS_ON]->(h2)
CREATE (h1)-[:TESTS]->(db) CREATE (h2)-[:TESTS]->(pay) CREATE (h3)-[:TESTS]->(db)
CREATE (h1)-[:FOR]->(inc) CREATE (h2)-[:FOR]->(inc) CREATE (h3)-[:FOR]->(inc) CREATE (c)-[:FOR]->(inc)
""", p_db=priority["orders-db"], p_pay=priority["payment-service"])

# %% [markdown]
# **Step 3 — phases by Cypher.** A step's phase = length of the longest `DEPENDS_ON` chain below it:
# `max(length(p))` over all paths `p` from the step. `coalesce(x, 0)` replaces `null` (no chain) with 0.

# %%
phase_rows = a.run("""
    MATCH (s:Step)
    OPTIONAL MATCH p=(s)-[:DEPENDS_ON*]->(:Step)
    WITH s, coalesce(max(length(p)), 0) AS phase
    RETURN phase, collect(s.id) AS steps, max(s.duration_s) AS phase_secs
    ORDER BY phase
""")
a.show(phase_rows)
cycles = a.run("MATCH (s:Step)-[:DEPENDS_ON*1..]->(s) RETURN s.id AS in_a_cycle")
a.check("graph plan has no cycles (Cypher)", cycles == [])

# %% [markdown]
# **Step 4 — the skill.** Read steps *from Neo4j*, build the DAG, and cross-check against Cypher.
# Two independent implementations (Cypher vs Python) agreeing is a cheap, strong test.

# %%
rows = a.run("""
    MATCH (s:Step)
    OPTIONAL MATCH (s)-[:DEPENDS_ON]->(d:Step)
    RETURN s.id AS id, s.description AS description, s.duration_s AS duration_s,
           s.priority AS priority, [x IN collect(d.id) WHERE x IS NOT NULL] AS depends_on
""")
tasks = dagplanner.tasks_from_dicts(rows)
dag = dagplanner.build_investigation_dag(tasks)
for p in dag.phases:
    print(f"Phase {p.index} est={p.estimated_duration_s}s  {p.task_ids}")
print(f"parallelism_factor={dag.parallelism_factor} total={dag.total_estimated_duration_s}s critical_path={dag.critical_path_len}")

skill_phases = {p.index: set(p.task_ids) for p in dag.phases}
a.check("Cypher phases == skill phases", {r["phase"]: set(r["steps"]) for r in phase_rows} == skill_phases)
a.check("h1 and h2 run in parallel in phase 0",
        set(dag.phases[0].task_ids) == {"h1_db_pool_exhaustion", "h2_payment_memory_pressure"})
a.check("parallel phase costs max (8s), not sum (14s)", dag.phases[0].estimated_duration_s == 8.0)

# a cycle is rejected by the skill too (in-memory only - we never write a bad plan to the graph)
bad = dagplanner.tasks_from_dicts([{"id": "x", "depends_on": ["y"]}, {"id": "y", "depends_on": ["x"]}])
try:
    dagplanner.build_investigation_dag(bad)
    rejected = False
except dagplanner.CycleError:
    rejected = True
a.check("skill raises CycleError on a deadlocked plan", rejected)

# %% [markdown]
# ## 4. Test hypotheses in parallel, isolating failures
#
# *Skill: `parallel-reconcile-merge`.* Each hypothesis is a **branch** run on a *copy* of the context; an
# exception in one branch is recorded, never propagated. A **merge** (reconcile) step combines results.
# **Quorum** = how many branches must succeed: `partial_coverage` (majority) vs `all_or_nothing`.
#
# Telemetry goes into the graph as `Metric` nodes; branches are generated *from the `Step -[:TESTS]->` edges*
# by Cypher, so adding a hypothesis to the graph adds a branch without code changes.

# %%
a.run("""
MATCH (db:Database {name:'orders-db'}), (pay:Service {name:'payment-service'})
CREATE (db)-[:HAS_METRIC]->(:Metric {name:'ConnectionPoolUtilization', value:1.0, at:'2026-10-24T08:40:00Z'})
CREATE (db)-[:HAS_METRIC]->(:Metric {name:'WaitingThreads', value:47, at:'2026-10-24T08:40:00Z'})
CREATE (pay)-[:HAS_METRIC]->(:Metric {name:'HeapUtilization', value:0.42, at:'2026-10-24T08:40:00Z'})
""")
targets = {r["step"]: r["component"] for r in a.run(
    "MATCH (s:Step {kind:'hypothesis'})-[:TESTS]->(c) RETURN s.id AS step, c.name AS component")}
print(targets)

# red-flag rules: metric name -> (predicate, flag)
RULES = {"ConnectionPoolUtilization": (lambda v: v >= 1.0, "pool_exhausted"),
         "HeapUtilization": (lambda v: v > 0.9, "memory_pressure")}


def make_branch(step_id, component):
    def branch(ctx):
        if step_id == "h3_network_partition":
            raise TimeoutError("probe to payment->db link timed out")
        ms = a.run("MATCH ({name:$c})-[:HAS_METRIC]->(m:Metric) RETURN m.name AS n, m.value AS v", c=component)
        vals = {m["n"]: m["v"] for m in ms}
        flags = [f for n, (pred, f) in RULES.items() if n in vals and pred(vals[n])]
        return {"component": component, "metrics": vals, "flags": flags}
    return branch


branches = {sid: make_branch(sid, comp) for sid, comp in targets.items()}
merge = merger.run_parallel_window(branches, {"incident": "INC-1001"}, mode="partial_coverage",
                                   merge_reducer=lambda x, y: {**x, **y})
print("succeeded:", merge.succeeded, "| failed:", merge.failed, "| flags:", merge.flags)
print("rationale:", merge.rationale)
strict = merger.run_parallel_window(branches, {}, mode="all_or_nothing")

# %% [markdown]
# `merged_value` uses our `merge_reducer` (a dict-merge), folding successful branch values; `flags` is the **union**
# of every branch's red flags, so one failure can't silently drop a warning. Now persist the outcome:
# a `Merge` node, plus step statuses (`tested` / `probe_failed`).

# %%
a.run("""
MATCH (inc:Incident {id:'INC-1001'})
CREATE (m:Merge {mode:$mode, completed:$done, flags:$flags, succeeded:size($ok), failed:size($bad)})-[:FOR]->(inc)
""", mode=merge.mode, done=merge.completed, flags=merge.flags, ok=merge.succeeded, bad=merge.failed)
a.run("UNWIND $ids AS id MATCH (s:Step {id:id}) SET s.status='tested'", ids=merge.succeeded)
a.run("UNWIND $ids AS id MATCH (s:Step {id:id}) SET s.status='probe_failed'", ids=merge.failed)
a.show(a.run("MATCH (s:Step) RETURN s.id AS step, s.status AS status ORDER BY step"))

a.check("crash isolated: network branch in failed", merge.failed == ["h3_network_partition"])
a.check("2/3 quorum completes", merge.completed and len(merge.succeeded) == 2)
a.check("pool_exhausted flag survives the merge", "pool_exhausted" in merge.flags)
a.check("all_or_nothing refuses a partial result", not strict.completed)
a.check("merged value was folded by our reducer", {"component", "metrics", "flags"} <= set(merge.merged_value))

# %% [markdown]
# **Trade-off:** `partial_coverage` for *gathering* (a dead probe shouldn't cancel the investigation);
# `all_or_nothing` for *confirming* something that must be complete (e.g. a compliance sign-off).

# %% [markdown]
# ## 5. Execute the DAG with early termination, grounded in graph evidence
#
# *Skill: `investigation-dag-planner` (`execute_with_early_termination`).* Phase by phase; once a hypothesis is
# **confirmed**, later phases are skipped. The test function reads the evidence from the merge result,
# which itself came from Neo4j. Then we write results back and run a multi-hop evidence query.

# %%
evidence = {}  # step id -> branch value; the merge's value dict is a flat dict-merge, so re-read per branch
for sid, comp in targets.items():
    if sid in merge.succeeded:
        evidence[sid] = branches[sid]({})


def test_fn(task):
    ev = evidence.get(task.id)
    if ev is None:
        return (False, "not evaluated (early termination or probe failed)")
    flagged = bool(ev["flags"])
    return (flagged, f"{ev['component']} metrics={ev['metrics']} flags={ev['flags']}")


exec_trace = dagplanner.execute_with_early_termination(dag, tasks, test_fn)
print(json.dumps({k: v for k, v in exec_trace.items() if k != "phase_results"}, indent=2))

a.run("MATCH (s:Step {id:$id}) SET s.status='confirmed'", id=exec_trace["confirmed"])
a.run("MATCH (s:Step) WHERE s.status='pending' SET s.status='not_needed'")

a.check("h1 confirmed", exec_trace["confirmed"] == "h1_db_pool_exhaustion")
a.check("later phase(s) skipped", exec_trace["phases_skipped"] >= 1)
a.check("exactly one confirmed Step in graph",
        len(a.run("MATCH (s:Step {status:'confirmed'}) RETURN s")) == 1)

# %% [markdown]
# **Multi-hop evidence query.** From the confirmed step to the component, then to the deployment that changed it,
# then to *every* service that transitively depends on that component (variable-length path *backwards*:
# `<-[:DEPENDS_ON*1..3]-`), plus the past fix via `OPTIONAL MATCH`.

# %%
verdict = a.run("""
    MATCH (inc:Incident {id:'INC-1001'})<-[:FOR]-(s:Step {status:'confirmed'})-[:TESTS]->(c)
    MATCH (d:Deployment)-[:CHANGED]->(c)
    OPTIONAL MATCH (c)<-[:DEPENDS_ON*1..3]-(up:Service)
    OPTIONAL MATCH (past:Incident)-[:ROOT_CAUSE]->(c) WHERE past.fix IS NOT NULL
    RETURN s.id AS confirmed, c.name AS component, d.id AS deploy, d.change AS change,
           collect(DISTINCT up.name) AS impacted_services, collect(DISTINCT past.fix) AS known_fix
""")
a.show(verdict)
a.check("deploy-481 is the culprit", verdict[0]["deploy"] == "deploy-481")
a.check("checkout-api is among transitively impacted services", "checkout-api" in verdict[0]["impacted_services"])

# %% [markdown]
# ## 6. Structured output contracts at the seams
#
# *Skill: `structured-output-contract-designer`* (book: "Structured Generation: the keystone of reliable
# communication"). Claim: most graph-agent failures are **communication breakdowns between nodes**, not bad
# reasoning. If a node emits free text that the next node or a graph write must parse, you rely on luck.
# Constraining generation (Outlines builds a **finite-state machine** that makes invalid tokens unreachable)
# makes the seam reliable *by construction*. Three **enforcement levels**, weakest to strongest:
# `FREE_TEXT` < `JSON_SCHEMA` < `GRAMMAR_CONSTRAINED` (closed vocabulary).
#
# We describe each pipeline node as a `NodeSpec` in the graph, derive a `NodeProfile` from Cypher, ask the skill
# for the level, and write the answer back with the matching `Contract`.

# %%
a.run("""
UNWIND [
  {name:'hypothesis_generator', consumed_by:'next_node',  valid:true,  vocab:false, crit:2, lat:2, ctype:'hypothesis'},
  {name:'remediation_proposer', consumed_by:'graph_write', valid:true,  vocab:true,  crit:3, lat:2, ctype:'remediation'},
  {name:'reasoning_trace',      consumed_by:'next_node',  valid:true,  vocab:false, crit:1, lat:1, ctype:'reasoning'},
  {name:'plan_checker',         consumed_by:'tool_call',  valid:true,  vocab:true,  crit:3, lat:1, ctype:'validation'},
  {name:'incident_summary',     consumed_by:'human',      valid:false, vocab:false, crit:0, lat:3, ctype:null}
] AS n
CREATE (:NodeSpec {name:n.name, consumed_by:n.consumed_by, needs_valid_parse:n.valid,
                   fixed_vocabulary:n.vocab, criticality:n.crit, latency_budget:n.lat, contract_type:n.ctype})
""")
specs = a.run("MATCH (n:NodeSpec) RETURN n.name AS name, n.consumed_by AS consumed_by, n.needs_valid_parse AS valid, "
              "n.fixed_vocabulary AS vocab, n.criticality AS crit, n.latency_budget AS lat, n.contract_type AS ctype")
for s in specs:
    prof = contracts.NodeProfile(consumed_by=s["consumed_by"], needs_valid_parse=s["valid"],
                                 fixed_vocabulary=s["vocab"], reliability_criticality=s["crit"],
                                 latency_budget=s["lat"])
    rec = contracts.recommend_enforcement(prof)
    a.run("MATCH (n:NodeSpec {name:$n}) SET n.enforcement=$lvl", n=s["name"], lvl=rec["recommended"])
    if s["ctype"]:
        a.run("MATCH (n:NodeSpec {name:$n}) MERGE (c:Contract {node_type:$t}) MERGE (n)-[:USES_CONTRACT]->(c)",
              n=s["name"], t=s["ctype"])
a.show(a.run("""
    MATCH (n:NodeSpec)
    OPTIONAL MATCH (n)-[:USES_CONTRACT]->(c:Contract)
    RETURN n.enforcement AS level, count(n) AS nodes, collect(n.name) AS node_names, collect(c.node_type) AS contracts
    ORDER BY level
"""))
levels = {r["name"]: r["level"] for r in a.run("MATCH (n:NodeSpec) RETURN n.name AS name, n.enforcement AS level")}
a.check("closed-vocabulary nodes -> GRAMMAR_CONSTRAINED",
        levels["remediation_proposer"] == levels["plan_checker"] == "GRAMMAR_CONSTRAINED")
a.check("machine seam without vocab -> JSON_SCHEMA", levels["hypothesis_generator"] == "JSON_SCHEMA")
a.check("terminal human prose -> FREE_TEXT", levels["incident_summary"] == "FREE_TEXT")

# %% [markdown]
# **Gate graph writes with the contract.** `validate_against_contract` checks required fields, types and enum
# vocabularies — deterministically, at the *receiving* node. `gated_write` is the pattern: *validate first, then
# write*; an invalid payload never reaches the graph.

# %%
def gated_write(node_type, payload, step_id):
    verdict = contracts.validate_against_contract(payload, contracts.schema_from_node_type(node_type))
    if verdict["valid"]:
        a.run("MATCH (s:Step {id:$id}) SET s.finding_json=$j, s.confidence=$c",
              id=step_id, j=json.dumps(payload), c=float(payload["confidence"]))
    return verdict


good = {"id": "h1", "statement": "orders-db pool exhausted after deploy-481", "confidence": 0.93,
        "evidence": ["pool 100%", "47 waiting threads"]}
bad_h = {"id": "h1", "statement": "pool", "confidence": "high"}  # wrong type, missing evidence
r_good = gated_write("hypothesis", good, "h1_db_pool_exhaustion")
r_bad = gated_write("hypothesis", bad_h, "h2_payment_memory_pressure")
print("good:", r_good)
print("bad :", r_bad["violations"])
r_vocab = contracts.validate_against_contract(
    {"action": "rollback deploy-481", "risk": "extreme", "rollback": "redeploy v2.13"},
    contracts.schema_from_node_type("remediation"))
print("vocab:", r_vocab["violations"])
gain = contracts.reliability_gain(0.97)
print("0.97 -> 1.00 parse rate eliminates", gain["failures_eliminated_per_million"], "failures per million")

a.check("valid payload accepted and written", r_good["valid"] and
        a.run("MATCH (s:Step {id:'h1_db_pool_exhaustion'}) RETURN s.confidence AS c")[0]["c"] == 0.93)
a.check("invalid payload rejected (missing + type)", not r_bad["valid"] and len(r_bad["violations"]) == 2)
a.check("rejected payload never reached the graph",
        a.run("MATCH (s:Step {id:'h2_payment_memory_pressure'}) RETURN s.finding_json AS j")[0]["j"] is None)
a.check("out-of-vocabulary risk caught", not r_vocab["valid"] and "vocabulary" in r_vocab["violations"][0])
a.check("reliability_gain math", gain["failures_eliminated_per_million"] == 30000)

# %% [markdown]
# ## 7. Loop router: refine vs fallback vs stop
#
# *Skill: `loop-pipeline-router`.* After each validation, one of four decisions: `proceed`, `refine` (a
# **correctable** problem and retries left), `fallback_strategy` (correctable but retries exhausted), or
# `terminate_with_partial` (a **fundamental** defect — retrying can't help). The retry cap guarantees
# the loop terminates. We exercise all four and record the evidence-gathering result in the graph.

# %%
state = {"n": 0}


def attempt(rc):
    state["n"] += 1
    return {"evidence_request": f"payment memory profile (attempt {state['n']})"}


def check_gap(cand):
    if state["n"] >= 2:
        return True, None
    return False, looprouter.ValidationError(looprouter.SEVERITY_CORRECTABLE, "memory profile missing")


loop_ok = looprouter.run_loop(attempt, check_gap)
always_bad = lambda c: (False, looprouter.ValidationError(looprouter.SEVERITY_CORRECTABLE, "still missing"))
loop_fb = looprouter.run_loop(lambda rc: {"try": rc}, always_bad, max_retries=2,
                              fallback_fn=lambda c: {"strategy": "use cached last-known profile"})
loop_fund = looprouter.run_loop(lambda rc: {"plan": "sync AND async"},
                                lambda c: (False, looprouter.ValidationError(looprouter.SEVERITY_FUNDAMENTAL, "contradiction")))
for name, r in [("correctable->fixed", loop_ok), ("retries exhausted", loop_fb), ("fundamental", loop_fund)]:
    print(f"{name:20} decision={r['decision']:24} iterations={r['iterations']} path={[t['decision'] for t in r['trace']]}")

a.run("""
UNWIND $rows AS r
MATCH (s:Step {id:'h2_payment_memory_pressure'})
CREATE (s)-[:COLLECTED]->(:Evidence {summary:r.name, decision:r.decision, refine_iterations:r.it})
""", rows=[{"name": n, "decision": r["decision"], "it": r["iterations"]}
           for n, r in [("memory profile", loop_ok), ("memory profile (cached)", loop_fb), ("plan check", loop_fund)]])
a.show(a.run("MATCH (:Step)-[:COLLECTED]->(e:Evidence) RETURN e.summary AS evidence, e.decision AS decision, e.refine_iterations AS refines"))

a.check("correctable gap -> proceed after >=1 refine", loop_ok["decision"] == looprouter.PROCEED and loop_ok["iterations"] >= 1)
a.check("exhausted retries -> fallback strategy used",
        loop_fb["decision"] == looprouter.FALLBACK and loop_fb["candidate"]["strategy"].startswith("use cached"))
a.check("loop is bounded by max_retries", loop_fb["iterations"] == 2)
a.check("fundamental defect stops at iteration 0",
        loop_fund["decision"] == looprouter.TERMINATE_PARTIAL and loop_fund["iterations"] == 0)

# %% [markdown]
# ## 8. Validate remediation plans against constraints and capability
#
# *Skill: `constraint-guided-plan-validator`.* A plan is scored against **constraints** (max steps, deadline,
# required/forbidden actions) and a **capability model** (what the agent may actually do, and at what privilege:
# read < write < admin). Capability and forbidden-action violations are **hard** failures: a plan the agent
# cannot legally run is not "mostly fine". The agent's permissions and the candidate plans live in Neo4j; the
# `OPTIONAL MATCH` below shows the gap before the skill even runs.

# %%
a.run("""
CREATE (ag:Agent {name:'diagnostic-agent', max_privilege:'read'})
WITH ag
UNWIND ['query_metrics','read_logs','record_incident','rollback_config'] AS act
CREATE (ag)-[:MAY_PERFORM]->(:Action {name:act})
""")
a.run("""
MATCH (inc:Incident {id:'INC-1001'})
CREATE (:Plan {id:'plan-A', label:'read-only'})-[:FOR]->(inc)
CREATE (:Plan {id:'plan-B', label:'write to database'})-[:FOR]->(inc)
CREATE (:Plan {id:'plan-C', label:'rollback pool config'})-[:FOR]->(inc)
WITH 1 AS _
UNWIND [
  {p:'plan-A', pos:1, action:'query_metrics',   privilege:'read'},
  {p:'plan-A', pos:2, action:'read_logs',       privilege:'read'},
  {p:'plan-A', pos:3, action:'record_incident', privilege:'read'},
  {p:'plan-B', pos:1, action:'query_metrics',   privilege:'read'},
  {p:'plan-B', pos:2, action:'modify_db',       privilege:'write', eta_days:45},
  {p:'plan-C', pos:1, action:'query_metrics',   privilege:'read'},
  {p:'plan-C', pos:2, action:'rollback_config', privilege:'write'},
  {p:'plan-C', pos:3, action:'record_incident', privilege:'read'}
] AS s
MATCH (p:Plan {id:s.p})
CREATE (p)-[:HAS_STEP]->(:PlanStep {pos:s.pos, action:s.action, privilege:s.privilege, eta_days:s.eta_days})
""")
a.show(a.run("""
    MATCH (p:Plan)-[:HAS_STEP]->(s:PlanStep)
    OPTIONAL MATCH (:Agent)-[:MAY_PERFORM]->(act:Action {name:s.action})
    RETURN p.id AS plan, s.pos AS pos, s.action AS action, s.privilege AS privilege, act IS NOT NULL AS action_allowed
    ORDER BY plan, pos
"""))

# %%
allowed = {r["x"] for r in a.run("MATCH (:Agent)-[:MAY_PERFORM]->(x:Action) RETURN x.name AS x")}
constraints = validator.extract_constraints({"max_steps": 6, "deadline_days": 30,
                                             "required_actions": ["record_incident"],
                                             "forbidden_actions": ["drop_table"]})


def plan_from_graph(pid):
    rows = a.run("MATCH (:Plan {id:$p})-[:HAS_STEP]->(s:PlanStep) RETURN s ORDER BY s.pos", p=pid)
    out = []
    for r in rows:
        s = r["s"]
        d = {"action": s["action"], "privilege": s["privilege"]}
        if s.get("eta_days") is not None:
            d["params"] = {"eta_days": s["eta_days"]}
        out.append(d)
    return validator.steps_from_dicts(out)


def verdict_for(pid, max_priv="read"):
    cap = validator.CapabilityModel(allowed_actions=allowed, max_privilege=max_priv)
    return validator.verify(plan_from_graph(pid), constraints, cap)


results = {pid: verdict_for(pid) for pid in ("plan-A", "plan-B", "plan-C")}
for pid, r in results.items():
    print(f"{pid}: score={r.score} passed={r.passed}")
    for v in r.violations:
        print("    -", v.constraint_kind, "|", v.message)
    a.run("MATCH (p:Plan {id:$p}) SET p.passed=$ok, p.score=$s", p=pid, ok=r.passed, s=r.score)

a.check("plan-A (read-only) passes with score 1.0", results["plan-A"].passed and results["plan-A"].score == 1.0)
a.check("plan-B blocked on capability AND deadline",
        not results["plan-B"].passed and {v.constraint_kind for v in results["plan-B"].violations} >= {"capability", "deadline_days"})
a.check("plan-C blocked: read-only agent may not write (even an allowed action)",
        not results["plan-C"].passed)
escalated = verdict_for("plan-C", max_priv="write")
a.check("plan-C passes once a human grants write privilege", escalated.passed)

# %% [markdown]
# Note plan-C: `rollback_config` is on the allow-list but needs `write` privilege; the agent holds `read`.
# The two-part model (allowed **actions** x allowed **privilege**) is why escalation is an explicit, auditable
# act. We stored the verdicts on the `Plan` nodes (`passed`, `score`); the agent below reads them back.

# %% [markdown]
# ## 9. The agent: plan -> act -> observe, with skills and Cypher as tools
#
# `a.MiniAgent` is a deliberately tiny loop. **Tools** are plain functions (our skills and Cypher queries). The
# **planner** (`next_action`) looks at what the agent has observed so far and picks the next tool — here by rules; in
# production an LLM would choose. Every call is logged by `AgentTrace` as
# `(:Run)-[:STEP]->(:ToolCall)-[:NEXT]->(:ToolCall)` in Neo4j. Failures are logged too (`ok=false`).

# %%
agent = a.MiniAgent("Find why checkout-api is slow and pick a safe remediation (INC-1001)")


@agent.tool("blast_radius")
def t_blast():
    return a.run("MATCH (:Incident {id:'INC-1001'})-[:AFFECTS]->(s)-[:DEPENDS_ON*1..3]->(d) "
                 "RETURN count(DISTINCT d) AS n")[0]["n"]


@agent.tool("route")
def t_route(radius):
    d = selector.route_query(ALERT_QUERY, complexity_fn=lambda q: max(selector.assess_task_complexity(q), min(1, radius / 8)))
    return d.final


@agent.tool("plan_dag")
def t_plan():
    return f"{len(dag.phases)} phases, parallelism={dag.parallelism_factor}"


@agent.tool("test_hypotheses")
def t_test():
    m = merger.run_parallel_window(branches, {}, mode="partial_coverage")
    return {"ok": m.succeeded, "failed": m.failed, "flags": m.flags}


@agent.tool("retry_network_probe")
def t_probe():
    return branches["h3_network_partition"]({})  # raises TimeoutError -> logged as ok=False


@agent.tool("graph_evidence")
def t_evidence():
    r = a.run("MATCH (d:Deployment)-[:CHANGED]->(c:Database) RETURN d.id AS deploy, c.name AS db, d.change AS change")
    return r[0]


@agent.tool("best_plan")
def t_best_plan():
    r = a.run("MATCH (p:Plan {passed:true}) RETURN p.id AS plan ORDER BY p.score DESC, p.id")
    return r[0]["plan"] if r else None


def next_action(mem, seen):
    if "blast_radius" not in mem: return "blast_radius", {}
    if "route" not in mem: return "route", {"radius": mem["blast_radius"]}
    if mem["route"] != "tree": return None, {}
    if "plan_dag" not in mem: return "plan_dag", {}
    if "test_hypotheses" not in mem: return "test_hypotheses", {}
    if mem["test_hypotheses"]["failed"] and "retry_network_probe" not in seen: return "retry_network_probe", {}
    if "pool_exhausted" in mem["test_hypotheses"]["flags"] and "graph_evidence" not in mem: return "graph_evidence", {}
    if "best_plan" not in mem: return "best_plan", {}
    return None, {}


seen = set()
for _ in range(10):  # hard cap: an agent loop must be bounded, just like the loop router
    tool, args = next_action(agent.memory, seen)
    if tool is None:
        break
    seen.add(tool)
    try:
        agent.act(tool, **args)
    except Exception as exc:
        print(f"  observe: {tool} failed ({type(exc).__name__}) -> continue with other evidence")
a.run("MATCH (r:Run {id:$r}), (i:Incident {id:'INC-1001'}) CREATE (r)-[:INVESTIGATES]->(i)", r=agent.trace.run_id)
print("agent conclusion:", agent.memory["graph_evidence"], "| safest passing plan:", agent.memory["best_plan"])

# %% [markdown]
# **Query the trace back.** Three views of the same run, using `collect` (ordered list of tools), a conditional
# count (`CASE WHEN`), and a **variable-length path** over `NEXT` edges (`[:NEXT*]`) that reconstructs the whole
# execution chain, with `NOT (last)-[:NEXT]->()` selecting the final call.

# %%
a.show(agent.trace.timeline())
summary = a.run("""
    MATCH (:Run {id:$id})-[:STEP]->(c:ToolCall)
    WITH c ORDER BY c.seq
    RETURN count(c) AS calls, sum(CASE WHEN c.ok THEN 0 ELSE 1 END) AS failures, collect(c.tool) AS tool_order
""", id=agent.trace.run_id)
a.show(summary)
chain = a.run("""
    MATCH (:Run {id:$id})-[:STEP]->(first:ToolCall {seq:1})
    MATCH p=(first)-[:NEXT*]->(last:ToolCall) WHERE NOT (last)-[:NEXT]->()
    RETURN length(p) AS hops, [n IN nodes(p) | n.tool] AS path
""", id=agent.trace.run_id)
a.show(chain)

a.check("trace persisted every call", summary[0]["calls"] == len(agent.trace.timeline()) >= 6)
a.check("the failed probe is recorded, not hidden", summary[0]["failures"] == 1)
a.check("NEXT chain covers all calls", chain[0]["hops"] == summary[0]["calls"] - 1)
a.check("agent routed to tree and found plan-A", agent.memory["route"] == "tree" and agent.memory["best_plan"] == "plan-A")

# %% [markdown]
# ## 10. Skill CLI and benchmarks
#
# Every skill is also a command-line tool, so cron jobs, CI or other agent harnesses can use it with zero imports.
# `--help` prints the SKILL.md description; `benchmark` runs the skill's own self-checks.

# %%
import subprocess, sys
cli = a.SKILLS_DIR / CH / "structured-output-contract-designer" / "cli.py"
p = subprocess.run([sys.executable, str(cli), "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace")
print("\n".join(p.stdout.splitlines()[:8]))
a.check("--help exits 0 and shows a usage line", p.returncode == 0 and "usage" in p.stdout.lower())

# %%
bench = {}
for slug in SLUGS:
    ok, line = a.run_benchmark(CH, slug)
    bench[slug] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] {slug:38} {line[:70]}")
a.check("all six benchmarks pass", all(bench.values()) and len(bench) == 6)

# %% [markdown]
# ## 11. 🎯 Vibe-coding corner
#
# Vibe coding = describing the goal to an AI assistant (Copilot / Claude Code) and steering the result. It only
# works if **you** can verify the output. A good prompt has **context** (which files/graph), **constraints**
# (stdlib only, no graph wipes), and an **acceptance check** (a command that must pass).
#
# **Prompt 1 — new Cypher query**
# > *Context:* Neo4j holds `(:Step)-[:DEPENDS_ON]->(:Step)` with `duration_s`. *Task:* write a Cypher query
# > returning the **critical path** (the dependency chain with the largest total `duration_s`) for incident
# > INC-1001. *Constraints:* read-only, use variable-length paths and `reduce()`. *Acceptance:* the longest chain
# > is `h3_network_partition -> h1_db_pool_exhaustion` with total 12 + 8 = 20 seconds; explain each clause.
#
# **Prompt 2 — new skill**
# > Write a new skill `skills/reasoning-planning/hypothesis-confidence-calibrator/` following the anatomy of
# > `skills/reasoning-planning/loop-pipeline-router/`: `SKILL.md` with the 7 sections + Source Attribution,
# > `lib.py` stdlib-only (production swaps marked TODO), `cli.py` with argparse, `--help` printing the SKILL.md
# > description and a `benchmark` subcommand. *Acceptance:* `python cli.py --help` and `python cli.py benchmark`
# > both exit 0, and `a.run_benchmark(...)` returns True.
#
# **Prompt 3 — harden an agent**
# > In my MiniAgent loop add a budget: stop after N tool calls or S seconds, and log a final `ToolCall` named
# > `budget_exhausted`. Add a test that proves it stops. Do not change `adv_helpers.py`.
#
# **Always verify:** run it, read the diff, and ask "what would make this wrong?" (empty graph? cycle? null?).
#
# **Worked example (as if vibe-coded).** Prompt: *"Write `parallel_speedup(dag, tasks)` returning
# sequential_total / dag.total_estimated_duration_s. Must not divide by zero."* Result + the test I wrote
# against an *independent* source of truth (the sum of `duration_s` taken straight from Neo4j):

# %%
def parallel_speedup(dag, tasks):
    """How many times faster the phased plan is than running every task one after another."""
    sequential = sum(t.duration_s for t in tasks.values())
    return sequential / dag.total_estimated_duration_s if dag.total_estimated_duration_s else 1.0


db_total = a.run("MATCH (s:Step) RETURN sum(s.duration_s) AS t")[0]["t"]
speedup = parallel_speedup(dag, tasks)
print(f"sequential={db_total}s phased={dag.total_estimated_duration_s}s speedup={speedup:.2f}x")
a.check("speedup matches graph-derived sum / phased total", abs(speedup - db_total / dag.total_estimated_duration_s) < 1e-9)
a.check("speedup > 1 because h1||h2 and h3||confirm overlap", speedup > 1.0)
a.check("empty plan does not divide by zero",
        parallel_speedup(dagplanner.build_investigation_dag({}), {}) == 1.0)

# %% [markdown]
# ## 12. Experiments — change a parameter, predict, then run
#
# 1. **Memory threshold.** `TREE_MEMORY_THRESHOLD` is 512 MB. *Predict:* which of 256 / 511 / 512 / 1024 MB keep the
#    tree? (Write your guess before running.)
# 2. **Retry probability.** In `estimate_latency("loop", ...)` raise `retry_probability` from 0.1 to 0.9 with cap 3.
#    *Predict:* does expected time grow without bound? (Hint: the cap truncates it.)
# 3. **Privilege.** Re-run `verdict_for("plan-B", "write")`. *Predict:* does it pass? (Two reasons it might not.)

# %%
for mem in (256, 511, 512, 1024):
    d = selector.route_query(ALERT_QUERY, complexity_fn=graph_complexity, available_memory_mb=mem)
    print(f"memory={mem:5}MB -> final={d.final}")
loop_times = [selector.estimate_latency("loop", [8, 6], retry_probability=p, max_iterations=3)["expected_seconds"]
              for p in (0.1, 0.5, 0.9)]
print("loop expected seconds for p=0.1/0.5/0.9:", loop_times)
b_write = verdict_for("plan-B", "write")
print("plan-B with write privilege:", b_write.passed, [v.constraint_kind for v in b_write.violations])

a.check("exp1: 511MB degrades, 512MB keeps tree",
        selector.route_query(ALERT_QUERY, complexity_fn=graph_complexity, available_memory_mb=511).degraded
        and not selector.route_query(ALERT_QUERY, complexity_fn=graph_complexity, available_memory_mb=512).degraded)
a.check("exp2: loop time grows with p but is capped (<= 3 passes)", loop_times[0] < loop_times[1] < loop_times[2] <= 14 * 3)
a.check("exp3: plan-B still fails (45-day deadline) even with write", not b_write.passed)

# %% [markdown]
# ## Common pitfalls
#
# * **Hard-coding what the graph knows.** If a priority or a permission is a literal in Python, it will drift. Read it
#   with Cypher (as `rank_rows` and `allowed` do).
# * **`DEPENDS_ON` means two things here** (Service->Service and Step->Step). Always label both ends
#   (`(:Step)-[:DEPENDS_ON]->(:Step)`) or phase queries will walk into your architecture graph.
# * **Unbounded variable-length paths** (`*`) on a big graph explode. Cap them (`*1..3`) unless the data is a DAG you trust.
# * **Writing before validating.** Validate a payload against its contract *then* write (`gated_write`), never the reverse.
# * **`optional MATCH` + aggregation.** Aggregate first (`WITH c, count(p)`), then `OPTIONAL MATCH`, or counts multiply.
# * **Loops without a cap.** Both the loop router and the agent loop above have hard limits; keep that habit.
# * **Treating a quorum as success.** `partial_coverage` can complete while a branch failed — read `failed` and `flags`.
# * **These notebooks wipe the DB** (`load_devops_graph`). Never point them at a graph you care about.

# %% [markdown]
# ## 13. Verification battery
#
# One last pass asking the graph (not Python variables) whether everything we claimed was recorded.

# %%
a.check("6 skills loaded", len(SLUGS) == 6)
a.check("graph has 4 Steps, 1 confirmed", a.run("MATCH (s:Step) RETURN count(s) AS n")[0]["n"] == 4
        and a.run("MATCH (s:Step {status:'confirmed'}) RETURN count(s) AS n")[0]["n"] == 1)
a.check("Decision, Merge and 3 Plans exist",
        a.run("MATCH (n) WHERE n:Decision OR n:Merge RETURN count(n) AS n")[0]["n"] == 2
        and a.run("MATCH (p:Plan) RETURN count(p) AS n")[0]["n"] == 3)
a.check("5 NodeSpecs with enforcement", a.run("MATCH (n:NodeSpec) WHERE n.enforcement IS NOT NULL RETURN count(n) AS n")[0]["n"] == 5)
a.check("only plan-A is marked passed", [r["p"] for r in a.run("MATCH (p:Plan {passed:true}) RETURN p.id AS p")] == ["plan-A"])
a.check("agent run linked to incident",
        a.run("MATCH (:Run)-[:INVESTIGATES]->(:Incident {id:'INC-1001'}) RETURN count(*) AS n")[0]["n"] == 1)
a.check("no cycles anywhere in the Step plan", a.run("MATCH (s:Step)-[:DEPENDS_ON*1..]->(s) RETURN s") == [])
print("\nALL CHECKS PASSED")

# %% [markdown]
# ## Recap
# * **Selector** — graph-aware complexity + resource-aware degradation + latency pricing; decision stored as a node.
# * **DAG planner** — ranking from Cypher aggregation, phases by variable-length paths, cycles rejected twice.
# * **Parallel merge** — isolated branches generated from graph edges; quorum modes; flags are a union.
# * **Contract designer** — enforcement level by seam; validate *before* writing to the graph.
# * **Loop router** — proceed / refine / fallback / terminate-partial under a hard cap.
# * **Plan validator** — constraints + capability x privilege; hard violations can't be averaged away.
# * **Agent** — skills and Cypher as tools; the whole run is a queryable graph.
#
# **Next:** `ch6-tool-orchestration.ipynb` (advanced tier).
