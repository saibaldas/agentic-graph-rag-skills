# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > **Beginner (tier 1/3) ← you are here**  →  [Intermediate (tier 2/3)](../notebooks_intermediate_simplified/ch5-reasoning-planning.ipynb)  →  [Advanced (tier 3/3)](../notebooks_advanced_simplified/ch5-reasoning-planning.ipynb)

# %% [markdown]
# # Chapter 5 (Simplified) — Reasoning & Planning, with a real Neo4j graph
#
# **Who this is for:** you are new to GraphRAG and agents. Every step is explained in plain language.
#
# **What you will learn**
# 1. How an agent *chooses a pipeline shape* (sequential / loop / tree) for a question.
# 2. How an investigation is stored as a **plan graph** (a DAG) in Neo4j and turned into parallel phases.
# 3. How to test several hypotheses in parallel **without one crash breaking the others**.
# 4. How to stop early once a hypothesis is confirmed, using evidence read from the graph.
# 5. How a *refine loop* decides between "try again" and "give up with a partial answer".
# 6. How a remediation plan is **validated** against constraints and what the agent is allowed to do.
#
# **The story:** it is 08:42, *checkout-api is slow* (p99 latency jumped from ~200 ms to 2.4 s).
# In Chapter 5 the agent does not just look things up — it **plans** an investigation.
#
# **Key idea of the chapter:** *"The graph plans. Events execute. Results reshape the graph."*
# So here the plan itself is stored as nodes and relationships in Neo4j.
#
# > **Before running:** Neo4j must be running (`docker compose -f docker-compose/docker-compose.yml up -d`).
# > Run the cells top to bottom. No API keys needed.

# %% [markdown]
# ## 0. Setup: connect to Neo4j and load the incident graph
#
# **Quick vocabulary**
# * **Node** = a thing (a service, a database, an incident). **Relationship** = a typed arrow between
#   two nodes, e.g. `(checkout-api)-[:DEPENDS_ON]->(orders-service)`.
# * **Cypher** = Neo4j's query language. `MATCH (a)-[:DEPENDS_ON]->(b) RETURN a, b` means
#   "find every a that depends on a b".
# * **Skill** = a small Python file (`skills/<chapter>/<name>/lib.py`) implementing one idea from the book.
#   We load them with `g.load_skill(chapter_folder, name)`.

# %%
import graph_helpers as g

counts = g.load_devops_graph()
print("Loaded into Neo4j:", counts)

# %% [markdown]
# The graph from earlier chapters is our ground truth. Let's recall what checkout-api depends on
# (follow `DEPENDS_ON` arrows 1 to 3 hops):

# %%
g.show(g.run("""
    MATCH (c:Service {name:'checkout-api'})-[:DEPENDS_ON*1..3]->(d)
    RETURN DISTINCT labels(d)[0] AS type, d.name AS depends_on_it
    ORDER BY type, depends_on_it
"""))

# %% [markdown]
# ## 1. Load the five Chapter 5 skills
#
# Every skill file is named `lib.py`, so `g.load_skill` imports each under its own name.
# They use only the Python standard library — no installs.

# %%
CH = "reasoning-planning"
selector = g.load_skill(CH, "pipeline-architecture-selector")
dagplanner = g.load_skill(CH, "investigation-dag-planner")
merger = g.load_skill(CH, "parallel-reconcile-merge")
looprouter = g.load_skill(CH, "loop-pipeline-router")
validator = g.load_skill(CH, "constraint-guided-plan-validator")
print("Loaded:", [m.__name__ for m in (selector, dagplanner, merger, looprouter, validator)])

# %% [markdown]
# ## 2. Choose the pipeline architecture
#
# *Skill: `pipeline-architecture-selector`.*
#
# The first design decision: **what shape should the work take?**
# * **sequential** – do step 1, then 2, then 3. Cheap; right for simple, certain questions.
# * **loop** – try, check, refine, repeat. Right for medium problems.
# * **tree** – explore several hypotheses in parallel. Right for open-ended "why" questions.
#
# The choice depends on two independent things: how **complex** the task is and how **uncertain** the
# answer is. A resource-aware wrapper also *degrades gracefully*: a tree on a memory-starved machine
# falls back to sequential instead of crashing.
#
# To make this concrete we first ask the graph how large the "blast radius" of the alert is
# (how many things could be involved) — a bigger radius means a harder question.

# %%
g.show(g.run("""
    MATCH (inc:Incident {id:'INC-1001'})-[:AFFECTS]->(s:Service)-[:DEPENDS_ON*1..3]->(d)
    RETURN inc.title AS incident, count(DISTINCT d) AS components_involved
"""))

# %%
ALERT_QUERY = ("Why did checkout latency spike from 200ms to 2.5s? "
               "Investigate root cause across services.")
FLAT_QUERY = "What is the checkout service error rate right now?"


def show_decision(decision, label):
    flag = "  [DEGRADED]" if decision.degraded else ""
    print(label)
    print(f"  complexity={decision.complexity}  uncertainty={decision.uncertainty}")
    print(f"  ideal={decision.architecture}  final={decision.final}{flag}")
    print(f"  reason: {decision.reason}\n")


alert_decision = selector.route_query(ALERT_QUERY)
flat_decision = selector.route_query(FLAT_QUERY)
degraded_decision = selector.route_query(ALERT_QUERY, available_memory_mb=128)

show_decision(flat_decision, "flat lookup ->")
show_decision(alert_decision, "latency alert ->")
show_decision(degraded_decision, "latency alert under 128MB ->")

# %% [markdown]
# **How to read it:** the simple lookup is *sequential*; the open-ended "why" question is a *tree*;
# the same "why" question on a tiny host keeps `ideal=tree` but `final=sequential_fallback`.

# %%
assert flat_decision.final == "sequential", flat_decision
assert alert_decision.architecture == "tree", alert_decision
assert degraded_decision.architecture == "tree" and degraded_decision.final == "sequential_fallback", degraded_decision
assert degraded_decision.degraded and "->" in degraded_decision.reason
print("PASS: flat->sequential, alert->tree, memory-starved alert->sequential_fallback (graceful).")

# %% [markdown]
# ## 3. Plan the investigation as a DAG — stored in Neo4j
#
# *Skill: `investigation-dag-planner`.*
#
# A **DAG** (directed acyclic graph) is a set of tasks with "must happen after" arrows and no loops.
# Our tasks are **hypotheses** (guesses about the root cause), ranked by structural plausibility
# (the dependency graph) and historical frequency (similar past incidents):
#
# 1. `h1` database connection pool exhaustion — most likely.
# 2. `h2` payment service memory pressure.
# 3. `h3` network partition between payment and database — only worth testing after ruling out h1.
# 4. `confirm` gather corroborating evidence — after h1 and h2 are done.
#
# We store each task as a `:Step` node and each "must come after" as a `DEPENDS_ON` arrow
# (`(h3)-[:DEPENDS_ON]->(h1)` = h3 needs h1 first). Steps also point at the component they test (`TESTS`)
# and the incident they belong to (`FOR`), tying the plan into the existing graph.

# %%
g.run("""
MATCH (inc:Incident {id:'INC-1001'})
MATCH (db:Database {name:'orders-db'}), (pay:Service {name:'payment-service'})
CREATE (h1:Step {id:'h1_db_pool_exhaustion', kind:'hypothesis', status:'pending', duration_s:8.0, priority:0.9,
        description:'Database connection pool exhaustion (caused similar symptoms 3x in 6 months)'})
CREATE (h2:Step {id:'h2_payment_memory_pressure', kind:'hypothesis', status:'pending', duration_s:6.0, priority:0.5,
        description:'Payment service memory pressure'})
CREATE (h3:Step {id:'h3_network_partition', kind:'hypothesis', status:'pending', duration_s:12.0, priority:0.2,
        description:'Network partition between payment and database; only test after ruling out h1'})
CREATE (c:Step {id:'confirm_root_cause', kind:'confirmation', status:'pending', duration_s:4.0, priority:0.0,
        description:'Gather corroborating evidence for the confirmed hypothesis'})
CREATE (h3)-[:DEPENDS_ON]->(h1)
CREATE (c)-[:DEPENDS_ON]->(h1)
CREATE (c)-[:DEPENDS_ON]->(h2)
CREATE (h1)-[:TESTS]->(db)
CREATE (h2)-[:TESTS]->(pay)
CREATE (h3)-[:TESTS]->(db)
CREATE (h1)-[:FOR]->(inc) CREATE (h2)-[:FOR]->(inc) CREATE (h3)-[:FOR]->(inc) CREATE (c)-[:FOR]->(inc)
""")
g.show(g.run("""
    MATCH (s:Step)
    OPTIONAL MATCH (s)-[:DEPENDS_ON]->(d:Step)
    RETURN s.id AS step, s.priority AS priority, s.duration_s AS secs, collect(d.id) AS must_come_after
    ORDER BY priority DESC
"""))

# %% [markdown]
# **Phases computed by Cypher.** A step's *phase* = the length of the longest dependency chain below it.
# Steps in the same phase don't depend on each other, so they can run **in parallel**.
# (`[:DEPENDS_ON*]` = follow the arrow any number of times; `length(p)` = number of hops.)

# %%
phase_rows = g.run("""
    MATCH (s:Step)
    OPTIONAL MATCH p=(s)-[:DEPENDS_ON*]->(:Step)
    WITH s, coalesce(max(length(p)), 0) AS phase
    RETURN phase, collect(s.id) AS steps, max(s.duration_s) AS phase_secs
    ORDER BY phase
""")
g.show(phase_rows)

# %% [markdown]
# A DAG is only valid if it has **no cycles** (a cycle = "A waits for B, B waits for A" = deadlock).
# Cypher can check: does any step reach itself?

# %%
cycles = g.run("MATCH (s:Step)-[:DEPENDS_ON*1..]->(s) RETURN s.id AS in_a_cycle")
print("steps in a cycle:", cycles)
assert cycles == []

# %% [markdown]
# **Now the skill.** We read the steps and edges back out of Neo4j, hand them to
# `investigation-dag-planner`, and compare with the Cypher answer.

# %%
rows = g.run("""
    MATCH (s:Step)
    OPTIONAL MATCH (s)-[:DEPENDS_ON]->(d:Step)
    RETURN s.id AS id, s.description AS description, s.duration_s AS duration_s,
           s.priority AS priority, [x IN collect(d.id) WHERE x IS NOT NULL] AS depends_on
""")
tasks = dagplanner.tasks_from_dicts(rows)
dag = dagplanner.build_investigation_dag(tasks)

for p in dag.phases:
    kind = "PARALLEL" if len(p.task_ids) > 1 else "single"
    print(f"Phase {p.index} [{kind}]  est={p.estimated_duration_s}s")
    for tid in p.task_ids:
        print(f"    - {tid}: {tasks[tid].description}")
print(f"\nparallelism_factor={dag.parallelism_factor}  "
      f"total_est={dag.total_estimated_duration_s}s  critical_path={dag.critical_path_len} phases")

# %% [markdown]
# **How to read it:** phase 0 holds h1 and h2 (independent → parallel); h3 and the confirmation come
# later. A parallel phase takes as long as its *slowest* task (max), not the sum — 8 s, not 14 s.

# %%
phase0 = set(dag.phases[0].task_ids)
assert phase0 == {"h1_db_pool_exhaustion", "h2_payment_memory_pressure"}, phase0
assert "h3_network_partition" in dag.phases[1].task_ids, dag.phases[1].task_ids
assert dag.phases[0].estimated_duration_s == 8.0, dag.phases[0]
assert dag.parallelism_factor >= 1
# the skill and the Cypher query agree on the phase grouping
cypher_phases = {r["phase"]: set(r["steps"]) for r in phase_rows}
skill_phases = {p.index: set(p.task_ids) for p in dag.phases}
assert cypher_phases == skill_phases, (cypher_phases, skill_phases)
print(f"PASS: h1+h2 parallel in phase 0 (est {dag.phases[0].estimated_duration_s}s = max, not sum); "
      f"h3 gated behind h1; Cypher and skill agree.")

# %% [markdown]
# ## 3.5 Test hypotheses in parallel with error isolation
#
# *Skill: `parallel-reconcile-merge`.*
#
# Each hypothesis is tested in its own **branch** (a function working on a copy of the context).
# If one branch crashes, the others must keep going — "one branch's failure shouldn't cascade".
# Only a **merge** step combines results. Two completion modes:
# * `partial_coverage` – a majority of branches is enough (good for gathering hypotheses);
# * `all_or_nothing` – every branch must succeed (good for a final confirmation).
#
# This time the branches are *real*: they query Neo4j. First, put the observed telemetry
# (what a monitoring system measured) into the graph as `Metric` nodes attached to the components:

# %%
g.run("""
MATCH (db:Database {name:'orders-db'}), (pay:Service {name:'payment-service'})
CREATE (db)-[:HAS_METRIC]->(:Metric {name:'ConnectionPoolUtilization', value:1.0, unit:'ratio', at:'2026-10-24T08:40:00Z'})
CREATE (db)-[:HAS_METRIC]->(:Metric {name:'WaitingThreads', value:47, unit:'count', at:'2026-10-24T08:40:00Z'})
CREATE (pay)-[:HAS_METRIC]->(:Metric {name:'HeapUtilization', value:0.42, unit:'ratio', at:'2026-10-24T08:40:00Z'})
""")
g.show(g.run("""
    MATCH (c)-[:HAS_METRIC]->(m:Metric)
    RETURN c.name AS component, m.name AS metric, m.value AS value, m.at AS at
"""))


# %%
def metric(component, name):
    """Read one metric value from Neo4j."""
    rows = g.run("MATCH (:{} {{name:$c}})-[:HAS_METRIC]->(m:Metric {{name:$m}}) RETURN m.value AS v".format(
        "Database" if component == "orders-db" else "Service"), c=component, m=name)
    return rows[0]["v"] if rows else None


def branch_db_pool(ctx):
    util = metric("orders-db", "ConnectionPoolUtilization")
    waiting = metric("orders-db", "WaitingThreads")
    return {"utilization": util, "waiting_threads": waiting,
            "flags": ["pool_exhausted"] if util >= 1.0 else []}


def branch_payment_mem(ctx):
    heap = metric("payment-service", "HeapUtilization")
    return {"heap_pct": heap, "flags": ["memory_pressure"] if heap > 0.9 else []}


def branch_network_partition(ctx):
    raise TimeoutError("probe to payment->db link timed out")


branches = {
    "db_pool": branch_db_pool,
    "payment_mem": branch_payment_mem,
    "network_partition": branch_network_partition,
}

merge = merger.run_parallel_window(branches, {"incident": "checkout-latency"}, mode="partial_coverage")
print("mode:", merge.mode, "| completed:", merge.completed)
print("succeeded:", merge.succeeded)
print("failed (isolated):", merge.failed)
print("flags (unioned):", merge.flags)
print("rationale:", merge.rationale)

# %% [markdown]
# **How to read it:** the network probe timed out, but that was *caught* and recorded under `failed`.
# The other two branches finished (2 of 3 = a majority), so the phase is complete, and the
# `pool_exhausted` red flag from the database branch is kept in the merged result.

# %%
assert "network_partition" in merge.failed, merge.failed
assert set(merge.succeeded) == {"db_pool", "payment_mem"}, merge.succeeded
assert merge.completed, "2/3 should clear the majority quorum"
assert "pool_exhausted" in merge.flags, merge.flags
strict = merger.run_parallel_window(branches, {}, mode="all_or_nothing")
assert not strict.completed, "all_or_nothing must fail when any branch failed"
print("PASS: crash isolated, 2/3 quorum completes, pool_exhausted flag surfaced, all_or_nothing correctly blocks.")

# %% [markdown]
# ## 3.75 Ground the test in evidence from the graph
#
# The merged result says the pool is exhausted. A good agent also asks the graph *why*: was there a
# recent change to that database? This is where a graph shines — one query connects the metric,
# the deployment that changed the database, and a past incident with the same root cause.

# %%
evidence_rows = g.run("""
    MATCH (db:Database {name:'orders-db'})-[:HAS_METRIC]->(m:Metric)
    WHERE m.name IN ['ConnectionPoolUtilization','WaitingThreads']
    WITH db, collect({metric:m.name, value:m.value}) AS metrics
    MATCH (d:Deployment)-[:CHANGED]->(db)
    OPTIONAL MATCH (past:Incident)-[:ROOT_CAUSE]->(db)
    RETURN db.name AS component, metrics, d.id AS deploy, d.at AS deployed_at, d.change AS change,
           past.id AS past_incident, past.fix AS past_fix
""")
g.show(evidence_rows)
db_pool_metrics = {"utilization": metric("orders-db", "ConnectionPoolUtilization"),
                   "waiting_threads": metric("orders-db", "WaitingThreads")}

# %%
assert db_pool_metrics["utilization"] == 1.0, db_pool_metrics
assert int(db_pool_metrics["waiting_threads"]) == 47, db_pool_metrics
assert evidence_rows and evidence_rows[0]["deploy"] == "deploy-481"
print("PASS: pool at 100% with 47 waiting threads, and deploy-481 changed the pool (100 -> 20).")

# %% [markdown]
# ## 4. Execute the DAG with early termination
#
# *Skill: `investigation-dag-planner` again — `execute_with_early_termination`.*
#
# We run phase by phase. As soon as a hypothesis is **confirmed**, testing the alternatives is a waste, so
# all later phases are skipped. The test function for h1 uses the numbers we just read from Neo4j.

# %%
POOL_EXHAUSTED = db_pool_metrics["utilization"] is not None and db_pool_metrics["utilization"] >= 1.0


def test_fn(task):
    if task.id == "h1_db_pool_exhaustion":
        return (POOL_EXHAUSTED,
                f"pool utilization {db_pool_metrics['utilization']:.0%}, "
                f"{int(db_pool_metrics['waiting_threads'])} waiting threads")
    if task.id == "h2_payment_memory_pressure":
        heap = metric("payment-service", "HeapUtilization")
        return (heap > 0.9, f"payment service heap {heap:.0%} (nominal)")
    return (False, "not evaluated (early termination)")


import json
exec_trace = dagplanner.execute_with_early_termination(dag, tasks, test_fn)
print(json.dumps(exec_trace, indent=2))

# %% [markdown]
# **Write the outcome back into the graph** ("results reshape the graph"): mark h1 confirmed,
# and the skipped steps as skipped.

# %%
g.run("MATCH (s:Step {id:$id}) SET s.status='confirmed'", id=exec_trace["confirmed"])
g.run("MATCH (s:Step) WHERE s.status='pending' AND s.id <> $id SET s.status='not_needed'", id=exec_trace["confirmed"])
g.show(g.run("MATCH (s:Step) RETURN s.id AS step, s.status AS status ORDER BY step"))

# %%
assert exec_trace["confirmed"] == "h1_db_pool_exhaustion", exec_trace["confirmed"]
assert exec_trace["phases_skipped"] >= 1, exec_trace
confirmed = g.run("MATCH (s:Step {status:'confirmed'}) RETURN s.id AS id")
assert [r["id"] for r in confirmed] == ["h1_db_pool_exhaustion"]
print(f"PASS: confirmed {exec_trace['confirmed']} in phase 0; "
      f"{exec_trace['phases_skipped']} phase(s) skipped by early termination.")

# %% [markdown]
# ## 5. Route the evidence-collection refine loop
#
# *Skill: `loop-pipeline-router`.*
#
# Sometimes evidence is incomplete (e.g. a memory profile hasn't arrived). The agent then enters a **loop**:
# collect → check → refine → check ... under a strict retry limit so it can never spin forever.
# The router separates two kinds of problem:
# * **correctable** (missing data) → refine and retry, up to the budget;
# * **fundamental** (the plan contradicts itself) → stop immediately with a *partial result*.

# %%
state = {"attempts": 0}


def collect_attempt(retry_count):
    state["attempts"] += 1
    return {"evidence_request": f"payment memory profile (attempt {state['attempts']})"}


def check_gap(candidate):
    if state["attempts"] >= 2:
        return True, None
    return False, looprouter.ValidationError(
        looprouter.SEVERITY_CORRECTABLE, "evidence incomplete: payment memory profile missing")


gap_loop = looprouter.run_loop(collect_attempt, check_gap, max_retries=looprouter.DEFAULT_MAX_RETRIES)
print("correctable evidence gap ->", gap_loop["decision"], f"(after {gap_loop['iterations']} refine)")

fundamental = looprouter.run_loop(
    lambda rc: {"plan": "route the same op through sync AND event-driven"},
    lambda c: (False, looprouter.ValidationError(
        looprouter.SEVERITY_FUNDAMENTAL, "plan contradicts itself")),
    max_retries=looprouter.DEFAULT_MAX_RETRIES,
)
print("fundamental defect      ->", fundamental["decision"], f"(after {fundamental['iterations']} refine)")

# %% [markdown]
# **Record the loop result in the graph** as an `Evidence` node, so a later reader can see how it was reached:

# %%
g.run("""
MATCH (s:Step {id:'h2_payment_memory_pressure'})
CREATE (s)-[:COLLECTED]->(:Evidence {summary:'payment memory profile', refine_iterations:$it, decision:$d})
""", it=gap_loop["iterations"], d=gap_loop["decision"])
g.show(g.run("MATCH (s:Step)-[:COLLECTED]->(e:Evidence) RETURN s.id AS step, e.summary AS evidence, e.refine_iterations AS refines, e.decision AS decision"))

# %%
assert gap_loop["decision"] == looprouter.PROCEED, gap_loop
assert gap_loop["iterations"] >= 1, gap_loop
assert fundamental["decision"] == looprouter.TERMINATE_PARTIAL, fundamental
assert fundamental["iterations"] == 0, fundamental
print("PASS: correctable gap -> proceed after refine; fundamental defect -> terminate_with_partial at iteration 0.")

# %% [markdown]
# ## 6. Validate the remediation plan against constraints + capability
#
# *Skill: `constraint-guided-plan-validator`.*
#
# Before a fix runs, the plan is checked against:
# * **constraints** – step budget, deadline, required actions, forbidden actions;
# * the agent's **capability model** – what it is actually *allowed* to do (e.g. read-only monitoring).
#
# Capability and forbidden-action violations are **hard**: a plan the agent isn't allowed to run
# is not "mostly fine". We store the agent's permissions and both candidate plans in Neo4j:
# * **Plan A** – read metrics, read logs, record the incident (all read-only);
# * **Plan B** – a `modify_db` *write* the agent may not do, taking 45 days against a 30-day deadline.

# %%
g.run("""
CREATE (a:Agent {name:'diagnostic-agent', max_privilege:'read'})
WITH a
UNWIND ['query_metrics','read_logs','record_incident','rollback_config'] AS act
CREATE (a)-[:MAY_PERFORM]->(:Action {name:act})
""")
g.run("""
MATCH (inc:Incident {id:'INC-1001'})
CREATE (pa:Plan {id:'plan-A', label:'read-only remediation'})-[:FOR]->(inc)
CREATE (pb:Plan {id:'plan-B', label:'write to database'})-[:FOR]->(inc)
WITH pa, pb
UNWIND [
  {p:'plan-A', pos:1, action:'query_metrics',   privilege:'read'},
  {p:'plan-A', pos:2, action:'read_logs',       privilege:'read'},
  {p:'plan-A', pos:3, action:'record_incident', privilege:'read'},
  {p:'plan-B', pos:1, action:'query_metrics',   privilege:'read'},
  {p:'plan-B', pos:2, action:'modify_db',       privilege:'write', eta_days:45}
] AS s
MATCH (p:Plan {id:s.p})
CREATE (p)-[:HAS_STEP]->(:PlanStep {pos:s.pos, action:s.action, privilege:s.privilege, eta_days:s.eta_days})
""")
g.show(g.run("""
    MATCH (p:Plan)-[:HAS_STEP]->(s:PlanStep)
    OPTIONAL MATCH (:Agent)-[:MAY_PERFORM]->(a:Action {name:s.action})
    RETURN p.id AS plan, s.pos AS step, s.action AS action, s.privilege AS privilege,
           a IS NOT NULL AS agent_may_do_it
    ORDER BY plan, step
"""))

# %% [markdown]
# Cypher already shows the problem: `modify_db` is not something the agent may perform.
# Now the skill does the full check, with the allowed actions *read from the graph*:

# %%
allowed = {r["a"] for r in g.run("MATCH (:Agent {name:'diagnostic-agent'})-[:MAY_PERFORM]->(x:Action) RETURN x.name AS a")}
capability = validator.CapabilityModel(allowed_actions=allowed, max_privilege="read")
constraints = validator.extract_constraints({
    "max_steps": 6,
    "deadline_days": 30,
    "required_actions": ["record_incident"],
    "forbidden_actions": ["drop_table"],
})


def plan_from_graph(plan_id):
    rows = g.run("""
        MATCH (:Plan {id:$p})-[:HAS_STEP]->(s:PlanStep) RETURN s ORDER BY s.pos
    """, p=plan_id)
    out = []
    for r in rows:
        s = r["s"]
        d = {"action": s["action"], "privilege": s["privilege"]}
        if s.get("eta_days") is not None:
            d["params"] = {"eta_days": s["eta_days"]}
        out.append(d)
    return validator.steps_from_dicts(out)


plan_a = plan_from_graph("plan-A")
plan_b = plan_from_graph("plan-B")
res_a = validator.verify(plan_a, constraints, capability)
res_b = validator.verify(plan_b, constraints, capability)

print(f"Plan A (read-only): score={res_a.score} passed={res_a.passed}")
for f in res_a.feedback:
    print("   ", f)
print(f"\nPlan B (unauthorized write): score={res_b.score} passed={res_b.passed}")
for f in res_b.feedback:
    print("   ", f)

# %% [markdown]
# **Store the verdicts** on the plans so they live with the incident:

# %%
g.run("MATCH (p:Plan {id:'plan-A'}) SET p.passed=$ok, p.score=$s", ok=res_a.passed, s=res_a.score)
g.run("MATCH (p:Plan {id:'plan-B'}) SET p.passed=$ok, p.score=$s", ok=res_b.passed, s=res_b.score)
g.show(g.run("MATCH (p:Plan) RETURN p.id AS plan, p.label AS label, p.passed AS passed, p.score AS score ORDER BY plan"))

# %%
assert res_a.passed and res_a.score == 1.0, res_a
assert not res_b.passed, "unauthorized write must not pass"
assert any(v.constraint_kind == "capability" for v in res_b.violations), res_b.violations
print("PASS: Plan A conforms (score 1.0); Plan B blocked on capability (unauthorized write).")

# %% [markdown]
# ## 7. CLI portability check
#
# Each skill also ships a command-line tool (`cli.py`), so any agent harness or a cron job can use it
# without Python imports. We run one scenario and then every skill's `benchmark` self-check.

# %%
import subprocess
import sys

RP_DIR = g.SKILLS_DIR / CH
proc = subprocess.run([sys.executable, str(RP_DIR / "investigation-dag-planner" / "cli.py"),
                       "scenario", "latency-investigation"], capture_output=True, text=True)
print(proc.stdout)
assert proc.returncode == 0, proc.stderr
print("PASS: investigation-dag-planner CLI scenario exits 0.")

# %%
results = {}
for name in ["pipeline-architecture-selector", "investigation-dag-planner", "parallel-reconcile-merge",
             "loop-pipeline-router", "constraint-guided-plan-validator"]:
    p = subprocess.run([sys.executable, str(RP_DIR / name / "cli.py"), "benchmark"],
                       capture_output=True, text=True)
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "(no output)"
    results[name] = (p.returncode, tail)
    print(f"[{'PASS' if p.returncode == 0 else 'FAIL'}] {name:36} {tail}")
assert all(rc == 0 for rc, _ in results.values()), results
print("\nPASS: all five Ch5 skill benchmark batteries green.")

# %% [markdown]
# ## 8. The diagnostic verdict — asked of the graph
#
# Everything is now in Neo4j, so the final answer is one query: the confirmed hypothesis, what it tests,
# the deployment that changed it, and the past incident that offers a fix.

# %%
verdict = g.run("""
    MATCH (inc:Incident {id:'INC-1001'})<-[:FOR]-(s:Step {status:'confirmed'})-[:TESTS]->(c)
    MATCH (d:Deployment)-[:CHANGED]->(c)
    OPTIONAL MATCH (past:Incident)-[:ROOT_CAUSE]->(c)
    RETURN inc.id AS incident, s.description AS confirmed_hypothesis, c.name AS component,
           d.id AS culprit_deploy, d.change AS change, past.fix AS known_fix
""")
g.show(verdict)
assert verdict and verdict[0]["culprit_deploy"] == "deploy-481"

# %% [markdown]
# **Diagnosis:** deploy-481 cut the `orders-db` connection pool from 100 to 20; the pool saturated
# (100% utilization, 47 waiting threads) and checkout latency rose. Remediation: roll back the
# configuration (`rollback_config`, an allowed action), add pool-utilization alerts, and require review of
# database parameter changes.
#
# ## Recap
# * **Pipeline selector** – sequential / loop / tree by complexity and uncertainty, with graceful fallback.
# * **DAG planner** – the plan is a graph (`Step` + `DEPENDS_ON`); phases come from dependency depth and
#   can be computed by Cypher *or* the skill — they agree. Parallel phase time = max, not sum.
# * **Parallel merge** – branches are isolated; one crash does not break the rest.
# * **Early termination** – stop once a hypothesis is confirmed; write the result back to the graph.
# * **Loop router** – retry correctable gaps within a budget; stop at fundamental defects.
# * **Plan validator** – hard checks against constraints and the agent's permissions.
#
# **Next:** `ch6-tool-orchestration.ipynb`.
