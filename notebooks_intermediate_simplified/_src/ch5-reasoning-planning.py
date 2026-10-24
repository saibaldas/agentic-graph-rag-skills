# %% [markdown]
# <!-- learning-path-banner -->
# > **Learning path** — three tiers, same chapters, same skills, same Neo4j:  
# > [Beginner (tier 1/3)](../notebooks_simplified/ch5-reasoning-planning.ipynb)  →  **Intermediate (tier 2/3) ← you are here**  →  [Advanced (tier 3/3)](../notebooks_advanced_simplified/ch5-reasoning-planning.ipynb)

# %% [markdown]
# # Chapter 5 — Reasoning & Planning (Intermediate)
#
# **Tier 2 of 3.** This assumes you have worked through `notebooks_simplified/ch5-reasoning-planning.ipynb`.
# Here you will inspect the skill interfaces, derive inputs from Cypher, and wire several skills together
# yourself. The tier-3 notebook adds a MiniAgent, broader experiments, and benchmarks for every skill.
#
# **Incident:** `checkout-api` p99 latency rose from about 200 ms to 2.4 s. We will use the graph to shape
# an investigation, test parallel hypotheses, and validate a read-only response. Neo4j is at
# `bolt://localhost:7687` (`neo4j` / `password`). No model calls, API keys, or AWS credentials are needed.
#
# The graph is reset by `load_devops_graph()` so this exercise starts with the repository's fictional
# DevOps data, not your own data.

# %%
import int_helpers as h

counts = h.load_devops_graph()
print("Loaded the sample graph:", counts)
CHAPTER = "reasoning-planning"
selector = h.load_skill(CHAPTER, "pipeline-architecture-selector")
dagplanner = h.load_skill(CHAPTER, "investigation-dag-planner")
merger = h.load_skill(CHAPTER, "parallel-reconcile-merge")
looprouter = h.load_skill(CHAPTER, "loop-pipeline-router")
validator = h.load_skill(CHAPTER, "constraint-guided-plan-validator")
print("Loaded five skills:", [m.__name__ for m in
      (selector, dagplanner, merger, looprouter, validator)])

# %% [markdown]
# ## 1. Let graph shape inform the route
#
# This variable-length path follows the checkout incident to affected services and their dependencies.
# `*1..3` means one to three `DEPENDS_ON` hops; `count(DISTINCT d)` avoids counting a component twice
# when multiple paths reach it. `MATCH` selects the starting nodes and `RETURN` names the result columns.
#
# We pass the resulting blast radius into the architecture selector rather than relying on query wording
# alone. A flat lookup should be sequential; an uncertain, multi-service incident can justify a tree.

# %%
blast_rows = h.run("""
    MATCH (inc:Incident {id:'INC-1001'})-[:AFFECTS]->(s:Service)-[:DEPENDS_ON*1..3]->(d)
    RETURN s.name AS service, count(DISTINCT d) AS blast_radius,
           collect(DISTINCT d.name) AS components
""")
h.show(blast_rows)
BLAST_RADIUS = blast_rows[0]["blast_radius"]

# %% [markdown]
# ## 2. Seed evidence and plan records, then aggregate
#
# The writes below add sample telemetry and a small investigation DAG. Each `Step` is work to do;
# `DEPENDS_ON` between steps means “must happen first,” while `TESTS` links a hypothesis to its component.
# `FOR` ties the work to this incident. We also add the agent's allowed actions for the later policy gate.
# The timestamps and measurements are fictional teaching data.

# %%
h.run("""
MATCH (inc:Incident {id:'INC-1001'})
MATCH (db:Database {name:'orders-db'}), (pay:Service {name:'payment-service'})
CREATE (h1:Step {id:'h1_db_pool_exhaustion', kind:'hypothesis', status:'pending',
                 duration_s:8.0, priority:0.9, description:'Database connection pool exhaustion'})
CREATE (h2:Step {id:'h2_payment_memory_pressure', kind:'hypothesis', status:'pending',
                 duration_s:6.0, priority:0.5, description:'Payment service memory pressure'})
CREATE (h3:Step {id:'h3_network_partition', kind:'hypothesis', status:'pending',
                 duration_s:12.0, priority:0.2, description:'Network partition; test after h1'})
CREATE (confirm:Step {id:'confirm_root_cause', kind:'confirmation', status:'pending',
                      duration_s:4.0, priority:0.0, description:'Gather corroborating evidence'})
CREATE (h3)-[:DEPENDS_ON]->(h1)
CREATE (confirm)-[:DEPENDS_ON]->(h1)
CREATE (confirm)-[:DEPENDS_ON]->(h2)
CREATE (h1)-[:TESTS]->(db)
CREATE (h2)-[:TESTS]->(pay)
CREATE (h3)-[:TESTS]->(db)
CREATE (h1)-[:FOR]->(inc)
CREATE (h2)-[:FOR]->(inc)
CREATE (h3)-[:FOR]->(inc)
CREATE (confirm)-[:FOR]->(inc)
CREATE (db)-[:HAS_METRIC]->(:Metric {name:'ConnectionPoolUtilization', value:1.0, unit:'ratio'})
CREATE (db)-[:HAS_METRIC]->(:Metric {name:'WaitingThreads', value:47, unit:'count'})
CREATE (pay)-[:HAS_METRIC]->(:Metric {name:'HeapUtilization', value:0.42, unit:'ratio'})
""")
action_rows = h.run("""
CREATE (agent:Agent {name:'diagnostic-agent', max_privilege:'read'})
WITH agent
UNWIND ['query_metrics','read_logs','record_incident'] AS action
CREATE (agent)-[:MAY_PERFORM]->(:Action {name:action})
RETURN collect(action) AS allowed_actions
""")
ALLOWED_ACTIONS = set(action_rows[0]["allowed_actions"])
print("Configured read-only actions:", sorted(ALLOWED_ACTIONS))

# %% [markdown]
# `collect()` gathers values into a list; `count(DISTINCT past)` counts prior incidents without duplicate
# rows. `WITH` passes one row per component into the optional history match (like a left join), keeping
# components even when they have no prior root-cause incident. This aggregation query is our metric input:
# the Python map below is built from rows returned by Neo4j, not hard-coded measurements.

# %%
metric_rows = h.run("""
    MATCH (component)-[:HAS_METRIC]->(m:Metric)
    WITH component, collect({name:m.name, value:m.value}) AS metrics
    OPTIONAL MATCH (past:Incident)-[:ROOT_CAUSE]->(component)
    RETURN component.name AS component, size(metrics) AS metric_count, metrics,
           count(DISTINCT past) AS past_incidents
    ORDER BY component
""")
h.show(metric_rows)
METRICS = {
    row["component"]: {item["name"]: item["value"] for item in row["metrics"]}
    for row in metric_rows
}

# %% [markdown]
# ## 3. Read the DAG inputs from Neo4j
#
# `OPTIONAL MATCH` preserves a step that has no predecessor; `collect()` produces its dependency list.
# The returned dictionaries match `tasks_from_dicts(rows)`. This is the third query pattern: the graph
# stores the plan, and the skill computes topological phases from those records.

# %%
step_rows = h.run("""
    MATCH (s:Step)-[:FOR]->(:Incident {id:'INC-1001'})
    OPTIONAL MATCH (s)-[:DEPENDS_ON]->(d:Step)
    RETURN s.id AS id, s.description AS description, s.duration_s AS duration_s,
           s.priority AS priority,
           [x IN collect(d.id) WHERE x IS NOT NULL] AS depends_on
""")
h.show(step_rows)

# %% [markdown]
# ## 4. Select a pipeline architecture
#
# `route_query(query, available_memory_mb=..., remaining_budget_s=..., complexity_fn=..., uncertainty_fn=...)`
# returns a `RouteDecision`. Its `complexity` and `uncertainty` are separate 0–1 signals; the returned
# `architecture` is the ideal shape, while `final` reflects resource constraints. Explicit degradation
# keeps a constrained host useful instead of letting a tree fan-out fail.

# %%
ALERT_QUERY = "Why did checkout latency spike from 200ms to 2.4s? Investigate root cause across services."
FLAT_QUERY = "What is the checkout service error rate right now?"


def graph_complexity(query):
    text_score = selector.assess_task_complexity(query)
    return max(text_score, min(1.0, BLAST_RADIUS / 8))


flat_route = selector.route_query(FLAT_QUERY)
alert_route = selector.route_query(ALERT_QUERY, complexity_fn=graph_complexity)
constrained_route = selector.route_query(
    ALERT_QUERY, complexity_fn=graph_complexity, available_memory_mb=128
)
for label, decision in [("lookup", flat_route), ("incident", alert_route), ("128MB host", constrained_route)]:
    print(f"{label:14} ideal={decision.architecture:10} final={decision.final:22} "
          f"complexity={decision.complexity:.2f} uncertainty={decision.uncertainty:.2f}")

# %% [markdown]
# **Output anatomy / trade-off.** `RouteDecision` contains `architecture`, `final`, `complexity`,
# `uncertainty`, `degraded`, and `reason`. `tree` explores independent hypotheses but costs coordination
# and memory; `sequential_fallback` is less parallel, yet safer under the memory floor. The estimators in
# `lib.py` are heuristics; production can replace the estimation seam without changing the routing API.

# %% [markdown]
# ## 5. Build phases from the graph-backed task rows
#
# `tasks_from_dicts(rows)` builds `{id: Task}` from `id`, `depends_on`, `duration_s`, and `priority`.
# `build_investigation_dag(tasks)` returns an `InvestigationDAG`; independent tasks share a phase and a
# parallel phase costs its slowest task (max), not the sum. We also execute one evidence callback: once
# h1's graph-derived metric confirms pool exhaustion, later phases can be skipped.

# %%
tasks = dagplanner.tasks_from_dicts(step_rows)
dag = dagplanner.build_investigation_dag(tasks)
for phase in dag.phases:
    print(f"phase {phase.index}: {phase.task_ids} (estimated {phase.estimated_duration_s}s)")


def test_hypothesis(task):
    if task.id == "h1_db_pool_exhaustion":
        confirmed = METRICS["orders-db"]["ConnectionPoolUtilization"] >= 1.0
        evidence = f"pool={METRICS['orders-db']['ConnectionPoolUtilization']:.0%}, " \
                   f"waiting={METRICS['orders-db']['WaitingThreads']}"
        return confirmed, evidence
    if task.id == "h2_payment_memory_pressure":
        return METRICS["payment-service"]["HeapUtilization"] > 0.9, "heap profile"
    return False, "not confirmed"


execution = dagplanner.execute_with_early_termination(dag, tasks, test_hypothesis)
print("confirmed:", execution["confirmed"], "| phases run:", execution["phases_run"],
      "| skipped:", execution["phases_skipped"])

# %% [markdown]
# **Output anatomy / why this matters.** A `Phase` exposes `index`, `task_ids`, and `estimated_duration_s`;
# `InvestigationDAG` exposes `phases`, `parallelism_factor`, `total_estimated_duration_s`, and
# `critical_path_len`. The execution trace reports `confirmed`, `phases_run`, `phases_skipped`, and
# per-phase test evidence. A cycle is a malformed plan, not a reason to retry forever; the skill raises
# `CycleError` rather than inventing an ordering.

# %% [markdown]
# ## 6. Reconcile parallel evidence safely
#
# `run_parallel_window(branch_fns, context, mode="partial_coverage", min_success=None, merge_reducer=None)`
# isolates each branch and returns a `MergeOutcome`. Here the branch inputs come from the aggregate
# Cypher result. The network probe intentionally times out to demonstrate isolation: two of three results
# still meet the default majority quorum, and flags are unioned rather than discarded.

# %%

def db_pool_branch(context):
    values = METRICS["orders-db"]
    flags = ["pool_exhausted"] if values["ConnectionPoolUtilization"] >= 1.0 else []
    return {"metrics": values, "flags": flags}


def payment_memory_branch(context):
    values = METRICS["payment-service"]
    flags = ["memory_pressure"] if values["HeapUtilization"] > 0.9 else []
    return {"metrics": values, "flags": flags}


def network_branch(context):
    raise TimeoutError("payment-to-database probe timed out")


branches = {"db_pool": db_pool_branch, "payment_memory": payment_memory_branch,
            "network": network_branch}
merge = merger.run_parallel_window(branches, {"incident": "INC-1001"}, mode="partial_coverage")
strict_merge = merger.run_parallel_window(branches, {"incident": "INC-1001"}, mode="all_or_nothing")
print("completed:", merge.completed, "succeeded:", merge.succeeded, "failed:", merge.failed)
print("flags:", merge.flags, "| rationale:", merge.rationale)
h.run("""
MATCH (inc:Incident {id:'INC-1001'})
CREATE (m:Merge {mode:$mode, completed:$completed, flags:$flags,
                 succeeded_count:$succeeded, failed_count:$failed})-[:FOR]->(inc)
""", mode=merge.mode, completed=merge.completed, flags=merge.flags,
       succeeded=len(merge.succeeded), failed=len(merge.failed))
merge_readback = h.run("""
    MATCH (m:Merge)-[:FOR]->(:Incident {id:'INC-1001'})
    RETURN m.mode AS mode, m.completed AS completed, m.flags AS flags,
           m.succeeded_count AS succeeded, m.failed_count AS failed
    ORDER BY m.succeeded_count DESC LIMIT 1
""")[0]
h.show([merge_readback])

# %% [markdown]
# **Output anatomy / trade-off.** `MergeOutcome` has `completed`, `mode`, `succeeded`, `failed`, `flags`,
# `merged_value`, and `rationale`. `partial_coverage` is useful for evidence gathering; use
# `all_or_nothing` when every branch is required for a safe decision. This implementation runs branches
# sequentially in-process—the guaranteed value here is isolation and deterministic reconciliation, not
# concurrency speed.

# %% [markdown]
# ## 7. Bound a refine loop, then validate the plan
#
# `run_loop(attempt_fn, validate_fn, max_retries=..., fallback_fn=...)` returns a dictionary containing
# `decision`, `candidate`, `iterations`, and a routing `trace`. Correctable gaps can refine; fundamental
# errors terminate with a partial result. A finite retry budget is the guard against an infinite loop.

# %%
loop_state = {"attempts": 0}


def collect_evidence(retry_count):
    loop_state["attempts"] += 1
    return {"request": f"payment heap profile, pass {loop_state['attempts']}"}


def validate_evidence(candidate):
    if loop_state["attempts"] >= 2:
        return True, None
    return False, looprouter.ValidationError(
        looprouter.SEVERITY_CORRECTABLE, "payment profile is not available yet"
    )


loop_result = looprouter.run_loop(collect_evidence, validate_evidence, max_retries=3)
fundamental_route = looprouter.route_after_validation(
    False, looprouter.ValidationError(looprouter.SEVERITY_FUNDAMENTAL, "contradictory plan"), 0
)
print("correctable gap:", loop_result["decision"], "after", loop_result["iterations"], "refine")
print("fundamental issue:", fundamental_route.decision, "—", fundamental_route.rationale)

# %% [markdown]
# `route_after_validation(is_valid, error, retry_count, max_retries=...)` returns `RouteResult(decision,
# retry_count, rationale)`. Next, `extract_constraints(spec)`, `steps_from_dicts(rows)`, and `verify(plan,
# constraints, capability)` produce `ValidationResult(score, violations, passed)`; `.feedback` turns
# violations into readable messages. Capability and forbidden-action violations are hard failures.
#
# ## 8. Validate and persist a read-only response plan
#
# We read the allowed-action list from Neo4j, validate the plan, write the validation result as a
# `PlanCheck`, and query that result back. The database remains authoritative for the permission boundary.

# %%
constraints = validator.extract_constraints({
    "max_steps": 4,
    "required_actions": ["record_incident"],
    "forbidden_actions": ["drop_table"],
})
capability = validator.CapabilityModel(allowed_actions=ALLOWED_ACTIONS, max_privilege="read")
response_plan = validator.steps_from_dicts([
    {"action": "query_metrics", "privilege": "read"},
    {"action": "read_logs", "privilege": "read"},
    {"action": "record_incident", "privilege": "read"},
])
validation = validator.verify(response_plan, constraints, capability)
print("passed:", validation.passed, "score:", validation.score)
for line in validation.feedback:
    print(" -", line)
h.run("""
MATCH (inc:Incident {id:'INC-1001'})
CREATE (v:PlanCheck {passed:$passed, score:$score, feedback:$feedback})-[:FOR]->(inc)
""", passed=validation.passed, score=validation.score, feedback=validation.feedback)
validation_readback = h.run("""
    MATCH (v:PlanCheck)-[:FOR]->(:Incident {id:'INC-1001'})
    RETURN v.passed AS passed, v.score AS score, v.feedback AS feedback
    ORDER BY v.score DESC LIMIT 1
""")[0]
h.show([validation_readback])

# %% [markdown]
# **Why this matters.** A perfect score is not permission to perform an unauthorized write: validate the
# plan before execution, and let explicit capability data constrain it. Here the plan only records a
# recommendation; this notebook never executes a remediation. The stored `PlanCheck` makes the gate
# auditable. **Common mistakes:** treating uncertainty and complexity as one dial; summing durations in a
# parallel phase; retrying a fundamental error; or trusting a high score despite a hard capability violation.
#
# ## 9. Hand-written pipeline: route → reconcile → validate
#
# This is ordinary Python, not `MiniAgent`: it composes three skills for the same checkout incident and
# prints a small decision report. It returns the signal and plan-gate result, not an automatic action.

# %%
def checkout_decision_report(query, metrics):
    route = selector.route_query(
        query, complexity_fn=lambda q: max(
            selector.assess_task_complexity(q), min(1.0, BLAST_RADIUS / 8)
        )
    )
    result = merger.run_parallel_window(
        {
            "db_pool": lambda ctx: {
                "flags": ["pool_exhausted"] if metrics["orders-db"]["ConnectionPoolUtilization"] >= 1.0 else [],
                "component": "orders-db",
            },
            "payment_memory": lambda ctx: {
                "flags": ["memory_pressure"] if metrics["payment-service"]["HeapUtilization"] > 0.9 else [],
                "component": "payment-service",
            },
            "network": network_branch,
        },
        {"incident": "INC-1001"}, mode="partial_coverage",
    )
    plan = validator.steps_from_dicts([
        {"action": "query_metrics"}, {"action": "read_logs"}, {"action": "record_incident"}
    ])
    gate = validator.verify(plan, constraints, capability)
    print("Decision report — INC-1001")
    print(f"  route: {route.final} (ideal {route.architecture}; uncertainty {route.uncertainty:.2f})")
    print(f"  evidence: {', '.join(result.flags) or 'no red flags'}; "
          f"{len(result.succeeded)}/{len(result.succeeded) + len(result.failed)} branches succeeded")
    print(f"  read-only plan gate: {'PASS' if gate.passed else 'BLOCK'} (score {gate.score:.2f})")
    print("  next: human reviews the pool-exhaustion evidence; no rollback is run here.")
    return route, result, gate


report = checkout_decision_report(ALERT_QUERY, METRICS)

# %% [markdown]
# ## Your turn 1 — query the service's direct dependencies
#
# **Task:** finish the Cypher helper so it counts unique services/components one hop from a named service.
# **Hint:** bind the start node with a parameter and use `count(DISTINCT d)`. Try your own query first,
# then run `your_turn` to compare with the reference answer.

# %%
def direct_dependency_count(service_name):
    rows = h.run("""
        MATCH (s:Service {name:$name})-[:DEPENDS_ON]->(d)
        RETURN count(DISTINCT d) AS count
    """, name=service_name)
    return rows[0]["count"]


h.your_turn("Count checkout-api direct dependencies", "Use one DEPENDS_ON hop and count distinct targets.",
            direct_dependency_count, 3, "checkout-api")

# %% [markdown]
# ## Your turn 2 — change the route under a tight budget
#
# **Task:** choose the route when complexity and uncertainty are both `0.5` but only 10 seconds remain.
# **Hint:** the middle uncertainty band selects a loop; `route_with_constraints` should degrade that loop
# to one best-effort pass when time is below the 30-second floor.

# %%
def tight_budget_route():
    decision = selector.route_with_constraints(0.5, 0.5, remaining_budget_s=10)
    return decision.architecture, decision.final, decision.degraded


h.your_turn("Route with only 10 seconds", "Inspect ideal and final separately.", tight_budget_route,
            ("loop", "single_pass_best_effort", True))

# %% [markdown]
# ## Trust checks and one concrete vibe-coding prompt
#
# Run one skill through its CLI and one self-check battery through `h.run_benchmark`. Tier 3 runs a
# benchmark for every chapter skill; here we keep the check focused. **Vibe-coding tip:** copy this prompt
# into your AI assistant when adapting the pipeline:
#
# > “In `checkout_decision_report`, add a read-only check for the latest deployment affecting `orders-db`.
# > Use a parameterized Cypher query, keep the existing three-skill pipeline, and do not execute rollback.”
#
# **Acceptance check:** rerun this notebook builder; confirm the report shows the deployment identifier,
# the existing `PlanCheck` is still passed in Neo4j, and every `h.check` below passes.

# %%
import os
import subprocess
import sys

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
cli_path = h.SKILLS_DIR / CHAPTER / "parallel-reconcile-merge" / "cli.py"
cli_result = subprocess.run(
    [sys.executable, str(cli_path), "scenario", "parallel-hypotheses"],
    capture_output=True, text=True, encoding="utf-8", env=env, timeout=120,
)
print("CLI output (first lines):")
print("\n".join(cli_result.stdout.splitlines()[:8]))
assert cli_result.returncode == 0, cli_result.stderr
benchmark_ok, benchmark_summary = h.run_benchmark(CHAPTER, "parallel-reconcile-merge")
print("benchmark:", benchmark_summary)

h.check("flat question routes sequential", flat_route.final == "sequential")
h.check("uncertain incident selects tree", alert_route.architecture == "tree")
h.check("memory constraint degrades tree", constrained_route.final == "sequential_fallback" and constrained_route.degraded)
h.check("DAG runs independent hypotheses together", set(dag.phases[0].task_ids) == {
    "h1_db_pool_exhaustion", "h2_payment_memory_pressure"})
h.check("parallel phase uses max duration, not sum", dag.phases[0].estimated_duration_s == 8.0)
h.check("confirmed hypothesis skips later phase",
        execution["confirmed"] == "h1_db_pool_exhaustion" and execution["phases_skipped"] >= 1)
h.check("parallel merge isolates failure and keeps flag",
        merge.completed and merge.failed == ["network"] and "pool_exhausted" in merge.flags)
h.check("all-or-nothing blocks a failed branch", not strict_merge.completed)
h.check("merge result persisted and read back",
        merge_readback["completed"] and "pool_exhausted" in merge_readback["flags"])
h.check("refine loop terminates within budget",
        loop_result["decision"] == looprouter.PROCEED and loop_result["iterations"] <= 3)
h.check("read-only plan validation persisted",
        validation.passed and validation_readback["passed"] and validation_readback["score"] == 1.0)
h.check("CLI and benchmark pass", cli_result.returncode == 0 and benchmark_ok)

# %% [markdown]
# ## Recap and next
#
# You routed from graph context, read a plan from Neo4j and built its DAG phases, isolated and reconciled
# parallel evidence, bounded a correction loop, validated a capability-limited plan, and persisted results
# back to the graph. The key boundary is that a planner can recommend and validate; authorization still
# decides what may execute.
#
# **Next:** continue to tier 3 at
# `../notebooks_advanced_simplified/ch5-reasoning-planning.ipynb` for the MiniAgent trace, all six skills,
# and a benchmark battery for every skill.