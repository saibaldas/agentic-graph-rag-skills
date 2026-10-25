"""The "what changed?" scenario for `notebooks_advanced_with_observability_plus_changes`.

The original notebooks investigate ONE problem: INC-1001, *checkout-api is slow*
(deploy-481 shrank the orders-db connection pool from 100 to 20).

This module adds the next chapter of the same company's story:

  * 11 NEW KINDS of data are loaded into the SAME Neo4j graph (feature flags, config changes, vendor status,
    secret rotations, queues, regions, cost records, metric samples, customer tickets, log events, runbooks ...).
  * A NEW, DIFFERENT problem appears: INC-2001, *customers are being charged twice* (plus a cost spike
    and a growing dead-letter queue). It is NOT a latency problem and the old root cause (db pool) is innocent.

Nothing here calls an LLM. It is plain, idempotent data loading (MERGE) so every notebook can call
`scn.load_changes()` safely, in any order, any number of times.

The ground truth lives in GROUND_TRUTH / GOLD_QA so notebooks can score their own investigations.
"""
from __future__ import annotations

import re

import real_helpers as h

# --------------------------------------------------------------------------- the new data
# Every node: (label, properties). Services / databases are keyed by `name`, everything else by `id`.
NAME_KEYED = {"Service", "Database", "Cache", "Vendor", "Queue", "Region"}

NODES: list[tuple[str, dict]] = [
    # ---- wave 1: topology (new services, regions, a vendor, queues)
    ("Service", {"name": "fraud-service", "tier": "high", "owner": "team-risk"}),
    ("Service", {"name": "notification-service", "tier": "low", "owner": "team-growth"}),
    ("Region", {"name": "eu-west-1", "city": "Dublin"}),
    ("Region", {"name": "us-east-1", "city": "N. Virginia"}),
    ("Vendor", {"name": "PayGate", "kind": "payment processor", "contract": "pay-per-call"}),
    ("Vendor", {"name": "MailPost", "kind": "email provider", "contract": "monthly"}),
    ("Queue", {"name": "payments-retry-queue", "engine": "sqs"}),
    ("Queue", {"name": "payments-dlq", "engine": "sqs", "purpose": "dead-letter queue for failed payment jobs"}),
    ("Engineer", {"id": "eng-priya", "name": "Priya", "team": "team-payments", "role": "on-call"}),
    ("Engineer", {"id": "eng-marco", "name": "Marco", "team": "team-payments", "role": "backend"}),
    ("Engineer", {"id": "eng-lena", "name": "Lena", "team": "team-orders", "role": "release manager"}),

    # ---- wave 2: change events (deployments, config changes, feature flags, secret rotations)
    ("Deployment", {"id": "deploy-482", "version": "v2.14.1", "at": "2026-10-24T09:10:00Z",
                    "change": "db connection pool 20 -> 100 (rollback of deploy-481)"}),
    ("ConfigChange", {"id": "cfg-310", "at": "2026-10-24T14:05:00Z", "key": "gateway_timeout_ms",
                      "old": "30000", "new": "3000", "ticket": "CHG-310",
                      "reason": "tighten tail latency after INC-1001"}),
    ("Deployment", {"id": "deploy-489", "version": "v3.2.0", "at": "2026-10-24T15:10:00Z",
                    "change": "cart-service: enable new-checkout-ui feature flag"}),
    ("Deployment", {"id": "deploy-490", "version": "v2.15.0", "at": "2026-10-24T15:20:00Z",
                    "change": "payment-service: enable payment-auto-retry (max_retries=5, idempotency_key=false)"}),
    ("FeatureFlag", {"id": "payment-auto-retry", "state": "on", "max_retries": 5, "idempotency_key": False,
                     "description": "automatically retry failed PayGate charge calls"}),
    ("FeatureFlag", {"id": "new-checkout-ui", "state": "on", "description": "redesigned checkout page"}),
    ("FeatureFlag", {"id": "fraud-score-v2", "state": "off", "description": "new fraud scoring model"}),
    ("ConfigChange", {"id": "cfg-311", "at": "2026-10-24T15:30:00Z", "key": "email_batch_size",
                      "old": "100", "new": "500", "ticket": "CHG-311", "reason": "faster marketing email sends"}),
    ("SecretRotation", {"id": "sec-22", "at": "2026-10-24T15:00:00Z", "secret": "paygate-api-key",
                        "by": "automation", "result": "ok, auth check returned 200"}),

    # ---- wave 3: external-world and operational signals
    ("VendorStatus", {"id": "vs-0", "at": "2026-10-01T00:00:00Z", "region": "us-east-1", "state": "healthy",
                      "p95_ms": 600}),
    ("VendorStatus", {"id": "vs-1", "at": "2026-10-24T15:25:00Z", "region": "eu-west-1", "state": "degraded",
                      "p95_ms": 4200, "summary": "PayGate eu-west-1 p95 latency 4.2s (was 0.8s), still returning 200"}),
    ("Alarm", {"id": "alarm-91", "metric": "duplicate_charge_rate_pct", "value": 0.9, "threshold": 0.05,
               "at": "2026-10-24T16:10:00Z"}),
    ("Alarm", {"id": "alarm-92", "metric": "payments_dlq_depth", "value": 5400, "threshold": 100,
               "at": "2026-10-24T16:20:00Z"}),
    ("Alarm", {"id": "alarm-93", "metric": "paygate_spend_usd_per_hour", "value": 131, "threshold": 60,
               "at": "2026-10-24T16:00:00Z"}),
    ("Incident", {"id": "INC-2001", "title": "Customers in the EU are being charged twice", "status": "open",
                  "opened_at": "2026-10-24T16:25:00Z", "at": "2026-10-24T16:25:00Z", "severity": "sev1"}),
    ("Incident", {"id": "INC-0650", "title": "Duplicate charges after enabling client retries", "status": "resolved",
                  "opened_at": "2026-03-14T10:00:00Z", "at": "2026-03-14T10:00:00Z",
                  "root_cause": "client retries enabled without an Idempotency-Key header on PayGate charges",
                  "fix": "send one Idempotency-Key per order and cap retries at 2",
                  "text": "Duplicate charges after enabling client retries. Root cause: automatic retries on PayGate "
                          "charge calls were enabled without an Idempotency-Key header, so every timeout that was "
                          "retried created a second real charge. Fix: send one Idempotency-Key per order, cap retries at 2."}),

    # ---- wave 4: documents (searchable text -> embeddings)
    ("Runbook", {"id": "RB-12", "title": "Duplicate charges",
                 "text": "Runbook RB-12 duplicate charges. 1. Disable the payment-auto-retry feature flag. 2. Confirm the "
                         "duplicates with the PayGate idempotency audit. 3. Refund duplicate charges. 4. Add an "
                         "Idempotency-Key header before re-enabling retries."}),
    ("Runbook", {"id": "RB-07", "title": "Rotate vendor API keys",
                 "text": "Runbook RB-07 rotating vendor API keys. Rotate the secret, run the auth check, watch for 401 "
                         "or 403 responses for 30 minutes, roll back the secret if authentication fails."}),
    ("Runbook", {"id": "RB-03", "title": "Database connection pool exhaustion",
                 "text": "Runbook RB-03 database pool exhaustion. Check max_connections, raise the pool size, "
                         "restart the waiting service, confirm p99 latency recovers."}),
    ("Runbook", {"id": "RB-15", "title": "Vendor degradation",
                 "text": "Runbook RB-15 vendor degradation. Set the client timeout to at least 1.5x the vendor p95, "
                         "enable a circuit breaker, never retry non-idempotent calls, and open a vendor ticket."}),
]

# customer tickets (text is embedded) -> (id, at, text, about, linked incident or None)
TICKETS = [
    ("T-9001", "2026-10-24T16:02:00Z", "I was charged twice for my order #55231, two identical 89.90 EUR charges on my card", "payment-service", "INC-2001"),
    ("T-9002", "2026-10-24T16:07:00Z", "Double charge on my credit card after checkout, please refund the extra payment", "payment-service", "INC-2001"),
    ("T-9003", "2026-10-24T16:12:00Z", "My bank app shows 3 pending payments from your shop for a single order", "payment-service", "INC-2001"),
    ("T-9004", "2026-10-24T16:15:00Z", "The payment page spun forever and afterwards I found two charges", "payment-service", "INC-2001"),
    ("T-9005", "2026-10-24T16:19:00Z", "Charged 2x for my subscription renewal in euros, please fix", "payment-service", "INC-2001"),
    ("T-9009", "2026-10-24T16:24:00Z", "Duplicate charge on my card for order #55302, I am in Germany", "payment-service", "INC-2001"),
    ("T-9006", "2026-10-24T15:40:00Z", "I cannot reset my password, the email never arrives", "notification-service", None),
    ("T-9007", "2026-10-24T08:50:00Z", "Checkout was extremely slow this morning", "checkout-api", "INC-1001"),
    ("T-9008", "2026-10-24T14:20:00Z", "Where can I download the invoice for order #54110?", "orders-service", None),
    ("T-9010", "2026-10-24T15:30:00Z", "The new checkout page layout is confusing, where is the coupon box?", "cart-service", None),
]

# log lines (text is embedded) -> (id, at, service, level, region, text)
LOGS = [
    ("log-01", "2026-10-24T09:12:00Z", "orders-service", "INFO", "eu-west-1", "orders-db pool size 100, active connections 31"),
    ("log-02", "2026-10-24T14:03:00Z", "payment-service", "INFO", "eu-west-1", "cfg-310 applied: gateway_timeout_ms=3000"),
    ("log-03", "2026-10-24T14:30:00Z", "payment-service", "INFO", "eu-west-1", "POST paygate/charge 200 in 812ms order=55001"),
    ("log-04", "2026-10-24T15:00:00Z", "payment-service", "INFO", "eu-west-1", "rotated PayGate API key sec-22, auth check 200 OK"),
    ("log-05", "2026-10-24T15:12:00Z", "cart-service", "INFO", "eu-west-1", "new-checkout-ui enabled, render time 120ms"),
    ("log-06", "2026-10-24T15:21:00Z", "payment-service", "WARN", "eu-west-1", "feature flag payment-auto-retry=ON max_retries=5 idempotency_key=false"),
    ("log-07", "2026-10-24T15:27:00Z", "payment-service", "WARN", "eu-west-1", "POST paygate/charge timeout after 3000ms (PayGate p95=4200ms) order=55120 attempt=1"),
    ("log-08", "2026-10-24T15:27:30Z", "payment-service", "WARN", "eu-west-1", "retrying charge order=55120 attempt=2 of 5 without Idempotency-Key header"),
    ("log-09", "2026-10-24T15:28:00Z", "payment-service", "ERROR", "eu-west-1", "PayGate answered 200 for order=55120 attempt=1 after 3900ms, late success ignored by client"),
    ("log-10", "2026-10-24T15:31:00Z", "notification-service", "INFO", "eu-west-1", "email_batch_size=500 applied"),
    ("log-11", "2026-10-24T16:04:00Z", "payment-service", "ERROR", "eu-west-1", "duplicate charge detected: order=55120 was charged 3 times by PayGate"),
    ("log-12", "2026-10-24T16:18:00Z", "payment-service", "ERROR", "eu-west-1", "payments-dlq depth=5400, jobs failed after 5 retries"),
    ("log-13", "2026-10-24T16:20:00Z", "payment-service", "INFO", "eu-west-1", "no 401 or 403 responses from PayGate in the last 2 hours"),
    ("log-14", "2026-10-24T16:30:00Z", "payment-service", "INFO", "us-east-1", "POST paygate/charge 200 in 590ms order=77410"),
]

# hourly metric samples for payment-service (eu-west-1): metric -> values for hours 12..17
_HOURS = [f"2026-10-24T{hh:02d}:00:00Z" for hh in range(12, 18)]
METRICS = {
    "gateway_timeouts": [3, 4, 6, 38, 2100, 2600],
    "retries": [0, 0, 0, 150, 9800, 11900],
    "duplicate_charges": [0, 0, 0, 11, 940, 1180],
    "p99_latency_ms": [700, 720, 710, 1500, 9100, 9800],
}
COSTS = {  # USD per hour
    "paygate-api-calls": [40, 41, 41, 58, 131, 152],
    "nat-egress": [6, 6, 6, 7, 19, 22],
}

# relationships: (label_a, key_a, REL, label_b, key_b)
RELS: list[tuple[str, str, str, str, str]] = [
    ("Service", "payment-service", "CALLS", "Vendor", "PayGate"),
    ("Service", "notification-service", "CALLS", "Vendor", "MailPost"),
    ("Service", "checkout-api", "DEPENDS_ON", "Service", "fraud-service"),
    ("Service", "payment-service", "PUBLISHES_TO", "Queue", "payments-retry-queue"),
    ("Service", "payment-service", "PUBLISHES_TO", "Queue", "payments-dlq"),
    ("Service", "payment-service", "RUNS_IN", "Region", "eu-west-1"),
    ("Service", "payment-service", "RUNS_IN", "Region", "us-east-1"),
    ("Service", "checkout-api", "RUNS_IN", "Region", "eu-west-1"),
    ("Service", "orders-service", "RUNS_IN", "Region", "eu-west-1"),
    ("Engineer", "eng-priya", "ON_CALL_FOR", "Service", "payment-service"),
    ("Engineer", "eng-marco", "AUTHORED", "ConfigChange", "cfg-310"),
    ("Engineer", "eng-lena", "AUTHORED", "Deployment", "deploy-490"),
    ("Engineer", "eng-lena", "AUTHORED", "Deployment", "deploy-489"),
    ("Deployment", "deploy-482", "CHANGED", "Database", "orders-db"),
    ("Deployment", "deploy-482", "DEPLOYED_TO", "Service", "orders-service"),
    ("Deployment", "deploy-489", "DEPLOYED_TO", "Service", "cart-service"),
    ("Deployment", "deploy-489", "ENABLED", "FeatureFlag", "new-checkout-ui"),
    ("Deployment", "deploy-490", "DEPLOYED_TO", "Service", "payment-service"),
    ("Deployment", "deploy-490", "ENABLED", "FeatureFlag", "payment-auto-retry"),
    ("FeatureFlag", "payment-auto-retry", "GATES", "Service", "payment-service"),
    ("FeatureFlag", "new-checkout-ui", "GATES", "Service", "cart-service"),
    ("FeatureFlag", "fraud-score-v2", "GATES", "Service", "fraud-service"),
    ("ConfigChange", "cfg-310", "CHANGED", "Service", "payment-service"),
    ("ConfigChange", "cfg-311", "CHANGED", "Service", "notification-service"),
    ("SecretRotation", "sec-22", "ROTATED_FOR", "Vendor", "PayGate"),
    ("SecretRotation", "sec-22", "ROTATED_FOR", "Service", "payment-service"),
    ("Vendor", "PayGate", "HAD_STATUS", "VendorStatus", "vs-0"),
    ("Vendor", "PayGate", "HAD_STATUS", "VendorStatus", "vs-1"),
    ("Alarm", "alarm-91", "FIRED_ON", "Service", "payment-service"),
    ("Alarm", "alarm-92", "FIRED_ON", "Queue", "payments-dlq"),
    ("Alarm", "alarm-93", "FIRED_ON", "Vendor", "PayGate"),
    ("Incident", "INC-2001", "AFFECTS", "Service", "payment-service"),
    ("Incident", "INC-2001", "AFFECTS", "Service", "checkout-api"),
    ("Incident", "INC-2001", "TRIGGERED_BY", "Alarm", "alarm-91"),
    ("Incident", "INC-0650", "AFFECTS", "Service", "payment-service"),
    ("Incident", "INC-0650", "ROOT_CAUSE", "FeatureFlag", "payment-auto-retry"),
    ("Runbook", "RB-12", "APPLIES_TO", "Service", "payment-service"),
    ("Runbook", "RB-07", "APPLIES_TO", "Vendor", "PayGate"),
    ("Runbook", "RB-03", "APPLIES_TO", "Database", "orders-db"),
    ("Runbook", "RB-15", "APPLIES_TO", "Vendor", "PayGate"),
]

# the story the notebooks should rediscover (never stored in the graph on purpose)
GROUND_TRUTH = {
    "incident": "INC-2001",
    "root_cause_change": "deploy-490",
    "root_cause_flag": "payment-auto-retry",
    "root_cause_config": "cfg-310",
    "trigger": "vs-1",
    "affected_region": "eu-west-1",
    "red_herrings": ["sec-22", "deploy-489", "deploy-482", "cfg-311", "orders-db"],
    "similar_incident": "INC-0650",
    "mitigation": "disable the payment-auto-retry feature flag",
    "fix": ["send an Idempotency-Key per order", "gateway timeout >= 1.5x vendor p95 (>= 6300 ms)",
            "cap retries / circuit breaker", "refund duplicate charges"],
    "summary": ("Retries were switched on without an idempotency key (deploy-490) at the same time the gateway timeout "
                "had been cut from 30s to 3s (cfg-310). When PayGate eu-west-1 slowed to 4.2s (vs-1) every charge "
                "timed out client-side but succeeded at PayGate, was retried, and charged the customer again."),
}

# deterministic gold set: question, keywords a correct answer must contain (any-of groups), reference for judges
GOLD_QA = [
    {"id": "q1", "input": "Which region is affected by INC-2001?", "must": [["eu-west-1"]],
     "reference": "eu-west-1 (Dublin). us-east-1 is healthy."},
    {"id": "q2", "input": "Which deployment turned on automatic retries for payments?",
     "must": [["deploy-490"], ["payment-auto-retry"]],
     "reference": "deploy-490 (payment-service v2.15.0) enabled the payment-auto-retry flag with 5 retries and no idempotency key."},
    {"id": "q3", "input": "Who changed the gateway timeout and to what value?",
     "must": [["cfg-310", "marco"], ["3000", "3 s", "3s", "3 seconds"]],
     "reference": "cfg-310 by Marco lowered gateway_timeout_ms from 30000 to 3000."},
    {"id": "q4", "input": "Was the PayGate secret rotation (sec-22) the cause?",
     "must": [["no", "not", "innocent", "unrelated", "ruled out"], ["401", "auth"]],
     "reference": "No. sec-22 succeeded (auth check 200) and there were no 401/403 responses; it is a red herring."},
    {"id": "q5", "input": "Which past incident is most similar to INC-2001?",
     "must": [["inc-0650"]],
     "reference": "INC-0650: duplicate charges after retries were enabled without an Idempotency-Key."},
    {"id": "q6", "input": "What is the fastest safe mitigation right now?",
     "must": [["disable", "turn off", "switch off"], ["payment-auto-retry", "retry", "retries"]],
     "reference": "Disable the payment-auto-retry feature flag (runbook RB-12), then refund duplicates."},
    {"id": "q7", "input": "Is orders-db or the connection pool implicated?",
     "must": [["no", "not", "unrelated", "innocent", "ruled out"], ["pool", "orders-db", "database"]],
     "reference": "No. The pool was restored to 100 by deploy-482 hours earlier and is healthy."},
    {"id": "q8", "input": "How much did hourly PayGate spend rise during the incident?",
     "must": [["41", "40"], ["152", "131"]],
     "reference": "From about $41/hour to $131-152/hour (roughly 3.5x) because retries multiply vendor calls."},
]


def keyword_score(answer: str, must: list[list[str]]) -> dict:
    """Deterministic check: every group in `must` needs at least one of its words in the answer."""
    low = (answer or "").lower()
    hit = [any(w.lower() in low for w in group) for group in must]
    return {"score": sum(hit) / len(hit), "passed": all(hit), "notes": f"groups hit {sum(hit)}/{len(hit)}"}


# --------------------------------------------------------------------------- loading
def _key(label: str) -> str:
    return "name" if label in NAME_KEYED else "id"


def counts_by_label() -> dict:
    return {r["l"]: r["c"] for r in h.run(
        "MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c ORDER BY c DESC")}


def load_changes() -> dict:
    """Add the new data to the CURRENT graph (idempotent). Returns {'before': ..., 'after': ..., 'added': ...}."""
    before = counts_by_label()
    n_before = h.run("MATCH (n) RETURN count(n) AS c")[0]["c"]
    r_before = h.run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    if not h.run("MATCH (s:Service {name:'checkout-api'}) RETURN s.name AS n"):
        raise RuntimeError("The base DevOps graph is missing - run the notebook's setup cell "
                           "(h.load_devops_graph()) before scn.load_changes().")
    for label, props in NODES:
        k = _key(label)
        h.run(f"MERGE (n:{label} {{{k}: $key}}) SET n += $props", key=props[k], props=props)
    for tid, at, text, about, inc in TICKETS:
        h.run("MERGE (t:Ticket {id:$id}) SET t.at=$at, t.text=$text, t.channel='support'", id=tid, at=at, text=text)
        h.run("MATCH (t:Ticket {id:$id}), (s:Service {name:$s}) MERGE (t)-[:ABOUT]->(s)", id=tid, s=about)
        if inc:
            h.run("MATCH (t:Ticket {id:$id}), (i:Incident {id:$i}) MERGE (t)-[:REPORTS]->(i)", id=tid, i=inc)
    for lid, at, svc, level, region, text in LOGS:
        h.run("MERGE (l:LogEvent {id:$id}) SET l.at=$at, l.level=$lv, l.text=$text", id=lid, at=at, lv=level, text=text)
        h.run("MATCH (l:LogEvent {id:$id}), (s:Service {name:$s}), (r:Region {name:$r}) "
              "MERGE (l)-[:LOGGED_BY]->(s) MERGE (l)-[:IN_REGION]->(r)", id=lid, s=svc, r=region)
    for metric, vals in METRICS.items():
        for at, v in zip(_HOURS, vals):
            mid = f"m-{metric}-{at[11:13]}"
            h.run("MERGE (m:MetricSample {id:$id}) SET m.metric=$metric, m.value=$v, m.at=$at, m.region='eu-west-1' "
                  "WITH m MATCH (s:Service {name:'payment-service'}) MERGE (m)-[:SAMPLE_OF]->(s)",
                  id=mid, metric=metric, v=v, at=at)
    for cat, vals in COSTS.items():
        for at, v in zip(_HOURS, vals):
            cid = f"cost-{cat}-{at[11:13]}"
            target = ("Vendor", "PayGate") if cat == "paygate-api-calls" else ("Service", "payment-service")
            h.run(f"MERGE (c:CostRecord {{id:$id}}) SET c.category=$cat, c.usd=$v, c.at=$at "
                  f"WITH c MATCH (t:{target[0]} {{name:$t}}) MERGE (c)-[:COST_OF]->(t)",
                  id=cid, cat=cat, v=float(v), at=at, t=target[1])
    for la, ka, rel, lb, kb in RELS:
        h.run(f"MATCH (a:{la} {{{_key(la)}: $ka}}), (b:{lb} {{{_key(lb)}: $kb}}) MERGE (a)-[:{rel}]->(b)", ka=ka, kb=kb)
    # the world moved on: the old incident was resolved by deploy-482
    h.run("MATCH (i:Incident {id:'INC-1001'}) SET i.at=i.opened_at, i.status='resolved', i.resolved_at='2026-10-24T09:15:00Z', "
          "i.root_cause='orders-db connection pool shrunk 100 -> 20 by deploy-481', i.fix='deploy-482 restored the pool to 100'")
    after = counts_by_label()
    n_after = h.run("MATCH (n) RETURN count(n) AS c")[0]["c"]
    r_after = h.run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    return {"nodes_before": n_before, "nodes_after": n_after, "rels_before": r_before, "rels_after": r_after,
            "new_labels": {k: v for k, v in after.items() if k not in before}, "labels_after": after}


def embed_new_documents() -> dict:
    """Embed the free-text node types and create their vector indexes (idempotent). Costs a few cents at most."""
    out = {}
    for label in ("Ticket", "LogEvent", "Runbook", "Incident"):
        h.ensure_vector_index(f"{label.lower()}_text_idx", label, "embedding")
        out[label] = h.embed_nodes(label, "text")
    return out


# --------------------------------------------------------------------------- reusable evidence (plain Cypher)
EVIDENCE = {
    "timeline": ("Everything that happened on or after a time, oldest first",
                 "MATCH (n) WHERE n.at >= $since AND any(l IN labels(n) WHERE l IN "
                 "['Deployment','ConfigChange','SecretRotation','Alarm','Incident','VendorStatus']) "
                 "RETURN n.at AS at, labels(n)[0] AS kind, n.id AS id, "
                 "coalesce(n.change, n.key + ': ' + n.old + ' -> ' + n.new, n.secret, n.metric, n.title, n.summary) AS what "
                 "ORDER BY at"),
    "changes_on_service": ("Every change that touched a service (deploys, config, flags)",
                           "MATCH (s:Service {name:$svc}) "
                           "OPTIONAL MATCH (d:Deployment)-[:DEPLOYED_TO]->(s) "
                           "OPTIONAL MATCH (c:ConfigChange)-[:CHANGED]->(s) "
                           "OPTIONAL MATCH (f:FeatureFlag)-[:GATES]->(s) "
                           "RETURN collect(DISTINCT {deploy:d.id, at:d.at, change:d.change}) AS deployments, "
                           "collect(DISTINCT {cfg:c.id, at:c.at, key:c.key, old:c.old, new:c.new}) AS configs, "
                           "collect(DISTINCT {flag:f.id, state:f.state}) AS flags"),
    "metric": ("Hourly series for one metric of payment-service",
               "MATCH (m:MetricSample {metric:$metric})-[:SAMPLE_OF]->(:Service {name:'payment-service'}) "
               "RETURN m.at AS at, m.value AS value ORDER BY at"),
    "cost": ("Hourly spend for one cost category",
             "MATCH (c:CostRecord {category:$cat}) RETURN c.at AS at, c.usd AS usd ORDER BY at"),
    "vendor": ("Vendor health history", "MATCH (:Vendor {name:$vendor})-[:HAD_STATUS]->(v:VendorStatus) "
               "RETURN v.at AS at, v.region AS region, v.state AS state, v.p95_ms AS p95_ms ORDER BY at"),
    "tickets": ("Tickets linked to an incident", "MATCH (t:Ticket)-[:REPORTS]->(:Incident {id:$inc}) "
                "RETURN t.id AS id, t.at AS at, t.text AS text ORDER BY at"),
    "logs": ("Log lines for a service in a window",
             "MATCH (l:LogEvent)-[:LOGGED_BY]->(:Service {name:$svc}) WHERE l.at >= $since AND l.at <= $until "
             "RETURN l.at AS at, l.level AS level, l.text AS text ORDER BY at"),
}


def q(name: str, **params):
    """Run a named EVIDENCE query: scn.q('timeline', since='2026-10-24T09:00:00Z')."""
    return h.run(EVIDENCE[name][1], **params)


def facts_text(limit_logs: int = 14, rich: bool = False) -> str:
    """A compact, LLM-ready dossier for INC-2001 built only from the graph (no ground truth leaks).

    rich=False : the first, naive dossier (timeline, metrics, spend, payment-service logs, tickets).
    rich=True  : adds vendor health per region and ALL services' logs with their region, so questions about
                 *where* and about *what is innocent* become answerable."""
    lines = ["TIMELINE (2026-10-24):"]
    for r in q("timeline", since="2026-10-24T08:00:00Z"):
        lines.append(f"- {r['at']} {r['kind']} {r['id']}: {r['what']}")
    lines.append("PAYMENT-SERVICE HOURLY (12:00..17:00 UTC):")
    for m in METRICS:
        lines.append(f"- {m}: " + ", ".join(str(r["value"]) for r in q("metric", metric=m)))
    lines.append("PAYGATE SPEND USD/HOUR: " + ", ".join(str(int(r["usd"])) for r in q("cost", cat="paygate-api-calls")))
    if rich:
        lines.append("PAYGATE HEALTH BY REGION:")
        for r in q("vendor", vendor="PayGate"):
            lines.append(f"- {r['at']} {r['region']}: {r['state']} p95={r['p95_ms']}ms")
        lines.append("LOGS (all services, with region):")
        for r in h.run("MATCH (l:LogEvent)-[:LOGGED_BY]->(s:Service), (l)-[:IN_REGION]->(g:Region) "
                       "WHERE l.at >= '2026-10-24T09:00:00Z' RETURN l.at AS at, l.level AS level, s.name AS svc, "
                       "g.name AS region, l.text AS text ORDER BY at"):
            lines.append(f"- {r['at']} {r['level']} [{r['svc']} / {r['region']}] {r['text']}")
    else:
        lines.append("LOGS:")
        for r in q("logs", svc="payment-service", since="2026-10-24T14:00:00Z", until="2026-10-24T17:00:00Z")[:limit_logs]:
            lines.append(f"- {r['at']} {r['level']} {r['text']}")
    lines.append("TICKETS:")
    for r in q("tickets", inc="INC-2001"):
        lines.append(f"- {r['id']}: {r['text']}")
    return "\n".join(lines)


def reset_new_data() -> None:
    """Delete everything this module added (labels only it creates). The base DevOps graph is kept."""
    h.run("MATCH (n) WHERE any(l IN labels(n) WHERE l IN ['Ticket','LogEvent','MetricSample','CostRecord','Runbook',"
          "'FeatureFlag','ConfigChange','SecretRotation','VendorStatus','Vendor','Queue','Region','Engineer']) DETACH DELETE n")
