"""Shared helpers for the notebooks in notebooks_simplified/.

Keeps the notebooks short: connect to Neo4j, load the DevOps incident graph,
run Cypher and print results as tables, and load a skill's lib.py.

Connection settings come from environment variables, defaulting to the
docker-compose/docker-compose.yml values:
    NEO4J_URI (bolt://localhost:7687)  NEO4J_USER (neo4j)  NEO4J_PASSWORD (password)
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

_driver = None


def connect(wait_seconds: int = 90):
    """Return a Neo4j driver, waiting for the database to finish starting."""
    global _driver
    if _driver is not None:
        return _driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    deadline = time.time() + wait_seconds
    while True:
        try:
            driver.execute_query("RETURN 1")
            break
        except Exception as exc:  # DB still starting, or not running at all
            if time.time() > deadline:
                raise RuntimeError(
                    "Cannot reach Neo4j. Start it with:\n"
                    "  docker compose -f docker-compose/docker-compose.yml up -d\n"
                    f"(tried {NEO4J_URI} as {NEO4J_USER}). Original error: {exc}"
                ) from exc
            time.sleep(3)
    _driver = driver
    return driver


def run(cypher: str, **params) -> list[dict]:
    """Run one Cypher query and return rows as plain dicts."""
    records, _, _ = connect().execute_query(cypher, parameters_=params)
    return [r.data() for r in records]


def show(rows: list[dict], max_rows: int = 25) -> None:
    """Print rows as a simple aligned text table."""
    if not rows:
        print("(no rows)")
        return
    cols = list(rows[0].keys())
    cells = [[str(r.get(c, "")) for c in cols] for r in rows[:max_rows]]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    if len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} more rows")


def clear_graph() -> None:
    """Delete every node and relationship (notebooks start from a clean slate)."""
    run("MATCH (n) DETACH DELETE n")


# The running scenario of the whole book: the checkout API is slow.
DEVOPS_GRAPH_CYPHER = """
CREATE (checkout:Service {name:'checkout-api',   tier:'critical', owner:'team-payments'})
CREATE (payment:Service  {name:'payment-service', tier:'critical', owner:'team-payments'})
CREATE (orders:Service   {name:'orders-service',  tier:'high',     owner:'team-orders'})
CREATE (cart:Service     {name:'cart-service',    tier:'medium',   owner:'team-orders'})
CREATE (db:Database      {name:'orders-db',       engine:'postgres', max_connections:100})
CREATE (cache:Cache      {name:'session-cache',   engine:'redis'})
CREATE (cart)-[:DEPENDS_ON]->(checkout)
CREATE (checkout)-[:DEPENDS_ON]->(payment)
CREATE (checkout)-[:DEPENDS_ON]->(orders)
CREATE (orders)-[:DEPENDS_ON]->(db)
CREATE (payment)-[:DEPENDS_ON]->(db)
CREATE (checkout)-[:DEPENDS_ON]->(cache)
CREATE (dep:Deployment {id:'deploy-481', version:'v2.14.0', at:'2026-10-24T08:30:00Z',
                        change:'db connection pool 100 -> 20'})
CREATE (dep)-[:CHANGED]->(db)
CREATE (dep)-[:DEPLOYED_TO]->(orders)
CREATE (alarm:Alarm {id:'alarm-77', metric:'p99_latency_ms', value:2400, threshold:800,
                     at:'2026-10-24T08:41:00Z'})
CREATE (alarm)-[:FIRED_ON]->(checkout)
CREATE (inc:Incident {id:'INC-1001', title:'Checkout API is slow', status:'open',
                      opened_at:'2026-10-24T08:42:00Z'})
CREATE (inc)-[:AFFECTS]->(checkout)
CREATE (inc)-[:TRIGGERED_BY]->(alarm)
CREATE (old:Incident {id:'INC-0877', title:'Checkout latency after pool change', status:'resolved',
                      opened_at:'2026-06-02T11:00:00Z', root_cause:'db connection pool too small',
                      fix:'raise pool size to 100'})
CREATE (old)-[:AFFECTS]->(checkout)
CREATE (old)-[:ROOT_CAUSE]->(db)
"""


def load_devops_graph(reset: bool = True) -> dict:
    """Create the shared DevOps incident graph in Neo4j and return node/edge counts."""
    if reset:
        clear_graph()
    run(DEVOPS_GRAPH_CYPHER)
    return {
        "nodes": run("MATCH (n) RETURN count(n) AS c")[0]["c"],
        "relationships": run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"],
    }


def load_skill(chapter_dir: str, slug: str, alias: str | None = None):
    """Import skills/<chapter_dir>/<slug>/lib.py under a unique module name.

    Every skill's file is called lib.py, so a plain `import lib` would collide;
    loading by file path with an alias avoids that.
    """
    path = SKILLS_DIR / chapter_dir / slug / "lib.py"
    alias = alias or slug.replace("-", "_") + "_lib"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module
