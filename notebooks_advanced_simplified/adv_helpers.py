"""Helpers for notebooks_advanced_simplified/ (builds on ../notebooks_simplified/graph_helpers.py).

Adds the pieces that make the advanced notebooks feel like real agent engineering:
  * skill_catalog()      - read every skills/*/*/SKILL.md frontmatter (the repo's "table of contents")
  * run_benchmark()      - run a skill's own `cli.py benchmark` self-check from a notebook
  * AgentTrace           - log every tool call of a mini agent run into Neo4j as a graph
  * MiniAgent            - a ~30-line plan->act->observe loop whose "tools" are repo skills + Cypher
  * check()              - tiny assertion helper that prints PASS/FAIL lines
Everything is stdlib + the neo4j driver. No LLM / API key needed.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "notebooks_simplified"))

from graph_helpers import (  # noqa: E402,F401  (re-exported for notebooks)
    PROJECT_ROOT, SKILLS_DIR, clear_graph, connect, load_devops_graph,
    load_skill, run, show,
)


def check(name: str, ok: bool) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert ok, f"check failed: {name}"
    return ok


def skill_catalog(chapter_dir: str | None = None) -> list[dict]:
    """Return [{chapter, slug, description}] parsed from each SKILL.md frontmatter."""
    rows = []
    for md in sorted(SKILLS_DIR.glob("*/*/SKILL.md")):
        chapter, slug = md.parent.parent.name, md.parent.name
        if chapter_dir and chapter != chapter_dir:
            continue
        text = md.read_text(encoding="utf-8")
        m = re.search(r"^description:\s*\|?\s*\n?(.*?)^\w[\w-]*:", text, re.S | re.M)
        desc = " ".join((m.group(1) if m else "").split())
        rows.append({"chapter": chapter, "slug": slug, "description": desc[:200]})
    return rows


def run_benchmark(chapter_dir: str, slug: str) -> tuple[bool, str]:
    """Run `python skills/<chapter>/<slug>/cli.py benchmark` and return (passed, last line)."""
    cli = SKILLS_DIR / chapter_dir / slug / "cli.py"
    sub = "prompt-budget" if slug == "mcp-gateway-two-meta-tools" else "benchmark"
    p = subprocess.run([sys.executable, str(cli), sub], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    lines = [l for l in (p.stdout + p.stderr).strip().splitlines() if l.strip()]
    lines = [l for l in lines if not set(l) <= {"="}]
    return p.returncode == 0, (lines[0] if lines else "")


class AgentTrace:
    """Persist an agent run as (:Run)-[:STEP]->(:ToolCall)-[:NEXT]->(:ToolCall) in Neo4j."""

    def __init__(self, goal: str):
        self.run_id = f"run-{uuid.uuid4().hex[:8]}"
        self.prev = None
        self.n = 0
        run("CREATE (:Run {id:$id, goal:$goal, started:$t})", id=self.run_id, goal=goal, t=time.time())

    def log(self, tool: str, args: dict, result_summary: str, ok: bool = True) -> None:
        self.n += 1
        cid = f"{self.run_id}-{self.n}"
        run("MATCH (r:Run {id:$rid}) "
            "CREATE (c:ToolCall {id:$cid, seq:$n, tool:$tool, args:$args, result:$res, ok:$ok}) "
            "CREATE (r)-[:STEP]->(c)", rid=self.run_id, cid=cid, n=self.n, tool=tool,
            args=str(args)[:200], res=result_summary[:200], ok=ok)
        if self.prev:
            run("MATCH (a:ToolCall {id:$a}), (b:ToolCall {id:$b}) CREATE (a)-[:NEXT]->(b)",
                a=self.prev, b=cid)
        self.prev = cid

    def timeline(self) -> list[dict]:
        return run("MATCH (:Run {id:$id})-[:STEP]->(c) RETURN c.seq AS seq, c.tool AS tool, "
                   "c.ok AS ok, c.result AS result ORDER BY c.seq", id=self.run_id)


class MiniAgent:
    """A deliberately tiny agent loop: for each planned step, call a registered tool, log it.

    The point is to show the *shape* of an agent (plan -> act -> observe -> log) with
    deterministic 'tools' (skills, Cypher). Swap `plan` for an LLM call to make it real.
    """

    def __init__(self, goal: str):
        self.goal = goal
        self.tools: dict = {}
        self.trace = AgentTrace(goal)
        self.memory: dict = {}

    def tool(self, name: str):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco

    def act(self, name: str, **args):
        try:
            out = self.tools[name](**args)
            self.trace.log(name, args, str(out), ok=True)
            self.memory[name] = out
            return out
        except Exception as exc:  # the trace records failures too
            self.trace.log(name, args, f"ERROR {exc}", ok=False)
            raise
