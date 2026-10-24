"""Helpers for notebooks_intermediate_simplified/ (tier 2 of 3).

Reuses the advanced-tier helpers (skills, Neo4j, MiniAgent) and adds teaching aids that
fit the middle step: guided "your turn" exercises with a runnable reference solution.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "notebooks_advanced_simplified"))
sys.path.insert(0, str(HERE.parent / "notebooks_simplified"))

from adv_helpers import (  # noqa: E402,F401  (re-exported for notebooks)
    AgentTrace, MiniAgent, PROJECT_ROOT, SKILLS_DIR, check, clear_graph, connect,
    load_devops_graph, load_skill, run, run_benchmark, show, skill_catalog,
)


def your_turn(title: str, hint: str, solution, expected, *args, **kwargs):
    """Guided exercise: print the task + hint, run the reference solution, check the answer.

    `solution` is a callable. Learners are told to first try their own version, then compare.
    The notebook stays runnable end to end because the reference solution always executes.
    """
    print(f"YOUR TURN: {title}")
    print(f"  hint: {hint}")
    got = solution(*args, **kwargs)
    print(f"  reference answer: {got}")
    check(f"exercise '{title}' matches expected", got == expected)
    return got
