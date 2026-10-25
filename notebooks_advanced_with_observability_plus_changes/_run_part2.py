"""Dev-only: execute ONLY the "Part 2" section of a notebook source, in a fresh kernel, against real OpenAI + Neo4j.

    python notebooks_advanced_with_observability_plus_changes/_run_part2.py ch1-defining-agentic-ai

Why: the full notebooks are long and cost real money; Part 2 is self-contained (it only needs `h`, `scn` and the base
DevOps graph), so you can iterate on it cheaply. It does NOT write any notebook.
Part 2 is the block that starts at the first markdown line `# ## Part 2` and ends at the next `# ## ` heading
(use `# ### ` for sub-headings inside Part 2).
"""
import os
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "notebooks_advanced_simplified"))
from _build import parse  # noqa: E402


def part2_source(text: str) -> str:
    lines = text.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("# ## Part 2"))
    # include the "# %% [markdown]" line that precedes the heading
    if start and lines[start - 1].startswith("# %%"):
        start -= 1
    end = len(lines)
    for i in range(start + 2, len(lines)):
        if lines[i].startswith("# ## ") and not lines[i].startswith("# ## Part 2"):
            end = i
            while end > start and not lines[end - 1].strip():
                end -= 1
            if end > start and lines[end - 1].startswith("# %%"):
                end -= 1
            break
    return "\n".join(lines[start:end])


def main():
    stem = sys.argv[1].replace(".ipynb", "").replace(".py", "")
    src = (HERE / "_src" / f"{stem}.py").read_text(encoding="utf-8")
    cells = parse(part2_source(src))
    pre = nbformat.v4.new_code_cell(
        'import sys; sys.path.insert(0, "."); import real_helpers as h\n'
        f'h.obs_start("{stem}-part2-dev")\nprint(h.load_devops_graph())')
    nb = nbformat.v4.new_notebook(cells=[pre] + cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    lock = HERE / ".run.lock"
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL)); break
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > 1800:
                lock.unlink(missing_ok=True)
            time.sleep(3)
    t0 = time.time()
    try:
        NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=True,
                       resources={"metadata": {"path": str(HERE)}}).execute()
    finally:
        lock.unlink(missing_ok=True)
    errs, passes, fails, cost = [], 0, 0, ""
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errs.append((i, o["ename"], o["evalue"][:400]))
            txt = o.get("text", "") if o.get("output_type") == "stream" else ""
            passes += len(re.findall(r"\[PASS\]", txt)); fails += len(re.findall(r"\[FAIL\]", txt))
            for l in txt.splitlines():
                if l.startswith("LLM calls="):
                    cost = l
    print(f"{stem} part2: {time.time() - t0:.0f}s cells={len(cells)} error_cells={len(errs)} PASS={passes} FAIL={fails}")
    print("LAST COST LINE:", cost)
    for i, en, ev in errs:
        print(f"  cell {i}: {en}: {ev}")
    if "--show" in sys.argv:
        for i, c in enumerate(nb.cells):
            if c.cell_type == "code":
                for o in c.get("outputs", []):
                    if o.get("output_type") == "stream":
                        print(f"--- cell {i}\n{o['text'][:1500]}")


if __name__ == "__main__":
    main()
