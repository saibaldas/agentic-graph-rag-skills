"""Dev-only: execute a built notebook against real OpenAI + Neo4j and report every failing cell.

    python notebooks_advanced/_run.py ch1-defining-agentic-ai [--save]

Runs with allow_errors so one pass lists ALL failures. --save writes the executed notebook (with outputs) back.
Notebooks share one Neo4j and reset it on start: run one at a time.
"""
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent


def main():
    stem = sys.argv[1].replace(".ipynb", "")
    path = HERE / f"{stem}.ipynb"
    nb = nbformat.read(path, as_version=4)
    lock = HERE / ".run.lock"  # one shared Neo4j: serialize runs across processes
    while True:
        try:
            import os
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
    cost = ""
    for c in nb.cells:
        for o in c.get("outputs", []) if c.cell_type == "code" else []:
            for l in (o.get("text", "") if o.get("output_type") == "stream" else "").splitlines():
                if l.startswith("LLM calls="):
                    cost = l
    errs, fails, passes = [], 0, 0
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errs.append((i, o["ename"], o["evalue"][:300]))
            txt = o.get("text", "") if o.get("output_type") == "stream" else ""
            passes += len(re.findall(r"\[PASS\]", txt)); fails += len(re.findall(r"\[FAIL\]", txt))
    print(f"{stem}: {time.time() - t0:.0f}s  error_cells={len(errs)}  PASS={passes}  FAIL={fails}")
    print("LAST COST LINE:", cost)
    for i, en, ev in errs:
        print(f"  cell {i}: {en}: {ev}")
    if "--save" in sys.argv:
        nbformat.write(nb, path)
        print("saved", path.name)


if __name__ == "__main__":
    main()
