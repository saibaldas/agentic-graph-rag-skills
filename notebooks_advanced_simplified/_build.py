"""Dev-only: convert percent-format .py sources into executed .ipynb files.

    python notebooks_simplified/_build.py <src.py> [--no-run]

Source format: cells start with `# %% [markdown]` (body lines prefixed `# `)
or `# %%` (code).
"""
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def parse(text: str):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
            cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.split("\n"):
        if line.startswith("# %% [markdown]"):
            flush(); kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush(); kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def main():
    src = Path(sys.argv[1])
    nb = nbformat.v4.new_notebook(cells=parse(src.read_text(encoding="utf-8")))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = src.parent.parent / (src.stem + ".ipynb")
    if "--no-run" not in sys.argv:
        # All notebooks share ONE Neo4j database and wipe it on start, so serialize runs.
        lock = Path(__file__).parent.parent / "notebooks_simplified" / ".run.lock"
        while True:
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL)); break
            except FileExistsError:
                time.sleep(2)
        try:
            NotebookClient(nb, timeout=300, kernel_name="python3",
                           resources={"metadata": {"path": str(out.parent)}}).execute()
        finally:
            lock.unlink(missing_ok=True)
    nbformat.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
