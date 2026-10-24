"""Dev-only builder. Two modes:

  python _build.py <src.py>            percent-format source -> executed .ipynb (one folder up from _src)
  python _build.py --banner <tier> <folder>   insert/refresh the 'learning path' banner as the
                                              first cell of every ch*/spike* .ipynb in <folder>
                                              and in its _src/*.py (no re-execution needed)
"""
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

TIERS = {
    "beginner": ("notebooks_simplified", 1, "Beginner"),
    "intermediate": ("notebooks_intermediate_simplified", 2, "Intermediate"),
    "advanced": ("notebooks_advanced_simplified", 3, "Advanced"),
}
ORDER = ["beginner", "intermediate", "advanced"]
MARK = "<!-- learning-path-banner -->"


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


def banner(tier: str, stem: str) -> str:
    lines = [MARK, "> **Learning path** — three tiers, same chapters, same skills, same Neo4j:  "]
    parts = []
    for t in ORDER:
        folder, n, label = TIERS[t]
        name = f"{label} (tier {n}/3)"
        parts.append(f"**{name} ← you are here**" if t == tier else f"[{name}](../{folder}/{stem}.ipynb)")
    lines.append("> " + "  →  ".join(parts))
    return "\n".join(lines)


def apply_banner(tier: str, folder: Path):
    root = folder.parent
    for nbp in sorted(folder.glob("*.ipynb")):
        stem = nbp.stem
        siblings = {t: (root / TIERS[t][0] / f"{stem}.ipynb").exists() for t in ORDER}
        text = banner(tier, stem)
        # only link tiers that really have this notebook
        for t in ORDER:
            if t != tier and not siblings[t]:
                folder_t = TIERS[t][0]
                text = "\n".join(l for l in text.split("\n"))
                text = text.replace(f"[{TIERS[t][2]} (tier {TIERS[t][1]}/3)](../{folder_t}/{stem}.ipynb)",
                                    f"{TIERS[t][2]} (tier {TIERS[t][1]}/3, no notebook for this topic)")
        nb = nbformat.read(nbp, as_version=4)
        if nb.cells and nb.cells[0].source.startswith(MARK):
            nb.cells[0].source = text
        else:
            nb.cells.insert(0, nbformat.v4.new_markdown_cell(text))
        nbformat.write(nb, nbp)
        src = folder / "_src" / f"{stem}.py"
        if src.exists():
            s = src.read_text(encoding="utf-8")
            block = "# %% [markdown]\n" + "\n".join("# " + l for l in text.split("\n")) + "\n\n"
            if s.startswith("# %% [markdown]\n# " + MARK):
                s = s[s.index("\n\n") + 2:]
            src.write_text(block + s, encoding="utf-8")
        print("banner:", nbp.name)


def main():
    if sys.argv[1] == "--banner":
        apply_banner(sys.argv[2], Path(sys.argv[3]).resolve())
        return
    src = Path(sys.argv[1])
    nb = nbformat.v4.new_notebook(cells=parse(src.read_text(encoding="utf-8")))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = src.parent.parent / (src.stem + ".ipynb")
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
