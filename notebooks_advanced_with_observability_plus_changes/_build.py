"""Dev-only: convert percent-format .py sources in _src/ into .ipynb files (NOT executed by default).

    python notebooks_advanced/_build.py <src.py>            # writes unexecuted notebook
    python notebooks_advanced/_build.py <src.py> --check    # also compile-checks every code cell

These notebooks need OPENAI_API_KEY, so building never runs them.
"""
import ast
import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks_advanced_simplified"))
from _build import parse  # noqa: E402


def main():
    src = Path(sys.argv[1])
    cells = parse(src.read_text(encoding="utf-8"))
    if "--check" in sys.argv:
        for i, c in enumerate(cells):
            if c.cell_type == "code":
                code = "\n".join(l for l in c.source.split("\n") if not l.lstrip().startswith(("%", "!")))
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    sys.exit(f"{src.name}: cell {i} syntax error: {e}")
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = src.parent.parent / (src.stem + ".ipynb")
    nbformat.write(nb, out)
    print("wrote", out, f"({len(cells)} cells)")


if __name__ == "__main__":
    main()
