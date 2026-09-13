"""Execute our own notebook's ordinary Python cells in order, without Jupyter.

This is intentionally not a safe runner for untrusted notebooks. Review code
before executing it. The project notebook has no shell magics or downloads.
"""
from pathlib import Path
import contextlib
import io
import json
from sync_notebook import expected_cells

ROOT = Path(__file__).resolve().parents[1]

def main():
    notebook = json.loads((ROOT / "notebooks/01_retrieval.ipynb").read_text(encoding="utf-8"))
    expected = expected_cells()
    ids = {c.get("id") for c in notebook["cells"]}
    if not set(expected).issubset(ids):
        raise AssertionError("Notebook setup cell missing")
    scope = {"__name__": "notebook_validation"}
    output = io.StringIO()
    count = 0
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell.get("id") in expected and source != expected[cell["id"]]:
                raise AssertionError("Run python scripts/sync_notebook.py before committing")
            if cell["cell_type"] == "code":
                exec(compile(source, "notebook:" + cell["id"], "exec"), scope)
                count += 1
    results = scope.get("NOTEBOOK_RESULTS", {})
    names = {"tf_weight", "sparse_dot", "maxsim", "rrf", "recall_at_k", "candidate_union"}
    if set(results) != names or any(v not in ("TODO", "PASS") for v in results.values()):
        raise AssertionError(output.getvalue())
    # Also grade the embedded reference through the notebook's learner checker.
    for name in names:
        scope[name] = getattr(scope["REFERENCE"], name)
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            if scope["check_exercise"](name) != "PASS":
                raise AssertionError(output.getvalue())
    print(f"Notebook OK: {count} code cells executed; all six reference exercise groups pass.")
    print("Starter TODOs are intentional. Colab hosting and GitHub CI are not tested by this script.")

if __name__ == "__main__":
    main()
