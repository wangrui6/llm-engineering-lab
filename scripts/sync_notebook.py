"""Refresh the notebook's embedded reference/checks from their source files."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("tf_weight", "sparse_dot", "maxsim", "rrf", "recall_at_k", "candidate_union")


def expected_cells():
    reference = (ROOT / "labs/retrieval/reference.py").read_text(encoding="utf-8")
    reference += "\nfrom types import SimpleNamespace\nREFERENCE = SimpleNamespace(" + ", ".join(n + "=" + n for n in GROUPS) + ")\n"
    checks = (ROOT / "tests/test_retrieval.py").read_text(encoding="utf-8")
    checks = checks.replace("from labs.retrieval import reference", "# REFERENCE is defined by the previous cell.")
    checks = checks.replace("api = reference", "api = REFERENCE")
    return {"reference-setup": reference, "checks-setup": checks}


def main():
    path = ROOT / "notebooks/01_retrieval.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    replacements = expected_cells()
    for cell in notebook["cells"]:
        if cell.get("id") in replacements:
            cell["source"] = replacements[cell["id"]].splitlines(keepends=True)
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Notebook reference and checks synchronized.")

if __name__ == "__main__":
    main()
