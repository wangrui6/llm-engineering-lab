"""Grade the learner's selected functions using explicit expected values."""
import argparse
import importlib
import sys
import unittest
from tests.test_retrieval import RetrievalChecks

GROUPS = ("tf_weight", "sparse_dot", "maxsim", "rrf", "recall_at_k", "candidate_union")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("exercises", "reference"), default="exercises")
    parser.add_argument("--only", choices=GROUPS)
    args = parser.parse_args()
    module = importlib.import_module("labs.retrieval." + args.impl)
    checks = type("SelectedChecks", (RetrievalChecks,), {"api": module})
    names = unittest.defaultTestLoader.getTestCaseNames(checks)
    if args.only:
        names = [n for n in names if n.startswith("test_" + args.only + "_")]
    suite = unittest.TestSuite(checks(n) for n in names)
    print("Checking " + args.impl + (": " + args.only if args.only else ": all six exercises"), flush=True)
    if args.impl == "exercises":
        print("Unfinished functions raise NotImplementedError. That is expected until you implement them.", flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(main())
