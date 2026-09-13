"""Run five small experiments. All data and similarities are constructed."""
from .reference import tf_weight, sparse_dot, maxsim, candidate_union, rrf, recall_at_k


def main():
    print("RETRIEVAL MECHANICS — constructed examples, not model benchmarks")
    print("\n1. BM25 TF saturation (same document length)")
    for tf in (1, 5, 10):
        print(f"   TF={tf:2d} -> multiplier {tf_weight(tf, 10, 10):.4f}")
    print("\n2. Expanded lexical coordinates")
    q = {"car": 1.5, "automobile": .7, "repair": 1.3, "maintenance": .9}
    d = {"automobile": 1.4, "repair": .8, "maintenance": 1.2}
    print(f"   Sparse dot product = {sparse_dot(q, d):.2f}")
    print("   These weights are supplied by us, not produced by SPLADE.")
    print("\n3. A zero padding value can win a maximum")
    matrix = [[-.2, -.5, 0]]
    print(f"   Unmasked = {maxsim(matrix):.1f}; masked = {maxsim(matrix, valid_doc=[True, True, False]):.1f}")
    print("\n4. Candidate screening can miss the best MaxSim document")
    docs = {"A": [[.95], [.05]], "B": [[.05], [.95]], "C": [[.8, 0], [0, .8]]}
    print("   Per-query Top-1 union:", ", ".join(sorted(candidate_union(docs))))
    print("   Full scores:", ", ".join(f"{name}={maxsim(m):.2f}" for name, m in docs.items()))
    print("   Increasing per_query to 2 includes C; it does not change the scoring function.")
    print("\n5. Fusion is not a metric")
    print("   RRF of [A,B] and [C,B]:", [(d, round(s, 4)) for d, s in rrf([["A", "B"], ["C", "B"]])])
    print(f"   One of four relevant documents found: recall = {recall_at_k({'A','B','C','D'}, ['X','A','Y'], 3):.2f}, Hit@3 = 1")
    print("\nNext: edit labs/retrieval/exercises.py, then run python -m labs.retrieval.check --only tf_weight")

if __name__ == "__main__":
    main()
