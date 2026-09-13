#!/usr/bin/env python3
"""Mechanics lab, not a neural-retrieval benchmark. Python 3.10+, standard library.

Run: python retrieval_scoring_lab.py

All sparse weights and similarity matrices below are hand-constructed. No model
is downloaded, trained, or evaluated. The BM25 convention uses unique query
terms and positive IDF; it is not a bit-for-bit clone of a production engine.
Sources and derivations are explained in the accompanying retrieval article.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import isclose, isfinite, log1p
from typing import Mapping, Sequence
import unittest


def bm25(query: Sequence[str], document: Sequence[str],
         corpus: Sequence[Sequence[str]], *, k1: float = 1.2,
         b: float = 0.75) -> float:
    """Conventional BM25. The document must belong to the supplied corpus."""
    if not corpus or not isfinite(k1) or k1 < 0 or not 0 <= b <= 1:
        raise ValueError("Need a nonempty corpus, finite k1 >= 0, and 0 <= b <= 1")
    if list(document) not in [list(d) for d in corpus]:
        raise ValueError("The scored document must belong to corpus")
    avg_length = sum(map(len, corpus)) / len(corpus)
    if avg_length == 0:
        return 0.0
    frequencies = Counter(document)
    result = 0.0
    for term in set(query):
        tf = frequencies[term]
        if tf == 0:
            continue
        df = sum(term in d for d in corpus)
        idf = log1p((len(corpus) - df + 0.5) / (df + 0.5))
        length = 1 - b + b * len(document) / avg_length
        result += idf * tf * (k1 + 1) / (tf + k1 * length)
    return result


def sparse_dot(query: Mapping[str, float], doc: Mapping[str, float]) -> float:
    return sum(weight * doc.get(term, 0.0) for term, weight in query.items())


def sparse_index_scores(query: Mapping[str, float],
                        documents: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Execute the same sparse dot product by traversing inverted postings."""
    postings: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for doc_id, vector in documents.items():
        for term, weight in vector.items():
            if weight:
                postings[term].append((doc_id, weight))
    scores = dict.fromkeys(documents, 0.0)
    for term, query_weight in query.items():
        for doc_id, doc_weight in postings.get(term, []):
            scores[doc_id] += query_weight * doc_weight
    return scores


def maxsim(matrix: Sequence[Sequence[float]], *,
           valid_doc: Sequence[bool] | None = None) -> float:
    """Sum row maxima: rows=query vectors, columns=document vectors.

    Mask columns BEFORE taking maxima. Reject an empty query/document instead
    of inventing a score. ColBERT query-augmentation vectors are not padding to
    discard; model-specific query masking is intentionally outside this lab.
    """
    if not matrix or not matrix[0]:
        raise ValueError("Need at least one query row and document column")
    n = len(matrix[0])
    if any(len(row) != n for row in matrix):
        raise ValueError("Ragged similarity matrix")
    if any(not isfinite(v) for row in matrix for v in row):
        raise ValueError("Similarity values must be finite")
    mask = list(valid_doc) if valid_doc is not None else [True] * n
    if len(mask) != n or not any(mask):
        raise ValueError("Need one mask value per column and a valid document token")
    return sum(max(v for v, keep in zip(row, mask) if keep) for row in matrix)


def rrf(rankings: Sequence[Sequence[str]], *, c: float = 60.0) -> list[tuple[str, float]]:
    """Fuse truncated, one-based ranked lists; missing documents contribute zero."""
    if not isfinite(c) or c < 0:
        raise ValueError("c must be finite and nonnegative")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("Deduplicate each input ranking before fusion")
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] += 1 / (c + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class MechanicsTests(unittest.TestCase):
    def test_bm25_saturation(self):
        # Padding equalizes document length, isolating term frequency.
        docs = [["target"] + ["filler"] * 9,
                ["target"] * 5 + ["filler"] * 5]
        ratio = bm25(["target"], docs[1], docs) / bm25(["target"], docs[0], docs)
        self.assertAlmostEqual(ratio, 11 / 6.2)

    def test_bm25_length_penalty(self):
        docs = [["target", "a"], ["target"] + ["a"] * 9]
        self.assertGreater(bm25(["target"], docs[0], docs),
                           bm25(["target"], docs[1], docs))
        self.assertAlmostEqual(bm25(["target"], docs[0], docs, b=0),
                               bm25(["target"], docs[1], docs, b=0))

    def test_sparse_dot_equals_postings(self):
        q = {"car": 1.5, "automobile": 0.7, "repair": 1.3, "maintenance": 0.9}
        docs = {"manual": {"automobile": 1.4, "repair": 0.8, "maintenance": 1.2},
                "racing": {"car": 2.0, "race": 1.1}}
        actual = sparse_index_scores(q, docs)
        self.assertAlmostEqual(actual["manual"], 3.10)
        for name, vector in docs.items():
            self.assertAlmostEqual(actual[name], sparse_dot(q, vector))

    def test_maxsim_axis_and_padding(self):
        matrix = [[0.9, 0.1, 0.0], [0.1, 0.8, 0.2]]
        self.assertAlmostEqual(maxsim(matrix), 1.7)
        self.assertAlmostEqual(maxsim([[-0.2, -0.5, 0.0]],
                                     valid_doc=[True, True, False]), -0.2)
        self.assertEqual(maxsim([[-0.2, -0.5, 0.0]]), 0.0)  # Unmasked padding bug.

    def test_topk_changes_preference(self):
        a, b = [0.95, 0.15, 0.10], [0.82, 0.79, 0.76]
        self.assertGreater(max(a), max(b))
        self.assertLess(sum(a) / 3, sum(b) / 3)

    def test_candidate_generation_can_miss_aggregate_winner(self):
        # All columns are feasible similarities to two orthogonal unit queries.
        scores = {"A": [[0.95], [0.05]], "B": [[0.05], [0.95]],
                  "C": [[0.80, 0.00], [0.00, 0.80]]}
        candidates = {max(scores, key=lambda name: max(scores[name][i])) for i in range(2)}
        winner = max(scores, key=lambda name: maxsim(scores[name]))
        self.assertEqual(candidates, {"A", "B"})
        self.assertEqual(winner, "C")

    def test_sum_sum_identity(self):
        q = [[1., 2.], [3., 4.]]
        d = [[2., -1.], [0., 3.], [1., 1.]]
        all_pairs = sum(sum(a * b for a, b in zip(qi, dj)) for qi in q for dj in d)
        pooled = sum(sum(qi[k] for qi in q) * sum(dj[k] for dj in d) for k in range(2))
        self.assertEqual(all_pairs, pooled)

    def test_rrf_consensus_and_duplicates(self):
        fused = rrf([["A", "B"], ["C", "B"]])
        self.assertEqual(fused[0][0], "B")
        self.assertAlmostEqual(fused[0][1], 2 / 62)
        with self.assertRaises(ValueError):
            rrf([["A", "A"]])

    def test_metric_distinction(self):
        relevant = {"A", "B", "C", "D"}
        retrieved = ["X", "A", "Y"]
        recall = len(set(retrieved) & relevant) / len(relevant)
        hit = int(bool(set(retrieved) & relevant))
        self.assertEqual(recall, 0.25)
        self.assertEqual(hit, 1)

    def test_compressed_payload(self):
        self.assertEqual(4 + 128 * 1 // 8, 20)
        self.assertEqual(4 + 128 * 2 // 8, 36)
        self.assertEqual(128 * 2, 256)  # FP16 baseline, excludes index metadata.

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            maxsim([])
        with self.assertRaises(ValueError):
            maxsim([[0.1]], valid_doc=[False])
        with self.assertRaises(ValueError):
            bm25(["x"], ["x"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
