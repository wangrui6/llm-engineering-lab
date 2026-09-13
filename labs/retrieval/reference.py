"""Reference scoring functions, not trained neural retrievers.

Sparse weights and token similarities are supplied by the caller. No model,
network connection, API key, GPU, or third-party package is used.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import isfinite, log1p
from typing import Mapping, Sequence


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



def tf_weight(tf: int, length: int, avg_length: float, *,
              k1: float = 1.2, b: float = 0.75) -> float:
    """BM25's length-adjusted TF factor, excluding IDF.

    Require integer 0 <= tf <= length, avg_length > 0, finite k1 >= 0,
    and finite 0 <= b <= 1. Empty documents score zero. This example
    uses analyzed-token lengths, not characters or bytes.
    """
    if (not isinstance(tf, int) or isinstance(tf, bool)
            or not isinstance(length, int) or isinstance(length, bool)
            or tf < 0 or length < tf
            or not isfinite(avg_length) or avg_length <= 0
            or not isfinite(k1) or k1 < 0
            or not isfinite(b) or not 0 <= b <= 1):
        raise ValueError("Invalid TF, length, average length, k1, or b")
    if tf == 0:
        return 0.0
    correction = 1 - b + b * length / avg_length
    return tf * (k1 + 1) / (tf + k1 * correction)


def recall_at_k(relevant: set[str], ranking: Sequence[str], k: int) -> float:
    """Fraction of relevant IDs found among the first k positions.

    Duplicate results occupy positions but count once. For an empty relevant
    set, recall is undefined: raise ValueError rather than invent a perfect
    or zero score. Evaluate unanswerable queries with a separate metric.
    """
    if not relevant:
        raise ValueError("Recall requires at least one relevant document")
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def candidate_union(matrices: Mapping[str, Sequence[Sequence[float]]], *,
                    per_query: int = 1) -> set[str]:
    """Union the best documents for each query-vector row.

    This is an exact toy screening rule, NOT PLAID or an ANN implementation.
    Each document first contributes its best token score for that query row;
    choose per_query documents, with alphabetical IDs breaking ties. Then
    union their IDs. All matrices have the same nonzero number of query rows.
    Values must be finite; columns need not have equal counts across docs.
    """
    if not isinstance(per_query, int) or isinstance(per_query, bool) or per_query <= 0:
        raise ValueError("per_query must be a positive integer")
    if not matrices:
        return set()
    rows = len(next(iter(matrices.values())))
    for matrix in matrices.values():
        maxsim(matrix)  # Validate nonempty rectangular matrices and finite values.
        if len(matrix) != rows:
            raise ValueError("All documents need the same query-row count")
    chosen: set[str] = set()
    for i in range(rows):
        ordered = sorted(matrices, key=lambda name: (-max(matrices[name][i]), name))
        chosen.update(ordered[:per_query])
    return chosen
