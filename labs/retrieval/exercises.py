"""Fill these six functions. Keep their signatures unchanged.

Run one group: python -m labs.retrieval.check --only maxsim
Run all:       python -m labs.retrieval.check
Read docs/exercises.md for inputs, edge cases, and hints.
Do not call the reference implementation from a solution.
"""
from __future__ import annotations
from typing import Mapping, Sequence


def tf_weight(tf: int, length: int, avg_length: float, *,
              k1: float = 1.2, b: float = 0.75) -> float:
    """Exercise 1: BM25 length-adjusted TF. Validate the documented inputs."""
    raise NotImplementedError("Start with Exercise 1 in docs/exercises.md")


def sparse_dot(query: Mapping[str, float], doc: Mapping[str, float]) -> float:
    """Exercise 2: multiply matching vocabulary coordinates, then sum."""
    raise NotImplementedError("Iterate over query coordinates; missing weights are zero")


def maxsim(matrix: Sequence[Sequence[float]], *,
           valid_doc: Sequence[bool] | None = None) -> float:
    """Exercise 3: rows=query, columns=document. Exclude padding before max."""
    raise NotImplementedError("Reduce document columns before summing query rows")


def rrf(rankings: Sequence[Sequence[str]], *, c: float = 60.0) -> list[tuple[str, float]]:
    """Exercise 4: reciprocal ranks; reject duplicates within a list."""
    raise NotImplementedError("Ranks start at one; sort ties by document ID")


def recall_at_k(relevant: set[str], ranking: Sequence[str], k: int) -> float:
    """Exercise 5: relevant-set recall, not the probability of at least one hit."""
    raise NotImplementedError("Count unique relevant IDs within the first k positions")


def candidate_union(matrices: Mapping[str, Sequence[Sequence[float]]], *,
                    per_query: int = 1) -> set[str]:
    """Exercise 6: per-query-vector screening can miss the best MaxSim document."""
    raise NotImplementedError("Choose top documents per query row, then union IDs")
