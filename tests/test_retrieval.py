"""Explicit expected values: reusable for both the reference and learner code."""
import unittest
from labs.retrieval import reference


class RetrievalChecks(unittest.TestCase):
    api = reference

    def test_tf_weight_average_length(self):
        self.assertAlmostEqual(self.api.tf_weight(1, 10, 10), 1.0)
        self.assertAlmostEqual(self.api.tf_weight(5, 10, 10), 11 / 6.2)

    def test_tf_weight_length_adjustment(self):
        self.assertAlmostEqual(self.api.tf_weight(1, 20, 10), 2.2 / 3.1)
        self.assertAlmostEqual(self.api.tf_weight(1, 5, 10), 2.2 / 1.75)

    def test_tf_weight_no_length_adjustment(self):
        self.assertEqual(self.api.tf_weight(1, 10, 10, b=0),
                         self.api.tf_weight(1, 100, 10, b=0))

    def test_tf_weight_zero(self):
        self.assertEqual(self.api.tf_weight(0, 0, 10, k1=0), 0.0)
        self.assertEqual(self.api.tf_weight(2, 10, 10, k1=0), 1.0)

    def test_tf_weight_invalid(self):
        for args in [(-1, 10, 10), (11, 10, 10), (1, 10, 0),
                     (1, 10, float("inf")), (1.5, 10, 10), (True, 10, 10)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.api.tf_weight(*args)
        for kwargs in [{"k1": -1}, {"k1": float("nan")}, {"b": 2}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.api.tf_weight(1, 10, 10, **kwargs)

    def test_sparse_dot_shared_keys(self):
        q = {"car": 1.5, "automobile": .7, "repair": 1.3, "maintenance": .9}
        d = {"automobile": 1.4, "repair": .8, "maintenance": 1.2}
        self.assertAlmostEqual(self.api.sparse_dot(q, d), 3.1)

    def test_sparse_dot_no_cross_coordinate_similarity(self):
        self.assertEqual(self.api.sparse_dot({"car": 2}, {"automobile": 3}), 0)

    def test_sparse_dot_empty(self):
        self.assertEqual(self.api.sparse_dot({}, {}), 0)
        self.assertEqual(self.api.sparse_dot({}, {"car": 1}), 0)

    def test_sparse_dot_commutative(self):
        a, b = {"x": 2, "y": 4}, {"y": 3, "z": 5}
        self.assertEqual(self.api.sparse_dot(a, b), 12)
        self.assertEqual(self.api.sparse_dot(b, a), 12)

    def test_maxsim_axes(self):
        self.assertAlmostEqual(self.api.maxsim([[.9, .1, 0], [.1, .8, .2]]), 1.7)

    def test_maxsim_padding(self):
        self.assertAlmostEqual(self.api.maxsim([[-.2, -.5, 0]],
                                              valid_doc=[True, True, False]), -.2)

    def test_maxsim_duplicate_columns(self):
        self.assertEqual(self.api.maxsim([[.9, .1], [.1, .8]]),
                         self.api.maxsim([[.9, .1, .9], [.1, .8, .1]]))

    def test_maxsim_not_boolean_and(self):
        self.assertGreater(self.api.maxsim([[1], [1], [.1]]),
                           self.api.maxsim([[.65], [.65], [.65]]))

    def test_maxsim_invalid_matrix(self):
        for matrix in [[], [[]], [[1], [2, 3]], [[float("nan")]], [[float("inf")]]]:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                self.api.maxsim(matrix)

    def test_maxsim_invalid_mask(self):
        for mask in [[], [True, False], [False]]:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.api.maxsim([[.1]], valid_doc=mask)

    def test_rrf_agreement(self):
        ranked = self.api.rrf([["A", "B"], ["C", "B"]])
        self.assertEqual([x[0] for x in ranked], ["B", "A", "C"])
        self.assertAlmostEqual(ranked[0][1], 2 / 62)

    def test_rrf_one_based(self):
        self.assertEqual(self.api.rrf([["B", "A"]], c=0), [("B", 1), ("A", .5)])

    def test_rrf_duplicate_rejection(self):
        with self.assertRaises(ValueError):
            self.api.rrf([["A", "A"]])

    def test_rrf_empty(self):
        self.assertEqual(self.api.rrf([[], []]), [])

    def test_rrf_invalid_c(self):
        for c in [-1, float("nan"), float("inf")]:
            with self.subTest(c=c), self.assertRaises(ValueError):
                self.api.rrf([["A"]], c=c)

    def test_recall_at_k_not_hit(self):
        self.assertEqual(self.api.recall_at_k({"A", "B", "C", "D"}, ["X", "A", "Y"], 3), .25)

    def test_recall_at_k_duplicate_positions(self):
        self.assertEqual(self.api.recall_at_k({"A", "B"}, ["A", "A", "B"], 2), .5)

    def test_recall_at_k_boundaries(self):
        self.assertEqual(self.api.recall_at_k({"A"}, ["A"], 0), 0)
        self.assertEqual(self.api.recall_at_k({"A"}, [], 9), 0)
        self.assertEqual(self.api.recall_at_k({"A"}, ["A"], 99), 1)

    def test_recall_at_k_undefined(self):
        with self.assertRaises(ValueError):
            self.api.recall_at_k(set(), ["A"], 1)

    def test_recall_at_k_invalid_k(self):
        for k in [-1, 1.5, True]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.api.recall_at_k({"A"}, ["A"], k)

    @staticmethod
    def screening_example():
        return {"A": [[.95], [.05]], "B": [[.05], [.95]],
                "C": [[.80, 0], [0, .80]]}

    def test_candidate_union_misses_aggregate_winner(self):
        self.assertEqual(self.api.candidate_union(self.screening_example()), {"A", "B"})

    def test_candidate_union_wider_screening(self):
        self.assertEqual(self.api.candidate_union(self.screening_example(), per_query=2), {"A", "B", "C"})

    def test_candidate_union_tie_breaking(self):
        self.assertEqual(self.api.candidate_union({"B": [[.5]], "A": [[.5]]}), {"A"})

    def test_candidate_union_empty(self):
        self.assertEqual(self.api.candidate_union({}), set())

    def test_candidate_union_invalid(self):
        for k in [0, -1, 1.5, True]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.api.candidate_union(self.screening_example(), per_query=k)
        for data in [{"A": [[.1]], "B": [[.2], [.3]]}, {"A": []}, {"A": [[float("inf")]]}]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.api.candidate_union(data)
