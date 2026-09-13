# LLM Engineering Lab

**Implement the mechanism. Predict the result. Find the failure case.**

The first module teaches retrieval through six small Python exercises: BM25 weighting, sparse scoring, ColBERT-style MaxSim, rank fusion, recall, and candidate screening. Each has a precise contract, tests, hints, and a reference solution.

**Python 3.10+ · Standard library only · No API keys · No GPU · No model downloads**

[Start the exercises](docs/exercises.md) · [Read the mechanisms](docs/retrieval.md) · [Open the notebook](notebooks/01_retrieval.ipynb) · [Sources](docs/sources.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wangrui6/llm-engineering-lab/blob/main/notebooks/01_retrieval.ipynb)

## Start here

Clone this repository, or choose **Code → Download ZIP** and unzip it. From its root:

```bash
python3 -m labs.retrieval.demo
python3 -m unittest discover -s tests -v
```

Use `python` instead of `python3` when that is your Python 3 command. On Windows, `py -3` is another option. The exercises need no installation step. The notebook is self-contained and can also be uploaded directly to Colab; it does not clone code or contact model APIs.

The demo includes a candidate-screening failure:

```text
Per-query Top-1 union: A, B
Full scores: A=1.00, B=1.00, C=1.60
```

C is the best document under the aggregate score. It never reaches that scorer when the candidate selector only returns A and B. Changing the scorer cannot repair that missing candidate.

## Practice, do not just run the answers

Open `labs/retrieval/exercises.py`. Complete one function, then run its checks:

```bash
python3 -m labs.retrieval.check --only tf_weight
python3 -m labs.retrieval.check --only maxsim
# After implementing all six:
python3 -m labs.retrieval.check
```

The starter functions intentionally raise `NotImplementedError`. A failing learner check is expected before you implement it. The repository test suite checks the reference implementation, not your unfinished work. Run `python3 -m labs.retrieval.check --impl reference` to see the full solution contract pass.

| Exercise | Implement | A question your code should answer |
|---|---|---|
| 1 | `tf_weight` | Why do five occurrences not count as five times the evidence? |
| 2 | `sparse_dot` | How can expanded vocabulary keys bridge different words? |
| 3 | `maxsim` | Why can a zero padding value beat valid evidence? |
| 4 | `rrf` | How can a result ranked second twice finish first? |
| 5 | `recall_at_k` | Why is one relevant hit not the same as full recall? |
| 6 | `candidate_union` | Can screening discard the best-scoring document? |

After a test passes, change one assumption and predict what happens before running the code. Add a counterexample as a test. That is more informative than copying the reference solution.

## What this repository does—and does not do

These are **mechanics exercises**, not production search systems or model benchmarks. Sparse weights and token-similarity matrices are constructed. We implement SPLADE-style sparse scoring, not its neural encoder; ColBERT-style MaxSim, not BERT training, residual compression, or PLAID search. BM25 is deliberately transparent and small-corpus oriented.

There are no claims that one trained retriever is better or faster than another. A model-backed comparison would need fixed data, checkpoints, relevance judgments, search settings, and hardware. See the [experiment protocol](docs/retrieval.md#the-next-experiment-is-a-different-task).

Only the retrieval module is shipped. The repository can hold later modules from the *Becoming an LLM Engineer* series when they are implemented and checked; they are not promised releases.

## Find your way around

```text
labs/retrieval/       # Demo, learner functions, reference, exercise grader
notebooks/           # Self-contained browser-friendly practice notebook
docs/                # Mechanisms, exercise contracts, hints, original sources
tests/               # Reference checks and the original scoring lab tests
examples/            # Original standalone scoring lab, preserved
.github/             # CI and focused issue/PR templates
scripts/             # Maintainer publishing and notebook verification tools
```

## Contribute one useful counterexample

Found a wrong explanation, an input that violates the stated contract, or a case where a claim needs a narrower scope? Open an issue with the smallest example that shows it. A correction plus a regression test is a useful contribution. Read [CONTRIBUTING.md](CONTRIBUTING.md).

Maintained by [Rui Wang](https://github.com/wangrui6). The material was developed with AI assistance. Runnable checks support the arithmetic and software contracts; they do not replace empirical retrieval evaluation. This is an independent educational project, not an employer product.

[MIT license](LICENSE). Original papers and linked implementations retain their own rights and licenses.
