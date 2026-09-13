# Retrieval mechanics: what the lab is actually teaching

A retrieval system needs a representation, a score, and a procedure that finds good candidates. A reranker occupies a pipeline stage. An inverted index or ANN index executes search. These are different decisions; do not place all their names on one accuracy ladder.

## BM25: changing the strength of lexical evidence

BM25 combines rarity across documents, term frequency within a document, and length adjustment. In the lab, its term-frequency factor is:

```text
tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_length))
```

At average length with `k1=1.2`, a term occurring five times gets a factor of approximately 1.7742, not five. Repetition has diminishing returns. Changing `b` changes how document length modifies that evidence. This is not a mechanism for creating a synonym match when analyzed terms do not overlap. See [Elastic's explanation of the BM25 variables](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) and the [original authors' technical report](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-356.pdf).

`tf_weight` isolates the factor. `bm25` in the reference includes corpus statistics and a positive IDF convention. Neither function is a production index. The simple implementation favors visible arithmetic over corpus-scale efficiency.

## Sparse does not mean non-neural

SPLADE encodes text into a sparse vector over vocabulary coordinates. Those coordinates can include tokens absent from the original text. Its score multiplies matching coordinates, so semantic expansion can occur before ordinary sparse matching. [Original SPLADE paper](https://arxiv.org/abs/2107.05720) and [authors' explanation](https://europe.naverlabs.com/blog/splade-a-sparse-bi-encoder-bert-based-model-achieves-effective-and-efficient-first-stage-ranking/).

Our sparse exercise starts after that encoding step. The weights are hand-written. Computing `automobile * automobile` is not the same as directly measuring similarity between the words `car` and `automobile`. An encoder may activate `automobile` for both texts; then the shared coordinate contributes to the dot product.

The same dot product can be executed by inverted postings. This changes access patterns, not the mathematical score. The lab keeps all document IDs in the result with default-zero scores for comparison, so even its postings implementation still has small-corpus bookkeeping. Do not infer production latency from it.

## Dense retrieval is a learned score plus a search problem

DPR separately encodes queries and passages, then scores them with a dot product. Passage embeddings can be precomputed. Training uses relevant passages and negatives to shape that geometry. [DPR paper](https://arxiv.org/abs/2004.04906).

This first lab does not load a dense encoder. The useful diagnostic for a later experiment is to hold embeddings fixed and compare exhaustive search with ANN. If exhaustive search succeeds but ANN misses, investigate search settings. If both rank irrelevant passages highly, more ANN effort does not repair the score. [FAISS index documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).

## ColBERT: choose evidence after the query arrives

ColBERT preserves contextual token vectors for queries and documents. The score is the sum, across query vectors, of the best document-token similarity for each query vector. [ColBERT paper](https://arxiv.org/abs/2004.12832).

```text
M = query_vectors @ document_vectors.T
score = sum(max(valid entries in row) for row in M)
```

The lab supplies M directly. It does not include a tokenizer, encoder, learned projection, normalization, compression, or corpus-scale candidate generator. That isolation makes the reduction axes easy to inspect.

With fixed similarities, repeating a column cannot increase the maximum. But a single accidental high similarity can dominate a row. Adding the row maxima also permits compensation: high scores for two conditions can outweigh a low score for a third. This is a deduction from the formula, not an empirical claim about a particular checkpoint.

Padding is another implementation trap. If valid entries are negative, a padded zero can win. Exclude padded positions before reducing. This exercise does not specify the treatment of ColBERT query-augmentation vectors; that is model-specific.

SPLADE v2 also uses max pooling, but across token positions while building a vocabulary vector. That happens before query–document matching. ColBERT preserves the document vectors until a query can select their matches. [SPLADE v2](https://arxiv.org/abs/2109.10086).

## Finding the maximum is not the same as defining it

Our three-document example makes this precise. A is best for the first query row, B for the second. C is second-best for both and has the highest sum. A candidate procedure that only retains A and B never evaluates the correct aggregate winner.

This is why real multi-vector retrieval requires more than the formula for MaxSim. [PLAID](https://arxiv.org/abs/2205.09707) describes a candidate refinement process. [ColBERTv2](https://arxiv.org/abs/2112.01488) addresses supervision and compact representations. The lab's candidate selector implements neither paper; it exposes the problem these systems must manage.

## Fusion and evaluation are separate

RRF combines ranked lists with contributions of `1/(c + rank)`. Ranks start at one. For lists `[A,B]` and `[C,B]` with `c=60`, B receives two contributions and finishes above A and C. [Original RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).

RRF is a fusion rule, not evidence that the combined ranking is better. It can reward agreement between systems with the same mistake. A later reranker may also improve ordering without retrieving a missing document. [Retrieve and rerank guide](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html).

Our Recall@K exercise uses the fraction of relevant documents returned. Hit@K asks only whether any relevant document was returned. When relevance labels are incomplete, state the judged-set limitation. When a query has no relevant documents, do not silently invent a recall value; define a separate unanswerable-query metric.

## The next experiment is a different task

A model-backed comparison should fix a public corpus, passage segmentation, and relevance judgments. Record model checkpoints and versions, index configuration, candidate depth, filters, runtime hardware, and query-encoding costs. Compare candidate recall and ordering separately. Include latency, index size, and query-type breakdowns.

That experiment is not part of this release. It should not be advertised as completed merely because these mechanics tests pass. [BEIR](https://arxiv.org/abs/2104.08663) provides a primary reference for evaluating retrieval across heterogeneous tasks.

## Transfer the lesson

Whenever a multi-stage system fails, ask two questions: did the right candidate survive long enough to be evaluated, and did the score prefer it once evaluated? The same distinction applies to recommendation pipelines, search tools, beam search, and other staged decision systems. This is an architectural analogy, not proof that their objectives or guarantees are identical.
