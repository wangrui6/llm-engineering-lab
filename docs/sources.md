# Primary sources and implementation explanations

These links explain the algorithms. They are not endorsements of every claim in the linked material, nor proof that a technique wins on a new dataset. This repository links to papers rather than redistributing their PDFs or figures.

| Source | What to use it for |
|---|---|
| [Robertson and Spärck Jones, Simple, proven approaches to text retrieval](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-356.pdf) | Early probabilistic weighting and inverted-file methods. |
| [Robertson and Zaragoza, BM25 and Beyond](https://doi.org/10.1561/1500000019) | The probabilistic framework and BM25 variants. |
| [Elastic, Practical BM25 Part 2](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) | An implementation-oriented explanation of the parameters. |
| [DPR](https://arxiv.org/abs/2004.04906) | Separate query/document encoders, training positives and negatives. |
| [SPLADE](https://arxiv.org/abs/2107.05720), [SPLADE v2](https://arxiv.org/abs/2109.10086) | Learned vocabulary-space representations and pooling variants. |
| [NAVER LABS, SPLADE introduction](https://europe.naverlabs.com/blog/splade-a-sparse-bi-encoder-bert-based-model-achieves-effective-and-efficient-first-stage-ranking/) | Author-written explanation of sparse semantic expansion. |
| [ColBERT](https://arxiv.org/abs/2004.12832) | Contextual token vectors and MaxSim. |
| [ColBERTv2](https://arxiv.org/abs/2112.01488), [PLAID](https://arxiv.org/abs/2205.09707) | Compact vectors and efficient multi-vector search. |
| [White Box Analysis of ColBERT](https://arxiv.org/abs/2012.09650), [authors' blog](https://europe.naverlabs.com/blog/a-white-box-analysis-of-colbert/) | Local score attribution and the limits of interpretability. |
| [RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) | Reciprocal rank fusion. |
| [Sentence Transformers retrieve/rerank guide](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) | The candidate-generation/reranking distinction. |
| [FAISS indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes), [BEIR](https://arxiv.org/abs/2104.08663) | Search trade-offs and heterogeneous retrieval evaluation. |

All numerical matrices and sparse weights in this lab are constructed unless explicitly stated otherwise. No model outputs or benchmark measurements are implied by those examples. No third-party paper figures, datasets, weights, or implementation source files are bundled.
