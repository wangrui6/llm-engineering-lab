# Launch notes for the maintainer

## Positioning

The current artifact is six tested retrieval exercises, not a complete LLM curriculum or a production retriever. Lead with a failure that readers can reproduce. Do not claim repository popularity, user counts, speedups, or benchmark wins that have not been measured.

One repository can hold the implemented modules of the series. Keep the front page focused on what works today. Link a public blog post when it is actually published; do not paste private Notion URLs into the README.

## Suggested launch copy

Use only after the repository is public and its links work:

> I turned my retrieval study notes into six Python exercises: BM25 weighting, sparse scoring, MaxSim, RRF, recall, and candidate screening.
>
> One example: the highest-scoring document never reaches the reranker. Another: a padded zero beats every real token match.
>
> No API keys, GPU, or model downloads. The repo includes tests, hints, reference answers, and a self-contained notebook. These are constructed mechanics examples, not model benchmarks.
>
> Try one exercise and send a counterexample or an unclear step.

Add the verified repository URL. Do not claim that posting this will produce a particular number of stars.

## Distribution and feedback

Publish the article and repo as complementary artifacts: the article explains a mechanism; the repository lets a reader test it. Share one self-contained code example or failure case with each announcement, rather than repeating a bare link.

Pin the repository on your GitHub profile and add precise topics. GitHub supports [profile pins](https://docs.github.com/en/account-and-profile/how-tos/profile-customization/pinning-items-to-your-profile) and [topic discovery](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics).

Ask early readers where they got stuck. Use issues for reproducible bugs and Discussions for explanations. Useful signals include completed exercises, reproducible bug reports, outside contributions, and returning readers. Do not equate stars, downloads, clones, or notebook opens with learning outcomes. This starter contains no telemetry or learner tracking.

The next substantive extension could be a measured comparison on a public dataset using actual model checkpoints. That is a separate project with dependencies and evaluation design, not something this release already implements.
