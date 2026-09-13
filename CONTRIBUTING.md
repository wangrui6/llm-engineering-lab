# Contributing

Useful contributions improve an explanation or make a failure reproducible. Start by running:

```bash
python3 -m unittest discover -s tests -v
python3 -m labs.retrieval.check --impl reference
python3 scripts/verify_notebook.py
```

The learner file intentionally contains unfinished functions. Do not replace the starter templates with completed answers in a pull request. Reference solutions belong in `labs/retrieval/reference.py`.

For a bug, provide the smallest input, expected output, actual output, and why the contract implies your expectation. Add a regression test. For an explanation, link a primary source and identify the exact claim that needs correction. Avoid claims about trained-model quality based on toy inputs.

Small first contributions include a TF/DF confusion example, a fusion counterexample, or a clearer hint that does not give away the full answer. These are suggestions, not pre-created GitHub issues.

Keep the core module dependency-free. Do not include credentials, private documents, copied paper figures, scraped personal data, or model weights. Record external dataset and code licenses before introducing them. Contributions are licensed under the repository's MIT license.

Notebook code is intentionally self-contained. When changing reference functions, regenerate its embedded reference cell with `python3 scripts/sync_notebook.py`, then execute the verification script. Do not commit local caches or learner credentials.
