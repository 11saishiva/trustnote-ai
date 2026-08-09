# Paper

IEEE conference-format write-up of this project.

## Build

```bash
latexmk -pdf trustnote-ai.tex
```

`IEEEtran.cls` and `IEEEtran.bst` are vendored here because many TeX Live
installs omit them and `tlmgr install ieeetran` needs admin rights. If your
distribution already provides them, delete the local copies.

## Open items

No placeholders remain in the text or the bibliography.

One value to verify against the primary source before submission: the
Introduction cites 222,639 counterfeit notes detected in 2023-24 with 95.3%
found at banks rather than the RBI. Those come from press coverage of the RBI
Annual Report 2023-24, not from the PDF itself. Confirm them against the
Currency Management chapter.

Resolved by the primary author:

- **Dataset counts** — 7,473 images (4,960 REAL / 2,513 FAKE), split 5,231 /
  1,121 / 1,121. Now in Table I.
- **Training hardware** — CPU only, Intel Core i5-1335U, 36.09 minutes over
  nine epochs. No GPU was involved at any stage, so the mixed-precision branch
  never executed.

## Numbers already verified against the repo

- Confusion matrix (722 / 22 / 12 / 365) reconstructed from
  `models/test_metrics.json` and `models/classification_report.txt`.
- Training trajectory from `models/training_history.csv`: stopped at epoch 9,
  best validation F1 0.9775 at epoch 4.
- Parameter counts and CPU latency measured by loading
  `models/best_model.pth` with `strict=True`. Latency was timed on a Ryzen 9
  7900X, which is not the training machine; re-running the benchmark on the
  i5-1335U would make the two numbers comparable.
