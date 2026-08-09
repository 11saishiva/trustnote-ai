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

Three placeholders remain, marked `\tofill{...}` in the `.tex` (they render in
red) and `[CITE: ...]` in `references.bib`:

1. **Training hardware** (`trustnote-ai.tex`, Optimization section) — GPU model
   and total wall-clock training time.
2. **Dataset counts** (`trustnote-ai.tex`, Table II) — `data/` and `reports/`
   are gitignored, so the train and validation rows are projected from the
   measured test partition under the 70/15/15 ratio. Run
   `python scripts/dataset_audit.py` and replace them with measured
   per-split and per-denomination counts.
3. **Citations** (`references.bib`) — the RBI annual report needs a specific
   year and table, and one slot is reserved for prior work on Indian banknote
   authentication.

## Numbers already verified against the repo

- Confusion matrix (722 / 22 / 12 / 365) reconstructed from
  `models/test_metrics.json` and `models/classification_report.txt`.
- Training trajectory from `models/training_history.csv`: stopped at epoch 9,
  best validation F1 0.9775 at epoch 4.
- Parameter counts and CPU latency measured by loading
  `models/best_model.pth` with `strict=True`.
