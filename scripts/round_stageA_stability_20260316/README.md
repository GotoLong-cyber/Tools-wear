# round_stageA_stability_20260316

This folder stores code/data/results for Stage-A stable feature filtering.

## Structure
- `code/`
  - `stageA_stability_filter.py`: Stage-A filtering logic (no model training).
  - `run_fold_stageA_template.sh`: template command for one fold.
- `data/`
  - filtered npz/csv outputs after Stage-A.
- `results/`
  - feature audit tables, keep/drop lists, summary json.

## Leakage rule (per fold)
- Fit filtering only on current fold train runs.
- Current fold test run can be exported with selected features, but must not be used for fitting/selection.

## Not executed in this turn
- Files are created only.
- No experiment command has been executed.
