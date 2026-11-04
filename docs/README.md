# Evidence index

The [project README](../README.md) gives the scope and selected comparisons.
The [experiment ledger](../EXPERIMENTS.md) retains the published reproduction
record through S17d and the S18 plan. Use [RUNBOOK.md](../RUNBOOK.md) for
execution; stage plans describe their own deviations from the default recipe.

Start with [What changed after S18](LATER_EXPERIMENTS.md) for the findings and
limitations, then use the stage records below for methods and complete tables.

The records below describe local experiments. Later implementation files and
full run bundles remain unpublished; the runbook covers the published code.

## Later experiments

| Stage | Question | Plan and evidence |
| :-- | :-- | :-- |
| S18 / S18a | What crosses the centre of the cycle? | [Methods](s18b/METHODS.md), [calibration](s18b/CALIBRATION.md), [S18a results](s18b/S18A_RESULTS.md) |
| S18b | Does a counterfactual centre target improve tracking? | [Contract](s18b/PLAN.md), [results](s18b/S18B_RESULTS.md) |
| S18c | Can the all-leg and centre-target adapters be combined? | [Plan](s18b/S18C_PLAN.md), [results](s18b/S18C_RESULTS.md) |
| S19 | Do the S18b changes transfer to five-shot inference? | [Plan](s18b/S19_PLAN.md), [results](s18b/S19_RESULTS.md) |
| S19b | What do multiple training references and orientation canonicalization add? | [Plan](s18b/S19B_PLAN.md), [results](s18b/S19B_RESULTS.md) |
| S20 | Do the changes transfer to SAM 3? | [Plan](s18b/S20_PLAN.md), [results](s18b/S20_RESULTS.md) |
| S21 | Does matching butterfly label conventions help? | [Plan](s18b/S21_PLAN.md), [results](s18b/S21_RESULTS.md) |
| S22 | Does support-pool filtering transfer to SAM 3? | [Plan](s18b/S22_PLAN.md), [results](s18b/S22_RESULTS.md) |
| S23 | Do shape losses or ventral oversampling recover thin traits? | [Plan](s18b/S23_PLAN.md), [results](s18b/S23_RESULTS.md) |
| S24 | Does tiled inference recover detail? | [Plan](s18b/S24_PLAN.md); full evaluation marked unrun |

The S18a failed gate remains part of the record. Later passing gates do not erase
it or establish that a treatment improved the scientific outcome. Plans retain
what was specified before each run; later result sections record what happened.

## Butterfly diagnostics

- [Photographed side and label consistency](s18b/MXP_DIAGNOSIS.md).
- [Lativitta ray annotations](s18b/LATIVITTA_DIAGNOSIS.md).
- [Query-level diagnosis](s18b/QUERY_DIAGNOSIS.md).
- [Alternative diagnostic readings](s18b/DIAGNOSTIC_READS.md), kept separate from the locked primary score.
- [Local zoom probe](s18b/ZOOM_PROBE.md) and [results](s18b/ZOOM_PROBE_RESULTS.md).

## Presentations

[Talk materials](talk/README.md) are historical snapshots. Their dates and stage
coverage matter; a slide deck is not the current implementation contract. Read
the linked methods and results when an older narrative conflicts with later work.
