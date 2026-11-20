# Visual findings

- V2@0.20 leaves many target-object cells eligible for reuse on held-out frames.
- V2@0.08 reduces those false-background cells but also leaves only about 25 of 256 cells reusable on average.
- Lowering the cutoff adds both useful foreground corrections and extra background protection; counts are per selected frame in `example_metrics.csv`.
- Very small items and narrow gripper/fixture structures are poorly represented by a 16×16 grid.
- Robot mount is unprompted and unvalidated; never treat its zero measured recall as safe background.
- Temporal overlays show source/current mask overlap only, not actual cached K/V correctness or policy safety.
