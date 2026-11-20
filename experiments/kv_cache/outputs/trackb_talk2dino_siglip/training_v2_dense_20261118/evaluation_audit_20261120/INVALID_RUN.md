# Superseded metric-audit run — do not cite

This preserved run contains complete per-image output, but its V2 held-out threshold application was incorrect: V2 predictions were summarized at 0.55 instead of the development-calibrated 0.20. The finalizer also stopped on a CSV output-column mismatch. These outputs are retained for provenance only and are superseded by `../evaluation_audit_20261120_corrected/`, which reruns the final split at the selected development-only thresholds and validates exact split coverage.

The saved development sweep and threshold calibration were reused; no training or checkpoint changes were made.
