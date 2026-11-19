# Independent vegetation annotation field guide — Version 12

This pack contains coordinates, not imagery patches or ground truth. Both reviewers must label the same licensed, georeferenced patches independently.

1. Record imagery source, license, acquisition date, patch footprint, resolution and scene ID.
2. Define target beforehand: 1 = vegetation visibly dominates fixed patch; 0 = nonvegetation dominates. Never derive independent labels from NDVI thresholds.
3. Flag clouds, shadows, missing imagery and ambiguous patches for exclusion/adjudication.
4. Reviewer A and B annotate independently, blinded to model output and each other's labels.
5. Preserve original labels; resolve disagreements with a third-party adjudicator.
6. Freeze spatial holdout before model tuning; check patch-footprint separation, scene overlap and any needed geographic buffer.
7. Use frozen CNN weights and preprocessing to produce probabilities on the independently annotated holdout. Compare only models predicting the same target.
8. Report exclusions, agreement, confusion matrix, per-class precision/recall/F1, balanced accuracy and spatially grouped uncertainty.

**Limitations:** 90 candidate-grid points do not represent an unbiased citywide sample. Deterministic holdout blocks do not guarantee physical separation. No image patches, independent labels, or saved CNN checkpoint are included; independent accuracy remains unestablished.
