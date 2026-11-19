# Independent vegetation classification validation

The archived CNN 98.29% figure is agreement with NDVI-derived pseudo-labels, **not independently validated accuracy**. Do not present it as ground-truth accuracy.

## Required evidence

1. Define vegetation versus non-vegetation patch labels, cloud/shadow exclusions, patch size and acquisition date.
2. Collect independent human annotations or authoritative independent land-cover references. Document imagery provenance, sampling, reviewer disagreements and adjudication. Do not generate reference labels from NDVI thresholds.
3. Freeze model weights, preprocessing and threshold; preserve source training data, sample IDs and scene IDs.
4. Hold out spatially disjoint geographic blocks and separate acquisition scenes, verifying non-overlapping footprints.
5. Compare CNN, RF (only if solving the same task), and a simple NDVI baseline on the same independent test set. Report confusion matrix, precision, recall, F1, balanced accuracy, ROC-AUC and sample support; repeat geographic folds.
6. If independently annotated test data or model weights cannot be obtained, omit claims of validated classification performance and describe the system as geospatial monitoring and park-proximity prioritization.

## Evaluation CSV contract

`sample_id,scene_id,spatial_block,reference_label,predicted_probability,label_source`

`reference_label` is independently annotated 0 or 1. Prediction probability lies in [0,1]. Keep training/test sample and geographic block IDs disjoint. Audit scene overlap and annotation provenance. Metadata alone cannot prove label independence.

The local Version 9 research bundle includes a runnable evaluation harness, unit tests and this protocol. Real independently annotated labels are **not** included and have not been fabricated.