# Version 11 validation status

The local archive includes CNN training-history and prediction figures, but no saved CNN checkpoint (.pt, .pth, .h5, .keras) or independently labelled satellite patch dataset was found.

A deterministic geographic sampling manifest has **90 unannotated candidate locations across 30 occupied spatial blocks**. This is not independent ground truth or image patches. 28 local software tests passed; these are not CNN performance tests.

## Evidence still needed
1. Frozen model weights, inference/preprocessing source and model-training provenance, or archived training imagery for reproducible retraining.
2. Licensed satellite scenes and imagery patches tied to sampling coordinates and acquisition dates.
3. Independently reviewed labels with disagreement adjudication.
4. Frozen spatial and temporal holdouts with training/test overlap checks.
5. CNN inference probabilities and independent confusion matrix, precision, recall, F1, balanced accuracy and ROC-AUC.

Until then, do not claim independently validated CNN accuracy or a submission-ready manuscript.
