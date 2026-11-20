"""Independent patch-level evaluation. Never treat NDVI pseudo-labels as ground truth.

Usage:
  python evaluate_independent_labels.py --predictions predictions.csv --labels reference_labels.csv --out evaluation
Labels CSV must contain patch_row,patch_col,reference_label,annotator,source_id.
reference_label is 0/1, or blank for unreviewed. Annotators must independently
inspect imagery; predictions/NDVI must not be used to assign reference labels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["patch_row", "patch_col"]

def evaluate(predictions, labels):
    required_p = KEYS + ["cnn_probability", "cnn_prediction_at_0_5"]
    required_l = KEYS + ["reference_label", "annotator", "source_id"]
    for title, frame, cols in [("predictions", predictions, required_p), ("labels", labels, required_l)]:
        missing = set(cols).difference(frame.columns)
        if missing:
            raise ValueError(f"{title} missing columns: {sorted(missing)}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{title} contains duplicate patch coordinates")
    labels = labels.copy()
    labels["reference_label"] = pd.to_numeric(labels["reference_label"], errors="coerce")
    reviewed = labels.dropna(subset=["reference_label"])
    if not reviewed["reference_label"].isin([0, 1]).all():
        raise ValueError("Reference labels must be binary 0 or 1")
    if reviewed[["annotator", "source_id"]].isna().any().any() or (reviewed[["annotator", "source_id"]].astype(str).apply(lambda s:s.str.strip()=="")).any().any():
        raise ValueError("Each reviewed label requires annotator and source_id")
    joined = predictions.merge(reviewed, on=KEYS, how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("No matching independently reviewed labels")
    probs = pd.to_numeric(joined["cnn_probability"], errors="raise").to_numpy()
    pred = pd.to_numeric(joined["cnn_prediction_at_0_5"], errors="raise").to_numpy()
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    if not np.array_equal(pred, (probs >= 0.5).astype(int)):
        raise ValueError("Predictions inconsistent with threshold 0.5")
    true = joined["reference_label"].astype(int).to_numpy()
    tp = int(((true == 1) & (pred == 1)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    def ratio(a, b):
        return float(a/b) if b else None
    metrics = {
        "n_evaluated":len(joined), "n_prediction_patches":len(predictions),
        "coverage_fraction":len(joined)/len(predictions),
        "tp":tp, "tn":tn, "fp":fp, "fn":fn,
        "accuracy":ratio(tp+tn,len(joined)),
        "precision_positive":ratio(tp,tp+fp),
        "recall_positive":ratio(tp,tp+fn),
        "specificity":ratio(tn,tn+fp),
        "f1_positive":ratio(2*tp,2*tp+fp+fn),
        "balanced_accuracy":None if (tp+fn==0 or tn+fp==0) else float((tp/(tp+fn)+tn/(tn+fp))/2),
        "brier_score":float(np.mean((probs-true)**2)),
        "warning":"Metrics describe only reviewed patches. Spatially independent sampling and reference quality must be documented; these figures do not prove citywide accuracy."
    }
    return metrics, joined

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",required=True)
    p.add_argument("--labels",required=True)
    p.add_argument("--out",default="independent_evaluation")
    args=p.parse_args()
    predictions=pd.read_csv(args.predictions)
    labels=pd.read_csv(args.labels)
    metrics,joined=evaluate(predictions,labels)
    dest=Path(args.out)
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    joined.to_csv(dest/"reviewed_predictions.csv",index=False)
    print(json.dumps(metrics,indent=2))

if __name__=="__main__":
    main()
