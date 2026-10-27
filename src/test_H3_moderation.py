"""Test H3 moderation using resource-level paired correctness differences.

Inputs (one per dataset):
  results/<dataset>/test_predictions_with_specialization.csv
or results/<dataset>/test_predictions.csv if it already has specialization.

Requirements: pandas, numpy, scipy, statsmodels.
Outputs:
  results/H3_moderation_summary.csv
  results/H3_resource_level_comparisons.csv

The regression includes dataset-by-prefix and model fixed effects and clusters
standard errors by resource (nested within dataset).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "results"

DATASETS_TO_TEST = [
    "BPIC2013",
    "BPIC2017",
    "BPIC2018",
    "BPIC2019",
]

KEYS = [
    "dataset", "prefix_length", "model", "test_sample_id", "resource_id"
]

CONTRASTS = [
    ("H3a", "SCap", "S2g", "positive"),
    ("H3b", "S2gR", "S2g", "negative"),
]


def load_predictions():
    dataframes = []
    for name in DATASETS_TO_TEST:
        dataset_dir = OUTPUT_DIR / name
        enriched = dataset_dir / "test_predictions_with_specialization.csv"
        path = enriched if enriched.is_file() else dataset_dir / "test_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"resource_id": "string"})
        if "activity_specialization" not in frame.columns:
            raise ValueError(f"H3 scores missing from {path}")
        if not frame["dataset"].eq(name).all():
            raise ValueError(f"Dataset mismatch in {path}")
        dataframes.append(frame)

    predictions = pd.concat(dataframes, ignore_index=True)
    if predictions.duplicated(KEYS + ["encoding"]).any():
        raise ValueError("Duplicate test resource predictions found")
    if predictions["activity_specialization"].isna().any():
        raise ValueError("Missing activity specialization scores")
    if not predictions["activity_specialization"].between(0, 1).all():
        raise ValueError("Activity specialization must be between zero and one")
    grouped = predictions.groupby(KEYS, dropna=False)
    if grouped["y_true"].nunique().gt(1).any():
        raise ValueError("Paired rows disagree on the true next activity")
    if grouped["activity_specialization"].nunique().gt(1).any():
        raise ValueError("Paired rows disagree on activity specialization")
    if (predictions["correct"] != predictions["y_true"].eq(predictions["y_pred"])).any():
        raise ValueError("Inconsistent correctness column")
    return predictions


def create_paired_comparisons(predictions):
    wide = predictions.pivot(
        index=KEYS,
        columns="encoding",
        values="correct",
    ).reset_index()
    scores = predictions.drop_duplicates(KEYS)[
        KEYS + ["activity_specialization"]
    ]
    wide = wide.merge(scores, on=KEYS, how="left", validate="one_to_one")

    comparisons = []
    for hypothesis, first, second, direction in CONTRASTS:
        paired = wide.dropna(subset=[first, second]).copy()
        paired["accuracy_difference"] = (
            paired[first].astype(int) - paired[second].astype(int)
        )
        paired["hypothesis"] = hypothesis
        paired["encoding_a"] = first
        paired["encoding_b"] = second
        paired["expected_direction"] = direction
        paired["setting"] = (
            paired["dataset"].astype(str)
            + "_prefix_"
            + paired["prefix_length"].astype(str)
        )
        paired["cluster_resource"] = (
            paired["dataset"].astype(str)
            + "::"
            + paired["resource_id"].astype(str)
        )
        comparisons.append(paired[
            KEYS + [
                "hypothesis", "encoding_a", "encoding_b",
                "expected_direction", "activity_specialization",
                "accuracy_difference", "setting", "cluster_resource",
            ]
        ])
    return pd.concat(comparisons, ignore_index=True)


def test_hypotheses(comparisons):
    rows = []
    for hypothesis, first, second, expected_direction in CONTRASTS:
        frame = comparisons.loc[
            comparisons["hypothesis"].eq(hypothesis)
        ].copy()
        if frame["activity_specialization"].nunique() < 2:
            raise ValueError(f"No specialization variation for {hypothesis}")

        fit = smf.ols(
            "accuracy_difference ~ activity_specialization "
            "+ C(setting) + C(model)",
            data=frame,
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": frame["cluster_resource"]},
        )
        variable = "activity_specialization"
        interval = fit.conf_int().loc[variable]
        slope = float(fit.params[variable])
        rows.append({
            "hypothesis": hypothesis,
            "comparison": f"{first} - {second}",
            "expected_slope": expected_direction,
            "n_paired_predictions": len(frame),
            "n_unique_resources": frame["cluster_resource"].nunique(),
            "n_model_prefix_settings": frame[["setting", "model"]].drop_duplicates().shape[0],
            "slope": slope,
            "cluster_robust_se": float(fit.bse[variable]),
            "ci95_lower": float(interval.iloc[0]),
            "ci95_upper": float(interval.iloc[1]),
            "p_value_two_sided": float(fit.pvalues[variable]),
            "direction_matches_hypothesis": (
                slope > 0 if expected_direction == "positive" else slope < 0
            ),
            "better": int(frame["accuracy_difference"].gt(0).sum()),
            "worse": int(frame["accuracy_difference"].lt(0).sum()),
            "equal": int(frame["accuracy_difference"].eq(0).sum()),
        })

    summary = pd.DataFrame(rows)
    summary["holm_adjusted_p"] = multipletests(
        summary["p_value_two_sided"], method="holm"
    )[1]
    summary["hypothesis_supported"] = (
        summary["direction_matches_hypothesis"]
        & summary["holm_adjusted_p"].lt(0.05)
    )
    return summary


def main():
    predictions = load_predictions()
    comparisons = create_paired_comparisons(predictions)
    summary = test_hypotheses(comparisons)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "H3_moderation_summary.csv", index=False)
    comparisons.to_csv(
        OUTPUT_DIR / "H3_resource_level_comparisons.csv", index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
