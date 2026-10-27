"""Attach H3 activity specialization to existing model predictions.

Place this script next to data_preparation.py and run it from the project.
It does not import the main experiment script and does not retrain models. No model training is performed.

For each dataset, the output is:
  results/<dataset>/test_predictions_with_specialization.csv
  results/<dataset>/test_resource_specialization.csv
Original test_predictions.csv files are preserved.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation import DATASETS, import_xes

REQUIRED_COLUMNS = ["concept:name", "org:resource", "time:timestamp"]


def calculate_activity_specialization(event_log, prefix_length):
    """Calculate 1 - normalized Shannon entropy from observed prefix events.

    Scores range from 0 to 1; larger values indicate greater specialization.
    A resource performing just one distinct activity receives a score of 1.
    The function uses only the first `prefix_length` events for each resource.
    """

    prefix_events = event_log.groupby(
        "org:resource", sort=False
    ).head(prefix_length)

    activity_counts = (
        prefix_events.groupby(
            ["org:resource", "concept:name"], sort=False
        )
        .size()
        .rename("activity_count")
        .reset_index()
    )

    totals = activity_counts.groupby("org:resource")[
        "activity_count"
    ].transform("sum")
    proportions = activity_counts["activity_count"] / totals
    activity_counts["entropy_term"] = -proportions * np.log(proportions)

    resource_stats = activity_counts.groupby("org:resource").agg(
        entropy=("entropy_term", "sum"),
        distinct_activities=("concept:name", "nunique"),
    )

    # log(1) = 0, so the single-activity case must be handled separately.
    denominators = np.log(resource_stats["distinct_activities"].clip(lower=2))
    scores = 1.0 - resource_stats["entropy"] / denominators
    scores.loc[resource_stats["distinct_activities"] == 1] = 1.0

    return scores.clip(lower=0.0, upper=1.0).rename(
        "activity_specialization"
    )

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "results"

DATASETS_TO_UPDATE = [
    "BPIC2013",
    "BPIC2017",
    "BPIC2018",
    "BPIC2019",
]


def main():
    for dataset_name in DATASETS_TO_UPDATE:
        input_path = OUTPUT_DIR / dataset_name / "test_predictions.csv"
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing predictions: {input_path}")

        predictions = pd.read_csv(
            input_path, dtype={"resource_id": "string"}
        )
        if predictions.empty:
            raise ValueError(f"No predictions in {input_path}")
        if not predictions["dataset"].eq(dataset_name).all():
            raise ValueError(f"Unexpected dataset in {input_path}")

        relative_path, _ = DATASETS[dataset_name]
        event_log = import_xes(str((PROJECT_DIR / relative_path).resolve()))
        event_log = event_log[REQUIRED_COLUMNS].copy()
        event_log["org:resource"] = event_log["org:resource"].astype("string")
        event_log = event_log.sort_values(
            ["org:resource", "time:timestamp"], kind="mergesort"
        ).reset_index(drop=True)

        predictions["activity_specialization"] = float("nan")
        for prefix_length in sorted(predictions["prefix_length"].unique()):
            specialization = calculate_activity_specialization(
                event_log, int(prefix_length)
            )
            selected = predictions["prefix_length"].eq(prefix_length)
            predictions.loc[selected, "activity_specialization"] = (
                predictions.loc[selected, "resource_id"]
                .map(specialization)
                .to_numpy()
            )

        if predictions["activity_specialization"].isna().any():
            missing = predictions.loc[
                predictions["activity_specialization"].isna(),
                ["prefix_length", "resource_id"],
            ].drop_duplicates()
            raise ValueError(
                f"No specialization found for {dataset_name}:\n{missing.head()}"
            )

        output_dir = OUTPUT_DIR / dataset_name
        enriched_path = output_dir / "test_predictions_with_specialization.csv"
        predictions.to_csv(enriched_path, index=False)

        resource_keys = [
            "dataset", "prefix_length", "test_sample_id", "resource_id"
        ]
        resource_scores = predictions[
            resource_keys + ["activity_specialization"]
        ].drop_duplicates()
        if resource_scores.duplicated(resource_keys).any():
            raise ValueError("Inconsistent specialization across model outputs")
        resource_scores.to_csv(
            output_dir / "test_resource_specialization.csv", index=False
        )
        print(
            f"{dataset_name}: {len(resource_scores)} test resource-prefix rows, "
            f"{len(predictions)} predictions enriched -> {enriched_path}"
        )


if __name__ == "__main__":
    main()
