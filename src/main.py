from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from data_preparation import (
    DATASETS,
    import_xes,
    create_activity_sequences,
)
from data_splitting import (
    split_resources,
    check_resource_split,
    check_target_classes,
    summarize_class_distribution,
    summarize_target_class_counts,
)
from sequence_encodings import (
    encode_baseline,
    encode_scap,
    encode_s2g,
    encode_s2gR,
    get_training_activity_vocabulary,
)
from input_encoding import (
    encode_tree_inputs,
    encode_sequence_inputs,
)
from tuning import (
    tune_tree_model,
    tune_neural_model,
    save_tuning_results,
)
from evaluation import evaluate_predictions


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3

RUN_DATASETS = [
     "BPIC2013",
    # "BPIC2017",
    # "BPIC2018",
    # "BPIC2019",
]

# Comment out prefix lengths in data_preparation.py to exclude them.
# All configured prefix lengths are used by default.

ENCODINGS_TO_RUN = [
    "baseline",
    "SCap",
    "S2g",
    "S2gR",
]

MODELS_TO_RUN = [
    "RandomForest",
    "LightGBM",
    "XGBoost",
    "LSTM",
    "Transformer",
]

TREE_N_ITER = 16
NEURAL_N_ITER = 16

TREE_MODELS = {"RandomForest", "LightGBM", "XGBoost"}
NEURAL_MODELS = {"LSTM", "Transformer"}

REQUIRED_COLUMNS = [
    "concept:name",
    "org:resource",
    "time:timestamp",
]


def load_dataset(dataset_name):
    relative_path, prefix_lengths = DATASETS[dataset_name]
    file_path = (PROJECT_DIR / relative_path).resolve()

    print(f"\nLoading {dataset_name} from {file_path}")

    # Explicit conversion also supports PM4Py versions requiring a string.
    event_log = import_xes(str(file_path))
    event_log = event_log[REQUIRED_COLUMNS].copy()

    # Preserve chronological ordering within each resource.
    event_log = event_log.sort_values(
        by=["org:resource", "time:timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)

    sequences_by_prefix = {
        prefix_length: create_activity_sequences(
            event_log,
            prefix_length,
        )
        for prefix_length in prefix_lengths
    }

    print(f"Events: {len(event_log):,}")
    print("Resources:", event_log["org:resource"].nunique())
    print(f"Prefix lengths: {list(sequences_by_prefix)}")

    return event_log, sequences_by_prefix


def create_resource_split(sequences_df, dataset_name, prefix_length):
    """Create and diagnose one fixed resource-level split."""

    train_df, test_df = split_resources(
        sequences_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    overlapping_resources = check_resource_split(train_df, test_df)
    if overlapping_resources:
        raise RuntimeError(
            "Resource leakage detected: resources occur in both "
            "training and test sets."
        )

    unseen_test_classes = check_target_classes(train_df, test_df)

    print(
        f"Training samples: {len(train_df):,} | "
        f"Test samples: {len(test_df):,}"
    )
    print(
        f"Training resources: {train_df['org:resource'].nunique()} | "
        f"Test resources: {test_df['org:resource'].nunique()}"
    )

    split_summaries = [
        summarize_class_distribution(
            train_df, dataset_name, prefix_length, "train",
        ),
        summarize_class_distribution(
            test_df, dataset_name, prefix_length, "test",
        ),
    ]

    class_summary = summarize_target_class_counts(
        sequences_df,
        train_df,
        test_df,
        dataset_name,
        prefix_length,
    )

    class_summary["unseen_in_train"] = (
        class_summary["target_activity"].isin(unseen_test_classes)
    )

    return train_df, test_df, split_summaries, class_summary


def build_encoding(
    encoding_name,
    train_df,
    test_df,
    event_log,
    prefix_length,
):
    """Build one encoding using the same training/test split."""

    activity_vocabulary = get_training_activity_vocabulary(train_df)

    if encoding_name == "baseline":
        X_train = encode_baseline(train_df)
        X_test = encode_baseline(test_df)

    elif encoding_name == "SCap":
        # Keep the established full-event-log SCap definition.
        X_train = encode_scap(train_df, event_log)
        X_test = encode_scap(test_df, event_log)

    elif encoding_name == "S2g":
        X_train = encode_s2g(
            train_df, activity_vocabulary,
        )
        X_test = encode_s2g(
            test_df, activity_vocabulary,
        )

    elif encoding_name == "S2gR":
        X_train = encode_s2gR(
            train_df, activity_vocabulary,
        )
        X_test = encode_s2gR(
            test_df, activity_vocabulary,
        )

    else:
        raise ValueError(f"Unknown encoding: {encoding_name}")

    # Ensure the test matrix has exactly the training feature columns.
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    return X_train, X_test


def run_model(
    model_name,
    X_train_raw,
    X_test_raw,
    y_train_raw,
    y_test_raw,
    test_df,
    dataset_name,
    prefix_length,
    encoding_name,
):
    """Tune and evaluate a model; return metrics and per-sample predictions."""

    print(f"\n--- {model_name} ---")

    # Every encoding must preserve the original test sample row ordering.
    if not X_test_raw.index.equals(test_df.index):
        raise ValueError(
            "Encoded test row indices differ from test_df; cannot safely "
            "align individual predictions across encodings."
        )

    if len(y_test_raw) != len(test_df):
        raise ValueError("Test target count differs from test_df row count.")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)

    tuning_dir = (
        OUTPUT_DIR
        / "tuning"
        / dataset_name
        / f"prefix_{prefix_length}"
        / encoding_name
    )
    tuning_file = tuning_dir / f"{model_name}.json"

    if model_name in TREE_MODELS:
        X_train, X_test = encode_tree_inputs(
            X_train_raw,
            X_test_raw,
        )

        X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
        X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Give LightGBM safe, unique feature names.
        if model_name == "LightGBM":
            if not X_train.columns.equals(X_test.columns):
                raise ValueError(
                    "Training and test feature columns do not match."
                )

            feature_names = [
                f"feature_{i}" for i in range(X_train.shape[1])
            ]

            X_train = X_train.copy()
            X_test = X_test.copy()

            X_train.columns = feature_names
            X_test.columns = feature_names

        estimator, best, candidates = tune_tree_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            n_iter=TREE_N_ITER,
            random_state=RANDOM_STATE,
            cv_folds=CV_FOLDS,
        )

        predicted_ids = estimator.predict(X_test)

    elif model_name in NEURAL_MODELS:
        (
            X_train_sequence,
            X_test_sequence,
            X_train_features,
            X_test_features,
            activity_to_id,
        ) = encode_sequence_inputs(
            X_train_raw,
            X_test_raw,
        )

        X_train_features = (
            X_train_features
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )
        X_test_features = (
            X_test_features
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        estimator, best, candidates = tune_neural_model(
            model_name=model_name,
            X_sequence_train=X_train_sequence.to_numpy(dtype=np.int64),
            X_features_train=X_train_features.to_numpy(dtype=np.float32),
            y_train=y_train,
            vocab_size=len(activity_to_id) + 1,
            num_classes=len(label_encoder.classes_),
            n_iter=NEURAL_N_ITER,
            random_state=RANDOM_STATE,
            cv_folds=CV_FOLDS,
        )

        X_test_combined = np.concatenate(
            [
                X_test_sequence.to_numpy(dtype=np.float32),
                X_test_features.to_numpy(dtype=np.float32),
            ],
            axis=1,
        )
        predicted_ids = estimator.predict(X_test_combined)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    predictions = label_encoder.inverse_transform(
        np.asarray(predicted_ids, dtype=int)
    )

    if len(predictions) != len(test_df):
        raise ValueError("Model returned an unexpected number of predictions.")

    metrics = evaluate_predictions(
        y_true=y_test_raw,
        y_pred=predictions,
        y_train=y_train_raw,
    )

    result = {
        "dataset": dataset_name,
        "prefix_length": prefix_length,
        "encoding": encoding_name,
        "model": model_name,
        **metrics,
        "best_params": best["params"],
        "best_cv_accuracy": best["cv_accuracy"],
        "cv_strategy": best["cv_strategy"],
        "cv_folds": best["cv_folds"],
        "tuning_status": best.get(
            "tuning_status",
            "completed" if best.get("cv_folds", 0) > 0
            else "skipped_rare_class"
        ),
        "min_class_count": best.get(
            "min_class_count",
            int(np.unique(y_train, return_counts=True)[1].min())
        ),
        "status": "success",
        "error": None,
    }

    # One row per test example per successful model/encoding.
    # test_sample_id is the original row index in sequences_df, stable
    # across encodings for the same dataset and prefix length.
    prediction_rows = pd.DataFrame({
        "dataset": dataset_name,
        "prefix_length": prefix_length,
        "model": model_name,
        "encoding": encoding_name,
        "test_sample_id": test_df.index.to_numpy(),
        "resource_id": test_df["org:resource"].to_numpy(),
        "y_true": np.asarray(y_test_raw),
        "y_pred": np.asarray(predictions),
    })
    prediction_rows["correct"] = (
        prediction_rows["y_true"] == prediction_rows["y_pred"]
    )
    prediction_rows["seen_in_train"] = prediction_rows["y_true"].isin(
        label_encoder.classes_
    )

    save_tuning_results(
        {"best": best, "candidates": candidates},
        tuning_file,
    )

    if best["cv_accuracy"] is not None:
        print(f"Best CV accuracy: {best['cv_accuracy']:.4f}")
    else:
        print("CV tuning skipped: singleton target class")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Class coverage: {metrics['class_coverage']:.4f}")
    print(f"Tuning results saved to: {tuning_file}")

    return result, prediction_rows


def save_experiment_outputs(
    results,
    split_summaries,
    class_summaries,
    prediction_frames,
    dataset_name,
):
    """Save accumulated metrics, diagnostics, and predictions by dataset."""

    dataset_dir = OUTPUT_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if results:
        pd.DataFrame(results).to_csv(
            dataset_dir / "experiment_results.csv",
            index=False,
        )

    if split_summaries:
        pd.DataFrame(split_summaries).to_csv(
            dataset_dir / "split_summaries.csv",
            index=False,
        )

    if class_summaries:
        pd.concat(class_summaries, ignore_index=True).to_csv(
            dataset_dir / "target_class_counts.csv",
            index=False,
        )

    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            dataset_dir / "test_predictions.csv",
            index=False,
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_runs = 0

    for dataset_name in RUN_DATASETS:
        # Reset buffers per dataset to prevent mixing results across datasets.
        results = []
        all_split_summaries = []
        all_class_summaries = []
        all_prediction_frames = []

        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name}")
        print("=" * 70)

        event_log, sequences_by_prefix = load_dataset(dataset_name)

        for prefix_length, sequences_df in sequences_by_prefix.items():
            print("\n" + "-" * 70)
            print(
                f"PREFIX LENGTH: {prefix_length} | "
                f"DATASET: {dataset_name}"
            )
            print("-" * 70)

            if sequences_df.empty:
                print("No sequences available. Skipping this prefix.")
                continue

            (
                train_df,
                test_df,
                split_summaries,
                class_summary,
            ) = create_resource_split(
                sequences_df,
                dataset_name,
                prefix_length,
            )

            all_split_summaries.extend(split_summaries)
            all_class_summaries.append(class_summary)

            y_train_raw = train_df["next_activity"].to_numpy()
            y_test_raw = test_df["next_activity"].to_numpy()

            for encoding_name in ENCODINGS_TO_RUN:
                print("\n" + "=" * 60)
                print(f"ENCODING: {encoding_name}")
                print("=" * 60)

                X_train_raw, X_test_raw = build_encoding(
                    encoding_name=encoding_name,
                    train_df=train_df,
                    test_df=test_df,
                    event_log=event_log,
                    prefix_length=prefix_length,
                )

                for model_name in MODELS_TO_RUN:
                    try:
                        result, prediction_rows = run_model(
                            model_name=model_name,
                            X_train_raw=X_train_raw,
                            X_test_raw=X_test_raw,
                            y_train_raw=y_train_raw,
                            y_test_raw=y_test_raw,
                            test_df=test_df,
                            dataset_name=dataset_name,
                            prefix_length=prefix_length,
                            encoding_name=encoding_name,
                        )
                        all_prediction_frames.append(prediction_rows)

                    except Exception as error:
                        print(
                            f"\nFAILED: {dataset_name} | "
                            f"prefix={prefix_length} | "
                            f"{encoding_name} | {model_name}"
                        )
                        print(f"{type(error).__name__}: {error}")
                        traceback.print_exc()

                        result = {
                            "dataset": dataset_name,
                            "prefix_length": prefix_length,
                            "encoding": encoding_name,
                            "model": model_name,
                            "status": "failed",
                            "error": (
                                f"{type(error).__name__}: {error}"
                            ),
                        }

                    results.append(result)

                    # Save this dataset's accumulated results after each run.
                    save_experiment_outputs(
                        results,
                        all_split_summaries,
                        all_class_summaries,
                        all_prediction_frames,
                        dataset_name,
                    )

        total_runs += len(results)
        print(f"\nFinished {dataset_name}: {len(results)} runs recorded.")
        if results:
            result_df = pd.DataFrame(results)
            print(result_df["status"].value_counts(dropna=False))

    print("\n" + "=" * 70)
    print("EXPERIMENTS FINISHED")
    print("=" * 70)
    print(f"Results directory: {OUTPUT_DIR.resolve()}")
    print(f"Successful/failed runs recorded: {total_runs}")


if __name__ == "__main__":
    main()
