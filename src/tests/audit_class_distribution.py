import pandas as pd
from data_preparation import DATASETS, prepare_dataset
from data_splitting import (
    split_resources,
    summarize_target_class_counts
)

all_class_summaries = []

for dataset, (file_path, prefix_lengths) in DATASETS.items():
    print(f"Auditing {dataset}...", flush=True)

    sequences_by_prefix = prepare_dataset(
        file_path,
        prefix_lengths
    )

    for prefix_length, sequences_df in sequences_by_prefix.items():
        train_df, test_df = split_resources(
            sequences_df,
            test_size=0.2,
            random_state=42
        )

        class_summary = summarize_target_class_counts(
            sequences_df,
            train_df,
            test_df,
            dataset,
            prefix_length
        )

        all_class_summaries.append(class_summary)

class_summary_df = pd.concat(
    all_class_summaries,
    ignore_index=True
)

class_summary_df = class_summary_df.sort_values(
    by=['dataset', 'prefix_length', 'total_samples']
)

class_summary_df.to_csv(
    'target_class_counts.csv',
    index=False
)

print("\nAudit complete.")
print("Saved to target_class_counts.csv")
print(class_summary_df.to_string(index=False))
