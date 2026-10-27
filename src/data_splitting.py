import pandas as pd
from sklearn.model_selection import train_test_split


# Split data by resource
def split_resources(sequences_df, test_size=0.2, random_state=42):
    resources = sequences_df['org:resource'].unique()

    train_resources, test_resources = train_test_split(
        resources,
        test_size=test_size,
        random_state=random_state
    )

    train_df = sequences_df[
        sequences_df['org:resource'].isin(train_resources)
    ].copy()

    test_df = sequences_df[
        sequences_df['org:resource'].isin(test_resources)
    ].copy()

    return train_df, test_df


# Check that resources do not overlap between splits
def check_resource_split(train_df, test_df):
    train_resources = set(train_df['org:resource'])
    test_resources = set(test_df['org:resource'])

    overlapping_resources = train_resources & test_resources

    print(f"Overlapping resources: {len(overlapping_resources)}")

    return overlapping_resources



# Check for target classes absent from either split
def check_target_classes(train_df, test_df):
    train_classes = set(train_df['next_activity'].unique())
    test_classes = set(test_df['next_activity'].unique())

    unseen_test_classes = test_classes - train_classes
    absent_test_classes = train_classes - test_classes

    print(f"Training classes: {len(train_classes)}")
    print(f"Test classes: {len(test_classes)}")
    print(f"Test classes absent from training: {len(unseen_test_classes)}")
    print(f"Training classes absent from test: {len(absent_test_classes)}")

    if unseen_test_classes:
        print("Test classes absent from training:")
        print(sorted(unseen_test_classes))

    return unseen_test_classes


# Summarize the class distribution
def summarize_class_distribution(
    sequences_df,
    dataset,
    prefix_length,
    split_name
):
    class_counts = sequences_df['next_activity'].value_counts()

    summary = {
        'dataset': dataset,
        'prefix_length': prefix_length,
        'split': split_name,
        'number_of_samples': len(sequences_df),
        'number_of_resources': sequences_df['org:resource'].nunique(),
        'number_of_classes': len(class_counts),
        'classes_with_1_sample': (class_counts == 1).sum(),
        'classes_with_fewer_than_5_samples': (class_counts < 5).sum()
    }

    return summary


# Summarize target-class counts before and after splitting
def summarize_target_class_counts(
    sequences_df,
    train_df,
    test_df,
    dataset,
    prefix_length
):
    all_counts = sequences_df['next_activity'].value_counts()
    train_counts = train_df['next_activity'].value_counts()
    test_counts = test_df['next_activity'].value_counts()

    class_summary = pd.DataFrame({
        'dataset': dataset,
        'prefix_length': prefix_length,
        'target_activity': all_counts.index,
        'total_samples': all_counts.values
    })

    class_summary['train_samples'] = (
        class_summary['target_activity']
        .map(train_counts)
        .fillna(0)
        .astype(int)
    )

    class_summary['test_samples'] = (
        class_summary['target_activity']
        .map(test_counts)
        .fillna(0)
        .astype(int)
    )

    return class_summary


