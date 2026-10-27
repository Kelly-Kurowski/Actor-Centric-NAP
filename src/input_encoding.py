import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Convert features into numerical inputs for tree-based models
def encode_tree_inputs(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    ).copy()

    # Identify activity-position columns
    activity_columns = [
        column for column in X_train.columns
        if column.startswith('activity_')
    ]

    # One-hot encode activity-position columns
    X_train_activities = pd.get_dummies(
        X_train[activity_columns],
        dtype=int
    )

    X_test_activities = pd.get_dummies(
        X_test[activity_columns],
        dtype=int
    )

    # Ensure training and test activity features match
    X_test_activities = X_test_activities.reindex(
        columns=X_train_activities.columns,
        fill_value=0
    )

    # Keep other features numerical
    other_columns = [
        column for column in X_train.columns
        if column not in activity_columns
    ]

    X_train_other = X_train[other_columns]
    X_test_other = X_test[other_columns]

    # Combine activity and numerical features
    X_train_encoded = pd.concat(
        [X_train_activities, X_train_other],
        axis=1
    )

    X_test_encoded = pd.concat(
        [X_test_activities, X_test_other],
        axis=1
    )

    return X_train_encoded, X_test_encoded



# Separate activity sequences and numerical features for neural models
def encode_sequence_inputs(X_train, X_test):
    # Identify activity-position columns
    activity_columns = [
        column for column in X_train.columns
        if column.startswith('activity_')
    ]

    # Separate activity sequences from additional features
    X_train_activities = X_train[activity_columns].copy()
    X_test_activities = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )[activity_columns].copy()

    other_columns = [
        column for column in X_train.columns
        if column not in activity_columns
    ]

    X_train_other = X_train[other_columns].copy()
    X_test_other = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )[other_columns].copy()

    # Build the activity vocabulary from training sequences only
    activities = sorted(
        activity
        for activity in pd.unique(
            X_train_activities.to_numpy().ravel()
        )
        if pd.notna(activity)
    )

    activity_to_id = {
        activity: index + 1
        for index, activity in enumerate(activities)
    }

    # ID 0 is reserved for unknown or missing activities
    X_train_encoded = X_train_activities.apply(
        lambda column: column.map(
            lambda activity: activity_to_id.get(activity, 0)
        )
    )

    X_test_encoded = X_test_activities.apply(
        lambda column: column.map(
            lambda activity: activity_to_id.get(activity, 0)
        )
    )

    return (
        X_train_encoded,
        X_test_encoded,
        X_train_other,
        X_test_other,
        activity_to_id
    )


# Encode target labels using the training targets only
def encode_target_labels(y_train, y_test):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Map unseen test classes to -1
    target_to_id = {
        target: index
        for index, target in enumerate(label_encoder.classes_)
    }

    y_test_encoded = y_test.map(
        lambda target: target_to_id.get(target, -1)
    )

    return y_train_encoded, y_test_encoded, label_encoder
