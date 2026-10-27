
import pandas as pd

from input_encoding import (
    encode_tree_inputs,
    encode_sequence_inputs,
    encode_target_labels
)


# Small example dataset
X_train = pd.DataFrame({
    'activity_1': ['A', 'B', 'A'],
    'activity_2': ['B', 'A', 'A']
})

X_test = pd.DataFrame({
    'activity_1': ['A', 'C'],
    'activity_2': ['A', 'B']
})

y_train = pd.Series(['C', 'B', 'A'])
y_test = pd.Series(['A', 'D'])


# Test 1: Tree-based inputs
print('\nTest 1: Tree-based inputs')

X_train_tree, X_test_tree = encode_tree_inputs(X_train, X_test)

print(X_train_tree)
print(X_test_tree)

assert all(
    pd.api.types.is_numeric_dtype(dtype)
    for dtype in X_train_tree.dtypes
)

assert list(X_train_tree.columns) == list(X_test_tree.columns)
assert len(X_train_tree) == len(X_train)
assert len(X_test_tree) == len(X_test)

print('Tree input checks passed.')


# Test 2: Neural sequence inputs
print('\nTest 2: Neural sequence inputs')

(
    X_train_seq,
    X_test_seq,
    X_train_other,
    X_test_other,
    activity_to_id
) = encode_sequence_inputs(X_train, X_test)

print('Activity mapping:', activity_to_id)
print(X_train_seq)
print(X_test_seq)

assert len(X_train_seq) == len(X_train)
assert len(X_test_seq) == len(X_test)
assert all(
    pd.api.types.is_integer_dtype(dtype)
    for dtype in X_train_seq.dtypes
)

# Unseen activity C should map to 0
assert X_test_seq.loc[1, 'activity_1'] == 0

print('Neural sequence checks passed.')


# Test 3: Target labels
print('\nTest 3: Target labels')

y_train_encoded, y_test_encoded, label_encoder = encode_target_labels(
    y_train,
    y_test
)

print('Target classes:', list(label_encoder.classes_))
print('Encoded training targets:', y_train_encoded)
print('Encoded test targets:', y_test_encoded)

assert len(y_train_encoded) == len(y_train)
assert len(y_test_encoded) == len(y_test)

# A is known, D is unseen in training
assert y_test_encoded.iloc[0] == label_encoder.transform(['A'])[0]
assert X_test_seq.loc[1, 'activity_1'] == 0

print('Target label checks passed.')

print('\nAll input encoding tests passed.')


# Test 4: Preserve numerical features in tree inputs
print('\nTest 4: Preserve numerical features')

X_train_mixed = pd.DataFrame({
    'activity_1': ['A', 'B', 'A'],
    'activity_2': ['B', 'A', 'A'],
    'A->B': [2, 0, 1],
    'avg_run_length': [1.5, 2.0, 1.0],
    'num_runs': [3, 2, 4],
    'capability_A': [1, 0, 1]
})

X_test_mixed = pd.DataFrame({
    'activity_1': ['A', 'C'],
    'activity_2': ['A', 'B'],
    'A->B': [1, 3],
    'avg_run_length': [2.5, 1.0],
    'num_runs': [2, 3],
    'capability_A': [1, 0]
})

X_train_mixed_encoded, X_test_mixed_encoded = encode_tree_inputs(
    X_train_mixed,
    X_test_mixed
)

print(X_train_mixed_encoded)
print(X_test_mixed_encoded)

# Numerical features must retain their original values
assert X_train_mixed_encoded['A->B'].tolist() == [2, 0, 1]
assert X_train_mixed_encoded['avg_run_length'].tolist() == [1.5, 2.0, 1.0]
assert X_train_mixed_encoded['num_runs'].tolist() == [3, 2, 4]
assert X_train_mixed_encoded['capability_A'].tolist() == [1, 0, 1]

# Training and test features must have the same columns
assert list(X_train_mixed_encoded.columns) == list(X_test_mixed_encoded.columns)

# An unseen activity should not create a new test-only feature
assert 'activity_1_C' not in X_test_mixed_encoded.columns

print('Numerical feature checks passed.')


import pandas as pd
from input_encoding import encode_sequence_inputs

# Example with activity, transition, and capability features
X_train = pd.DataFrame({
    'activity_1': ['A', 'B', 'A'],
    'activity_2': ['B', 'A', 'A'],
    'A->B': [1, 0, 2],
    'capability_A': [1, 0, 1]
})

X_test = pd.DataFrame({
    'activity_1': ['A', 'C'],
    'activity_2': ['B', 'A'],
    'A->B': [1, 3],
    'capability_A': [1, 0]
})

(
    X_train_seq,
    X_test_seq,
    X_train_other,
    X_test_other,
    activity_to_id
) = encode_sequence_inputs(X_train, X_test)

print("Activity mapping:", activity_to_id)
print("\nTraining sequences:")
print(X_train_seq)
print("\nTest sequences:")
print(X_test_seq)
print("\nAdditional training features:")
print(X_train_other)
print("\nAdditional test features:")
print(X_test_other)

assert 'A->B' in X_train_other.columns
assert 'capability_A' in X_train_other.columns
assert 'A->B' not in X_train_seq.columns
assert X_test_seq.loc[1, 'activity_1'] == 0

print("\nNeural input separation checks passed.")
