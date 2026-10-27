
import pandas as pd

from sequence_encodings import (
    get_activity_window,
    get_activity_vocabulary,
    encode_baseline,
    encode_scap,
    encode_s2g,
    encode_s2gR
)


# Create a small example dataset
sequences_df = pd.DataFrame({
    'activity_1': ['A', 'B', 'A'],
    'activity_2': ['B', 'A', 'A'],
    'activity_3': ['A', 'C', 'B'],
    'activity_4': ['A', 'A', 'C'],
    'next_activity': ['C', 'B', 'A'],
    'org:resource': ['R1', 'R2', 'R3']
})

# Create a matching event log
event_log = pd.DataFrame({
    'concept:name': [
        'A', 'B', 'A', 'A', 'C',
        'B', 'A', 'C', 'A',
        'A', 'A', 'B', 'C'
    ],
    'org:resource': [
        'R1', 'R1', 'R1', 'R1', 'R1',
        'R2', 'R2', 'R2', 'R2',
        'R3', 'R3', 'R3', 'R3'
    ]
})

unique_activities = get_activity_vocabulary(event_log)


# Test 1: Sliding window
print('\nTest 1: Sliding window')

window = get_activity_window(sequences_df, window_size=2)

print(window)

assert window.iloc[0].tolist() == ['A', 'A']
assert window.iloc[1].tolist() == ['C', 'A']
assert window.iloc[2].tolist() == ['B', 'C']


# Test 2: Baseline encoding
print('\nTest 2: Baseline encoding')

baseline = encode_baseline(sequences_df, window_size=2)

print(baseline)

assert baseline.shape == (3, 2)


# Test 3: Sequence Capability encoding
print('\nTest 3: SCap encoding')

scap = encode_scap(sequences_df, event_log, window_size=2)

print(scap)

assert len(scap) == len(sequences_df)
assert any(column.startswith('capability_') for column in scap.columns)


# Test 4: S2g encoding
print('\nTest 4: S2g encoding')

s2g = encode_s2g(
    sequences_df,
    window_size=2,
    unique_activities=unique_activities
)

print(s2g)

assert len(s2g) == len(sequences_df)


# Test 5: S2gR encoding
print('\nTest 5: S2gR encoding')

s2gR = encode_s2gR(
    sequences_df,
    window_size=2,
    unique_activities=unique_activities
)

print(s2gR)

assert len(s2gR) == len(sequences_df)
assert 'avg_run_length' in s2gR.columns
assert 'num_runs' in s2gR.columns


# Test 6: Check target and resource preservation
print('\nTest 6: Target and resource preservation')

for encoded_df in [baseline, scap, s2g, s2gR]:
    assert encoded_df.index.equals(sequences_df.index)

assert sequences_df['next_activity'].tolist() == ['C', 'B', 'A']
assert sequences_df['org:resource'].tolist() == ['R1', 'R2', 'R3']

print('All tests passed.')


# Test 7: Exact transition counts
print('\nTest 7: Exact transition counts')

# Each window contains two activities, so each has one transition
assert s2g.loc[0, 'A->A'] == 1
assert s2g.loc[1, 'C->A'] == 1
assert s2g.loc[2, 'B->C'] == 1

# Transitions that do not occur should have a count of zero
assert s2g.loc[0, 'A->B'] == 0
assert s2g.loc[1, 'A->A'] == 0

# S2g and S2gR should have identical transition counts
transition_columns = [
    column for column in s2g.columns
    if '->' in column
]

pd.testing.assert_frame_equal(
    s2g[transition_columns],
    s2gR[transition_columns]
)

print('Transition counts are correct.')


# Test 8: Exact repetition features
print('\nTest 8: Exact repetition features')

# R1 has A -> A: one run of length 2
assert s2gR.loc[0, 'avg_run_length'] == 2.0
assert s2gR.loc[0, 'num_runs'] == 1

# R2 has C -> A: two runs of length 1
assert s2gR.loc[1, 'avg_run_length'] == 1.0
assert s2gR.loc[1, 'num_runs'] == 2

# R3 has B -> C: two runs of length 1
assert s2gR.loc[2, 'avg_run_length'] == 1.0
assert s2gR.loc[2, 'num_runs'] == 2

print('Repetition features are correct.')


# Test 9: Capability features match the correct resource
print('\nTest 9: Resource-capability matching')

capability_test_log = pd.DataFrame({
    'concept:name': ['A', 'B', 'C'],
    'org:resource': ['R1', 'R2', 'R3']
})

scap_test = encode_scap(
    sequences_df,
    capability_test_log,
    window_size=2
)

assert scap_test.loc[0, 'capability_A'] == 1
assert scap_test.loc[0, 'capability_B'] == 0
assert scap_test.loc[0, 'capability_C'] == 0

assert scap_test.loc[1, 'capability_A'] == 0
assert scap_test.loc[1, 'capability_B'] == 1
assert scap_test.loc[1, 'capability_C'] == 0

assert scap_test.loc[2, 'capability_A'] == 0
assert scap_test.loc[2, 'capability_B'] == 0
assert scap_test.loc[2, 'capability_C'] == 1

print('Capability features match the correct resources.')


# Test 10: Feature consistency and leakage checks
print('\nTest 10: Feature consistency and leakage checks')

encoded_datasets = {
    'Baseline': baseline,
    'SCap': scap,
    'S2g': s2g,
    'S2gR': s2gR
}

# All encodings must preserve the same sample order
for encoding_name, encoded_df in encoded_datasets.items():
    assert encoded_df.index.equals(sequences_df.index), encoding_name

# Resource identifiers and targets must not be in encoded feature columns
for encoding_name, encoded_df in encoded_datasets.items():
    assert 'next_activity' not in encoded_df.columns, encoding_name
    assert 'org:resource' not in encoded_df.columns, encoding_name

# S2g and S2gR must use identical transition columns
s2g_transition_columns = [
    column for column in s2g.columns if '->' in column
]

s2gR_transition_columns = [
    column for column in s2gR.columns if '->' in column
]

assert s2g_transition_columns == s2gR_transition_columns

print('Sample order, target/resource separation, and transition schemas are correct.')
