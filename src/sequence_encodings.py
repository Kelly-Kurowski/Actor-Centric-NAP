import pandas as pd
from collections import defaultdict

from data_preparation import get_activity_sequences


def get_training_activity_vocabulary(train_df):
    """Get unique activities appearing in training prefixes only."""
    sequences = get_activity_sequences(train_df)
    activities = pd.unique(sequences.to_numpy().ravel())
    return sorted(activity for activity in activities if pd.notna(activity))


# Baseline (SeqOnly)
def encode_baseline(sequences_df):
    sequences = get_activity_sequences(sequences_df)

    return sequences


# Sequence Capability (SCap)
def encode_scap(sequences_df, event_log):
    sequences = get_activity_sequences(sequences_df)

    # Create a resource-activity matrix from the full event log
    capability_matrix = event_log.pivot_table(
        index='org:resource',
        columns='concept:name',
        aggfunc='size',
        fill_value=0
    )

    # Convert activity counts to binary indicators
    capability_matrix = (capability_matrix > 0).astype(int)

    # Add a prefix to distinguish capability features from sequence features
    capability_matrix.columns = [
        f"capability_{activity}"
        for activity in capability_matrix.columns
    ]

    # Match each sequence to its resource's capability vector
    resources = sequences_df['org:resource']

    capabilities = capability_matrix.reindex(
        resources.values
    ).fillna(0)

    capabilities.index = sequences.index

    # Combine sequence and capability features
    result_df = pd.concat(
        [sequences, capabilities],
        axis=1
    )

    return result_df

# Sequence two-gram (S2g)
def encode_s2g(sequences_df, unique_activities):
    sequences = get_activity_sequences(sequences_df)

    # Generate all possible transitions using a shared vocabulary
    all_possible_transitions = [
        (a, b)
        for a in unique_activities
        for b in unique_activities
    ]

    transition_counts = []

    # Count transitions for each sequence
    for _, row in sequences.iterrows():
        activities = row.dropna().values
        transitions = defaultdict(int)

        for i in range(len(activities) - 1):
            transition = (activities[i], activities[i + 1])
            transitions[transition] += 1

        row_counts = {
            transition: transitions.get(transition, 0)
            for transition in all_possible_transitions
        }

        transition_counts.append(row_counts)

    # Create transition feature DataFrame
    transitions_df = pd.DataFrame(
        transition_counts,
        columns=all_possible_transitions,
        index=sequences.index
    )

    transitions_df.columns = [
        f"{a}->{b}"
        for a, b in transitions_df.columns
    ]

    # Combine positional and transition features
    result_df = pd.concat(
        [sequences, transitions_df],
        axis=1
    )

    return result_df


# Sequence two-gram Repetition (S2gR)
def encode_s2gR(sequences_df, unique_activities):
    sequences = get_activity_sequences(sequences_df)

    # Generate all possible transitions using a shared vocabulary
    all_possible_transitions = [
        (a, b)
        for a in unique_activities
        for b in unique_activities
    ]

    transition_counts = []
    repeat_pattern_features = []

    # Calculate transition and repetition features for each sequence
    for _, row in sequences.iterrows():
        activities = row.dropna().values
        transitions = defaultdict(int)

        # Transition counts
        for i in range(len(activities) - 1):
            transition = (activities[i], activities[i + 1])
            transitions[transition] += 1

        row_counts = {
            transition: transitions.get(transition, 0)
            for transition in all_possible_transitions
        }

        transition_counts.append(row_counts)

        # Repeat-pattern features
        run_lengths = []

        if len(activities) > 0:
            current_run = 1

            for i in range(1, len(activities)):
                if activities[i] == activities[i - 1]:
                    current_run += 1
                else:
                    run_lengths.append(current_run)
                    current_run = 1

            run_lengths.append(current_run)

            avg_run_length = sum(run_lengths) / len(run_lengths)
            num_runs = len(run_lengths)

        else:
            avg_run_length = 0
            num_runs = 0

        repeat_pattern_features.append({
            'avg_run_length': avg_run_length,
            'num_runs': num_runs
        })

    # Create transition features
    transitions_df = pd.DataFrame(
        transition_counts,
        columns=all_possible_transitions,
        index=sequences.index
    )

    transitions_df.columns = [
        f"{a}->{b}"
        for a, b in transitions_df.columns
    ]

    # Create repeat-pattern features
    repeat_df = pd.DataFrame(
        repeat_pattern_features,
        index=sequences.index
    )

    # Combine sequence, transition, and repetition features
    result_df = pd.concat(
        [sequences, transitions_df, repeat_df],
        axis=1
    )

    return result_df
