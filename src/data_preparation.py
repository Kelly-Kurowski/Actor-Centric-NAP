import pm4py
import pandas as pd


PREFIX_2013 = [10, 20, 30, 40, 50, 75, 100, 125, 150]
PREFIX_2017 = [100, 150, 200, 400, 800, 1000, 1200, 1400, 1500, 2000]
PREFIX_2018 = [100, 150, 200, 400, 800, 1000, 1200, 1400, 1500, 2000, 2500]
PREFIX_2019 = [100, 150, 200, 300, 400, 500, 600, 700, 800]


DATASETS = {
    "BPIC2013": ("../data/BPI_Challenge_2013_incidents.xes", PREFIX_2013),
    "BPIC2017": ("../data/BPI_Challenge_2017.xes", PREFIX_2017),
    "BPIC2018": ("../data/BPI_Challenge_2018.xes", PREFIX_2018),
    "BPIC2019": ("../data/BPI_Challenge_2019.xes", PREFIX_2019)
}


def import_xes(file_path):
    log = pm4py.read_xes(str(file_path))
    event_log = pm4py.convert_to_dataframe(log)
    return event_log


def create_activity_sequences(df, prefix_length):
    sequences = []
    next_activities = []
    resources = []

    for resource, resource_df in df.groupby('org:resource'):
        activities = resource_df['concept:name'].values

        if len(activities) >= prefix_length + 1:
            prefix = activities[:prefix_length]
            next_activity = activities[prefix_length]

            sequences.append(prefix)
            next_activities.append(next_activity)
            resources.append(resource)

    sequences_df = pd.DataFrame(
        sequences,
        columns=[f"activity_{i+1}" for i in range(prefix_length)]
    )

    sequences_df['next_activity'] = next_activities
    sequences_df['org:resource'] = resources

    return sequences_df

def get_activity_sequences(sequences_df):
    activity_columns = [
        column
        for column in sequences_df.columns
        if column.startswith("activity_")
    ]
    return sequences_df[activity_columns].copy()


def prepare_dataset(file_path, prefix_lengths):
    event_log = import_xes(file_path)

    df = event_log[
        ['concept:name', 'org:resource', 'time:timestamp']
    ].copy()

    df = df.sort_values(
        by=['org:resource', 'time:timestamp']
    )

    sequences_by_prefix = {}

    for prefix_length in prefix_lengths:
        sequences_df = create_activity_sequences(df, prefix_length)
        sequences_by_prefix[prefix_length] = sequences_df

    return sequences_by_prefix
