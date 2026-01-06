import pandas as pd
from difflib import get_close_matches

path_to_beacons_dataset : str = '../data/beacons_dataset.csv'


#Helpers: String Similarity functions#########################################################################
def replace_with_most_similar(value, choices) -> str:
    if isinstance(value, str):
        match = get_close_matches(value, choices, n=1, cutoff=0.65)
        if match:
            return match[0]
    return value

def replace_with_most_similar_dict(value, choices:dict) -> str:
    if isinstance(value, str):
        match = get_close_matches(value, choices.keys(), n=1, cutoff=0.7)
        if match:
            return choices[match[0]]
    return value

def replace_if_contains(text, mapping) -> str:
    if isinstance(text, str):
        for key, value in mapping.items():
            if key in text: 
                return value
    return text
#############################################################################################################



def fix_room_names(df) -> None:
    rooms_reference = ['bathroom', 'laundryroom', 'livingroom', 'kitchen', 'bedroom', 'office']

    keywords_dict = {
        'bath': 'bathroom',
        'laundry': 'laundryroom',
        'tv': 'livingroom',
        'sit': 'livingroom',
        'seat': 'livingroom',
        'dinner': 'kitchen',
        'diner': 'kitchen',
        'desk': 'office',
        'work': 'office',
        'hall': 'hall',
        'out': 'garden',
        'entr': 'entrance',
        'bed': 'bedroom'
    }
    
    df['room'] = df['room'].str.lower()
    df['room'] = df['room'].str.replace(r'[^a-z]+', '', regex=True)
    df['room'] = df['room'].apply(lambda x: replace_with_most_similar(x, rooms_reference))
    df['room'] = df['room'].apply(lambda x: replace_with_most_similar_dict(x, keywords_dict))
    df['room'] = df['room'].apply(lambda x: replace_if_contains(x, keywords_dict))


def remove_erronous_users(df) -> pd.DataFrame:
    mask = df['part_id'].astype(str).str.len() == 4
    df = df[mask] #Remove non 4-digit values 
    return df[df['part_id'].apply(lambda x: x.isnumeric())].copy()


def get_percent_of_time_per_room(df) -> pd.DataFrame:

    keep_cols = ['kitchen', 'bedroom', 'bathroom', 'livingroom']

    percent_of_time_per_room_df = pd.DataFrame()
    unique_ids : list = df['part_id'].unique()
    for user_id in unique_ids:
        user_df = df.loc[df['part_id'] == user_id]
        unique_days : list = user_df['ts_date'].unique()
        full_data_per_user = pd.DataFrame()
        for day in unique_days:
            user_in_day_df = user_df.loc[user_df['ts_date'] == day].copy()
            user_in_day_df['ts_time_delta'] = pd.to_timedelta(user_in_day_df['ts_time'])
            user_in_day_df['time_diff'] = (user_in_day_df['ts_time_delta'].shift(-1) - user_in_day_df['ts_time_delta']).dt.total_seconds()

            total_time_diff = (user_in_day_df['ts_time_delta'].iloc[-1] - user_in_day_df['ts_time_delta'].iloc[0]).total_seconds()
            user_in_day_df = user_in_day_df.drop('ts_time_delta', axis=1)
            user_in_day_df = user_in_day_df[:-1]
            user_in_day_df = user_in_day_df.groupby(['room'])['time_diff'].sum().reset_index()
            user_in_day_df['time_diff'] = (user_in_day_df['time_diff'] / total_time_diff) * 100

            pivoted_df = user_in_day_df.pivot_table(values='time_diff', columns='room', aggfunc='sum').reset_index(drop=True)

            pivoted_df = pivoted_df[[c for c in pivoted_df.columns if c in keep_cols]]

            pivoted_df.insert(0, 'part_id', user_id)
            pivoted_df.insert(1, 'date', day)

            full_data_per_user = pd.concat([full_data_per_user, pivoted_df], ignore_index=True)
            full_data_per_user = full_data_per_user.fillna(0)
            full_data_per_user = full_data_per_user.groupby('part_id').mean(numeric_only=True).reset_index()
        percent_of_time_per_room_df = pd.concat([percent_of_time_per_room_df, full_data_per_user.drop("date", axis=1)], ignore_index=True)
    return percent_of_time_per_room_df.fillna(0).copy()


def get_preprocessed_data() -> pd.DataFrame:
    df = pd.read_csv(path_to_beacons_dataset, sep=';')

    df = remove_erronous_users(df)
    fix_room_names(df)
    df = get_percent_of_time_per_room(df)
    return df.copy()