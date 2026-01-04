
import pandas as pd
import json
import math
from difflib import get_close_matches

path_to_beacons_dataset : str = '../data/beacons_dataset.csv'
path_to_clinical_dataset : str = '../data/clinical_dataset.csv'



def remove_errors(df: pd.DataFrame) -> None:
    df.replace(to_replace = [999, r'^(?i)test.*'], value = math.nan, regex=True, inplace=True)


#clinical
def nominal_to_numerical(df: pd.DataFrame) -> None:
    nominal_to_numerical_dict : dict = {}
    for column in df.columns:
        if isinstance(df[column][0], str):
            nominal_to_numerical_dict[column] = {}
            possible_values : list = df[column].unique()
            for i in range(len(possible_values)):
                if not pd.isna(possible_values[i]):
                    nominal_to_numerical_dict[column][possible_values[i]] = i
            df[column] = df[column].map(nominal_to_numerical_dict[column])

    with open('../data/map_nominals_to_numerical.json', 'w') as output:
        json.dump(nominal_to_numerical_dict, output, indent=4)

#clinical
def num_NaN_rows_by_column(df) -> None:
    NaN_rows_by_col_dict = {}
    for column in df.columns:
        NaN_rows_by_col_dict[column] = len(df.loc[pd.isna(df[column])])
        with open('../data/num_NaN_rows_by_column.json', 'w') as output:
            json.dump(NaN_rows_by_col_dict, output, indent=4)


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

#beacons
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


    print(df['room'].unique())
    print(len(df['room'].unique()))


def remove_erronous_users(df) -> pd.DataFrame:
    mask = df['part_id'].astype(str).str.len() == 4
    df = df[mask] #Remove non 4-digit values 
    return df[df['part_id'].apply(lambda x: x.isnumeric())].copy()


def get_percent_of_time_per_room(df) -> pd.DataFrame:

    #df = df.loc[(df['part_id'] == '3089') & (df['ts_date'] == 20170915)] #DELETE THIS
    keep_cols = ['kitchen', 'bedroom', 'bathroom', 'livingroom']

    percent_of_time_per_room_df = pd.DataFrame()
    unique_ids : list = df['part_id'].unique()
    for user_id in unique_ids:
        user_df = df.loc[df['part_id'] == user_id]
        unique_days : list = user_df['ts_date'].unique()
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
            percent_of_time_per_room_df = pd.concat([percent_of_time_per_room_df, pivoted_df], ignore_index=True)
    #print(percent_of_time_per_room_df)
    return percent_of_time_per_room_df.fillna(0).copy()
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(percent_of_time_per_room_df)

#df = pd.read_csv(path_to_clinical_dataset, sep=';')


# print(df['gait_speed_slower'].loc[df['part_id'] == 3078]) #Has a Test not adequate
# print(df['balance_single'].loc[df['part_id'] == 1008]) #Has a test non realizable

# remove_errors(df)
# print(df['gait_speed_slower'].loc[df['part_id'] == 3078])
# print(df['balance_single'].loc[df['part_id'] == 1008])

# nominal_to_numerical(df)
# print(df['gait_speed_slower'].loc[df['part_id'] == 3078])
# print(df['balance_single'].loc[df['part_id'] == 1008])

df = pd.read_csv(path_to_beacons_dataset, sep=';')
print(df[:20])
fix_room_names(df)
#3089  20180214
#df = df.loc[(df['part_id'] == '3089') & (df['ts_date'] == 20180214)] #DELETE THIS
#print(df)

df = remove_erronous_users(df)
df = get_percent_of_time_per_room(df)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df[:20])