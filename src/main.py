
import pandas as pd
import json
import math
from difflib import SequenceMatcher, get_close_matches

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


def replace_with_most_similar(value, choices):
    if isinstance(value, str):
        match = get_close_matches(value, choices, n=1, cutoff=0.65)
        if match:
            return match[0]
    return value

def replace_with_most_similar_dict(value, choices:dict):
    if isinstance(value, str):
        match = get_close_matches(value, choices.keys(), n=1, cutoff=0.7)
        if match:
            return choices[match[0]]
    return value

def replace_if_contains(text, mapping):
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
        'desk': 'office',
        'work': 'office',
        'hall': 'hall',
        'out': 'garden',
        'entr': 'entrance',
        'bed': 'bedroom'
    }
    
    for column in df.columns:
        if isinstance(df[column][0], str):
            df[column] = df[column].str.lower()
            df[column] = df[column].str.replace(r'[^a-z]+', '', regex=True)
            df[column] = df[column].apply(lambda x: replace_with_most_similar(x, rooms_reference))
            df[column] = df[column].apply(lambda x: replace_with_most_similar_dict(x, keywords_dict))
            df[column] = df[column].apply(lambda x: replace_if_contains(x, keywords_dict))


    print(df['room'].unique())
    print(len(df['room'].unique()))


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
fix_room_names(df)