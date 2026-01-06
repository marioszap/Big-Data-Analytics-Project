
import pandas as pd
import json
import math


path_to_clinical_dataset : str = '../data/clinical_dataset.csv'


def remove_errors(df: pd.DataFrame) -> None:
    df.replace(to_replace = [999, r'^(?i)test.*'], value = math.nan, regex=True, inplace=True)


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


def num_NaN_rows_by_column(df) -> None:
    NaN_rows_by_col_dict = {}
    for column in df.columns:
        NaN_rows_by_col_dict[column] = len(df.loc[pd.isna(df[column])])
        with open('../data/num_NaN_rows_by_column.json', 'w') as output:
            json.dump(NaN_rows_by_col_dict, output, indent=4)


def handle_missing_values(df):
    pass


def get_preprocessed_data() -> pd.DataFrame:
    df = pd.read_csv(path_to_clinical_dataset, sep=';')
    remove_errors(df)
    nominal_to_numerical(df)
    df['part_id'] = df['part_id'].astype(str)
    return df.copy()