
import pandas as pd
import clinical_preprocessor
import beacons_preprocessor

clinical_df : pd.DataFrame = clinical_preprocessor.get_preprocessed_data()
beacons_df : pd.DataFrame = beacons_preprocessor.get_preprocessed_data()

clinical_preprocessor.num_NaN_rows_by_column(clinical_df)
# print(clinical_df)
# print()
# print(beacons_df)
full_data_df = clinical_df.merge(beacons_df, on='part_id', how='inner')
print(full_data_df)