import os

import pandas as pd
import plotly.express as px

# df_1 = pd.read_csv('./athlete_csv_file/updated_data/big_air_10676.csv',
#                    usecols=['cluster_name', 'cur_cluster_size', 'cur_cluster_length', 'length_size_ratio',
#                             'expected_race'])
#
# # fig = px.scatter(df_1, x="cur_cluster_size", y="cur_cluster_length", color="expected_race")
# # fig.show()
# # df_1.drop(columns=['is_a_race_after_length', 'is_a_race_after_size', 'Unnamed: 0', 'is_race',
# #                    'is_a_race_after_length_size_ratio'], inplace=True, axis=1)
# df_2 = pd.read_csv('./athlete_csv_file/updated_data/big_air_10655.csv',
#                    usecols=['cluster_name', 'cur_cluster_size', 'cur_cluster_length', 'length_size_ratio',
#                             'expected_race'])
#
# # fig = px.scatter(df_2, x="cur_cluster_size", y="cur_cluster_length", color="expected_race")
# # fig.show()
# # df_2.drop(columns=['next_cluster', 'cur_cluster_distance', 'cur_cluster_min', 'cur_cluster_max', 'Unnamed: 0'],
# #           inplace=True, axis=1)
#
# df_3 = pd.read_csv('./athlete_csv_file/updated_data/big_air_10652.csv',
#                    usecols=['cluster_name', 'cur_cluster_size', 'cur_cluster_length', 'length_size_ratio',
#                             'expected_race'])
#
# # fig = px.scatter(df_3, x="cur_cluster_size", y="cur_cluster_length", color="expected_race")
# # fig.show()
# # df_3.drop(
# #     columns=['Unnamed: 0', 'cluster', 'next_cluster', 'cur_cluster_distance', 'cur_cluster_min', 'cur_cluster_max'],
# #     inplace=True, axis=1)
#
# # df_2 = pd.read_csv('./athlete_csv_file/big_air_csv_10688_cleaned.csv')
# # df_2.drop(columns=['Unnamed: 0', 'cluster', 'next_cluster', 'cur_cluster_min', 'cur_cluster_max', 'cur_cluster_distance'], axis=1, inplace=True)
#
# combine_df = pd.concat([df_1, df_2, df_3], ignore_index=True)
#
# fig = px.scatter(combine_df, x="cur_cluster_size", y="cur_cluster_length", color="expected_race")
# fig.show()
#
# print(len(combine_df))


file_path = './athlete_csv_file/all_v3'

all_csv = []
for file in os.listdir(file_path):
    full_path = f'{file_path}/{file}'
    df = pd.read_csv(full_path,
                     usecols=['cluster_name', 'cur_cluster_size', 'cur_cluster_length', 'length_size_ratio',
                              'is_race'])

    all_csv.append(df)

combine_df = pd.concat(all_csv, ignore_index=True)

# fig = px.scatter(combine_df, x="cur_cluster_size", y="cur_cluster_length", color="is_race")
# fig.show()

combine_df.to_csv(f'{file_path}/combined_3.csv')