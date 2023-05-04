import pandas as pd

df = pd.read_csv('./athlete_csv_file/big_air_3_friction.csv')

df_2 = df[['is_a_race_after_length', 'is_a_race_after_size', 'is_a_race_after_length_size_ratio']]

result = []
for index, row in df_2.iterrows():
    a = (row['is_a_race_after_length'] + row['is_a_race_after_size'] + row['is_a_race_after_length_size_ratio']) / 3
    result.append(round(a, 2))

df['is_race'] = result
print('jkhkjh')