import pandas as pd
import plotly.express as px

df = pd.read_csv('./athlete_csv_file/all_v2/combined_2.csv')
# df = df[df['cur_cluster_length'] < 400000]
# fig = px.scatter(df, x="cur_cluster_length", y="length_size_ratio", color="expected_race")
# fig.show()
#
# fig2 = px.scatter(df, x="cur_cluster_size", y="length_size_ratio", color="expected_race")
# fig2.show()


fig3 = px.scatter(df, x="cur_cluster_size", y="cur_cluster_length", color="is_race")
fig3.show()

# df_2 = pd.read_csv('./athlete_csv_file/26-2-23/id_10688.csv')
# # df_2 = df_2[df_2['cur_cluster_length'] < 400000]
# fig = px.scatter(df_2, x="cur_cluster_size", y="cur_cluster_length", color="expected_race")
# fig.show()
print("")
