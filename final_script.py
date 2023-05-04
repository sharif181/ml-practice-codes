from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime
from operator import itemgetter
import itertools
from yellowbrick.cluster import KElbowVisualizer
import pickle


def stream_position_to_time(stream_position):
    if not stream_position:
        return None
    return datetime.fromtimestamp(stream_position / 1000.0).strftime("%H:%M:%S")


temp_sentences = []
athlete_list = []
stream_pos = []
with open('./athlete_list_txt_file/big_air_athlete_list_10688.txt') as f:
    sentences = f.read().splitlines()
    sentences = [tuple(x.split(',')) for x in sentences]
    sentences = [(int(float(x[0])), x[1], stream_position_to_time(int(float(x[0])))) for x in sentences]
    sentences = sorted(sentences, key=itemgetter(0))

# map athlete in the frame
cluster_dict = {}
cluster_list = []
for key, group in itertools.groupby(sentences, key=itemgetter(0)):
    group_content = list(group)
    athletes = [x[1] for x in group_content]
    athlete_names = ' ,'.join(athletes).upper()  # at least avoiding case-insensitive
    athlete_list.append(athlete_names)
    stream_pos.append(group_content[0][0])

    temp_sentences.append((group_content[0][0], athlete_names, group_content[0][2]))

# create DataFrame
df = pd.DataFrame({'athlete': athlete_list, 'stream_pos': stream_pos})

### make column transformer
transformer = make_column_transformer(
    (OneHotEncoder(), ['athlete']),
    remainder='passthrough',
    sparse_threshold=0
)

transformed = transformer.fit_transform(df)
transformed_df = pd.DataFrame(
    transformed,
    columns=transformer.get_feature_names_out()
)

# keep stream pos col name
remainder_col = transformed_df.columns[-1]
result = []

# initialize clustering model for elbow method
model = AgglomerativeClustering()

# iterate each athlete
for athlete in transformed_df.columns[0:-1]:

    # for simplicity, ignoring multiple athlete name in same frame
    if len(athlete.split(',')) > 1:
        continue

    new_data_frame = transformed_df[[athlete, remainder_col]]  # slice dataframe for each athlete
    new_data_frame = new_data_frame[new_data_frame != 0].dropna()  # remove rows where athlete not present

    # for KElbowVisualizer k parameter
    if len(new_data_frame) <= 10:
        continue
    # silhouette
    visualizer = KElbowVisualizer(model, k=(2, 10), metric='silhouette', timings=True)
    visualizer.fit(new_data_frame[remainder_col].to_numpy().reshape(-1, 1))  # Fit the data to the visualizer
    k_value = visualizer.elbow_value_
    index = list(new_data_frame.index.values)  # getting index value for mapping

    # if cluster number is none then do nothing, we can also use k_value = 1
    if k_value is None:
        continue
    # Agglomerative clustering
    clustering_model = AgglomerativeClustering(distance_threshold=None, n_clusters=k_value,
                                               metric='euclidean', linkage='complete', compute_distances=False)
    clustering_model.fit(new_data_frame)
    labels = clustering_model.labels_

    # map athletes to clusters
    cluster_dict = defaultdict(list)
    for idx, cluster in zip(index, labels):
        if cluster == -1:
            continue
        cluster_dict[cluster].append(temp_sentences[idx])

    cluster_dict = sorted(cluster_dict.items(), key=lambda k_v: k_v[1][0][0])  # sort cluster based on stream pos

    i = 0  # cluster number for each athlete
    # creating cluster info
    for cluster_index in range(len(cluster_dict) - 1):
        # current cluster
        cur_cluster = cluster_dict[cluster_index][1]
        cur_cluster_min = min(cur_cluster, key=lambda t: t[0])[0]
        cur_cluster_max = max(cur_cluster, key=lambda t: t[0])[0]

        cur_cluster_size = len(cur_cluster)
        cur_cluster_length = cur_cluster_max - cur_cluster_min
        next_cluster_max = max(cluster_dict[cluster_index + 1][1], key=lambda t: t[0])[0]
        cur_cluster_distance_with_next_cluster = next_cluster_max - cur_cluster_min

        data = {
            f"cluster_name": f"{athlete}_{i}",
            "cluster": cur_cluster,
            "next_cluster": cluster_dict[cluster_index + 1][1],
            "cur_cluster_size": cur_cluster_size,
            "cur_cluster_length": cur_cluster_length,
            "cur_cluster_distance": cur_cluster_distance_with_next_cluster,
            "cur_cluster_min": cur_cluster_min,
            "cur_cluster_max": cur_cluster_max
        }
        result.append(data)
        i += 1

    # for last cluster
    cur_cluster = cluster_dict[-1][1]
    cur_cluster_min = min(cur_cluster, key=lambda t: t[0])[0]
    cur_cluster_max = max(cur_cluster, key=lambda t: t[0])[0]

    cur_cluster_size = len(cur_cluster)
    cur_cluster_length = cur_cluster_max - cur_cluster_min

    data = {
        f"cluster_name": f"{athlete}_{i}",
        "cluster": cur_cluster,
        "next_cluster": [],
        "cur_cluster_size": cur_cluster_size,
        "cur_cluster_length": cur_cluster_length,
        "cur_cluster_distance": 0,
        "cur_cluster_min": cur_cluster_min,
        "cur_cluster_max": cur_cluster_max
    }
    result.append(data)

# creating pd dataframe from list of clusters
cluster_df = pd.DataFrame.from_dict(result)

# new column for ratio
cluster_df['length_size_ratio'] = [x[0] / x[1] for x in
                                   zip(cluster_df['cur_cluster_length'], cluster_df['cur_cluster_size'])]

# load scaler
scl = './saved_models/updated_model/sc_all_features.pkl'
scaler = pickle.load(open(scl, 'rb'))

# scale features
x = cluster_df[['cur_cluster_size', 'cur_cluster_length', 'length_size_ratio']]
x = scaler.transform(x)

# load model
filename = 'saved_models/updated_model/combined_model_all_features.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# predict result
predictions = loaded_model.predict(x)

# create new column in cluster dataframe
cluster_df['is_race'] = predictions
# only keep clusters which are race
cluster_df = cluster_df[cluster_df['is_race'] == 1]

race = []
for index, row in cluster_df.iterrows():
    race_info = {
        "athlete_name": row['cluster_name'],
        "stream_start_pos": row['cur_cluster_min'],
        "stream_end_pos": row['cur_cluster_max'],
        "cluster_elements": row['cluster']
    }
    race.append(race_info)
print("")
