"""
This is a simple application for sentence embeddings: clustering

Sentences are mapped to sentence embeddings and then agglomerative clustering with a threshold is applied.
"""
import collections
from datetime import datetime
from operator import itemgetter

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from itertools import groupby
import numpy as np
from collections import defaultdict
import pandas as pd

from yellowbrick.cluster import KElbowVisualizer
import pickle

from mysports import group_list_by_similarity
from rapidfuzz.process import extract, extractOne


def stream_position_to_time(stream_position):
    if not stream_position:
        return None
    return datetime.fromtimestamp(stream_position / 1000.0).strftime("%H:%M:%S")


embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus with example sentences
with open('./athlete_list_txt_file/big_air_athlete_list_10688.txt') as f:
    sentences = f.read().splitlines()
    sentences = [tuple(x.split(',')) for x in sentences]
    sentences = [(float(x[0]), x[1], stream_position_to_time(float(x[0]))) for x in sentences]

unique, counts = np.unique([x[0] for x in sentences], return_counts=True)

corpus = [x[1] for x in sentences]

# for i_app in range(1):

corpus_embeddings = embedder.encode(corpus)

# Normalize the embeddings to unit length
corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

# Perform kmean clustering
clustering_model = AgglomerativeClustering(n_clusters=None,
                                           distance_threshold=1.5)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(sentences[sentence_id])


# experimental code
# output = {}
# for i, cluster in clustered_sentences.items():
#     cluster_list = []
#     cluster_list.append(dict(collections.Counter([x[1] for x in cluster])))
#     cluster_list_string = [max(x, key=x.get) for x in cluster_list][0].upper()
#
#     keys = output.keys()
#     if len(keys) == 0:
#         output[cluster_list_string] = cluster
#         continue
#
#     res = extractOne(cluster_list_string, keys, score_cutoff=60)
#     if res:
#         output[res[0]].extend(cluster)
#         temp_list = output[res[0]]
#         del output[res[0]]
#         cluster_list_2 = []
#         cluster_list_2.append(dict(collections.Counter([x[1] for x in temp_list])))
#         cluster_list_string = [max(x, key=x.get) for x in cluster_list_2][0].upper()
#         output[cluster_list_string] = temp_list
#     else:
#         output[cluster_list_string] = cluster
#     print()
# print("")
model = AgglomerativeClustering()
result = []
for i, cluster in clustered_sentences.items():
    # cluster_list = []
    # cluster_list.append(dict(collections.Counter([x[1] for x in cluster])))
    # if len([x for x in cluster_list if sum(x.values()) > 100]) == 0:
    #     continue
    # cluster_list_string = [max(x, key=x.get) for x in cluster_list][0]
    stream_pos = [x[0] for x in cluster]
    # stream_pos = sorted(stream_pos)
    if len(stream_pos) < 12:
        continue
    stream_pos = np.array(stream_pos)
    # print('')
    # silhouette
    visualizer = KElbowVisualizer(model, k=(2, 12), metric='silhouette', timings=True)
    visualizer.fit(stream_pos.reshape(-1, 1))  # Fit the data to the visualizer
    # visualizer.show()
    k_value = visualizer.elbow_value_

    if k_value is None:
        continue
    #
    # # DBSCAN
    # # cluster_dbscan = DBSCAN(eps=120000, min_samples=10).fit(transformed_df)
    # # labels = cluster_dbscan.labels_
    #
    # # Agglomerative clustering
    clustering_model = AgglomerativeClustering(distance_threshold=None, n_clusters=k_value,
                                               metric='euclidean', linkage='complete', compute_distances=True)
    clustering_model.fit(stream_pos.reshape(-1, 1))
    labels = clustering_model.labels_
    # # # map athletes to clusters
    cluster_dict = defaultdict(list)
    for cluster_index, cluster_id in enumerate(labels):
        if cluster_id == -1:
            continue
        cluster_dict[cluster_id].append(cluster[cluster_index])

    cluster_dict = sorted(cluster_dict.items(), key=lambda k_v: k_v[1][0][0])

    i = 0
    cluster_list = []
    cluster_list.append(dict(collections.Counter([x[1] for x in cluster])))
    cluster_list_string = [max(x, key=x.get) for x in cluster_list][0]

    # cluster to cluster distance
    for cluster_index in range(len(cluster_dict) - 1):
        cur_cluster = cluster_dict[cluster_index][1]
        # current cluster
        cur_cluster_min = min(cur_cluster, key=lambda t: t[0])[0]
        cur_cluster_max = max(cur_cluster, key=lambda t: t[0])[0]

        cur_cluster_size = len(cur_cluster)
        cur_cluster_length = cur_cluster_max - cur_cluster_min
        length_size_ratio = cur_cluster_length // cur_cluster_size
        next_cluster_max = max(cluster_dict[cluster_index + 1][1], key=lambda t: t[0])[0]
        cur_cluster_distance_with_next_cluster = next_cluster_max - cur_cluster_min

        # cluster_1 = min(cluster_dict[cluster_index][1], key=lambda t: t[0])
        # cluster_2 = max(cluster_dict[cluster_index + 1][1], key=lambda t: t[0])
        # distance = cluster_2[0] - cluster_1[0]
        data = {
            f"cluster_name": f"{cluster_list_string}_{i}",
            "cluster": cur_cluster,
            "next_cluster": cluster_dict[cluster_index + 1][1],
            "cur_cluster_size": cur_cluster_size,
            "cur_cluster_length": cur_cluster_length,
            "length_size_ratio": length_size_ratio,
            "cur_cluster_distance": cur_cluster_distance_with_next_cluster,
            "cur_cluster_min": cur_cluster_min,
            "cur_cluster_max": cur_cluster_max
        }
        result.append(data)
        i += 1

    cur_cluster = cluster_dict[-1][1]
    cur_cluster_min = min(cur_cluster, key=lambda t: t[0])[0]
    cur_cluster_max = max(cur_cluster, key=lambda t: t[0])[0]

    cur_cluster_size = len(cur_cluster)
    cur_cluster_length = cur_cluster_max - cur_cluster_min
    length_size_ratio = cur_cluster_length // cur_cluster_size

    data = {
        f"cluster_name": f"{cluster_list_string}_{i}",
        "cluster": cur_cluster,
        "next_cluster": [],
        "cur_cluster_size": cur_cluster_size,
        "cur_cluster_length": cur_cluster_length,
        "length_size_ratio": length_size_ratio,
        "cur_cluster_distance": 0,
        "cur_cluster_min": cur_cluster_min,
        "cur_cluster_max": cur_cluster_max
    }
    result.append(data)

cluster_df = pd.DataFrame.from_dict(result)
x = cluster_df[['cur_cluster_size', 'cur_cluster_length']]
# x = cluster_df[['cur_cluster_size', 'cur_cluster_length', 'length_size_ratio']]
# load scaler
scl = './saved_models/final_test/combined_data_sc_without_ratio.pkl'
# scl = './saved_models/updated_model/sc_data_4_all_features.pkl'
scaler = pickle.load(open(scl, 'rb'))
x = scaler.transform(x)

filename = 'saved_models/final_test/combined_data_without_ratio.sav'
# filename = 'saved_models/updated_model/combined_model_all_features.sav'
loaded_model = pickle.load(open(filename, 'rb'))
predictions = loaded_model.predict(x)
cluster_df['is_race'] = predictions
print("")
# cluster_list = []

# for i, cluster in clustered_sentences.items():
#     cluster_list.append(dict(collections.Counter(cluster)))
#     # print("Cluster ", i + 1)
#     c = set(cluster)
#         # cluster_list.append(c)
#     #     print(c)
#     #     print("")
#     #
#     # print(cluster_list)
#     cluster_list_string = [max(x, key=itemgetter(1)) for x in cluster_list if sum(x.values()) > 12]
#     corpus = cluster_list_string

# print("")
