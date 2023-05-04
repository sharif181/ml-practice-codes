from sentence_transformers import SentenceTransformer
import hdbscan
from datetime import datetime
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import collections
from mysports import group_list_by_similarity
from collections import defaultdict
import pandas as pd
import pickle


# def find_nearest(array, value):
#     idx = (np.abs(array - value)).argmin()
#     return array[idx]


def stream_position_to_time(stream_position):
    if not stream_position:
        return None
    return datetime.fromtimestamp(stream_position / 1000.0).strftime("%H:%M:%S")


with open('./athlete_list_txt_file/ski_cross.txt') as f:
    sentences = f.read().splitlines()
    sentences = [tuple(x.split(',')) for x in sentences]
    sentences = [(float(x[0]), x[1], stream_position_to_time(float(x[0]))) for x in sentences]

unique, counts = np.unique([x[0] for x in sentences], return_counts=True)

corpus = [x[1] for x in sentences]

embedder = SentenceTransformer('all-MiniLM-L6-v2')

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

output = {}
keys = []
values = []

# taking one athlete from every group
for i, cluster in clustered_sentences.items():
    cluster_list = []
    cluster_list.append(dict(collections.Counter([x[1] for x in cluster])))
    cluster_list_string = [max(x, key=x.get) for x in cluster_list][0].upper()

    keys.append(cluster)
    values.append(cluster_list_string)

# merge similar groups
res = group_list_by_similarity(listN=values, data_source=keys)


# remove cluster which contains one word as athlete name
# res_2 = []
# for cluster in res:
#     cluster_list = []
#     cluster_list.append(dict(collections.Counter([x[1] for x in cluster])))
#     cluster_list_string = [max(x, key=x.get) for x in cluster_list][0].upper()
#     if len(cluster_list_string.split(" ")) > 1:
#         res_2.append(cluster)

# cluster object creation for dataframe
def cluster_object_creator(cur_cluster, cluster_id=None):
    cluster_list = []
    cluster_list.append(dict(collections.Counter([x[1] for x in cur_cluster])))
    cluster_list_string = [max(x, key=x.get) for x in cluster_list][0].upper()

    cur_cluster_min = min(cur_cluster, key=lambda t: t[0])[0]
    cur_cluster_max = max(cur_cluster, key=lambda t: t[0])[0]
    cur_cluster_size = len(cur_cluster)
    cur_cluster_length = cur_cluster_max - cur_cluster_min
    length_size_ratio = cur_cluster_length // cur_cluster_size
    data = {
        f"cluster_name": f"{cluster_list_string}_{cluster_id}",
        "cluster": cur_cluster,
        "cur_cluster_size": cur_cluster_size,
        "cur_cluster_length": cur_cluster_length,
        "length_size_ratio": length_size_ratio,
        "cur_cluster_min": cur_cluster_min,
        "cur_cluster_max": cur_cluster_max
    }
    return data


result = []
for i, cluster in enumerate(res):
    stream_pos = [[x[0], 0] for x in cluster]
    stream_pos = np.array(stream_pos)
    clusterer = hdbscan.HDBSCAN(gen_min_span_tree=True, allow_single_cluster=True, cluster_selection_epsilon=120000)
    # clusterer = hdbscan.HDBSCAN(gen_min_span_tree=True, allow_single_cluster=True, cluster_selection_epsilon=120000)
    clusterer.fit(stream_pos)

    # clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
    #                                       edge_alpha=0.9,
    #                                       node_size=120,
    #                                       edge_linewidth=4)
    # plt.show()
    #
    # clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    # plt.show()
    # clusterer.condensed_tree_.plot()
    # plt.show()
    # clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
    # plt.show()
    # k_value = math.floor((len(np.unique(np.array(clusterer.labels_)))) / 2)
    # k_value = len(np.unique(np.array(clusterer.labels_)))
    # if k_value <= 0:
    #     k_value = 1
    #
    # clustering_model = AgglomerativeClustering(distance_threshold=None, n_clusters=k_value, linkage='complete')
    #
    # clustering_model.fit(stream_pos)

    # labels = clustering_model.labels_
    labels = clusterer.labels_
    cluster_dict = defaultdict(list)
    for cluster_index, cluster_id in enumerate(labels):
        if cluster_id == -1:
            continue
        cluster_dict[cluster_id].append(cluster[cluster_index])

    cluster_dict = sorted(cluster_dict.items(), key=lambda k_v: k_v[1][0][0])
    for item_id, item in enumerate(cluster_dict):
        result.append(cluster_object_creator(item[1], cluster_id=item_id))

cluster_df = pd.DataFrame.from_dict(result)
x = cluster_df[['cur_cluster_size', 'cur_cluster_length']]
# x = cluster_df[['cur_cluster_size', 'cur_cluster_length', 'length_size_ratio']]
# load scaler
# scl = './saved_models/final_test/combined_data_sc_without_ratio.pkl'
scl = './saved_models/26-2-2023/combined_data_sc_without_ratio.pkl'
# scl = './saved_models/updated_model/sc_data_4_all_features.pkl'
scaler = pickle.load(open(scl, 'rb'))
x = scaler.transform(x)

# filename = 'saved_models/final_test/combined_data_without_ratio.sav'
filename = 'saved_models/26-2-2023/combined_data_without_ratio.sav'
# filename = 'saved_models/updated_model/combined_model_all_features.sav'
loaded_model = pickle.load(open(filename, 'rb'))
predictions = loaded_model.predict(x)
cluster_df['is_race'] = predictions
print("")
