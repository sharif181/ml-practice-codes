from sentence_transformers import SentenceTransformer
import hdbscan
from datetime import datetime
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import collections
from mysports import group_list_by_similarity
from mysports_2 import group_merge_helper, find_player_names_per_group
from collections import defaultdict
import pandas as pd
import pickle
from rapidfuzz.distance.Levenshtein import normalized_similarity
from rapidfuzz.process import extractOne
from rapidfuzz.fuzz import ratio
import matplotlib.pyplot as plt
from rapidfuzz import process

from time import time

# def find_nearest(array, value):
#     idx = (np.abs(array - value)).argmin()
#     return array[idx]


def stream_position_to_time(stream_position):
    if not stream_position:
        return None
    return datetime.fromtimestamp(stream_position / 1000.0).strftime("%H:%M:%S")


with open('./all_text_files/test_data/FreeSkiBigAir_333.txt') as f:
    sentences = f.read().splitlines()
    sentences = [tuple(x.split(',')) for x in sentences]
    sentences = [(float(x[0]), x[1], stream_position_to_time(float(x[0]))) for x in sentences]

# find athlete names in each frame
# key: stream_pos
# value: athlete's name

start_1 = time()
temp_sentences = {}
for x in sentences:
    if x[0] not in temp_sentences:
        temp_sentences[x[0]] = [x[1]]
    temp_sentences[x[0]].append(x[1])

end_1 = time() - start_1

temp_athlete_stream_pos = list(temp_sentences.keys())
temp_athlete_names_per_frame = list(temp_sentences.values())

athlete_names = list(temp_sentences.values())
athlete_groups = []
current_group = []


def check_athlete_in_group(frame_athlete_names, current_group):
    score = process.cdist(frame_athlete_names, current_group, score_cutoff=80, scorer=ratio, dtype=np.uint8)
    if np.count_nonzero(score) > 0:
        return True
    return False
    # for athlete in frame_athlete_names:
    #     athlete_extracted = extractOne(athlete,
    #                                    current_group,
    #                                    score_cutoff=85,
    #                                    scorer=ratio
    #                                    )
    #     if athlete_extracted:
    #         return True
    # return False

start_2 = time()
for i, frame_athlete_names in enumerate(athlete_names):
    if len(current_group) == 0:
        current_group.append(frame_athlete_names)
    # elif check_athlete_in_group(frame_athlete_names, current_group[-1]):
    # elif check_athlete_in_group(frame_athlete_names, find_player_names_per_group([a for x in current_group for a in x])):
    elif check_athlete_in_group(frame_athlete_names, [a for x in current_group for a in x]):
        current_group.append(frame_athlete_names)
        # if len(current_group[-1]) > len(frame_athlete_names):
        #     current_group.append(current_group[-1])
        # else:
        #     current_group.append(frame_athlete_names)
    # elif set(frame_athlete_names).intersection(set(current_group[-1])):
    #     current_group.append(frame_athlete_names)
    else:
        athlete_groups.append((i - len(current_group), i - 1))
        current_group = [frame_athlete_names]
end_2 = time() - start_2
if len(current_group) > 0:
    athlete_groups.append((len(athlete_names) - len(current_group), len(athlete_names) - 1))

start_3 = time()
clusters = []
for group in athlete_groups:
    cluster = []
    for i in range(group[0], group[1] + 1):
        cluster.append([temp_athlete_stream_pos[i], temp_athlete_names_per_frame[i],
                        stream_position_to_time(temp_athlete_stream_pos[i])])

    if len(cluster) > 3:
        # athlete_names_in_group = set([x.split(" ")[-1].upper() for a in cluster for x in a[1]])
        # athlete_names_in_group = set([x.upper() for a in cluster for x in a[1]])
        # athlete_names_in_group = sorted(list(athlete_names_in_group))
        # string_from_list = " ".join(athlete_names_in_group)
        # clusters.append((string_from_list, cluster))

        # new implementation
        athlete_names = [x.upper() for a in cluster for x in a[1]]
        group_athlete_names = find_player_names_per_group(athlete_names)
        string_from_list = " ".join(sorted(group_athlete_names))
        clusters.append((string_from_list, cluster))
end_3 = time() - start_3

start_4 = time()
sentences = []
for cluster in clusters:
    frame_string = cluster[0]
    for val in cluster[1]:
        sentences.append((val[0], frame_string, stream_position_to_time(val[0]), cluster))
end_4 = time() - start_4
# athlete_dict = {}
# # temp_sentences = sorted(temp_sentences)
# for key, value in temp_sentences.items():
#     value = sorted(value)
#     frame_string = " ".join(value).upper()
#     if len(athlete_dict) == 0:
#         athlete_dict[frame_string] = [(key, frame_string, stream_position_to_time(key))]
#         continue
#     extract_key = extractOne(frame_string, athlete_dict.keys(), score_cutoff=50, scorer=ratio)
#     if extract_key:
#         athlete_dict[extract_key[0]].append((key, frame_string, stream_position_to_time(key)))
#     else:
#         athlete_dict[frame_string] = [(key, frame_string, stream_position_to_time(key))]

# way 1
# sentences = []
# for key, value in temp_sentences.items():
#     value = sorted(value)
#     frame_string = " ".join(value).upper()
#     sentences.append((key, frame_string, stream_position_to_time(key)))
# print()

# way 2
# athlete_dict = {}
# i = 0
# for key, value in temp_sentences.items():
#     value = sorted(value)
#     if len(athlete_dict) == 0:
#         athlete_dict[0] = [(value, key)]
#         continue
#
#     if len(value) == 1:
#         i += 1
#         athlete_dict[i] = [(value, key)]
#         continue
#
#     from_list = set([a for x in athlete_dict[i] for a in x[0]])
#     count = 0
#     for val in value:
#         extract_value = extractOne(val, from_list, score_cutoff=75)
#         if extract_value:
#             count += 1
#
#     if count > 0:
#         athlete_dict[i].append((value, key))
#     else:
#         i += 1
#         athlete_dict[i] = [(value, key)]

# sentences = []
# for key, value in athlete_dict.items():
#     from_list_2 = sorted(set([a for x in value for a in x[0]]))
#
#     # from_list_2 = sorted([a for x in value for a in x[0]])
#     cluster_list = []
#     cluster_list.append(dict(collections.Counter([x for x in from_list_2])))
#     # # if len(from_list_2) == 1:
#     # #     frame_string = [max(x, key=x.get) for x in cluster_list][0].upper()
#     # count = sum([cluster_list[0][x] for x in cluster_list[0]]) / len(cluster_list[0])
#     # from_list_2 = sorted([a for x in cluster_list for a in x if x[a] > count])
#
#     # naming issue
#     frame_string = [max(x, key=x.get) for x in cluster_list][0].upper()
#     # frame_string = " ".join(from_list_2).upper()
#     for val in value:
#         sentences.append((val[1], frame_string, stream_position_to_time(val[1]), from_list_2))

unique, counts = np.unique([x[0] for x in sentences], return_counts=True)

start_5 = time()
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
end_5 = time() - start_5
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
    # cluster_list = []
    # cluster_list.append(dict(collections.Counter([x[1] for x in cluster])))
    # cluster_list_string = [max(x, key=x.get) for x in cluster_list][0].upper()
    cluster_list_string = clustered_sentences[i][0][1]
    keys.append(cluster)
    values.append(cluster_list_string)

# merge similar groups
# res = group_list_by_similarity(listN=values, data_source=keys, score_cutoff=0.25)
res = group_merge_helper(listN=values, data_source=keys, score_cutoff=60)


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

    # cluster_athlete_names = [b for x in cur_cluster for a in x[3][1] for b in a[1]]
    # cluster_list_string = find_player_names_per_group(cluster_athlete_names, flag=True)

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
    # plt.scatter(x=stream_pos[:, 0], y=stream_pos[:, 1])
    # plt.show()

    clusterer = hdbscan.HDBSCAN(gen_min_span_tree=True, allow_single_cluster=True, cluster_selection_epsilon=200000)
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

# result = sorted(result, key=lambda k: k['cur_cluster_min'])
cluster_df = pd.DataFrame.from_dict(result)
x = cluster_df[['cur_cluster_size', 'cur_cluster_length']]
# x = cluster_df[['cur_cluster_size', 'cur_cluster_length', 'length_size_ratio']]
# load scaler
# scl = './saved_models/final_test/combined_data_sc_without_ratio.pkl'
scl = './saved_models/all_v2/combined_data_sc_without_ratio.pkl'
# scl = './saved_models/updated_model/sc_data_4_all_features.pkl'
scaler = pickle.load(open(scl, 'rb'))
x = scaler.transform(x)

# filename = 'saved_models/final_test/combined_data_without_ratio.sav'
filename = 'saved_models/all_v2/combined_data_without_ratio.sav'
# filename = 'saved_models/updated_model/combined_model_all_features.sav'
loaded_model = pickle.load(open(filename, 'rb'))
predictions = loaded_model.predict(x)
cluster_df['is_race'] = predictions
print("")
