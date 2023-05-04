from yellowbrick.cluster import KElbowVisualizer
from sentence_transformers import SentenceTransformer
import hdbscan
from datetime import datetime
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
import collections
from mysports import group_list_by_similarity
from collections import defaultdict
import math
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from scipy.stats import norm


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def stream_position_to_time(stream_position):
    if not stream_position:
        return None
    return datetime.fromtimestamp(stream_position / 1000.0).strftime("%H:%M:%S")


with open('./athlete_list_txt_file/big_air_athlete_list_10688.txt') as f:
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
for i, cluster in clustered_sentences.items():
    cluster_list = []
    cluster_list.append(dict(collections.Counter([x[1] for x in cluster])))
    cluster_list_string = [max(x, key=x.get) for x in cluster_list][0].upper()

    keys.append(cluster)
    values.append(cluster_list_string)

res = group_list_by_similarity(listN=values, data_source=keys)


def cluster_object_creator(cur_cluster, cluster_index=None):
    cur_cluster_min = min(cur_cluster, key=lambda t: t[0])[0]
    cur_cluster_max = max(cur_cluster, key=lambda t: t[0])[0]
    cur_cluster_size = len(cur_cluster)
    cur_cluster_length = cur_cluster_max - cur_cluster_min
    length_size_ratio = cur_cluster_length // cur_cluster_size
    # next_cluster_max = max(cluster_dict[cluster_index + 1][1], key=lambda t: t[0])[0]
    # cur_cluster_distance_with_next_cluster = next_cluster_max - cur_cluster_min
    data = {
        # f"cluster_name": f"{cluster_list_string}_{cluster_index}",
        "cluster": cur_cluster,
        # "next_cluster": cluster_dict[cluster_index + 1][1],
        "cur_cluster_size": cur_cluster_size,
        "cur_cluster_length": cur_cluster_length,
        "length_size_ratio": length_size_ratio,
        # "cur_cluster_distance": cur_cluster_distance_with_next_cluster,
        "cur_cluster_min": cur_cluster_min,
        "cur_cluster_max": cur_cluster_max
    }
    return data


model = AgglomerativeClustering()
result = []
all_distance = []

for_plot_x = []
for_plot_y = []
for i, cluster in enumerate(res):
    stream_pos = [x[0] for x in cluster]
    stream_pos = np.array(stream_pos)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True, alpha=1.0, cluster_selection_epsilon=3000)
    # clusterer = hdbscan.RobustSingleLinkage(cut=10000, k=15)
    clusterer.fit(stream_pos.reshape(-1, 1))
    # k_value = math.floor((len(np.unique(np.array(clusterer.labels_)))) / 2)
    k_value = len(np.unique(np.array(clusterer.labels_)))
    if k_value <= 0:
        k_value = 1
    # print(f"cluster: {cluster[0][1]}, agglomerative: {k_value}, hdbscan: {len(np.unique(np.array(clusterer.labels_)))}")

    # clusters = hdbscan.HDBSCAN(min_cluster_size = 1,
    #                            metric='euclidean',
    #                            cluster_selection_method='eom').fit(stream_pos.reshape(-1, 1))

    clustering_model = AgglomerativeClustering(distance_threshold=None, n_clusters=k_value, linkage='complete')
    # clustering_model = KMeans(n_clusters=k_value)
    clustering_model.fit(stream_pos.reshape(-1, 1))

    stream_pos_2 = []
    y = []
    # if cluster[0][1] == 'SANDRA EIE' or cluster[0][1] == 'TESS LEDEUX':
    # if cluster[0][1] == 'TESS LEDEUX':

    if k_value > 1:
        # print(len(np.unique(np.array(clusterer.labels_))))
        # print(clustering_model.cluster_centers_)
        # dists = euclidean_distances(clustering_model.cluster_centers_)
        # tri_dists = dists[np.triu_indices(k_value, 1)]
        # max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
        # print(np.diag(dists, k=1).mean())
        # # print(dists)
        # print(np.diag(dists, k=1))
        labels = clustering_model.labels_
        # # print(np.unique(np.array(clusterer.labels_))[-1] + 1)
        # # clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        # # clusterer.condensed_tree_.plot()
        # # clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
        cluster_dict = defaultdict(list)
        for cluster_index, cluster_id in enumerate(labels):
            if cluster_id == -1:
                continue
            cluster_dict[cluster_id].append(cluster[cluster_index])

        cluster_dict = sorted(cluster_dict.items(), key=lambda k_v: k_v[1][0][0])
        # print(cluster_dict)
        cluster_2 = []
        for yy, item in enumerate(cluster_dict):
            cluster_2.append(item[1])
            stream_pos_2.append(item[1][0][0])
            # result.append(item)
            # print(item)
            # if len(item[1]) > 30:
            #     cluster_2.append(item[1])
            #     stream_pos_2.append(item[1][0][0])
            #     y.append(0)

        stream_pos_2 = np.array(stream_pos_2)
        # print(stream_pos_2)
        dists = euclidean_distances(stream_pos_2.reshape(-1, 1))
        distances = np.diag(dists, k=1)

        # all distance
        all_distance.extend(distances.tolist())

        # val = find_nearest(distances, distances.mean())
        # for_plot_x = []
        # for_plot_y = []
        # for j in range(len(distances)):
        #     for_plot_x.append(stream_pos_2[j])
        #     for_plot_y.append(distances[j])

        # for_plot_x = np.array(for_plot_x)
        # for_plot_y = np.array(for_plot_y)

        # find line of best fit
        # a, b = np.polyfit(for_plot_x, for_plot_y, 1)
        #
        # # add points to plot
        # plt.scatter(for_plot_x, for_plot_y)
        #
        # # add line of best fit to plot
        # plt.plot(for_plot_x, a * for_plot_x + b)
    # print(np.diag(dists, k=1).min())
    # print(stream_pos_2)
    # plt.scatter(for_plot_x, for_plot_y)
    # plt.show()

    # visualizer = KElbowVisualizer(model, k=(1, len(stream_pos_2)))
    # visualizer.fit(stream_pos_2)  # Fit the data to the visualizer
    # # visualizer.show()
    # k_value = visualizer.elbow_value_

    # clustering_model = AgglomerativeClustering(distance_threshold=distances.min() * 2, n_clusters=None,
    #                                            linkage='complete')
    # clustering_model.fit(stream_pos_2.reshape(-1, 1))
    # labels = clustering_model.labels_
    # # # print(stream_pos_2)
    # # clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True, alpha=1.0, min_samples=1, cluster_selection_epsilon=np.diag(dists, k=1).min())
    # # # # clusterer = hdbscan.RobustSingleLinkage(cut=300000, k=10)
    # # clusterer.fit(stream_pos_2.reshape(-1, 1))
    # # labels = clusterer.labels_
    # # # print(cluster[0][1])
    # # # print(stream_pos_2)
    # # print(labels)
    # # # print("second", len(cluster_2), len(labels))
    # cluster_dict = defaultdict(list)
    # for cluster_index, cluster_id in enumerate(labels):
    #     if cluster_id == -1:
    #         # print(cluster_2[cluster_index])
    #         result.append(cluster_2[cluster_index])
    #         continue
    #     cluster_dict[cluster_id].extend(cluster_2[cluster_index])
    #
    # cluster_dict = sorted(cluster_dict.items(), key=lambda k_v: k_v[1][0][0])
    # # print(cluster_dict)
    # for item in cluster_dict:
    #     # print(item)
    #     # print(item[1])
    #     result.append(cluster_object_creator(item[1]))

all_distance = sorted(all_distance)


# Creating a Function.
def normal_dist(x, mean, sd):
    prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density


# Calculate mean and Standard deviation.
mean = np.mean(np.array(all_distance))
sd = np.std(np.array(all_distance))

pdf = normal_dist(np.array(all_distance), mean, sd)

# Plotting the Results
plt.plot(np.array(all_distance), pdf, color='red')
plt.xlabel('Data points')
plt.ylabel('Probability Density')
plt.show()
#
# from scipy.stats import percentileofscore
#
# a = np.array([percentileofscore(np.array(all_distance), i, kind='strict') for i in np.array(all_distance)])
# a = np.array([round(x) for x in a])
# a_uniqe = np.unique(a, return_counts=True)
# vals = {}
# percetial_list = [25, 50, 75, 100]
# start = 0
# for per in percetial_list:
#     vals[per] = len(a[np.where((a > start) & (a <= per))])
#     start = per
val = np.percentile(np.array(all_distance), (1 - np.array(pdf).argmax() / len(pdf)) * 100) * 1.75
# val = mean / 2
# val = np.percentile(np.array(all_distance), 40)
# val2 = norm(np.array(all_distance))
for i, cluster in enumerate(res):
    stream_pos = [x[0] for x in cluster]
    stream_pos = np.array(stream_pos)

    clustering_model = AgglomerativeClustering(distance_threshold=val, n_clusters=None,
                                               linkage='complete')
    clustering_model.fit(stream_pos.reshape(-1, 1))
    labels = clustering_model.labels_

    cluster_dict = defaultdict(list)
    for cluster_index, cluster_id in enumerate(labels):
        if cluster_id == -1:
            continue
        cluster_dict[cluster_id].append(cluster[cluster_index])

    cluster_dict = sorted(cluster_dict.items(), key=lambda k_v: k_v[1][0][0])
    for item in cluster_dict:
        result.append(cluster_object_creator(item[1]))

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
