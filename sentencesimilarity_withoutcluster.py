import itertools
import operator
import time
from collections import defaultdict
from datetime import datetime
from operator import itemgetter

import Levenshtein
import numpy as np
from matplotlib import pyplot as plt
from numpy.testing._private.parameterized import param
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.neighbors import kneighbors_graph

from mysports import group_list_by_similarity


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def merge_set(paraphrases):
    def merge_set_inner(key, row):
        # set all values for the given key
        mc = my_collection_merged.get(key, [])
        mc.append(key)
        mc.extend(row)
        my_collection_merged[key] = mc

    def get_values_by_key(key):
        return my_collection.get(key, [])

    def execute_ex(key, results=[], processed=set()):
        if key not in processed:
            row = get_values_by_key(key)
            results.append(key)
            results.extend(row)
            processed.add(key)
            for r in row:
                execute_ex(r, results, processed)
        return results

        # expand values

    # (0, 1)
    # paraphrase_list_copy = paraphrase_list.copy()
    my_collection = defaultdict(list)

    # x-values
    paraphrases_x = [x for x, y in paraphrases]

    # y-values
    paraphrases_y = [y for x, y in paraphrases]

    # build dictionary
    for i, j in zip(paraphrases_x, paraphrases_y):
        my_collection[i].append(j)

    my_collection_merged = {}
    processed = set()

    # take first element

    for k, v in my_collection.items():
        if k in processed:
            continue
        final_row = execute_ex(k, results=[], processed=set())
        processed = processed.union(set(final_row))
        my_collection_merged[k] = list(set(final_row))
    return my_collection_merged


def stream_position_to_time(stream_position):
    if not stream_position:
        return None
    return datetime.fromtimestamp(stream_position / 1000.0).strftime("%H:%M:%S")


model = SentenceTransformer('all-MiniLM-L6-v2')

with open('./athletes_skicross.txt') as f:
    sentences = f.read().splitlines()
    sentences = [tuple(x.split(',')) for x in sentences]
    sentences = [(int(x[0]), x[1], stream_position_to_time(int(x[0]))) for x in sentences]
    sentences = sorted(sentences, key=itemgetter(0))

# unique, counts = np.unique([x[0] for x in sentences], return_counts=True)
# X = np.concatenate((unique, counts), axis=1)
# X =np.array(list(zip(list(unique), list(counts))))
# X = np.array(unique, counts)

# cluster detections into time groups with a max distance of around 1500ms
# X = np.array([[x[0], 0] for x in sentences])
# dbscan = DBSCAN(eps=1500, min_samples=2).fit(X)
# labels = dbscan.labels_  # getting the labels

# Perform kmean clustering
# clustering_model = AgglomerativeClustering(distance_threshold=0.8, n_clusters=None, affinity='euclidean', linkage='complete')
# clustering_model.fit(X)
# labels = clustering_model.labels_
#
# # Plot the clusters
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="plasma")  # plotting the clusters
# plt.xlabel("Time")  # X-axis label
# plt.ylabel("Athletes")  # Y-axis label
# plt.show()  # showing the plot

# map every athlete in the same frame
cluster_dict = {}
cluster_list = []
for key, group in itertools.groupby(sentences, key=itemgetter(0)):
    group_content = list(group)
    athletes = [x[1] for x in group_content]
    athlete_names = ' ,'.join(athletes)
    cluster_element = {
        'stream_pos_time': group_content[0][2],
        'athlete_names': athlete_names,
        'stream_position': group_content[0][0],
        'athletes_raw': athletes
    }
    cluster_dict[key] = cluster_element
    cluster_list.append(cluster_element)
    # for key, group:

    # s[0] = Stream Position
    # s[1] = Detected text
    # s[2] = Stream Position Human readable
    # cluster_dict[s[0]].append(s)

# TODO: That's good code but not needed right now
# for each cluster find most common athletes
race_segments = [] # TODO: Make sure one cluster has only data of a single race segment and not also of a rpevious race
for cluster in cluster_dict.values():
    stream_pos_min = min(cluster, key=itemgetter(0))[0]
    stream_pos_max = max(cluster, key=itemgetter(0))[0]

    string_groups = group_list_by_similarity(sorted([x[1].upper() for x in cluster]))

    string_group_values_median = []
    score = sum([y[1] for x in string_groups for y in x])
    for item in string_groups:
        string_group_values_median.append(Levenshtein.setmedian([x[0] for x in item]))

    names = sorted(string_group_values_median) # sort and transform names into list
    race_segments.append({
        "key": ' '.join(names),
        "stream_pos_min_time": stream_position_to_time(stream_pos_min),
        "stream_pos_max_time": stream_position_to_time(stream_pos_max),
        "stream_pos_min": stream_pos_min,
        "stream_pos_max": stream_pos_max,
        "score": score,
        "names": names,
    })

# find similarity of two clusters

# athlete_conv = defaultdict(list)
# for k,v in cluster_dict.items():

# for idx, race_segment in enumerate(race_segments):
#
#     athlete_names1 = ' '.join(race_segment['names'])
#
#     for next_race_segment in race_segments[idx:]:
#         athlete_names2 = ' '.join(next_race_segment['names'])
#         ratio = fuzz.ratio(athlete_names1, athlete_names2)
#         if ratio > 80:
#             athlete_conv[athlete_names1].append((athlete_names2, ratio, race_segment['stream_pos_min'], next_race_segment['stream_pos_max']))
#

# athlete_names = [' '.join(x['names']) for x in race_segments]
# string_groups = group_list_by_similarity(athlete_names)


# foo_list = group_list_by_similarity(athlete_names, score_cutoff=0.5)
# lol = 2

# embeddings = model.encode(athlete_names, convert_to_tensor=True)
# cosine_scores = util.cos_sim(embeddings, embeddings)
#
# pairs = []
# threshold = 0.9
# for i in range(len(cosine_scores)-1):
#     for j in range(i+1, len(cosine_scores)):
#         if cosine_scores[i][j] < threshold:
#             continue
#         pairs.append({'index': [i,j], 'score': cosine_scores[i][j]})
# pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
#
# pair_strings = []
# for pair in pairs:
#     i, j = pair['index']
#     pair_strings.append((cluster_list[i], cluster_list[j], pair['score']))


# paraphrases_old = util.paraphrase_mining(model, athlete_names)
#
# athlete_names = [x['athlete_names'] for x in cluster_list]
# print(f"start embedding: {time.time()}")
# athlete_embedding = model.encode(athlete_names, convert_to_tensor=True)
# print(f"finish embedding: {time.time()}")
# cosine_scores = util.cos_sim(athlete_embedding, athlete_embedding)
# cosine_array = np.asarray(cosine_scores)
#
# #TODO: use proper nparray mechanism to pull values to 0 and 1
# print(f"start pull to 0 and 1: {time.time()}")
# threshold = 0.9
# # cosine_scores[cosine_scores >= threshold] = 1
# # cosine_scores[cosine_scores < threshold] = 0
#
#
# outliers_removed = reject_outliers(cosine_array)
#
# X_nonzero = np.nonzero(cosine_array)
#
# funky_cluster = defaultdict(list)
# processed = set()
# for x in X_nonzero[0]:
#     if x in processed:
#         continue
#     for y in X_nonzero[1]:
#         funky_cluster[x].append(cluster_list[y])
#         processed.add(y)
#
#
#
# X_nonzero = np.where(cosine_array > 0.9)
# X = np.array([X_nonzero[0], X_nonzero[1]], dtype=np.dtype('float')).transpose()
#
#
#
# paraphrases = util.paraphrase_mining(model, athlete_names)
# paraphrases_local = paraphrases
# paraphrases_local = [(s,x,y) for s,x,y in paraphrases if (idx == x or idx == y)]

# idea
# check similarity between two adjoining groups
# if they are similar, then merge them
# keep going until they are not similar anymore
# continue with next group

my_cluster_dict = defaultdict(list)
cluster_id = 0
idx = 0
start = time.time()
# break loop after iterating all element of cluster list
while idx < len(cluster_list):
    # for idx, cluster in enumerate(cluster_list):
    if idx == 0:
        my_cluster_dict[cluster_id].append(cluster_list[idx])
        idx += 1
        continue

    my_cluster_list = my_cluster_dict[cluster_id]
    # athlete_names = [x['athlete_names'] for x in my_cluster_list] + [cluster_list[idx]['athlete_names']]
    cluster_embeddings = model.encode([x['athlete_names'] for x in my_cluster_list], convert_to_tensor=True)
    athlete_embedding = model.encode([cluster_list[idx]['athlete_names']], convert_to_tensor=True)

    cosine_scores = util.cos_sim(cluster_embeddings, athlete_embedding)

    # paraphrases = util.paraphrase_mining(model, athlete_names)
    # paraphrases_local = paraphrases
    # paraphrases_local = [(s,x,y) for s,x,y in paraphrases if (idx == x or idx == y)]
    mean = sum([x for x in cosine_scores]) / len(my_cluster_list)
    # mean = sum([s for s, x, y in paraphrases_local]) / len(paraphrases_local)

    # if mean > 0.8:
    if any([x for x in cosine_scores if x > 0.8]):
        my_cluster_list.append(cluster_list[idx])
    else:
        cluster_id += 1
        my_cluster_dict[cluster_id].append(cluster_list[idx])
        pass
    idx += 1
end = time.time()
print("total time ", end - start)
# paraphrases_orig = paraphrases

cluster_group = defaultdict(set)
# [cluster_group[i].add(j) for score, i, j in paraphrases if score > 0.8]
# paraphrases = [(race_segments[i], race_segments[j]) for score, i, j in paraphrases if score > 0.8]

# threshold filter
paraphrases = [(i, j) for score, i, j in paraphrases if score >= 0.9]
paraphrases = sorted(paraphrases, key=itemgetter(0))
paraphrases_merged = merge_set(paraphrases)

merged_clusters = []

for idx, x in enumerate(paraphrases_merged.values()):
    if len(merged_clusters) < idx + 1:
        merged_clusters.append([])
        for ix in x:
            merged_clusters[idx].append(cluster_list[ix])
        merged_clusters[idx].sort(key=lambda x: x['stream_position'])

# merged_clusters = [x for x in paraphrases_merged.values()]

# cluster detections into time groups with a max distance of around 1500ms
for merged_cluster in merged_clusters:
    X = np.array([[x['stream_position'], 0] for x in merged_cluster])
    # dbscan = DBSCAN(eps=3000, min_samples=2).fit(X)
    # labels = dbscan.labels_ # getting the labels

    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=5000, affinity='manhattan',
                                               linkage='complete', compute_distances=True)
    clustering_model.fit(X)
    labels = clustering_model.labels_

    # map athletes to clusters
    cluster_dict = defaultdict(list)
    for idx, cluster in enumerate(labels):
        if cluster == -1:
            continue
        cluster_dict[cluster].append(merged_cluster[idx])

# for paraphrase in [(athlete_names[i], athlete_names[j], score) for score, i, j in paraphrases if score > 0.8]:
#     score, i, j = paraphrase
#     print("{} \t\t {} \t\t Score: {:.4f}".format(athlete_names[i], athlete_names[j], score))

corpus_embeddings = model.encode(athlete_names)
corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

# Perform kmean clustering
clustering_model = AgglomerativeClustering(n_clusters=None,
                                           distance_threshold=1.5)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences_inner = defaultdict(list)
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences_inner[cluster_id].append(race_segments[sentence_id])
#
# clustered_sentences_inner = [x for x in clustered_sentences_inner.values() if len(x) > 7]
#
# if len(clustered_sentences_inner) > 0:
#     clustered_sentences[k] = clustered_sentences_inner


for race_segment in race_segments:
    athlete_names = [''.joinx['names'] for x in race_segment]

    athlete_names = [x[1] for x in v]
    corpus_embeddings = model.encode(athlete_names)
    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=1.5)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences_inner = defaultdict(list)
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences_inner[cluster_id].append(cluster_dict[k][sentence_id])

    clustered_sentences_inner = [x for x in clustered_sentences_inner.values() if len(x) > 7]

    if len(clustered_sentences_inner) > 0:
        clustered_sentences[k] = clustered_sentences_inner
        # if cluster_id not in clustered_sentences[k]:
        #     clustered_sentences[k] = []
        #
        # # clustered_sentences[cluster_id].append(sentences[sentence_id]) # adds the whole sentence object
        # try:
        #     clustered_sentences[k].append(cluster_dict[cluster][sentence_id][1])
        # except:
        #     pass # TODO
for i, cluster in clustered_sentences.items():
    if len(cluster) == 0:
        continue

    print("Cluster ", i + 1)
    print(' \n'.join(cluster))
    print("")
