from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from Levenshtein import distance as levenshtein_distance
from rapidfuzz import process, fuzz
from rapidfuzz.distance.Levenshtein import normalized_similarity
from mysports import group_list_by_similarity
import umap

#
# # Define a list of names to be clustered
# # names = ['John', 'Jonathan', 'Jon', 'Johnny', 'Joan', 'Joanne']
with open('./athlete_list_txt_file/big_air_athlete_list_10688.txt') as f:
    sentences = f.read().splitlines()
    sentences = [tuple(x.split(',')) for x in sentences]
    sentences = [(float(x[0]), x[1]) for x in sentences]

unique, counts = np.unique([x[0] for x in sentences], return_counts=True)

names = [x[1] for x in sentences]
#
#
# # Define a function to compute the distance between pairs of names
# def name_distance(name1, name2):
#     return levenshtein_distance(name1, name2)
#
#
# # Compute the pairwise distance matrix between names
# n_names = len(names)
# # dist_matrix = np.zeros((n_names, n_names))
# # for i in range(n_names):
# #     for j in range(i + 1, n_names):
# #         dist_matrix[i, j] = name_distance(names[i], names[j])
# #         dist_matrix[j, i] = dist_matrix[i, j]
# # dist_matrix = process.cdist(names, names, scorer=fuzz.ratio, dtype=np.uint8,
# #                             score_cutoff=.75)
# res = group_list_by_similarity(listN=names)
# # Perform agglomerative clustering on the distance matrix
# # n_clusters = 2
# clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete',
#                                      distance_threshold=20)
# # clustering.fit(dist_matrix)
# # cluster_assignment = clustering.labels_
# #
# # clustered_sentences = {}
# # for sentence_id, cluster_id in enumerate(cluster_assignment):
# #     if cluster_id not in clustered_sentences:
# #         clustered_sentences[cluster_id] = []
# #
# #     clustered_sentences[cluster_id].append(sentences[sentence_id])
#

# sentences = ["K. Sharif", "Sharif", "Khan"]

embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(names)


def generate_clusters(message_embeddings,
                      n_neighbors,
                      n_components,
                      min_cluster_size,
                      random_state=None):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    """

    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors,
                                 n_components=n_components,
                                 metric='cosine',
                                 random_state=random_state)
                       .fit_transform(message_embeddings))

    # clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
    #                            metric='euclidean',
    #                            cluster_selection_method='eom').fit(umap_embeddings)
    clusters = AgglomerativeClustering(n_clusters=min_cluster_size, affinity='precomputed', linkage='complete').fit(
        umap_embeddings)

    return clusters


cluster = generate_clusters(message_embeddings=embeddings, n_neighbors=15, n_components=5,
                            min_cluster_size=10)
print()
# import numpy as np
# from matplotlib import pyplot as plt
#
# # Set the figure size
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True
#
# # Random data of 100×3 dimension
# # data = np.array(np.random.random((100, 3)))
#
# # Scatter plot
# plt.scatter(embeddings[:, 0], embeddings[:, 1], c=embeddings[:, 2], cmap='hot')
#
# # Display the plot
# plt.show()
# Print the cluster assignments for each name
# for i in range(n_names):
#     print(names[i], 'belongs to cluster', clustering.labels_[i])


print()
