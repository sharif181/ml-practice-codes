from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import itertools
from operator import itemgetter
from datetime import datetime
# Elbow Method for K means
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
import scipy.cluster.hierarchy as sch

from sklearn.metrics import davies_bouldin_score
from sklearn.mixture import GaussianMixture


# Gap Statistic for K means
def optimalK(data, nrefs=2, maxClusters=5):
    """
    Calculates KMeans optimal K using Gap Statistic
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp
        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)


# Davies-Bouldin Index
def get_kmeans_score(cluster_df, center):
    '''
    returns the kmeans score regarding Davies Bouldin for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the Davies Bouldin score for the kmeans model fit to the data
    '''
    # instantiate kmeans
    kmeans = KMeans(n_clusters=center)
    # Then fit the model to your data using the fit method
    model = kmeans.fit_predict(cluster_df)

    # Calculate Davies Bouldin score
    score = davies_bouldin_score(cluster_df, model)

    return score


def stream_position_to_time(stream_position):
    if not stream_position:
        return None
    return datetime.fromtimestamp(stream_position / 1000.0).strftime("%H:%M:%S")


### read data
temp_sentences = []
athlete_list = []
stream_pos = []
with open('./athlete_list.txt') as f:
    sentences = f.read().splitlines()
    sentences = [tuple(x.split(',')) for x in sentences]
    sentences = [(int(x[0]), x[1], stream_position_to_time(int(x[0]))) for x in sentences]
    sentences = sorted(sentences, key=itemgetter(0))

# map every athlete in the same frame
cluster_dict = {}
cluster_list = []
for key, group in itertools.groupby(sentences, key=itemgetter(0)):
    group_content = list(group)
    athletes = [x[1] for x in group_content]
    athlete_names = ' ,'.join(athletes).upper()  # at least avoiding case-insensitive

    # # my code for pd dataframe
    # for athlete in athletes:
    #     athlete_list.append(athlete)
    #     stream_pos.append(group_content[0][0])

    athlete_list.append(athlete_names)
    stream_pos.append(group_content[0][0])

    temp_sentences.append((group_content[0][0], athlete_names, group_content[0][2]))

    cluster_element = {
        'stream_pos_time': group_content[0][2],
        'athlete_names': athlete_names,
        'stream_position': group_content[0][0],
        'athletes_raw': athletes
    }
    cluster_dict[key] = cluster_element
    cluster_list.append(cluster_element)

# create DataFrame
df = pd.DataFrame({'athlete': athlete_list,
                   'stream_pos': stream_pos})

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

# keep remainder col name
remainder_col = transformed_df.columns[-1]
result = []

model = KMeans()

# iterate for each athlete
for athlete in transformed_df.columns[0:-1]:
    # for simplicity, ignoring multiple athlete name in same frame
    if len(athlete.split(',')) > 1:
        continue

    new_data_frame = transformed_df[[athlete, remainder_col]]  # slice dataframe for each athlete

    new_data_frame = new_data_frame[new_data_frame != 0].dropna()

    if len(new_data_frame) <= 5:
        continue

    stream_data = new_data_frame[remainder_col].to_numpy()
    stream_data = stream_data.reshape(-1, 1)

    # Gap Statistic
    score_g, df = optimalK(stream_data, nrefs=2, maxClusters=6)
    plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b')
    plt.xlabel('K')
    plt.ylabel('Gap Statistic')
    plt.title('Gap Statistic vs. K')
    plt.show()

    # Elbow Method
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2, 6), timings=True)
    visualizer.fit(stream_data)  # Fit data to visualizer
    visualizer.show()  # Finalize and render figure
    value = visualizer.elbow_value_

    # silhouette
    visualizer = KElbowVisualizer(model, k=(2, 6), metric='silhouette', timings=True)
    visualizer.fit(stream_data)  # Fit the data to the visualizer
    visualizer.show()
    k_value = visualizer.elbow_value_

    # calinski_harabasz
    visualizer = KElbowVisualizer(model, k=(2, 6), metric='calinski_harabasz', timings=True)
    visualizer.fit(stream_data)  # Fit the data to the visualizer
    visualizer.show()
    k_value = visualizer.elbow_value_

    # Davies-Bouldin Index
    scores = []
    centers = list(range(2, 6))
    k_nums = []
    for center in centers:
        scores.append(get_kmeans_score(stream_data, center))
        k_nums.append(center)

    plt.plot(centers, scores, linestyle='--', marker='o', color='b');
    plt.xlabel('K')
    plt.ylabel('Davies Bouldin score')
    plt.title('Davies Bouldin score vs. K')
    plt.show()
    value = min(scores)
    index = scores.index(value)
    cluster_num = k_nums[index]
    # minimum score will be result

    # dendrogram
    plt.figure(figsize=(20, 10))
    sch.dendrogram(sch.linkage(new_data_frame, method="complete"))
    plt.title("Dendrogram complete")
    plt.xlabel(athlete)
    plt.ylabel("Euclidean distances")
    plt.show()

    # Bayesian information criterion
    n_components = range(2, 6)
    covariance_type = ['spherical', 'tied', 'diag', 'full']
    score = []
    for cov in covariance_type:
        for n_comp in n_components:
            gmm = GaussianMixture(n_components=n_comp, covariance_type=cov)
            gmm.fit(stream_data)
            score.append((cov, n_comp, gmm.bic(stream_data)))
    s = min(score, key=itemgetter(2))
    cluster_num = s[1]
    print("end")
    # min score will be result
