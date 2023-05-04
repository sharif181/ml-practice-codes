from collections import defaultdict
from mysports import group_list_by_similarity
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from datetime import datetime
from operator import itemgetter
import itertools
from yellowbrick.cluster import KElbowVisualizer
import Levenshtein
from scipy.signal import find_peaks
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

### for dendrogram
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import ward, median, centroid, weighted, average, complete, single, fcluster
from scipy.spatial.distance import pdist


# find outlier
def find_outliers_IQR(df):
    q1 = df.quantile(0.25)

    q3 = df.quantile(0.75)

    IQR = q3 - q1

    outliers = df[((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR)))]

    return outliers


# drop outlier
def drop_outliers_IQR(df):
    q1 = df.quantile(0.25)

    q3 = df.quantile(0.75)

    IQR = q3 - q1

    not_outliers = df[~((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR)))]

    outliers_dropped = not_outliers.dropna().reset_index()

    return outliers_dropped


### dendrogram method
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


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

# for each cluster find most common athletes
# race_segments = [] # TODO: Make sure one cluster has only data of a single race segment and not also of a previous race
# for cluster in cluster_dict.values():
#     stream_pos_min = min(cluster, key=itemgetter(0))[0]
#     stream_pos_max = max(cluster, key=itemgetter(0))[0]
#
#     string_groups = group_list_by_similarity(sorted([x[1].upper() for x in cluster]))
#
#     string_group_values_median = []
#     score = sum([y[1] for x in string_groups for y in x])
#     for item in string_groups:
#         string_group_values_median.append(Levenshtein.setmedian([x[0] for x in item]))
#
#     names = sorted(string_group_values_median) # sort and transform names into list
#     race_segments.append({
#         "key": ' '.join(names),
#         "stream_pos_min_time": stream_position_to_time(stream_pos_min),
#         "stream_pos_max_time": stream_position_to_time(stream_pos_max),
#         "stream_pos_min": stream_pos_min,
#         "stream_pos_max": stream_pos_max,
#         "score": score,
#         "names": names,
#     })

# athlete = list(set(athlete_list))
# stream_pos = list(cluster_dict.keys())

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

# find non zero count per column
# for column_name in transformed_df.columns[0:-1]:
#     column = transformed_df[column_name]
#     # Get the count of non-Zeros values in column
#     count_of_non_zeros = (column != 0).sum()
#     if count_of_non_zeros <= 10:
#         transformed_df.drop([column_name], axis=1, inplace=True)

# ## DBSCAN
# cluster_dbscan = DBSCAN(eps=2000, min_samples=2).fit(transformed_df)
# labels = cluster_dbscan.labels_
# print(labels)

# clustering_model = AgglomerativeClustering(distance_threshold=20000, n_clusters=None, metric='euclidean',
#                                            linkage='ward')
# clustering_model.fit(transformed_df)
# labels = clustering_model.labels_
#
# plt.title("Hierarchical Clustering Dendrogram")
# # plot the top three levels of the dendrogram
# plot_dendrogram(clustering_model, truncate_mode="level", p=3)
# plt.xlabel("Athletes")
# plt.ylabel("Stream pos")
# plt.show()
#
# # map athletes to clusters
# cluster_dict = defaultdict(list)
# for idx, cluster in enumerate(labels):
#     if cluster == -1:
#         continue
#     cluster_dict[cluster].append(sentences[idx])
#
# print(cluster_dict)

# print(transformed_df.columns[0:-1])
#

# keep remainder col name
remainder_col = transformed_df.columns[-1]
result = []
cluster_distance = []  # store cluster distance difference

# model = KMeans(n_init='auto')
model = AgglomerativeClustering()
# iterate for each athlete
for athlete in transformed_df.columns[0:-1]:

    # for simplicity, ignoring multiple athlete name in same frame
    if len(athlete.split(',')) > 1:
        continue

    new_data_frame = transformed_df[[athlete, remainder_col]]  # slice dataframe for each athlete

    # data points plotting
    # new_data_frame.plot.scatter(x=remainder_col, y=athlete)
    # plt.show()

    new_data_frame = new_data_frame[new_data_frame != 0].dropna()  # remove rows where athlete not present

    if len(new_data_frame) <= 14:
        continue

    # new_data_frame.plot.scatter(x=remainder_col, y=athlete)
    # plt.show()

    # find outliers
    # dropped_outlier = drop_outliers_IQR(new_data_frame[remainder_col])

    # px plotting
    # fig = px.scatter(x=new_data_frame[remainder_col], y=new_data_frame[athlete])
    # fig.show()
    # silhouette
    visualizer = KElbowVisualizer(model, k=(2, 12), metric='silhouette', timings=True)
    visualizer.fit(new_data_frame[remainder_col].to_numpy().reshape(-1, 1))  # Fit the data to the visualizer
    # visualizer.show()
    k_value = visualizer.elbow_value_
    # k_value = 1

    # best fit line
    # athlete_pos = new_data_frame[athlete].to_numpy()
    # time_pos = new_data_frame[remainder_col].to_numpy()
    #
    # a, b = np.polyfit(time_pos, athlete_pos, 1)
    #
    # # add points to plot
    # plt.scatter(time_pos, athlete_pos)
    #
    # # add line of best fit to plot
    # plt.plot(time_pos, a * time_pos + b)
    #
    # plt.show()

    ### clusting for every single node
    # clustering_model = AgglomerativeClustering(distance_threshold=None, n_clusters=len(new_data_frame),
    #                                            metric='euclidean', linkage='single')
    # clustering_model.fit(new_data_frame)
    # labels = clustering_model.labels_

    # std
    # std = transformed_df[remainder_col].std()
    # mean = transformed_df[remainder_col].mean()
    # describe = transformed_df.describe()[[remainder_col]]

    ## dendrogram
    # plt.figure(figsize=(20, 10))
    # sch.dendrogram(sch.linkage(new_data_frame, method="single"))
    # plt.title("Dendrogram single")
    # plt.xlabel(athlete)
    # plt.ylabel("Euclidean distances")
    # plt.show()

    # plt.figure(figsize=(20, 10))
    # sch.dendrogram(sch.linkage(new_data_frame, method="ward"))
    # plt.title("Dendrogram ward")
    # plt.xlabel(athlete)
    # plt.ylabel("Euclidean distances")
    # plt.show()

    # plt.figure(figsize=(20, 10))
    # sch.dendrogram(sch.linkage(new_data_frame, method="complete"))
    # plt.title("Dendrogram complete")
    # plt.xlabel(athlete)
    # plt.ylabel("Euclidean distances")
    # plt.show()

    # plt.figure(figsize=(20, 10))
    # sch.dendrogram(sch.linkage(new_data_frame, method="average"))
    # plt.title("Dendrogram average")
    # plt.xlabel(athlete)
    # plt.ylabel("Euclidean distances")
    # plt.show()

    # derivative_1 = new_data_frame.diff()
    # derivative_1.fillna(0, inplace=True)
    # derivative_1 = derivative_1.rename(columns={remainder_col: "derivative_1"})
    #
    # derivative_2 = derivative_1.diff()
    # derivative_2.fillna(0, inplace=True)
    # derivative_2 = derivative_2.rename(columns={'derivative_1': 'derivative_2'})
    #
    # derivative_1 = derivative_1['derivative_1']
    # derivative_2 = derivative_2['derivative_2']
    # abs_derivative_2_val = derivative_2.abs()
    # new_data_frame['derivative_1'] = derivative_1
    # new_data_frame['derivative_2'] = derivative_2
    # new_data_frame['abs_derivative_2'] = abs_derivative_2_val
    # best fit line
    # athlete_pos = new_data_frame['derivative_1'].to_numpy()
    # time_pos = new_data_frame[remainder_col].to_numpy()
    #
    # a, b = np.polyfit(time_pos, athlete_pos, 1)
    #
    # # add points to plot
    # plt.scatter(time_pos, athlete_pos)
    #
    # # add line of best fit to plot
    # plt.plot(time_pos, a * time_pos + b)
    #
    # plt.show()
    # if athlete in ['onehotencoder__athlete_HENRIK KRISTOFFERSEN', 'onehotencoder__athlete_MARC ROCHAT']:
    #     new_data_frame.plot(x=remainder_col, y='abs_derivative_2')
    #     plt.show()
    #
    # # density plot
    # # stream_pos_val = new_data_frame[remainder_col]
    # # stream_pos_val.plot.kde()
    # # plt.show()
    #
    index = list(new_data_frame.index.values)  # getting index value for mapping
    #
    # # # convert to 1D array
    # number_column = new_data_frame.loc[:, 'abs_derivative_2']
    # numbers = number_column.values
    # #
    # # # finding peaks for 1D array
    # # # peaks = find_peaks(numbers, height = 300, threshold = 1, distance = 5)
    # #
    # # peaks = find_peaks(numbers, height=30000, threshold=None, distance=10)
    # peaks = find_peaks(numbers, height=30000, threshold=None)
    # heights = peaks[1]['peak_heights']  # list of height of peaks
    # sorted_heights = sorted(heights, reverse=True)
    # # # peak_pos = peaks[0]
    # # # print(peaks)
    # # if len(height) > 0:
    # #     height_sum = sum(height)
    # #     height_avg = height_sum / len(height)
    # #
    # #     height_sum = 0
    # #     n = 0
    # #     for h in height:
    # #         if h >= height_avg:
    # #             height_sum += h
    # #             n += 1
    # #
    # #     height_avg = height_sum
    # #
    # # else:
    # #     height_avg = new_data_frame[remainder_col].sum()
    # # # find max value from stream col
    # # min_stream_pos = new_data_frame[remainder_col].min()
    # # max_stream_pos = new_data_frame[remainder_col].max()
    # # #
    # # stream_diff = (max_stream_pos - min_stream_pos)
    #
    # # make peak value from second derivative
    # if len(sorted_heights) < 2:
    #     threshold_value = new_data_frame['abs_derivative_2'].max() + 1000
    # else:
    #     threshold_value = (sorted_heights[0] - sorted_heights[1])
    #     # threshold_value = sorted_height[0] - 1000
    #
    # # error message
    # # ValueError: Found array with 1 sample(s) (shape=(1, 2)) while a minimum of 2 is required by AgglomerativeClustering.
    # if len(new_data_frame) < 2:  # avoiding error
    #     continue
    #
    if k_value is None:
        continue
    #
    # # DBSCAN
    # cluster_dbscan = DBSCAN(eps=120000, min_samples=10).fit(transformed_df)
    # labels = cluster_dbscan.labels_
    #
    # # Agglomerative clustering
    clustering_model = AgglomerativeClustering(distance_threshold=None, n_clusters=k_value,
                                               metric='euclidean', linkage='complete', compute_distances=True)
    clustering_model.fit(new_data_frame)
    labels = clustering_model.labels_
    # # #
    # # # map athletes to clusters
    cluster_dict = defaultdict(list)
    for idx, cluster in zip(index, labels):
        if cluster == -1:
            continue
        cluster_dict[cluster].append(temp_sentences[idx])

    cluster_dict = sorted(cluster_dict.items(), key=lambda k_v: k_v[1][0][0])

    temp_cluster = []
    i = 0
    # cluster to cluster distance
    for cluster_index in range(len(cluster_dict) - 1):
        # current cluster
        cur_cluster = cluster_dict[cluster_index][1]
        cur_cluster_min = min(cur_cluster, key=lambda t: t[0])[0]
        cur_cluster_max = max(cur_cluster, key=lambda t: t[0])[0]

        cur_cluster_size = len(cur_cluster)
        cur_cluster_length = cur_cluster_max - cur_cluster_min
        next_cluster_max = max(cluster_dict[cluster_index + 1][1], key=lambda t: t[0])[0]
        cur_cluster_distance_with_next_cluster = next_cluster_max - cur_cluster_min

        # cluster_1 = min(cluster_dict[cluster_index][1], key=lambda t: t[0])
        # cluster_2 = max(cluster_dict[cluster_index + 1][1], key=lambda t: t[0])
        # distance = cluster_2[0] - cluster_1[0]
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
        # cluster_distance.append(data)
        i += 1
        temp_cluster.append(data)
        # cluster_name_wise_distance.append(distance)
        # cluster_name.append(f"{athlete}_{i}")

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

    # athlete wise plot
    temp_cluster.append(data)
    # if athlete in ['onehotencoder__athlete_ANNA GASSER', 'onehotencoder__athlete_NICK PUENTER']:
    #     temp_cluster_df = pd.DataFrame.from_dict(temp_cluster)
    #     # cluster_df.sort_values(by='cur_cluster_min', ascending=True, inplace=True)
    #     temp_cluster_df = temp_cluster_df[['cur_cluster_size', 'cur_cluster_length', 'cur_cluster_distance', 'cur_cluster_min']]
    #     # cluster_df = cluster_df[cluster_df['cur_cluster_size'] >= 12]
    #     # cluster_df = cluster_df[cluster_df['cur_cluster_length'] <= 15000]
    #     describe = temp_cluster_df.describe()
    #     fig = px.scatter(temp_cluster_df, x="cur_cluster_min",
    #                      y=['cur_cluster_length', 'cur_cluster_size', 'cur_cluster_distance'])
    #     fig.show()

    # temp_result = []
    # median_list = []
    # median_list_cluster_num = []
    # for item in cluster_dict.values():
    #     temp_result.append(item)
    #     median_list.append([np.median([x[0] for x in item]), 0])
    #     median_list_cluster_num.append([np.median([x[0] for x in item])])

    # median_list_cluster_num = np.array(median_list_cluster_num)
    # silhouette
    # visualizer = KElbowVisualizer(model, k=(1, 1), metric='silhouette', timings=True)
    # visualizer.fit(median_list_cluster_num)  # Fit the data to the visualizer
    # # visualizer.show()
    # k_value_2 = visualizer.elbow_value_

    # clustering_model = AgglomerativeClustering(distance_threshold=70000, n_clusters=None,
    #                                            metric='euclidean', linkage='complete')
    # clustering_model.fit(median_list)
    # labels = clustering_model.labels_
    #
    # cluster_dict = defaultdict(list)
    # for idx, cluster in enumerate(labels):
    #     if cluster == -1:
    #         continue
    #     cluster_dict[cluster].extend(temp_result[idx])
    # # for item in cluster_dict.values():
    # #     result.append(item)
    # result.extend(cluster_dict.values())
    #
    # plt.title("Hierarchical Clustering Dendrogram")
    # # plot the top three levels of the dendrogram
    # plot_dendrogram(clustering_model)
    # plt.xlabel(athlete)
    # plt.ylabel("Stream position distance")
    # plt.show()
    # print("")

# sorted_result = sorted(result, key=lambda i: i[0])
# print(sorted_result)

# final_result = [res for res in sorted_result if res[-1][0] - res[0][0] > 10000]
# print(final_result)

# cluster_name = []
# cluster_name_wise_distance = []
# for cluster in cluster_distance:
#     keys = list(cluster.keys())
#     keys.remove('cluster_1')
#     keys.remove('cluster_2')
#     athlete_name = keys[0].split('_')[-2:]
#     athlete_name = athlete_name[0] + "_" + athlete_name[1]
#     cluster_name.append(athlete_name)
#     cluster_name_wise_distance.append(cluster[keys[0]])
#
# fig = px.scatter(x=cluster_name, y=cluster_name_wise_distance)
# fig.show()

# output = []
# for cluster in cluster_distance:
#     keys = list(cluster.keys())
#     cluster_1 = cluster['cluster_1']
#     cluster_2 = cluster['cluster_2']
#     keys.remove('cluster_1')
#     keys.remove('cluster_2')
#     distance = cluster[keys[0]]
#     if 45000 <= distance <= 70000:
#         output.append(cluster_1 + cluster_2)
#     elif distance > 70000:
#         output.append(cluster_1)
#         output.append(cluster_2)
#     else:
#         output.append(cluster_1 + cluster_2)
#
# output = sorted(output, key=lambda i: i[0][0])
cluster_df_m = pd.DataFrame.from_dict(result)
# cluster_df.sort_values(by='cur_cluster_min', ascending=True, inplace=True)
cluster_df = cluster_df_m[['cur_cluster_size', 'cur_cluster_length', 'cur_cluster_distance', 'cur_cluster_min']]
# cluster_df = cluster_df[cluster_df['cur_cluster_length'] <= 300000]
# describe = cluster_df.describe()
# cluster_df_m = cluster_df_m[cluster_df_m['cur_cluster_length'] <= 300000]
cluster_df_m['length_size_ratio'] = [x[0] / x[1] for x in
                                     zip(cluster_df_m['cur_cluster_length'], cluster_df_m['cur_cluster_size'])]


# cluster_df_m = cluster_df_m[cluster_df_m['length_size_ratio'] <= 10000]


# cluster_df_m = cluster_df_m[cluster_df_m['length_size_ratio'] <= 10000]
# best fit line
# y_axis = cluster_df_m['cur_cluster_length'].to_numpy()
# x_axis = cluster_df_m['cur_cluster_min'].to_numpy()
# #
# a, b = np.polyfit(x_axis, y_axis, 1)
# fig = px.scatter(cluster_df_m, x="cur_cluster_min", y=['cur_cluster_length', 'cur_cluster_size', 'cur_cluster_distance'])
# # # add line of best fit to plot
# fig_line = px.line(x=x_axis, y=a * y_axis + b)
# #
# fig_3 = go.Figure(data=fig.data + fig_line.data)
# fig_3.show()

def create_regression_functions(x_axis, y_axis):
    # extract x, y value
    x = cluster_df_m[x_axis].to_numpy().reshape(-1, 1)
    y = cluster_df_m[y_axis].to_numpy().reshape(-1, 1)

    # Splitting the dataset into training and test set.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # st_x = StandardScaler()
    # x_train = st_x.fit_transform(x_train)
    # y_train = st_x.fit_transform(y_train)
    classifier = LinearRegression()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    x_test_1 = x_test.flatten()
    y_pred_1 = y_pred.flatten()
    new_data_frame = pd.DataFrame({"x_test": x_test_1, "y_pred": y_pred_1})
    fig_scatter = px.scatter(cluster_df_m, x=x_axis, y=y_axis)
    fig_line = px.line(new_data_frame, x='x_test', y='y_pred')
    comb_fig = go.Figure(data=fig_scatter.data + fig_line.data,
                         layout={"xaxis": {"title": x_axis}, "yaxis": {"title": y_axis}})
    comb_fig.show()
    return classifier


is_a_race_after_length_size_ratio_func = create_regression_functions('cur_cluster_min', 'length_size_ratio')
is_a_race_after_length_func = create_regression_functions('cur_cluster_min', 'cur_cluster_length')
is_a_race_after_size_func = create_regression_functions('cur_cluster_min', 'cur_cluster_size')
is_a_race_after_length_and_size_func = create_regression_functions('cur_cluster_size', 'cur_cluster_length')
# create more features in cluster_df
# cluster_df_m = cluster_df_m[cluster_df_m['cur_cluster_length'] <= 300000]
# cluster_df_m['is_a_race_after_length_size_ratio'] = [1 if 1000 <= x <= 3000 else 0 for x in
#                                                      cluster_df_m['length_size_ratio']]

cluster_df_m['is_a_race_after_length_size_ratio'] = [
    1 if is_a_race_after_length_size_ratio_func.predict([[x[0]]])[0][0] < x[1] else 0 for x in
    zip(cluster_df_m['cur_cluster_min'], cluster_df_m['length_size_ratio'])]
cluster_df_m['is_a_race_after_length'] = [1 if is_a_race_after_length_func.predict([[x[0]]])[0][0] < x[1] else 0 for x
                                          in
                                          zip(cluster_df_m['cur_cluster_min'], cluster_df_m['cur_cluster_length'])]
cluster_df_m['is_a_race_after_size'] = [1 if is_a_race_after_size_func.predict([[x[0]]])[0][0] < x[1] else 0 for x in
                                        zip(cluster_df_m['cur_cluster_min'], cluster_df_m['cur_cluster_size'])]
# cluster_df_m['is_a_race_after_size'] = [0 if x < 35 else 1 for x in cluster_df_m['cur_cluster_size']]
# cluster_df_m['is_a_race_after_length_and_size'] = [1 if (x[0] > 50000 and x[1] > 35) else 0 for x in
#                                                    zip(cluster_df_m['cur_cluster_length'],
#                                                        cluster_df_m['cur_cluster_size'])]
# cluster_df_m['is_a_race_after_length_and_size'] = [
#     1 if is_a_race_after_length_and_size_func.predict([[x[1]]])[0][0] < x[0] else 0
#     for x in zip(cluster_df_m['cur_cluster_length'],
#                  cluster_df_m['cur_cluster_size'])]
# cluster_df_m['is_a_race_after_length_and_size_2'] = [1 if (x[0] == 1 and x[1] == 1) else 0 for x in
#                                                      zip(cluster_df_m['is_a_race_after_size'],
#                                                          cluster_df_m['is_a_race_after_length'])]
# cluster_df_m['is_race_after_combination'] = [1 if not (x[0] ^ x[1]) else 0 for x in
#                                              zip(cluster_df_m['is_a_race_after_length'],
#                                                  cluster_df_m['is_a_race_after_length_and_size'])]

cluster_df_m['is_race'] = [1 if x[0] + x[1] + x[2] >= 2 else 0 for x in
                           zip(cluster_df_m['is_a_race_after_length_size_ratio'],
                               cluster_df_m['is_a_race_after_length'], cluster_df_m['is_a_race_after_size'])]

# try to find cluster number on cluster length
# pca = PCA(2)
# # model = KMeans()
# visualizer = KElbowVisualizer(model, k=(2, 6), metric='silhouette', timings=True)
# visualizer.fit(cluster_df['cur_cluster_length'].to_numpy().reshape(-1, 1))  # Fit the data to the visualizer
# # visualizer.show()
# k_value = visualizer.elbow_value_
#
# # Initialize the class object
# kmeans = KMeans(n_clusters=2)

# predict the labels of clusters.
# df = cluster_df[['cur_cluster_length', 'cur_cluster_size']]
# df['is_race'] = kmeans.fit_predict(df)
# pca_data = pca.fit_transform(df)
#
# results = pd.DataFrame(pca_data, columns=['pca1', 'pca2'])
# import seaborn as sns
#
# sns.scatterplot(x="pca1", y="pca2", hue=df['is_race'], data=results)
# plt.title('K-means Clustering with 2 dimensions')
# plt.show()
# # Getting unique labels
#
# u_labels = np.unique(label)
# #
# # # plotting the results:
# #
#
# for i in u_labels:
#     plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
# plt.legend()
# plt.show()


print("")

# # check multi correlatation
# from patsy import dmatrices
# from statsmodels.stats.outliers_influence import variance_inflation_factor
#
# #find design matrix for regression model using 'rating' as response variable
# y, X = dmatrices('cur_cluster_min ~ cur_cluster_length+cur_cluster_size', data=cluster_df, return_type='dataframe')
#
# #create DataFrame to hold VIF values
# vif_df = pd.DataFrame()
# vif_df['variable'] = X.columns
#
# #calculate VIF for each predictor variable
# vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#
# #view VIF for each predictor variable
# print(vif_df)
