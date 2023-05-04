import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import scale, StandardScaler
from sklearn.inspection import permutation_importance

sc = StandardScaler()


def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()


df = pd.read_csv('athlete_csv_file/all_v2/combined_2.csv')
# df = df[df['cur_cluster_length'] < 300000]
x = df[['cur_cluster_size', 'cur_cluster_length']]
y = df['is_race']

x_train_scaled = sc.fit_transform(x)

# cv = CountVectorizer()
# cv.fit(df)
# print(len(cv.vocabulary_))
# print(cv.get_feature_names())
# X_train = cv.transform(x)

svm = SVC(kernel='rbf')
svm.fit(x_train_scaled, y)
plot_coefficients(svm, svm.feature_names_in_)
print('')
