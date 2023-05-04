import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import scale, StandardScaler

sc = StandardScaler()

# df = pd.read_csv('./athlete_csv_file/combine_train_df_2.csv')
df = pd.read_csv('athlete_csv_file/all_v2/combined_2.csv')
x = df[['cur_cluster_size', 'cur_cluster_length']]
y = df['is_race']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.fit_transform(x_test)

pca = PCA()
x_train_pca = pca.fit_transform(x_train_scaled)
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = [str(x) for x in range(1, len(per_var) + 1)]
plt.bar(x=range(1, len(per_var) + 1), height=per_var)
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
plt.ylabel("Percentage of Explained variance")
plt.xlabel("Principal Components")
plt.title("PCA plot")

plt.show()

train_pc1_coords = x_train_pca[:, 0]
train_pc2_coords = x_train_pca[:, 1]

pca_train_scaled = scale(np.column_stack((train_pc1_coords, train_pc2_coords)))

param_grid = [
    {'C': [0.5, 1, 10, 100],
     'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001, 'auto'],
     'kernel': ['rbf', 'linear']}
]
optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=0
)
optimal_params.fit(pca_train_scaled, y_train)
best_params = optimal_params.best_params_

optimized_clf_svm = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
optimized_clf_svm.fit(pca_train_scaled, y_train)

x_test_pca = pca.fit_transform(x_test_scaled)
test_pc1_coords = x_test_pca[:, 0]
test_pc2_coords = x_test_pca[:, 1]

x_min = test_pc1_coords.min() - 1
x_max = test_pc1_coords.max() + 1

y_min = test_pc2_coords.min() - 1
y_max = test_pc2_coords.max() + 1

xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),
                     np.arange(start=y_min, stop=y_max, step=0.1)
                     )

z = optimized_clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))
z = z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 10))
ax.contourf(xx, yy, z, alpha=0.1)
cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])

scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_test,
                     cmap=cmap, s=100, edgecolors='k', alpha=0.7)
legend = ax.legend(scatter.legend_elements()[0],
                   scatter.legend_elements()[1],
                   loc="upper right")

legend.get_texts()[0].set_text("No race")
legend.get_texts()[1].set_text("race")

ax.set_ylabel("PC2")
ax.set_xlabel("PC1")
ax.set_title("")
plt.show()

df2 = pd.read_csv('./athlete_csv_file/big_air_3.csv')
x = df2[['cur_cluster_size', 'cur_cluster_length', 'length_size_ratio']]
y = df2['expected_race']

x_test_scaled_test = scale(x)

x_test_pca = pca.fit_transform(x_test_scaled_test)
test_pc1_coords = x_test_pca[:, 0]
test_pc2_coords = x_test_pca[:, 1]

x_min = test_pc1_coords.min() - 1
x_max = test_pc1_coords.max() + 1

y_min = test_pc2_coords.min() - 1
y_max = test_pc2_coords.max() + 1

xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),
                     np.arange(start=y_min, stop=y_max, step=0.1)
                     )

z = optimized_clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))
z = z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 10))
ax.contourf(xx, yy, z, alpha=0.1)
cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])

scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y,
                     cmap=cmap, s=100, edgecolors='k', alpha=0.7)
legend = ax.legend(scatter.legend_elements()[0],
                   scatter.legend_elements()[1],
                   loc="upper right")

legend.get_texts()[0].set_text("No race")
legend.get_texts()[1].set_text("race")

ax.set_ylabel("PC2")
ax.set_xlabel("PC1")
ax.set_title("")
plt.show()
print("")
