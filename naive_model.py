import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
import pickle
import numpy as np
import matplotlib.colors as colors
from sklearn.tree import DecisionTreeClassifier

sc = StandardScaler()

df = pd.read_csv('./athlete_csv_file/all_v2/combined_2.csv')
x = df[['cur_cluster_size', 'cur_cluster_length']]
y = df['is_race']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)
#
# x_train_scaled[:, 0] = 5 * x_train_scaled[:, 0]
# x_test_scaled[:, 0] = 5 * x_test_scaled[:, 0]
#
# x_train_scaled[:, 1] = .5 * x_train_scaled[:, 1]
# x_test_scaled[:, 1] = .5 * x_test_scaled[:, 1]

# clf_svm = GaussianNB()
clf_svm = GaussianNB()

param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
grid_search = GridSearchCV(clf_svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train_scaled, y_train)
best_params = grid_search.best_params_

optimized_clf_svm = GaussianNB(var_smoothing=best_params['var_smoothing'])

optimized_clf_svm.fit(x_train_scaled, y_train)
predictions = optimized_clf_svm.predict(x_test_scaled)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

cm = confusion_matrix(y_test, predictions, labels=optimized_clf_svm.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=optimized_clf_svm.classes_)
disp.plot()
plt.title("Naive Bayes model")
plt.show()

test_pc1_coords = x_test_scaled[:, 0]
test_pc2_coords = x_test_scaled[:, 1]

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

ax.set_ylabel("length")
ax.set_xlabel("Size")
ax.set_title("")
plt.show()
#
# param_grid = [
#     {"priors": [0.3, 0.4, 0.3], "var_smoothing": [1e-8, 1e-7, 1e-9]}
# ]
#
# optimal_params = GridSearchCV(
#     GaussianNB(),
#     param_grid,
#     cv=5,
#     scoring='accuracy',
#     verbose=0
# )
# optimal_params.fit(x_train_scaled, y_train)
# best_params = optimal_params.best_params_
#
# optimized_clf_svm = GaussianNB(priors=best_params['priors'], var_smoothing=best_params['var_smoothing'])
# optimized_clf_svm.fit(x_train_scaled, y_train)
#
# optimized_predictions = clf_svm.predict(x_test_scaled)
#
# optimized_accuracy = accuracy_score(y_test, optimized_predictions)
# optimized_precision = precision_score(y_test, optimized_predictions)
# optimized_recall = recall_score(y_test, optimized_predictions)
#
# op_cm = confusion_matrix(y_test, optimized_predictions, labels=optimized_clf_svm.classes_)
# op_disp = ConfusionMatrixDisplay(confusion_matrix=op_cm, display_labels=optimized_clf_svm.classes_)
# op_disp.plot()
# plt.title("Optimized SVM model")
# plt.show()

# # save the model to disk
filename = './saved_models/all_v4/combined_data_without_ratio.sav'
pickle.dump(optimized_clf_svm, open(filename, 'wb'))

# saving the scalling object for using on new data
filename = './saved_models/all_v4/combined_data_sc_without_ratio.pkl'
pickle.dump(sc, open(filename, 'wb'))

print()
