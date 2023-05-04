import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
import pickle

sc = StandardScaler()

df = pd.read_csv('./athlete_csv_file/all_v2/combined_2.csv')
# df = df[df['cur_cluster_length'] < 400000]
x = df[['cur_cluster_size', 'cur_cluster_length']]
y = df[['is_race']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)

# x_train_scaled = scale(x_train)
# x_test_scaled = scale(x_test)

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

clf_svm = SVC()
clf_svm.fit(x_train_scaled, y_train)
predictions = clf_svm.predict(x_test_scaled)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

cm = confusion_matrix(y_test, predictions, labels=clf_svm.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_svm.classes_)
disp.plot()
plt.title("SVM model")
plt.show()

param_grid = [
    {'C': [0.5, 1, 10, 100],
     'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001, 'auto'],
     'kernel': ['rbf']}
]

optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=0
)
optimal_params.fit(x_train_scaled, y_train)
best_params = optimal_params.best_params_

optimized_clf_svm = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
optimized_clf_svm.fit(x_train_scaled, y_train)

optimized_predictions = clf_svm.predict(x_test_scaled)

optimized_accuracy = accuracy_score(y_test, optimized_predictions)
optimized_precision = precision_score(y_test, optimized_predictions)
optimized_recall = recall_score(y_test, optimized_predictions)

op_cm = confusion_matrix(y_test, optimized_predictions, labels=optimized_clf_svm.classes_)
op_disp = ConfusionMatrixDisplay(confusion_matrix=op_cm, display_labels=optimized_clf_svm.classes_)
op_disp.plot()
plt.title("Optimized SVM model")
plt.show()

# # save the model to disk
filename = './saved_models/all_v4/combined_data_without_ratio.sav'
pickle.dump(optimized_clf_svm, open(filename, 'wb'))

# saving the scalling object for using on new data
filename = './saved_models/all_v4/combined_data_sc_without_ratio.pkl'
pickle.dump(sc, open(filename, 'wb'))

print()
