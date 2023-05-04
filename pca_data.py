import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv('./athlete_csv_file/combine_train_df_2.csv')
x = df[['cur_cluster_size', 'cur_cluster_length', 'length_size_ratio']]
y = df['expected_race']

scaling = StandardScaler()

# Use fit and transform method
scaling.fit(x)
Scaled_data = scaling.transform(x)

# Set the n_components=3
principal = PCA(0.90)
principal.fit(Scaled_data)
x1 = principal.transform(Scaled_data)

# Check the dimensions of data after PCA

# plt.figure(figsize=(10, 10))
plt.scatter(x1[:, 0], x1[:, 1], c=y, cmap='plasma')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()
x_train_scaled, x_test_scaled, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=0, shuffle=True)
clf_svm = SVC()
clf_svm.fit(x_train_scaled, y_train)
predictions = clf_svm.predict(x_test_scaled)

accuracy = accuracy_score(predictions, y_test)
precision = precision_score(predictions, y_test)
recall = recall_score(predictions, y_test)

cm = confusion_matrix(y_test, predictions, labels=clf_svm.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_svm.classes_)
disp.plot()
plt.title("PCA SVM model")
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

optimized_accuracy = accuracy_score(optimized_predictions, y_test)
optimized_precision = precision_score(optimized_predictions, y_test)
optimized_recall = recall_score(optimized_predictions, y_test)

op_cm = confusion_matrix(y_test, optimized_predictions, labels=optimized_clf_svm.classes_)
op_disp = ConfusionMatrixDisplay(confusion_matrix=op_cm, display_labels=optimized_clf_svm.classes_)
op_disp.plot()
plt.title("PCA Optimized SVM model")
plt.show()

df = pd.read_csv('./athlete_csv_file/big_air_3.csv')
x = df[['cur_cluster_size', 'cur_cluster_length', 'length_size_ratio']]

y = df['expected_race']

scaling.fit(x)
scaled_test_data = scaling.transform(x)
principal.fit(scaled_test_data)
scaled_test_data = principal.transform(scaled_test_data)
test_prediction = optimized_clf_svm.predict(scaled_test_data)
test_accuracy = accuracy_score(test_prediction, y)
test_precision = precision_score(test_prediction, y)
test_recall = recall_score(test_prediction, y)

test_cm = confusion_matrix(y, test_prediction, labels=optimized_clf_svm.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels=optimized_clf_svm.classes_)
disp.plot()
plt.title("Test PCA SVM model")
plt.show()

print()
