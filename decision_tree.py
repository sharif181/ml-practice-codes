import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
import pickle

df = pd.read_csv('./athlete_csv_file/combine_train_df_2.csv')
x = df[['cur_cluster_length', 'length_size_ratio']]
y = df[['expected_race']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)

x_train_scaled = scale(x_train)
x_test_scaled = scale(x_test)

clf_tree = DecisionTreeClassifier()
clf_tree.fit(x_train_scaled, y_train)
predictions = clf_tree.predict(x_test_scaled)

accuracy = accuracy_score(predictions, y_test)
precision = precision_score(predictions, y_test)
recall = recall_score(predictions, y_test)

cm = confusion_matrix(y_test, predictions, labels=clf_tree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_tree.classes_)
disp.plot()
plt.title("Decision Tree model")
plt.show()

df_test = pd.read_csv('./athlete_csv_file/big_air_3.csv')
x = df_test[['cur_cluster_length', 'length_size_ratio']]

y = df_test['expected_race']
x = scale(x)

test_pred = clf_tree.predict(x)
test_acc = accuracy_score(test_pred, y)
test_prec = precision_score(test_pred, y)
test_rec = recall_score(test_pred, y)

cm_2 = confusion_matrix(y, test_pred, labels=clf_tree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_2, display_labels=clf_tree.classes_)
disp.plot()
plt.title("Decision Tree model")
plt.show()

print()
