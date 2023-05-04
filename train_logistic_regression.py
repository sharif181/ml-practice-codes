import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

df = pd.read_csv('./athlete_csv_file/big_air_3_without_size.csv')
x = df[['cur_cluster_length', 'length_size_ratio']]
y = df[['expected_race']]
# lab = LabelEncoder()
# y = lab.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=0, shuffle=True)

# # extract expected output
# y_train.drop('expected_race', inplace=True, axis=1)
# expected_is_race = y_test['expected_race']
# y_test.drop('expected_race', inplace=True, axis=1)

model = LogisticRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
# Calculate metrics
accuracy = accuracy_score(predictions, y_test)
precision = precision_score(predictions, y_test)
recall = recall_score(predictions, y_test)

cm = confusion_matrix(y_test, predictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.title("logistic regression model")
plt.show()

# save the model to disk
filename = './saved_models/logistic_regression_model_3.sav'
pickle.dump(model, open(filename, 'wb'))


