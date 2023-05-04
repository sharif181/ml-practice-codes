import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv('./big_air_3.csv')
x = df[['cur_cluster_size', 'cur_cluster_length', 'length_size_ratio']]
y = df[['is_race', 'expected_race']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=0, shuffle=True)

# extract expected output
y_train.drop('expected_race', inplace=True, axis=1)
expected_is_race = y_test['expected_race']
y_test.drop('expected_race', inplace=True, axis=1)

# st_x = StandardScaler()
# x_train = st_x.fit_transform(x_train)
# x_test = st_x.transform(x_test)

models = {}

# Logistic Regression
from sklearn.linear_model import LogisticRegression

models['Logistic Regression'] = LogisticRegression()

# Support Vector Machines
from sklearn.svm import LinearSVC

models['Support Vector Machines'] = LinearSVC()

# Decision Trees
from sklearn.tree import DecisionTreeClassifier

models['Decision Trees'] = DecisionTreeClassifier()

# Random Forest
from sklearn.ensemble import RandomForestClassifier

models['Random Forest'] = RandomForestClassifier()

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

models['Naive Bayes'] = GaussianNB()

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

models['K-Nearest Neighbor'] = KNeighborsClassifier()

from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy, precision, recall, prediction = {}, {}, {}, {}

for key in models.keys():
    # Fit the classifier
    models[key].fit(x_train, y_train)

    # Make predictions
    predictions = models[key].predict(x_test)

    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)
    prediction[key] = predictions

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()

for key, val in prediction.items():
    cm = confusion_matrix(y_test, val, labels=models[key].classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=models[key].classes_)
    disp.plot()
    plt.title(key)
    plt.show()
print("")
