import pickle
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

# for svm
from sklearn.preprocessing import scale, StandardScaler

# load scaler
scl = './saved_models/all_v3/combined_data_sc_without_ratio.pkl'
scaler = pickle.load(open(scl, 'rb'))

df = pd.read_csv('./athlete_csv_file/test_data/SnowboardSlopestyle_11029_wr.csv')
x = df[['cur_cluster_size', 'cur_cluster_length']]

y = df['is_race']
# svm
# x = scale(x)
x = scaler.transform(x)

filename = './saved_models/all_v3/combined_data_without_ratio.sav'
loaded_model = pickle.load(open(filename, 'rb'))
predictions = loaded_model.predict(x)
df['is_expected'] = predictions

accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)

cm = confusion_matrix(y, predictions, labels=loaded_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=loaded_model.classes_)
disp.plot()
plt.title("SVM model")
plt.show()

# fig = px.scatter(df, x="cur_cluster_size", y="cur_cluster_length", color="is_race")
# fig.show()
#
# fig = px.scatter(df, x="cur_cluster_size", y="cur_cluster_length", color="is_expected")
# fig.show()

# fig_1 = px.scatter(df, x="cur_cluster_size", y="length_size_ratio", color="expected_race")
# fig_1.show()
#
# fig_2 = px.scatter(df, x="cur_cluster_size", y="cur_cluster_length", color="expected_race")
# fig_2.show()
print("")
