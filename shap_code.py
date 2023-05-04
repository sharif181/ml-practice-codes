import shap
import pandas as pd
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from numpy.random import sample
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from shap.explainers.explainer import Explainer

sc = StandardScaler()
df = pd.read_csv('athlete_csv_file/all_v2/combined_2.csv')
x = df[['cur_cluster_size', 'cur_cluster_length']]
y = df['is_race']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.fit_transform(x_test)

param_grid = [
    {'C': [100, 250, 750, 850, 1000],
     'gamma': ['scale'],
     'kernel': ['rbf'],
     'class_weight': [{0: 100.0, 1: 1.0}, {0: 50.0, 1: 1.0}, {0: 10, 1: 1}, {0: 1000.0, 1: 1.0}]
     }
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

optimized_clf_svm = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'],
                        class_weight=best_params['class_weight'], probability=True)

optimized_clf_svm.fit(x_train_scaled, y_train)

# estimator = GradientBoostingRegressor(random_state=42)
# estimator.fit(x_train_scaled, y_train)

explainer = shap.KernelExplainer(optimized_clf_svm.predict_proba, x_train_scaled)
shap_values = explainer.shap_values(x_test_scaled)
shap.summary_plot(shap_values, max_display=15, show=False)
plt.show()
print()
# shap_values =

# explainer = shap.Explainer(optimized_clf_svm)
# shap_values = explainer(df)
# shap.plots.bar(shap_values)
# print()
