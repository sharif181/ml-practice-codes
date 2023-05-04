import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./athlete_csv_file/all_v2/combined_2.csv')
# df = df[df['cur_cluster_length'] < 400000]

cols = ['cur_cluster_size', 'cur_cluster_length', 'length_size_ratio']

stdsc = StandardScaler()
X_std = stdsc.fit_transform(df[cols].iloc[:, range(0, 3)].values)
cov_mat = np.cov(X_std.T)
plt.figure(figsize=(10, 10))
sns.set(font_scale=1.5)
hm = sns.heatmap(cov_mat,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 12},
                 cmap='coolwarm',
                 yticklabels=cols,
                 xticklabels=cols)
plt.title('Covariance matrix showing correlation coefficients', size=18)
plt.tight_layout()
plt.show()
