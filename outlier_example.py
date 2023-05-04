import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./athlete_csv_file/big_air_3_without_size.csv')

sns.boxplot(df['cur_cluster_length'])
plt.show()
sns.boxplot(df['length_size_ratio'])
plt.show()
sns.boxplot(df['cur_cluster_size'])
plt.show()
print('')
