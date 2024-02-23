import os
from collections import defaultdict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

dic_stock_return = defaultdict(int)
for dirname, _, filenames in os.walk('./kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        df = pd.read_csv(os.path.join(dirname, filename))
        df['return'] = (df.Close - df.Open) / (df.Open)
        dic_stock_return[df.Name[0]] = df['return']
df_stock_return = pd.DataFrame(dic_stock_return)
df_stock_return.head()
tech_px = df_stock_return[['GOOGL', 'AAPL']]
tech_px = tech_px.drop(3019)
kmeans = KMeans(n_clusters=4)  # Initialize the KMeans model with inputs and num_clusters
kmeans.fit(tech_px)
tech_px['cluster'] = kmeans.labels_
centers = pd.DataFrame(kmeans.cluster_centers_, columns=['GOOGL', 'AAPL'])
print(centers)
fig, ax = plt.subplots(figsize=(14, 10))
ax = sns.scatterplot(x='GOOGL', y='AAPL', hue='cluster', style='cluster',
                     ax=ax, data=tech_px)
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(-0.1, 0.1)
centers.plot.scatter(x='GOOGL', y='AAPL', ax=ax, s=50, color='black')
plt.tight_layout()
plt.show()
nan_rows = df_stock_return[df_stock_return.isnull().T.any()]
df_stock_return = df_stock_return.drop(list(nan_rows.index))
syms = sorted(['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE',
               'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE',
               'PG', 'TRV', 'UTX', 'UNH', 'VZ', 'WMT', 'GOOGL', 'AMZN', 'AABA'])
top_df = df_stock_return[syms]
kmeans = KMeans(n_clusters=4).fit(top_df)
from collections import Counter
print(Counter(kmeans.labels_))
inertia = []
for n_clusters in range(2, 15):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(top_df)
    inertia.append(kmeans.inertia_ / n_clusters)
inertias = pd.DataFrame({'n_clusters': range(2, 15), 'inertia': inertia})
ax = inertias.plot(x='n_clusters', y='inertia')
plt.xlabel('Number of clusters(k)')
plt.ylabel('Average Within-Cluster Squared Distances')
plt.ylim((0, 1.1 * inertias.inertia.max()))
ax.legend().set_visible(False)
plt.tight_layout()
plt.show()
top_df = top_df.transpose()
Z = linkage(top_df, method='complete')
fig, ax = plt.subplots(figsize=(8, 5))
dendrogram(Z, labels=list(top_df.index), color_threshold=0)
plt.xticks(rotation=90)
ax.set_ylabel('distance')
plt.tight_layout()
plt.show()