import pandas as pd

df = pd.read_csv("./new.csv")
import numpy as np
from sklearn.neighbors import NearestNeighbors

# n_neighbors = 5 as kneighbors function returns distance of point to itself
nbrs = NearestNeighbors(n_neighbors=5).fit(df)
neigh_dist, neigh_ind = nbrs.kneighbors(df)
sort_neigh_dist = np.sort(neigh_dist, axis=0)
import matplotlib.pyplot as plt

k_dist = sort_neigh_dist[:, 4]
plt.plot(k_dist)
plt.ylabel("k-NN distance")
plt.xlabel("Sorted observations (4th NN)")
plt.show()
from kneed import KneeLocator

kneedle = KneeLocator(x=range(1, len(neigh_dist) + 1), y=k_dist, S=1.0,
                      curve="concave", direction="increasing", online=True)
print(kneedle.knee_y)
from sklearn.cluster import DBSCAN

clusters = DBSCAN(eps=4.54, min_samples=4).fit(df)
clusters.labels_
# check unique clusters
set(clusters.labels_)
from collections import Counter

Counter(clusters.labels_)
import seaborn as sns
import matplotlib.pyplot as plt

p = sns.scatterplot(data=df, x="t-SNE-1", y="t-SNE-2", hue=clusters.labels_, legend="full", palette="deep")
sns.move_legend(p, "upper right", bbox_to_anchor=(1.17, 1.), title='Clusters')
plt.show()
