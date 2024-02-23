from sklearn.cluster import KMeans
import timeit
import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=10)
# 3 clusters
m = MiniBatchKMeans(n_clusters=3)
# fit the model
m.fit(X)
# predict the cluster for each data point
p = m.predict(X)
# unique clusters
cl = np.unique(p)
# plot the data points and cluster centers
for c in cl:
    r = np.where(c == p)
    pyplot.title('Mini Batch K-means')
    pyplot.scatter(X[r, 0], X[r, 1])
# show the plot
pyplot.show()
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=10)
# start timer for Mini Batch K-Means
t1_mkm = timeit.default_timer()
m = MiniBatchKMeans(n_clusters=2)
m.fit(X)
p = m.predict(X)
# stop timer for Mini Batch K-Means
t2_mkm = timeit.default_timer()
# start timer for K-Means
t1_km = timeit.default_timer()
m = KMeans(n_clusters=2)
m.fit(X)
p = m.predict(X)
# stop timer for K-Means
t2_km = timeit.default_timer()
# print time difference
print("Time difference between Mini Batch K-Means and K-Means = ",
      (t2_km-t1_km)-(t2_mkm-t1_mkm))