import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=10)

# initialize the model
m = AffinityPropagation(damping=0.9)
# fit the model
m.fit(X)
# predict the cluster for each data point
p = m.predict(X)
# unique clusters
cl = np.unique(p)
# plot the data points and cluster centers
for c in cl:
    r = np.where(c == p)
    pyplot.title('Affinity Propagation Clustering')
    pyplot.scatter(X[r, 0], X[r, 1])
# show the plot
pyplot.show()