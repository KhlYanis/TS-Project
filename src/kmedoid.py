import numpy as np
import random
from src.dtw import get_dtw_distance_vectorized
from src.dba import DBA, get_approximate_medoid_index


class kmedoid():
    def __init__(self, nb_clusters, max_iter=100, eps=0.1):
        self.nb_clusters = nb_clusters
        self.max_iter = max_iter
        self.eps = eps
    
    def _init_centroids(self, X, dtw_matrix):
        nb_clusters = self.nb_clusters
        centroids = []
        n_sample, _ = X.shape
        indices = list(range(n_sample))
        first_centroid = np.random.randint(0, n_sample)
        centroids.append(first_centroid)
        for _ in range(nb_clusters-1):
            min_dist = np.min(dtw_matrix[:, sorted(centroids)], axis=1) 
            prob = min_dist/np.sum(min_dist) # already selected centroids have a probability of 0
            new_centroid = random.choices(indices, weights=prob, k=1)[0]
            centroids.append(new_centroid)
        return sorted(centroids)

    def fit(self, X, dtw_matrix):
        centroids = self._init_centroids(X, dtw_matrix)
        for _ in range(self.max_iter):
            dic_centroids = {i:[] for i in centroids}
            closest_centroid = np.argmin(dtw_matrix[:,centroids], axis=1)
            for i,el in enumerate(list(closest_centroid)):
                dic_centroids[centroids[el]].append(i)
            dic_centroids_sorted = {key:sorted(value) for key, value in dic_centroids.items()}
            print(dic_centroids_sorted)
            new_centroids = [get_approximate_medoid_index(dtw_matrix, dic_centroids_sorted[i],len(dic_centroids_sorted[i])) for i in dic_centroids_sorted]
            if np.linalg.norm(X[new_centroids,:] - X[centroids,:]) < self.eps:
                return X[new_centroids,:]
            centroids = sorted(new_centroids)
        return X[centroids,:]
