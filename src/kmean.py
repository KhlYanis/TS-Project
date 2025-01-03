import numpy as np
import random
from src.dtw import get_dtw_distance_vectorized
from src.dba import DBA


class kmeans():
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
        centroids = X[self._init_centroids(X, dtw_matrix),:]
        for _ in range(self.max_iter):
            new_centroids = []
            dic_clusters = {i:[] for i in range(self.nb_clusters)}
            for indice_signal in range(X.shape[0]):
                cluster_signal = np.argmin(get_dtw_distance_vectorized(X[indice_signal,:], centroids))
                dic_clusters[cluster_signal].append(indice_signal)
            dic_clusters_sorted = {i:sorted(liste) for i,liste in dic_clusters.items()}
            print(dic_clusters_sorted)
            for centroid in dic_clusters_sorted.keys():
                seq_to_avg = [X[i,:] for i in dic_clusters_sorted[centroid]]
                new_centroid = DBA(seq_to_avg, dic_clusters_sorted[centroid], 4, dtw_matrix, len(dic_clusters_sorted[centroid]))
                new_centroids.append(new_centroid)
            new_centroids = np.array(new_centroids)
            if np.linalg.norm(new_centroids - centroids) < self.eps:
                return new_centroids
            centroids = new_centroids
        return centroids



            





                



            