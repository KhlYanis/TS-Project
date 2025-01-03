import numpy as np
import random
from src.dtw import get_dtw_distance_vectorized
from src.dba import DBA, get_approximate_medoid_index


def find_cluster_to_merge(dic_dist):
    # Initialisation des variables pour la recherche du cluster à fusionner
    min_dist = np.inf
    min_cluster1 = None
    min_cluster2 = None
    
    # Recherche du cluster avec la plus petite distance
    for cluster1, cluster2 in dic_dist.items():
        if  cluster2[0] < min_dist:
                min_dist = cluster2[0]
                min_cluster1 = cluster1
                min_cluster2 = cluster2[1]
    return min_cluster1, min_cluster2

class AGH():
    def __init__(self, nb_clusters, max_iter=100, eps=0.1, dba_iter=10):
        # Initialisation des hyperparamètres du K-Means
        self.nb_clusters = nb_clusters
        self.max_iter = max_iter
        self.eps = eps
        self.dba_iters = dba_iter

    def fit(self, X, dtw_matrix):
        """
            Fonction pour entrainer le modèle de clustering hierarchique avec DBA

            Entrées :
                - X : np.array (2D)
                    Array avec les séries temporelle de l'ensemble d'entrainement
                - dtw_matrix : np.array (2D)
                    Matrice DTW associée à l'ensemble d'entrainement
        """
        # Initialisation des clusters
        clusters = set([frozenset([i]) for i in range(X.shape[0])])

        while len(clusters) > self.nb_clusters:
            min_dist = {cluster:None for cluster in clusters}

            # On cherche, pour chaque cluster, le cluster le plus proche
            for cluster in clusters:
                min_cluster = (np.inf, None)
                for other_cluster in clusters-set([cluster]):
                    dist  = np.min(dtw_matrix[sorted(list(cluster)), sorted(list(other_cluster))])
                    if dist < min_cluster[0]:
                        min_cluster = (dist, other_cluster)
                min_dist[cluster] = min_cluster
            
            # On merge les deux clusters les plus proches
            cluster1, cluster2 = find_cluster_to_merge(min_dist)
            clusters.remove(cluster1)
            clusters.remove(cluster2)
            clusters.add(cluster1|cluster)
        # Calcul de la DBA pour chaque cluster
        centroids = []
        for cluster in clusters:
            list_cluster = sorted(list(cluster))
            seq_to_avg = [X[i,:] for i in list_cluster]
            dba_cluster = DBA(X, seq_to_avg, list_cluster , self.dba_iters, dtw_matrix, len(list_cluster))
            centroids.append(dba_cluster)
        
        self.centroids = np.array(centroids)

    def predict(self, X : np.array):
        # Vérifier que les centroides sont calculées (i.e. le classifieur est déjà entrainé)
        assert hasattr(self, 'centroids'), "The model needs to be trained first"

        test_set_size = X.shape[0]

        predictions = np.array([test_set_size])

        for idx, ts in enumerate(X):
            # Calcul de la distance DTW par rapport aux centroides des K clusters
            dtw_distances = get_dtw_distance_vectorized(ts, self.centroids)

            # Le cluster prédit est celui pour lequel la DTW est minimale
            predictions[idx] = np.argmin(dtw_distances)

        return predictions
