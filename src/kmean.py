import numpy as np
import random
from src.dtw import get_dtw_distance_vectorized
from src.dba import DBA


class KMeans():
    def __init__(self, nb_clusters, max_iter=100, eps=0.1, dba_iter = 10):
        # Initialisation des hyperparamètres du K-Means
        self.nb_clusters = nb_clusters
        self.max_iter = max_iter
        self.eps = eps

        self.dba_iters = dba_iter
    
    def _init_centroids(self, X, dtw_matrix):
        # Initialisation des centroides avec la méthode K-Means++
        centroids = []
        n_sample, _ = X.shape
        indices = list(range(n_sample))

        # Sélection du premier centroide
        first_centroid = np.random.randint(0, n_sample)
        centroids.append(first_centroid)

        for _ in range(self.nb_clusters-1):
            min_dist = np.min(dtw_matrix[:, sorted(centroids)], axis=1) 
            prob = min_dist/np.sum(min_dist) # already selected centroids have a probability of 0
            new_centroid = random.choices(indices, weights=prob, k=1)[0]
            centroids.append(new_centroid)

        return sorted(centroids)

    def fit(self, X: np.array, dtw_matrix: np.array):
        """
            Fonction pour entrainer le modèle K-Means avec DBA

            Entrées :
                - X : np.array (2D)
                    Array avec les séries temporelle de l'ensemble d'entrainement
                - dtw_matrix : np.array (2D)
                    Matrice DTW associée à l'ensemble d'entrainement
        """
        centroids = X[self._init_centroids(X, dtw_matrix), :]
        for _ in range(self.max_iter):
            new_centroids = []
            dic_clusters = {i:[] for i in range(self.nb_clusters)}
            # Récupération des séries contenues dans les clusters
            for indice_signal in range(X.shape[0]):
                cluster_signal = np.argmin(get_dtw_distance_vectorized(X[indice_signal,:], centroids))
                dic_clusters[cluster_signal].append(indice_signal)
            dic_clusters_sorted = {i:sorted(liste) for i,liste in dic_clusters.items()}

            # Mise à jour des centroides en utilisant DBA
            for centroid in dic_clusters_sorted.keys():
                seq_to_avg = [X[i,:] for i in dic_clusters_sorted[centroid]]
                new_centroid = DBA(X, seq_to_avg, dic_clusters_sorted[centroid], self.dba_iters, dtw_matrix, len(dic_clusters_sorted[centroid]))
                new_centroids.append(new_centroid)

            new_centroids = np.array(new_centroids)
            # Arret prématuré
            if np.linalg.norm(new_centroids - centroids) < self.eps:
                print(f"Convergence criterion satisfied before reaching {self.max_iter} iterations")
                break

            centroids = new_centroids

        self.centroids = new_centroids

    def predict(self, X : np.array):
        # Vérifier que les centroides sont calculées (i.e. le classifieur est déjà entrainé)
        assert hasattr(self, 'centroids'), "The model needs to be trained first"

        test_set_size = X.shape[0]

        predictions = np.zeros([test_set_size])

        for idx, ts in enumerate(X):
            # Calcul de la distance DTW par rapport aux centroides des K clusters
            dtw_distances = get_dtw_distance_vectorized(ts, self.centroids)

            # Le cluster prédit est celui pour lequel la DTW est minimale
            predictions[idx] = np.argmin(dtw_distances)

        return predictions



            





                



            