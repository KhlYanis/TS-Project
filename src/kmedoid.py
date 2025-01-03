import numpy as np
import random
from src.dtw import get_dtw_distance_vectorized
from src.dba import DBA, get_approximate_medoid_index


class kmedoid():
    def __init__(self, nb_clusters, max_iter=100, eps=0.1):
        # Initialisation des hyperparamètres du K-Means
        self.nb_clusters = nb_clusters
        self.max_iter = max_iter
        self.eps = eps
    
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

    def fit(self, X, dtw_matrix):
        # Initialisation des centroids avec la méthode kmean++
        centroids = self._init_centroids(X, dtw_matrix)


        for _ in range(self.max_iter):
            dic_centroids = {i:[] for i in centroids}

            # Calcul du centroïde le plus proche pour chaque point
            closest_centroid = np.argmin(dtw_matrix[:,centroids], axis=1)

            # Création des clusters
            for i,el in enumerate(list(closest_centroid)):
                dic_centroids[centroids[el]].append(i)
            dic_centroids_sorted = {key:sorted(value) for key, value in dic_centroids.items()}

            # Calcul des nouveaux centroides à partir des clusters
            new_centroids = [get_approximate_medoid_index(dtw_matrix, dic_centroids_sorted[i], len(dic_centroids_sorted[i])) for i in dic_centroids_sorted]

            # Si les centroides ne changent pas, on arrete la boucle
            if np.linalg.norm(X[new_centroids,:] - X[centroids,:]) < self.eps:
                print(f"Convergence criterion satisfied before reaching {self.max_iter} iterations")
                break
            centroids = sorted(new_centroids)
        
        self.centroids = X[centroids,:]

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
