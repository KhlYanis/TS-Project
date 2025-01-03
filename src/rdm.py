import numpy as np
import random
from src.dtw import get_dtw_distance_vectorized
from src.dba import DBA


class RDm():
    def __init__(self, nb_clusters):
        # Initialisation des hyperparamètres du K-Means
        self.nb_clusters = nb_clusters
    

    def fit(self, X: np.array):
        """
            Fonction pour entrainer le modèle Random, qui sélectionne de manière aléatoire nb_clusters pour réduire 
            la taille du dataset initial

            Entrées :
                - X : np.array (2D)
                    Array avec les séries temporelle de l'ensemble d'entrainement
        """
        # Echantillonage des indices
        indexes = sorted(np.random.choice(range(0, X.shape[0]), self.nb_clusters, replace = False))

        self.centroids = X[indexes,:]

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
