import numpy as np
import src.dtw as dtw

def get_approximate_medoid_index(dtw_matrix : np.array, id_D: list, subset_size : int, seed = 42) -> int :
    """
        Fonction pour calculer une approximation du médoide

        Entrées :
            dtw_matrix : np.array (2D)
                Matrice DTW de l'ensemble d'entrainement
            id_D : list of int
                Liste des indices des séquences que l'on veut moyenner
            subset_size : int
                Nombre de séquences utilisées pour calculer le medoide

        Sortie :
            Indice du médoide 
    """

    np.random.seed(seed)
    set_size = len(id_D)

    assert subset_size < set_size, f"Le nombre d'échantillon pour approximer doit être inférieure à la taille de l'ensemble"
    
    # Echantillonage des indices
    indexes = np.random.choice(range(0, set_size), subset_size, replace = False)

    # Calcul de la matrice DTW du sous-ensemble
    sub_dtw = dtw.extract_sub_dtw_mat(dtw_matrix, id_D[indexes]) 

    # Indice du médoide
    return np.argmin(np.sum(sub_dtw, axis = 1))


def DBA_update(T_init: np.array, D : list) -> np.array:
    """
        Fonction pour mettre à jour la séquence moyenne d'un ensemble de séquences
        à partir de l'alignement DTW

        Entrées :
            T_init : np.array (1D)
                Séquence moyenne initiale
            D : list de np.array (1D)
                Liste des séquences que l'on veut moyenner

        Sortie :
            T_updated : np.array (1D)
                Séquence moyenne mise à jour
    """

    L = len(T_init)
    alignment = [set() for _ in range(L)]

    for S in D :
        S_alignment = dtw.get_dtw_alignment(T_init, S)
        for i in range(L):
            alignment[i].update([S[j] for j in S_alignment[i]])

        
    T_updated = np.zeros([L])
    for i in range(L):
        if alignment[i] :
            T_updated[i] = np.mean(list(alignment[i]))

    return T_updated

def DBA(D : list, id_D, nb_iter : int, dtw_matrix : np.array, subset_size : int) -> np.array :
    """
       Fonction pour calculer une séquence moyenne pour un ensemble de séquences 
       en utilisant la méthode DBA

       Entrées : 
            D : list de np.array (1D)
                Ensemble de séquences à moyenner
            id_D : list de int
                Liste des indices des séquences à moyenner
            nb_iter : int
                Nombre d'itération de l'algorithme DBA pour le calcul de la séquence moyenne
            dtw_matrix : np.array (2D)
                Matrice DTW de l'ensemble d'entrainement
            subset_size : int
                Nombre de séquences utilisés pour l'approximation du médoide

        Sortie :
            average_T : np.array (1D)
                Séquence moyenne
    """

    medoid_index = get_approximate_medoid_index(dtw_matrix, id_D, subset_size)

    T = D[medoid_index]

    for iter in range(nb_iter):
        T = DBA_update(T, D)

    return T