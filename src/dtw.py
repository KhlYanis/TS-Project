import numpy as np

def get_dtw_distance(x: np.array, y: np.array) -> np.float32:
    """
    Calcule la distance DTW entre deux séries temporelles x et y.

        Entrées :
            x : np.array (1D) Série temporelle de taille m.
            y : np.array (1D) Série temporelle de taille n.

        Sortie :
            - float : Distance DTW x et y.
    """
    m = len(x)
    n = len(y)

    D = np.zeros([m, n])    # Matrice de distance
    C = np.zeros([m, n])    # Matrice de coût

    # Calcul des distances euclidiennes
    for i in range(m):
        for j in range(n):
            D[i, j] = (x[i] - y[j])**2

    C[0, 0] = D[0, 0]
    # Initialisation de la 1re colonne de la matrice de coût
    for i in range(1, m):
        C[i, 0] = C[i-1, 0] + D[i, 0]
    # Initialisation de la 1re ligne de la matrice de coût 
    for j in range(1, n):
        C[0, j] = C[0, j-1] + D[0, j]


    for i in range(1, m):
        for j in range(1, n):
            C[i, j] = D[i, j] + min(C[i-1, j-1], C[i-1, j], C[i, j-1])


    dtw_distance = np.sqrt(C[m-1, n-1])

    return dtw_distance


def extract_sub_dtw_mat(dtw_matrix : np.array, labels : np.array, indexes : list):
    """
        Extrait la matrice DTW et les classes pour une sous-partie de l'ensemble d'entrainement

        Entrées :
            dtw_matrix : np.array (2D) 
                Matrice DTW de l'ensemble d'entrainement
            labels : np.array (1D) 
                Classes de l'ensemble d'entrainement
            indexes : (list) 
                Liste des indices du sous-ensemble

        Sorties:
            np.array (2D) : Matrice DTW du sous-ensemble 
            np.array (1D) : Liste des classes pour le sous-ensemble
    """
    size = len(indexes)
    # Initialisation de la matrice DTW pour le sous-ensemble d'entrainement
    sub_dtw_matrix = np.zeros([size, size])

    for i in range(size):
        for j in range(1, size):
            sub_dtw_matrix[i, j] = dtw_matrix[indexes[i], indexes[j]]
            sub_dtw_matrix[j, i] = sub_dtw_matrix[i, j]

    return sub_dtw_matrix, labels[indexes]