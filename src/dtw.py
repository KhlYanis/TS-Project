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