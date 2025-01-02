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


def extract_sub_dtw_mat(dtw_matrix : np.array, indexes : list):
    """
        Extrait la matrice DTW et les classes pour une sous-partie de l'ensemble d'entrainement

        Entrées :
            dtw_matrix : np.array (2D) 
                Matrice DTW de l'ensemble d'entrainement
            indexes : (list) 
                Liste des indices du sous-ensemble

        Sorties:
            np.array (2D) : Matrice DTW du sous-ensemble 
    """
    size = len(indexes)
    # Initialisation de la matrice DTW pour le sous-ensemble d'entrainement
    sub_dtw_matrix = np.zeros([size, size])

    for i in range(size):
        for j in range(1, size):
            sub_dtw_matrix[i, j] = dtw_matrix[indexes[i], indexes[j]]
            sub_dtw_matrix[j, i] = sub_dtw_matrix[i, j]

    return sub_dtw_matrix


def get_dtw_alignment(S_ref: np.array, S: np.array):
    """
        Renvoie l'alignement correspondant à la DTW 

        Entrées :
            S_ref : np.array (1D) 
                Séquence de référence
            S : np.array (1D)
                Séquence à aligner
        
        Sortie :
            alignement : list de sets
                Alignement entre les indices de S_ref et S
    """
    
    n, m = len(S_ref), len(S)
    C = np.zeros([n, m])    # Matrice de coût

    C[0, 0] = (S_ref[0] - S[0])**2
    for i in range(1, n) :
        C[i, 0] = C[i-1, 0] + (S_ref[i] - S[0])**2
    for j in range(1, n) :
        C[0, j] = C[0, j-1] + (S_ref[0] - S[j])**2

    # Remplissage de la matrice de coût
    for i in range(1, n):
        for j in range(1, m):
            C[i, j] = (S_ref[i] - S[j])**2 + min(C[i-1, j-1], C[i-1, j], C[i, j-1])
    
    
    alignment = [set() for _ in range(n)]
    # On commence en bas à droite de la matrice de coût
    i = n - 1
    j = m - 1

    # Backtracking pour retrouver l'alignement associé à la DTW
    while i > 0 or j > 0 :
        alignment[i].add(j)

        if i == 0 : 
            j += -1
        elif j == 0 :
            i += -1
        else :
            if C[i-1, j-1] <= C[i-1, j] and C[i-1, j-1] <= C[i, j-1]:
                i += -1
                j += -1
            elif C[i-1, j] <= C[i, j-1]:
                i += -1
            else : 
                j += -1

    alignment[i].add(j)

    return alignment

def get_dtw_distance_vectorized(x: np.array, y: np.array):
    """
    Calcule la distance DTW entre une série temporelle x et chaque ligne de y.

        Entrées :
            x : np.array (1D) Série temporelle de taille m.
            y : np.array (2D) de taille k*n, k séries temporelle de taille n.
    """
    m = len(x)
    k, n = y.shape

    D = np.zeros((k, m, n)) 
    C = np.zeros((k, m, n))  

    # Utilisation du broadcasting pour la distance euclidienne
    for i in range(k):
        D[i] = (x[:, None] - y[i, :]) ** 2 

    C[:, 0, 0] = D[:, 0, 0]

    for i in range(1, m):
        C[:, i, 0] = C[:, i-1, 0] + D[:, i, 0]

    for j in range(1, n):
        C[:, 0, j] = C[:, 0, j-1] + D[:, 0, j]

    for i in range(1, m):
        for j in range(1, n):
            C[:, i, j] = D[:, i, j] + np.minimum.reduce([
                C[:, i-1, j-1],  # Diagonal
                C[:, i-1, j],    # Vertical
                C[:, i, j-1]     # Horizontal
            ])

    dtw_distances = np.sqrt(C[:, m-1, n-1])

    return dtw_distances
