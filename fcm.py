"""Algorithme Fuzzy C-means (FCM)
"""

import numpy as np


def initialiser_U(N, C, graine=0):
    """Étape 1 : U(0) aléatoire. Chaque ligne a une somme égale à 1."""
    rng = np.random.default_rng(graine)
    U = rng.random((N, C))
    U = U / U.sum(axis=1, keepdims=True)
    return U


def calculer_centres(X, U, m):
    """Étape 2 : c_j = somme(u_ij^m * x_i) / somme(u_ij^m)."""
    Um = U ** m                                  # (N, C)
    centres = (Um.T @ X) / Um.sum(axis=0)[:, None]
    return centres


def calculer_distances(X, centres):
    """Distance ||x_i - c_j|| (norme euclidienne) pour tous les i et j."""
    diff = X[:, None, :] - centres[None, :, :]   # (N, C, d)
    D = np.linalg.norm(diff, axis=2)             # (N, C)
    # Ajout par rapport à l'article : on évite une distance nulle
    # (division par zéro si un point est exactement sur un centre).
    return np.maximum(D, 1e-10)


def mettre_a_jour_U(X, centres, m):
    """Étape 3 : u_ij = 1 / somme_k (||x_i - c_j|| / ||x_i - c_k||)^(2/(m-1))."""
    D = calculer_distances(X, centres)
    # Même formule que l'article, écrite autrement :
    # u_ij = D_ij^(-2/(m-1)) / somme_k D_ik^(-2/(m-1))
    inverse = D ** (-2.0 / (m - 1.0))
    U = inverse / inverse.sum(axis=1, keepdims=True)
    return U


def fcm(X, C, m=2.0, epsilon=0.01, graine=0, iterations_max=100):
    """Fuzzy C-means.

    Renvoie U, les centres et le nombre d'itérations.
    """
    U = initialiser_U(len(X), C, graine)         # étape 1

    for iteration in range(1, iterations_max + 1):
        centres = calculer_centres(X, U, m)      # étape 2
        U_nouveau = mettre_a_jour_U(X, centres, m)   # étape 3

        changement = np.abs(U_nouveau - U).max()     # étape 4
        U = U_nouveau
        if changement < epsilon:
            break

    return U, centres, iteration
