"""Segmentation floue d'une image en niveaux de gris avec le Fuzzy C-means.

Partie 1 : K = 2 clusters, m = 2  -> fig1 et fig2
Partie 2 (extension) : effet du paramètre m -> fig3

Lancer avec :  python segmentation.py
"""

import os

import matplotlib
matplotlib.use("Agg")          # on enregistre les figures, pas de fenêtre
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fcm import fcm, mettre_a_jour_U

DOSSIER = os.path.dirname(os.path.abspath(__file__))
IMAGE = os.path.join(DOSSIER, "images", "milky-way-nvg.jpg")
RESULTATS = os.path.join(DOSSIER, "resultats")

K = 2              # nombre de clusters
M = 2.0            # paramètre de flou pour la partie 1
EPSILON = 0.01     # critère d'arrêt 
GRAINE = 0         # graine pour l'initialisation aléatoire de U
VALEURS_M = [1.5, 2.0, 3.0, 5.0]   # valeurs testées pour l'extension


def charger_image(chemin):
    """Charge l'image en niveaux de gris (valeurs de 0 à 255)."""
    image = Image.open(chemin).convert("L")
    return np.array(image, dtype=float)


def lancer_fcm(image, m):
    """Applique le FCM sur les pixels. Chaque pixel est un point en 1D.

    On range les clusters du plus sombre au plus clair pour que
    le cluster 1 soit toujours le fond sombre.
    """
    X = image.reshape(-1, 1)                  # N pixels, 1 caractéristique
    U, centres, iterations = fcm(X, K, m, EPSILON, GRAINE)
    ordre = np.argsort(centres[:, 0])         # du plus sombre au plus clair
    U = U[:, ordre]
    centres = centres[ordre]
    return U, centres, iterations


def segmentation_nette(U, forme):
    """Chaque pixel va dans le cluster où son appartenance est maximale."""
    return np.argmax(U, axis=1).reshape(forme)


def figure_heatmaps(U, centres, iterations, forme, chemin):
    """Fig 1 : une heatmap par cluster."""
    fig, axes = plt.subplots(1, K, figsize=(10, 4.8))
    for j in range(K):
        carte = axes[j].imshow(U[:, j].reshape(forme), cmap="hot", vmin=0, vmax=1)
        axes[j].set_title(f"Cluster {j + 1} (centre = {centres[j, 0]:.1f})")
        axes[j].axis("off")
        fig.colorbar(carte, ax=axes[j], fraction=0.046, pad=0.02,
                     label="degré d'appartenance")
    fig.suptitle(f"K = {K}, m = {M}, epsilon = {EPSILON} ({iterations} itérations)")
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def figure_segmentation(image, U, forme, chemin):
    """Fig 2 : image d'origine et segmentation nette (appartenance maximale)."""
    nette = segmentation_nette(U, forme)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Image d'origine")
    axes[1].imshow(nette, cmap="viridis")
    axes[1].set_title("Segmentation nette (appartenance maximale)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def figure_effet_de_m(image, chemin):
    """Fig 3 (extension) : effet de m.

    Ligne 1 : heatmap du cluster clair pour chaque m.
    Ligne 2 : courbes d'appartenance en fonction du niveau de gris.
    """
    niveaux = np.arange(256, dtype=float).reshape(-1, 1)
    fig, axes = plt.subplots(2, len(VALEURS_M), figsize=(13, 6.5))
    lignes = []

    for colonne, m in enumerate(VALEURS_M):
        U, centres, iterations = lancer_fcm(image, m)
        part_floue = np.mean(U.max(axis=1) < 0.8)   # pixels "vraiment flous"
        lignes.append(f"m = {m} : centres = {centres[0, 0]:.1f} et "
                      f"{centres[1, 0]:.1f}, {iterations} itérations, "
                      f"pixels avec appartenance max < 0.8 : {100 * part_floue:.1f} %")

        axes[0, colonne].imshow(U[:, 1].reshape(image.shape), cmap="hot",
                                vmin=0, vmax=1)
        axes[0, colonne].set_title(f"m = {m} ({iterations} itérations)")
        axes[0, colonne].axis("off")

        courbes = mettre_a_jour_U(niveaux, centres, m)
        axes[1, colonne].plot(niveaux, courbes[:, 0], label="cluster 1 (sombre)")
        axes[1, colonne].plot(niveaux, courbes[:, 1], label="cluster 2 (clair)")
        axes[1, colonne].set_xlabel("niveau de gris")
        axes[1, colonne].set_ylim(0, 1)
        if colonne == 0:
            axes[1, colonne].set_ylabel("degré d'appartenance")
            axes[1, colonne].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    return lignes


def main():
    os.makedirs(RESULTATS, exist_ok=True)
    image = charger_image(IMAGE)
    print(f"Image : {image.shape[1]} x {image.shape[0]} pixels")

    # Partie 1 : K = 2, m = 2
    U, centres, iterations = lancer_fcm(image, M)
    texte = [f"Partie 1 : K = {K}, m = {M}, epsilon = {EPSILON}",
             f"Centres trouvés : {centres[0, 0]:.2f} (sombre) et "
             f"{centres[1, 0]:.2f} (clair)",
             f"Nombre d'itérations : {iterations}"]
    nette = segmentation_nette(U, image.shape)
    for j in range(K):
        texte.append(f"Pixels du cluster {j + 1} (segmentation nette) : "
                     f"{100 * np.mean(nette == j):.1f} %")

    figure_heatmaps(U, centres, iterations, image.shape,
                    os.path.join(RESULTATS, "fig1_heatmaps_clusters.png"))
    figure_segmentation(image, U, image.shape,
                        os.path.join(RESULTATS, "fig2_segmentation_nette.png"))

    # Partie 2 : effet de m
    texte.append("")
    texte.append("Partie 2 : effet de m (K = 2, mêmes réglages sinon)")
    texte += figure_effet_de_m(image,
                               os.path.join(RESULTATS, "fig3_effet_de_m.png"))

    print("\n".join(texte))
    with open(os.path.join(RESULTATS, "resultats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(texte) + "\n")


if __name__ == "__main__":
    main()
