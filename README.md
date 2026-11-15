# TP Fuzzy C-means
## Bilal D.
## M2VMI

Implémentation de l'algorithme Fuzzy C-means
et application à la segmentation floue d'une image de la Voie lactée en niveaux de gris.

## Contenu

- `fcm.py` : l'algorithme (U, centres, m, epsilon).
- `segmentation.py` : charge l'image, lance le FCM (K = 2), produit les figures.
- `images/` : images de test (`milky-way-nvg.jpg` est utilisée, `milky-way.jpg` ne l'est pas).
- `resultats/` : figures et `resultats.txt`

## Lancer

```bash
pip install -r requirements.txt
python segmentation.py
```

## Résultats produits

- `fig1_heatmaps_clusters.png` : une heatmap par cluster (K = 2, m = 2).
- `fig2_segmentation_nette.png` : image d'origine et segmentation à appartenance maximale.
- `fig3_effet_de_m.png` : extension, effet du paramètre m (1,5 ; 2 ; 3 ; 5).
- `resultats.txt` : centres trouvés et nombre d'itérations pour chaque valeur de m.


