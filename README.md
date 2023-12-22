# Colorisation d'image par réseau de neurone

## Installation

Utiliser Python 3, et installer les dépendances avec `pip install -r requirements.txt`.

## Utilisation

Pour lancer l'application web, lancez `python app.py` et rendez-vous sur http://localhost:5000.

Pour utiliser différents modèles, vous pouvez spécifier les poids à charger en ajoutant un argument. Exemple : `python app.py models/modelflower0%.pth`.

Les différents modèles disponibles sur le dépôt sont :
- `models/model1%.pth` : modèle entraîné sur Coco dataset avec 1% de pixels de couleur (propagation forte)
- `models/model5%.pth` : modèle entraîné sur Coco dataset avec 5% de pixels de couleur (propagation moyenne)
- `models/model25%.pth` : modèle entraîné sur Coco dataset avec 25% de pixels de couleur (propagation faible)
- `models/modelflower0%.pth` : modèle entraîné sur Flowers102 dataset avec 0% de pixels de couleur (pixels de couleur non pris en compte)

## Fichiers

Il est possible de récupérer les fichiers générés par le modèle dans le dossier `static`.