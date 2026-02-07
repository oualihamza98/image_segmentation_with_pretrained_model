# Image Segmentation App with DeepLabV3+ResNet50

Application Flask de segmentation sémantique d'images utilisant un modèle de Deep Learning.

## Modèle utilisé
**DeepLabV3+ avec backbone ResNet50**
- Architecture : Convolutions dilatées (atrous convolution)
- Dataset : PASCAL VOC (21 classes)
- Tâche : Segmentation sémantique (classification pixel par pixel)

## Prétraitement (étapes critiques)

### Pipeline de transformation
```python
1. Conversion RGB       → Assure 3 canaux couleur
2. ToTensor()          → [0, 255] → [0.0, 1.0]
3. Normalize()         → mean=[0.485, 0.456, 0.406]
                         std=[0.229, 0.224, 0.225]
                         (statistiques ImageNet)
4. unsqueeze(0)        → Ajoute dimension batch [1, C, H, W]
```

**Pourquoi normaliser ?** Le modèle a été entraîné sur ImageNet avec ces statistiques spécifiques. Sans cette normalisation exacte, les prédictions seraient incorrectes.


Accès : http://localhost:3000

## Output
- Image segmentée avec masque coloré superposé
- Liste des classes détectées (person, car, dog, etc.)