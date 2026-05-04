# Modèle CLIP pour l'alignement multimodal image-texte

Implémentation from scratch d'un modèle CLIP (Contrastive Language-Image Pretraining) pour apprendre un espace latent commun entre images et descriptions textuelles.

**Projet de M2 Bioinformatique** · Université de Montpellier (2025-2026)

## Objectif

Construire, entraîner et évaluer une chaîne complète comprenant :

1. **Classifieur d'images** basé sur un CNN
2. **Classifieur de textes** basé sur SmallBERT
3. **Transformation** de ces classifieurs en encodeurs produisant des embeddings comparables
4. **Assemblage dans un modèle CLIP** entraîné par loss contrastive pour la recherche image → texte et texte → image

## Données

Sous-ensemble du dataset **Flickr** contenant 600 images réparties en 4 catégories (ball, bike, dog, water) avec leurs descriptions textuelles associées.

## Architecture

```
Image (224x224x3) ──► CNN Encoder ──► L2 Norm ──┐
                                                  ├──► Loss Contrastive
Texte (32 tokens) ──► SmallBERT   ──► L2 Norm ──┘
```

- **Encodeur image :** 2 blocs convolutionnels + projection Dense(128)
- **Encodeur texte :** SmallBERT + projection Dense(128)
- **Loss :** contrastive symétrique (température = 0.07)
- **Embedding :** 128 dimensions, normalisation L2

## Résultats

| Composant | Accuracy validation | Commentaire |
|-----------|-------------------|-------------|
| CNN (classifieur image) | 37.5% | Limité par la taille du dataset |
| SmallBERT (classifieur texte) | 70.8% | Le texte contient souvent explicitement la classe |
| CLIP (alignement) | · | Surapprentissage, pas d'alignement sémantique réel |

### Analyse

- SmallBERT surpasse le CNN de 33 points car les descriptions contiennent des indices textuels directs
- Le modèle CLIP souffre d'un manque de données pour l'apprentissage contrastif (600 paires vs 400M pour le CLIP original d'OpenAI)
- La matrice de similarité ne montre pas de diagonale distincte, confirmant l'absence de généralisation
- Ce résultat est attendu et pédagogiquement riche : il illustre que les architectures seules ne suffisent pas sans données à l'échelle

### Pistes d'amélioration identifiées

- Utilisation d'encodeurs pré-entraînés (ResNet, BERT complet)
- Dataset plus conséquent
- Data augmentation sur les images

## Structure du projet

```
├── data/                    # Dataset Flickr (images + captions)
├── rapport.pdf              # Rapport complet du projet
├── notebook.pdf             # Notebook d'analyse complet
├── README.md
```

## Outils

Python · TensorFlow/Keras · SmallBERT · CNN · Loss contrastive · NumPy · matplotlib

## Auteur

Hermine KIOSSOU · M2 Bioinformatique, Université de Montpellier
