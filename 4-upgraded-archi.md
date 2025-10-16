# Étape 4 : Architecture plus puissante

## Le problème

e modèle simple atteint ses limites. On peut l'améliorer avec des techniques plutot récentes

## Les nouvelles techniques

- **BatchNormalization** : stabilise l'entraînement
- **Dropout** : réduit le surapprentissage
- **Architecture plus profonde** : plus de capacité

## Construction du code étape par étape

### Étape 4.1 : Ajout de BatchNormalization

Modifie tes couches Conv2D :

```python
# Premier bloc avec BatchNormalization
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
layers.BatchNormalization(),  # NOUVEAU !
layers.MaxPooling2D(2, 2),
```

**BatchNormalization expliqué :**
- Normalise les données entre chaque couche
- **Problème résolu :** Les données changent de distribution entre couches
- **Avantages :** Entraînement plus rapide et stable
- **Coût :** Quelques paramètres supplémentaires

### Étape 4.2 : Troisième bloc convolutionnel

ajoute un bloc supplémentaire :

```python
# Troisième bloc - NOUVEAU
layers.Conv2D(128, (3, 3), activation='relu'),
layers.BatchNormalization(),
layers.MaxPooling2D(2, 2),
```

**Progression logique des filtres :** 32 → 64 → 128

**Pourquoi cette progression ?**
- **Premières couches :** Détectent formes simples → peu de filtres
- **Couches intermédiaires :** Combinent en formes complexes → plus de filtres
- **Dernières couches :** Reconnaissent objets entiers → beaucoup de filtres

### Étape 4.3 : Dropout pour la régularisation

Modifie tes couches Dense :

```python
# Classification avec régularisation
layers.Flatten(),
layers.Dropout(0.5), # NOUVEAU
layers.Dense(128, activation='relu'), # Plus de neurones
layers.Dropout(0.5), # NOUVEAU
layers.Dense(3, activation='softmax')
```

**Dropout(0.5) expliqué :**
- Désactive aléatoirement 50% des neurones pendant l'entraînement
- **Pourquoi ça marche :** Force le modèle à ne pas dépendre de neurones spécifiques
- **Important :** Ne s'active QUE pendant l'entraînement, pas lors des prédictions

### Étape 4.4 : Plus d'époques pour modèle complexe

```python
# Plus d'époques pour le modèle plus complexe
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator
)
```

**Pourquoi plus d'époques ?**
- Modèle plus complexe = plus de paramètres à optimiser
- Besoin de plus de temps pour converger

## Fichier modele.py complet - Étape 4

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Data Augmentation (comme étape 3)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'data/validation',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# modele amélioré avec batchNorm et dropoout
model = models.Sequential([
    # premier bloc convolutionnel
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    # Deuxième bloc convolutionnel
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    # troisième bloc convolutionnel - NOUVEAU
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    # Classification avec régularisation
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator
)

model.save('mon_modele.h5')

# Graphique
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Précision - Modèle amélioré')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Perte - Modèle amélioré')
plt.legend()

plt.show()

print("Étape 4 terminée !")
```

