# Étape 3 : Data Augmentation

## Le problème détecté

Si tu vois du surapprentissage (validation qui stagne alors que train continue), c'est que ton modèle manque de variété dans les données.

## La solution : Data Augmentation

Au lieu de collecter plus d'images, on va **transformer** les images existantes pour créer de la variété artificielle.

## Construction du code étape par étape

### Étape 3.1 : Générateur avec transformations

Remplace ton `train_datagen` par celui ci :

```python
# Générateur avec augmentation pour l'entraînement
train_datagen = ImageDataGenerator(
    rescale=1./255, # normalisation
    rotation_range=20, # rotation aléatoire ±20°
    width_shift_range=0.2, # décalage horizontal ±20%
    height_shift_range=0.2, # décalage vertical ±20%
    shear_range=0.2, # cisaillement (déformation)
    zoom_range=0.2, # zoom aléatoire ±20%
    horizontal_flip=True, # miroir horizontal
    fill_mode='nearest' # remplissage des pixels manquants
)
```

**Chaque transformation expliquée :**

**rotation_range=20 :**
- Fait tourner l'image entre -20° et +20°
- Un chat reste un chat même s'il a la tête penchée

**width_shift_range=0.2 :**
- Décale l'image horizontalement de ±20%
- Simule un animal pas parfaitement centré

**shear_range=0.2 :**
- Cisaillement : déforme l'image comme si tu tirais sur un coin
- Simule des angles de vue différents

**zoom_range=0.2 :**
- Zoom entre 80% et 120%
- Simule différentes distances de l'animal

**horizontal_flip=True :**
- Miroir horizontal (gauche ↔ droite)
- **ATTENTION :** Pas de vertical_flip ! Un chat à l'envers n'est pas naturel

### Étape 3.2 : Validation sans augmentation

**Important :** garde le validation_datagen sans transformation :

```python
# Validation sans augmentation ca permet d'avoir une mesure objective
val_datagen = ImageDataGenerator(rescale=1./255)
```

**Pourquoi seulement sur train ?**
- La validation doit rester "pure" pour mesurer les vraies performances
- Augmenter la validation serait tricher !

### Étape 3.3 : Plus d'époques

Modifie ton entraînement :

```python
# Entraînement avec augmentation
history = model.fit(
    train_generator,
    epochs=20,  # plus d'époques car augmentation ralentit
    validation_data=val_generator
)
```

**Pourquoi plus d'époques ?**
- L'augmentation rend l'apprentissage plus "difficile"
- Le modèle a besoin de plus de temps pour converger
- Mais il généralise mieux !

## Impact attendu

**Avantages :**
- Réduit le surapprentissage
- Améliore la généralisation
- Équivaut à avoir plus de données

**Inconvénients :**
- Entraînement plus lent
- Peut parfois diminuer la précision d'entraînement (normal)

## Fichier modele.py complet - Étape 3

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

# Générateur AVEC augmentation pour train, SANS pour validation
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

# Modèle (identique)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement avec augmentation
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator
)

model.save('mon_modele.h5')

# Graphique
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Précision avec Data Augmentation')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Perte avec Data Augmentation')
plt.legend()

plt.show()

print("Étape 3 terminée !")
```
