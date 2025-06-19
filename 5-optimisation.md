# Étape 5 : Optimisation des hyperparamètres

## Le problème

Le modèle peut encore mieux apprendre avec des réglages plus fins du learning rate afin de lui permettre de mieuw converger vers la bonne prediction

## La solution : Learning Rate adaptatif
Au lieu d'un learning rate fixe, on va l'adapter automatiquement pendant l'entraînement.

## Construction du code étape par étape

### Étape 5.1 : Optimiseur avec LR explicite

Modifie la compilation :

```python
# NOUVEAU : Optimiseur avec learning rate explicite
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Changement :**
- Avant : `optimizer='adam'` (LR par défaut)
- Maintenant : LR explicite à 0.001

### Étape 5.2 : Callback Learning Rate adaptatif

Ajoute ce callback avant l'entraînement :

```python
# NOUVEAU : Learning Rate adaptatif
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', # Surveille la perte de validation
    factor=0.5, # Divise le LR par 2
    patience=3, # Attend 3 époques sans amélioration
    min_lr=0.0001, # LR minimum
    verbose=1 # Affiche quand ça change
)
```

**Comment ça marche :**

**ReduceLROnPlateau :**
- Quand l'apprentissage stagne, divise le learning rate par 2
- **monitor='val_loss'** : surveille la perte de validation
- **factor=0.5** : divise par 2 (0.001 → 0.0005 → 0.00025...)
- **patience=3** : attend 3 époques sans amélioration
- **min_lr=0.0001** : ne descend pas en dessous

### Étape 5.3 : Entraînement avec callback

Modifie ton entraînement :

```python
# Entraînement avec callback
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[lr_scheduler]  # NOUVEAU
)
```

**Stratégie typique :**
1. Début : LR = 0.001 (apprentissage rapide)
2. Plateau détecté : LR = 0.0005 (affinage)
3. Nouveau plateau : LR = 0.00025 (peaufinage)

## Explications techniques

### Learning Rate (LR)
Contrôle la "taille des pas" lors de l'apprentissage.

**LR trop élevé (0.1) :**
- Le modèle "saute" par-dessus la solution optimale
- Apprentissage instable

**LR trop faible (0.00001) :**
- Apprentissage très lent
- Risque de rester bloqué

**LR idéal (0.001) :**
- Bon compromis vitesse/stabilité

## Fichier modele.py complet - Étape 5

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

# Data Augmentation
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

# Modèle amélioré comme a l'étape précédente
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

# NOUVEAU : Optimiseur avec learning rate adaptatif
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# NOUVEAU : Learning Rate adaptatif
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.0001,
    verbose=1
)

# Entraînement avec callback
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[lr_scheduler]
)

model.save('mon_modele.h5')

# Graphique
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Précision - LR adaptatif')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Perte - LR adaptatif')
plt.legend()

plt.show()

print("Étape 5 terminée !")
```

Tu devrais voir dans les logs quand le LR se réduit automatiquement !