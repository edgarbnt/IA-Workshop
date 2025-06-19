# Étape 6 : Automatisation avec callbacks

## Le problème

Comment savoir quand arrêter l'entraînement ? Comment sauvegarder automatiquement le meilleur modèle

## La solution : callbacks automatiques

On va ajouter des "fonctions automatiques" qui s'exécutent pendant l'entraînement

## Construction du code étape par étape

### Étape 6.1 : EarlyStopping

Ajoute ce callback :

```python
tf.keras.callbacks.EarlyStopping(
    patience=5, # arrete après 5 époques sans amélioration
    restore_best_weights=True, # revient aux meilleurs poids
    monitor='val_accuracy'# surveille la précision validation
)
```

**early stopping expliqué :**
- **Problème :** Comment savoir quand arrêter ?
- **Solution :** Arrêt automatique quand la validation n'améliore plus
- **patience=5** : attend 5 époques consécutives sans amélioration
- **restore_best_weights=True** : revient aux meilleurs poids (pas les derniers)

### Étape 6.2 : ModelCheckpoint

Ajoute ce callback :

```python
tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5', # Sauvegarde le meilleur modèle
    save_best_only=True, # Seulement si amélioration
    monitor='val_accuracy' # Critère d'amélioration
)
```

**ModelCheckpoint expliqué :**
- **Problème :** Et si l'ordinateur plante ?
- **Solution :** Sauvegarde automatique à chaque amélioration
- **save_best_only=True** : ne sauvegarde que les améliorations

### Étape 6.3 : Combinaison des callbacks

Combine tous les callbacks :

```python
# NOUVEAU : callbacks automatiques
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor='val_accuracy'
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=3,
        monitor='val_loss'
    )
]
```

### Étape 6.4 : Entraînement automatisé

```python
# Entraînement automatisé
history = model.fit(
    train_generator,
    epochs=50,  # Beaucoup d'époques, mais arrêt automatique
    validation_data=val_generator,
    callbacks=callbacks
)
```

**Stratégie :**
- On met 50 époques mais l'EarlyStopping arrêtera probablement vers 20-30
- Le meilleur modèle sera automatiquement sauvé

## Explications techniques

### Ordre d'exécution des callbacks
Les callbacks s'exécutent dans l'ordre de la liste :
1. **EarlyStopping** : vérifie s'il faut arrêter
2. **ModelCheckpoint** : sauvegarde si amélioration
3. **ReduceLROnPlateau** : réduit le LR si stagnation

### Combinaison intelligente
Les 3 callbacks travaillent ensemble :
1. **ReduceLROnPlateau** : essaie d'améliorer en réduisant le LR
2. **ModelCheckpoint** : sauvegarde chaque amélioration
3. **EarlyStopping** : arrête quand plus rien ne marche

## Fichier modele.py complet - Étape 6

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

# Modèle amélioré
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

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# NOUVEAU : Callbacks automatiques
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor='val_accuracy'
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=3,
        monitor='val_loss'
    )
]

# Entraînement automatisé
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=callbacks
)

# Graphique
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Précision - Entraînement automatisé')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Perte - Entraînement automatisé')
plt.legend()

plt.show()

print("Étape 6 terminée !")
print("Meilleur modèle sauvé dans 'best_model.h5'")
```

Maintenant ton modèle s'entraîne en "pilote automatique" ! c'est a dire qu'il gere lui même quand s'arrété pour sauvegarde le meilleur modèle !