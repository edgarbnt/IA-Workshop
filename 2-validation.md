# Étape 2 : Ajout de la validation

## Le problème
Comment savoir si le modèle ne fait pas que du par coeur ?
Effectivement le problème qu'il peut y avoir est que le modèle apprend les images plutôt qu'essayer de comprendre ce qui fait les caractéristiques de tel ou tel animal.

## Solution : Données de validation
n garde des image a part pour tester pendant l'entraînement.

## Fichier modele.py complet - Étape 2

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

# Chargement des données avec validation
train_datagen = ImageDataGenerator(rescale=1./255)
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

# Modèle (identique à l'étape 1)
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

# Entraînement avec validation
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator  # NOUVEAU !
)

model.save('mon_modele.h5')

# Graphique simple
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Précision')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Perte')
plt.legend()

plt.show()

print("Étape 2 terminée !")
```

## Nouveautés expliquées

### validation_data=val_generator
À chaque époque, ça teste les performances sur des données jamais vues pour l'apprentissage.

### Deux générateurs
- **train_datagen** : pour apprendre
- **val_datagen** : pour tester (même configuration)

### 4 métriques au lieu de 2
- `loss` et `accuracy` : sur l'entraînement
- `val_loss` et `val_accuracy` : sur la validation

### Détection du surapprentissage
- Si train continue de monter mais validation stagne : surapprentissage
- Si écart train-validation > 0.1 : problème de généralisation