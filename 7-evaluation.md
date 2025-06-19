# Étape 7 : Évaluation finale et sauvegarde

## Le problème

Comment séparer l'entraînement du test ? Comment sauvegarder proprement le modèle pour l'utiliser plus tard ?

## La solution

- **Script d'entraînement** : entraîne et sauvegarde le modèle
- **Script de test séparé** : charge le modèle et teste les prédictions

## Construction du code étape par étape

### Étape 7.1 : Générateur de test

Ajoute ce générateur pour l'évaluation finale :

```python
# NOUVEAU : Générateur de test pour évaluation finale
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Important pour l'analyse
)
```

**Structure attendue :**
```
data/
├── train/      (80% des images)
├── validation/ (10% des images)
└── test/       (10% des images - jamais vues)
    ├── chat/
    ├── chien/
    └── oiseau/
```

### Étape 7.2 : Évaluation finale

Ajoute après l'entraînement :

```python
# NOUVEAU : Test sur données jamais vues
best_model = tf.keras.models.load_model('best_model.h5')
test_loss, test_acc = best_model.evaluate(test_generator)
print(f"Précision finale sur test : {test_acc:.3f}")
```

**Pourquoi charger le modèle ?**
- On utilise le meilleur modèle sauvé par ModelCheckpoint
- Pas forcément le dernier de l'entraînement


## Fichier modele.py complet - Étape 7 (Entraînement)

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# configuration
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

print("=== ENTRAÎNEMENT DU MODÈLE ===")

# data Augmentation
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
test_datagen = ImageDataGenerator(rescale=1./255)

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

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())

# modèle complet
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
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# callbacks
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

# entraînement
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=callbacks
)

# evaluation finale
best_model = tf.keras.models.load_model('best_model.h5')
test_loss, test_acc = best_model.evaluate(test_generator)
print(f"Précision finale sur test : {test_acc:.3f}")

# Graphique final
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Précision finale')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Perte finale')
plt.legend()

plt.show()

print("\nENTRAÎNEMENT TERMINÉ !")
print("Fichiers sauvegardés :")
print("  - best_model.h5 (modèle)")
print("\nPour tester : python test_prediction.py")
```

## Script de test séparé

Crée un nouveau fichier `test_prediction.py` :

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# charger le modèle
model = tf.keras.models.load_model('best_model.h5')
class_names = ['chat', 'chien', 'oiseau']

image_path = 'data/test/chat/image1.jpg'  # Change ce chemin

img = image.load_img(image_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

predictions = model.predict(img_array, verbose=0)
predicted_class = class_names[np.argmax(predictions[0])]

print(predicted_class)
```

## Utilisation

1. **Entraîne le modèle :**
```bash
python modele.py
```

2. **Teste une prédiction :**
   - Modifie `image_path` dans `test_prediction.py`
   - Lance :
```bash
python test_prediction.py
```

Maintenant tu peux entraîner une fois et tester autant d'images que tu veux !