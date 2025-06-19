# Étape 1 : Premier modèle qui fonctionne

## Théorie rapide

Un CNN (Convolutional Neural Network) traite les images avec des filtres qui détectent des formes simples puis complexes :
- **Conv2D** : Détecte des motifs (contours, textures)
- **MaxPooling2D** : Réduit la taille des données
- **Dense** : Couches finales pour la classification

## Construction du code étape par étape

### Étape 1.1 : Les imports et configuration

commence par créer `modele.py` et ajoute ces import :

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Configuration
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
```

**Qu'est-ce qui se passe ici ?**

- **tensorflow** : La bibliothèque principale pour le machine learning
- **layers, models** : Pour construire notre réseau de neurones couche par couche
- **ImageDataGenerator** : Pour charger et transformer nos images automatiquement
- **matplotlib** : Pour faire des graphiques

**Pourquoi ces valeurs de configuration ?**
- **IMG_HEIGHT/WIDTH = 150** : Toutes nos images seront redimensionnées à 150x150 pixels. Plus grand = plus de détails mais plus lent
- **BATCH_SIZE = 32** : On traite 32 images à la fois. C'est un bon compromis entre vitesse et mémoire

### Étape 1.2 : Chargement des données

Ajoute cette partie :

```python
# Chargement des données
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
```

**Explications détaillées :**

**ImageDataGenerator(rescale=1./255)** :
- Les pixels d'une image vont de 0 à 255 (noir à blanc)
- `rescale=1./255` divise chaque pixel par 255
- Résultat : pixels entre 0 et 1 au lieu de 0-255
- Pourquoi ? Les réseaux de neurones apprennent mieux avec des petites valeurs

**flow_from_directory** :
- Lit automatiquement les images dans des dossiers
- `'data/train'` : cherche dans ce dossier
- `target_size=(150, 150)` : redimensionne toutes les images à 150x150
- `class_mode='categorical'` : pour classification multi-classes (chat, chien, oiseau)

**Ce qui se passe concrètement :**
```
data/train/
├── chat/       ← Classe 0
├── chien/      ← Classe 1  
└── oiseau/     ← Classe 2
```

Lance ce bout de code. Tu devrais voir :
```
Found 2400 images belonging to 3 classes.
```

### Étape 1.3 : Construction du modèle

Maintenant, on construit notre réseau de neurones :

```python
# Modèle simple
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])
```

**Décortiquons couche par couche :**

**Sequential** : Les couches s'enchaînent une après l'autre (comme des perles sur un collier)

**Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))** :
- `32` : nombre de filtres (détecteurs de motifs)
- `(3, 3)` : taille de chaque filtre (3x3 pixels)
- `activation='relu'` : fonction d'activation (si négatif → 0)
- `input_shape=(150, 150, 3)` : images 150x150 en couleur (RGB = 3 canaux)

**Que fait cette couche ?** 
Imagine 32 "loupes" différentes qui parcourent l'image pour détecter des motifs : lignes horizontales, verticales, courbes...

**MaxPooling2D(2, 2)** :
- Réduit la taille de l'image de moitié
- 150x150 → 75x75
- Garde seulement les informations importantes
- But : réduire les calculs et éviter le surapprentissage

**Conv2D(64, (3, 3), activation='relu')** :
- Deuxième couche de convolution
- Plus de filtres (64) pour détecter des motifs plus complexes
- Maintenant on combine les motifs simples en formes plus élaborées

**MaxPooling2D(2, 2)** :
- Encore une réduction : 75x75 → 37x37

**Flatten()** :
- Transforme l'image 2D en liste 1D
- Nécessaire pour passer aux couches Dense
- 37x37x64 → une longue liste de nombres

**Dense(64, activation='relu')** :
- Couche "classique" avec 64 neurones
- Chaque neurone regarde TOUS les éléments de la liste
- Apprend les relations complexes

**Dense(3, activation='softmax')** :
- Couche finale : 3 neurones pour 3 classes
- `softmax` : transforme en probabilités qui somment à 1
- Exemple : [0.7, 0.2, 0.1] = 70% chien, 20% chat, 10% oiseau

### Étape 1.4 : Compilation du modèle

```python
# Compilation
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

**Qu'est-ce que la compilation ?**
On dit au modèle comment apprendre :

**optimizer='adam'** :
- Algorithme qui ajuste les poids du réseau
- Adam = adaptatif, il s'ajuste tout seul
- Alternative : SGD (plus simple mais demande plus de réglages)

**loss='categorical_crossentropy'** :
- Fonction qui mesure l'erreur
- Compare la prédiction [0.1, 0.8, 0.1] avec la vraie réponse [0, 1, 0]
- Plus l'erreur est grande, plus le modèle va se corriger

**metrics=['accuracy']** :
- Métrique facile à comprendre : % de bonnes réponses
- On pourrait ajouter d'autres métriques mais accuracy suffit pour commencer

### Étape 1.5 : Visualisation de la structure

```python
# Affichage de la structure
model.summary()
```

Lance ce code. Tu verras quelque chose comme :
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 32)      896       
 max_pooling2d (MaxPooling2  (None, 74, 74, 32)        0         
 conv2d_1 (Conv2D)           (None, 72, 72, 64)        18496     
 max_pooling2d_1 (MaxPoolin  (None, 36, 36, 64)        0         
 flatten (Flatten)           (None, 82944)              0         
 dense (Dense)               (None, 64)                 5308480   
 dense_1 (Dense)             (None, 3)                  195       
=================================================================
Total params: 5327971 (20.32 MB)
```

**Comment lire ce tableau ?**
- **Output Shape** : Taille des données après chaque couche
- **Param #** : Nombre de paramètres (poids) à apprendre
- **Total params** : 5,3 millions de paramètres ! C'est beaucoup mais normal

**Pourquoi la taille change ?**
- (150, 150, 3) → (148, 148, 32) : Conv2D réduit un peu + 32 filtres
- (148, 148, 32) → (74, 74, 32) : MaxPooling divise par 2
- etc...

### Étape 1.6 : Entraînement

```python
# Entraînement
print("Début de l'entraînement...")
history = model.fit(train_generator, epochs=5)
```

**Qu'est-ce qui va se passer ?**
- Le modèle va voir toutes tes images 5 fois (5 époques)
- À chaque image, il va prédire et corriger ses erreurs
- Tu verras défiler :
```
Epoch 1/5
75/75 [==============================] - 45s 590ms/step - loss: 1.0856 - accuracy: 0.4254
```

**Comment interpréter ces chiffres ?**
- **loss: 1.0856** : Erreur du modèle (plus bas = mieux)
- **accuracy: 0.4254** : 42,54% de bonnes réponses
- **75/75** : 75 batchs de 32 images = 2400 images total

**Évolution typique :**
- Époque 1 : ~40% (mieux que le hasard à 33%)
- Époque 2 : ~55%
- Époque 3 : ~65%
- Époque 4 : ~70%
- Époque 5 : ~75%

### Étape 1.7 : Sauvegarde et visualisation

```python
# Sauvegarde
model.save('mon_modele.h5')
print("Modèle sauvegardé !")

# Graphique simple
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Précision')
plt.xlabel('Époque')
plt.ylabel('Précision')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Perte (erreur)')
plt.xlabel('Époque')
plt.ylabel('Perte')

plt.tight_layout()
plt.show()

print("Modèle de base terminé !")
```

**Qu'est-ce que history ?**
- `history` contient toutes les métriques de chaque époque
- `history.history['accuracy']` = [0.42, 0.55, 0.65, 0.70, 0.75]
- `history.history['loss']` = [1.08, 0.89, 0.75, 0.68, 0.62]

**Que devraient montrer les graphiques ?**
- **Précision** : courbe qui monte (bon signe)
- **Perte** : courbe qui descend (bon signe aussi)

si les courbes stagnent ou font n'importe quoi, il y a un problème.

## fichier complet

voici le fichier `modele.py` complet :

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Configuration
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Chargement des données
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Modèle simple
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes
])

# Compilation
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Affichage de la structure
print("Structure du modèle :")
model.summary()

# Entraînement
print("\nDébut de l'entraînement...")
history = model.fit(train_generator, epochs=5)

# Sauvegarde
model.save('mon_modele.h5')
print("Modèle sauvegardé !")

# Graphiques
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Précision')
plt.xlabel('Époque')
plt.ylabel('Précision')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Perte (erreur)')
plt.xlabel('Époque')
plt.ylabel('Perte')

plt.tight_layout()
plt.show()

print("Modèle de base terminé !")
print(f"Précision finale : {history.history['accuracy'][-1]:.3f}")
```

Lance ce code avec `python modele.py`