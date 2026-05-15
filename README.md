# Système de Détection des Deepfakes par Intelligence Artificielle

Un système complet de détection de deepfakes basé sur l'intelligence artificielle, utilisant des réseaux de neurones convolutifs (CNN) pour analyser les images et vidéos.

## 🎯 Objectif

Développer un système capable de détecter efficacement les deepfakes dans les images et vidéos avec une précision élevée et un temps d'analyse rapide.

## 🏗️ Architecture du Système

### Classes Principales

- **User**: Représente l'utilisateur qui interagit avec l'application
- **Media**: Contient les données à analyser (image ou vidéo)
- **Preprocessor**: Prépare les données avant l'analyse
- **Frame**: Représente une frame vidéo extraite
- **Dataset**: Ensemble de données pour l'entraînement
- **Model**: Classe abstraite pour les modèles d'IA
- **CNNModel**: Implémentation CNN avec Xception, ResNet50, EfficientNet
- **Classifier**: Classe responsable de la décision finale
- **Result**: Contient le résultat avec label et score de confiance

## 🚀 Installation

1. Clonez le repository :
```bash
git clone <repository-url>
cd deepfake_detection
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Créez les répertoires nécessaires :
```bash
mkdir -p data/real data/fake models uploads logs
```

## 📁 Structure du Projet

```
deepfake_detection/
├── src/
│   ├── __init__.py
│   ├── models.py              # Classes de données
│   ├── preprocessor.py        # Prétraitement
│   ├── dataset_manager.py     # Gestion des datasets
│   ├── deep_learning_models.py # Modèles CNN
│   └── classifier.py          # Classification
├── data/                      # Données d'entraînement
├── models/                    # Modèles entraînés
├── uploads/                   # Fichiers uploadés
├── logs/                      # Logs
├── main.py                    # Point d'entrée principal
├── config.py                  # Configuration
├── requirements.txt           # Dépendances
└── README.md                  # Documentation
```

## 💡 Utilisation

### Analyser un fichier

```bash
python main.py --mode analyze --input chemin/vers/fichier.jpg --user-name "Votre Nom"
```

### Entraîner le modèle

```bash
python main.py --mode train --input data/ --epochs 20 --architecture Xception
```

### Utiliser un modèle pré-entraîné

```bash
python main.py --mode analyze --input video.mp4 --model models/Xception_trained.h5
```

## 🎮 Options Disponibles

### Architectures de Modèles
- **Xception**: Haute précision, traitement plus lent
- **ResNet50**: Bon équilibre précision/vitesse
- **EfficientNetB0**: Rapide et efficace

### Paramètres d'Entraînement
- `--epochs`: Nombre d'époques (défaut: 10)
- `--architecture`: Architecture du modèle
- `--batch-size`: Taille du batch (configurable dans config.py)

## 📊 Datasets Supportés

### FaceForensics++
Structure attendue :
```
data/
├── real/
│   ├── image1.jpg
│   └── image2.jpg
└── fake/
    ├── fake1.jpg
    └── fake2.jpg
```

### Celeb-DF
Utilisez le fichier metadata pour charger les données.

## 🔧 Configuration

Le fichier `config.py` contient tous les paramètres configurables :

- **Modèles**: Taille d'entrée, taux d'apprentissage, etc.
- **Prétraitement**: Taille cible, nombre de frames, etc.
- **Classification**: Seuil de décision, stratégie vidéo
- **Entraînement**: Early stopping, taux d'apprentissage fine-tune

## 📈 Performance

### Métriques Évaluées
- **Accuracy**: Précision globale
- **Precision**: Précision par classe
- **Recall**: Rappel par classe
- **F1-Score**: Score F1
- **AUC-ROC**: Aire sous la courbe ROC

### Stratégies de Classification Vidéo
- **Majority Vote**: Vote majoritaire des frames
- **Average**: Moyenne des confiances
- **Weighted**: Moyenne pondérée (frames centrales plus importantes)

## 🛠️ Technologies Utilisées

- **Python 3.8+**
- **TensorFlow/Keras**: Deep Learning
- **OpenCV**: Traitement d'images
- **Scikit-learn**: Métriques et utilitaires
- **Face Recognition**: Détection de visages
- **NumPy/Pandas**: Manipulation de données

## 🎯 Fonctionnalités

### ✅ Implémentées
- [x] Détection de visages
- [x] Extraction de frames vidéo
- [x] Prétraitement des images
- [x] Entraînement CNN
- [x] Classification binaire
- [x] Évaluation des performances
- [x] Support multi-architectures

### 🔄 En Développement
- [ ] Interface web (Streamlit/Flask)
- [ ] Interface graphique desktop
- [ ] Traitement par lots
- [ ] Export des résultats

## 📝 Exemple d'Utilisation

```python
from src import User, Media, Preprocessor, CNNModel, Classifier

# Créer un utilisateur
user = User(id=1, name="Journaliste")

# Initialiser le système
preprocessor = Preprocessor()
model = CNNModel(architecture="Xception")
classifier = Classifier()

# Analyser un média
media = Media(type="image", path="test.jpg", format="jpg")
processed_media = preprocessor.preprocess_media(media)

# Faire une prédiction
if processed_media.frames:
    prediction = model.predict(np.array([processed_media.frames[0].image]))
    result = classifier.classify(prediction[0])
    
    # Afficher le résultat
    print(user.view_result(result))
```

## 🤝 Contributeurs


- **Kdah Fatimzahra**
- **Amine Hajar**

## 📄 Licence

Ce projet est réalisé dans le cadre d'un projet de recherche en ingénierie informatique et réseaux.

## 🔮 Perspectives

- Amélioration de la détection de deepfakes avancés
- Intégration de nouvelles architectures (Transformers)
- Déploiement en production
- API REST pour intégration externe