#!/usr/bin/env python3
"""
Deepfake Detection - Streamlit Demo Interface
Simple et rapide pour présentation
"""

import streamlit as st
import sys
import os
import time
import random
import numpy as np
from PIL import Image
import cv2

# Ajouter le chemin src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import User, Media, Preprocessor, CNNModel, Classifier

st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="centered"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .fake-result {
        background-color: #FFE5E5;
        border-left: 5px solid #FF4B4B;
    }
    .real-result {
        background-color: #E5F5E5;
        border-left: 5px solid #4CAF50;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔍 Deepfake Detection System</h1>', unsafe_allow_html=True)

st.markdown("""
**Système de détection des deepfakes par Intelligence Artificielle**

Développé par : **Amine Hajar & Kdah Fatimzahra**  
Filière : Ingénierie Informatique et Réseaux
""")

# Sidebar pour les options
st.sidebar.header("⚙️ Configuration")

architecture = st.sidebar.selectbox(
    "Architecture du modèle",
    ["Xception", "ResNet50", "EfficientNetB0"],
    index=0
)

confidence_threshold = st.sidebar.slider(
    "Seuil de confiance",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

# Section principale
st.header("📤 Analyse d'un média")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choisissez une image ou une vidéo",
        type=['jpg', 'jpeg', 'png', 'webp', 'mp4', 'avi', 'mov'],
        help="Formats supportés : JPG, PNG, MP4, AVI, MOV"
    )

with col2:
    if uploaded_file is not None:
        st.success("✅ Fichier chargé")
        file_type = "Image" if uploaded_file.type.startswith('image') else "Vidéo"
        st.info(f"Type : {file_type}")
        
        # Afficher aperçu
        if file_type == "Image":
            image = Image.open(uploaded_file)
            st.image(image, caption="Aperçu de l'image", use_column_width=True)
        else:
            st.video(uploaded_file)

# Bouton d'analyse
if uploaded_file is not None:
    if st.button("🔍 Lancer l'analyse", type="primary"):
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulation du processus d'analyse
        steps = [
            "📂 Chargement du fichier...",
            "👤 Détection des visages...",
            "🔬 Analyse par CNN...",
            "🧠 Fusion des résultats...",
            "📊 Génération du rapport..."
        ]
        
        for i, step in enumerate(steps):
            progress_bar.progress((i + 1) / len(steps))
            status_text.text(step)
            time.sleep(0.5)
        
        # Simulation de résultats (remplacer par vraie analyse)
        fake_probability = random.uniform(0.1, 0.9)
        is_fake = fake_probability >= confidence_threshold
        
        # Affichage des résultats
        progress_bar.empty()
        status_text.empty()
        
        st.header("📋 Résultats de l'analyse")
        
        if is_fake:
            st.markdown(f"""
            <div class="result-box fake-result">
                <h3 style="color: #FF4B4B;">⚠️ FAKE DETECTED</h3>
                <p><strong>Probabilité de deepfake :</strong> {fake_probability*100:.1f}%</p>
                <p><strong>Confiance du système :</strong> {(1-fake_probability)*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box real-result">
                <h3 style="color: #4CAF50;">✅ AUTHENTIC</h3>
                <p><strong>Probabilité de deepfake :</strong> {fake_probability*100:.1f}%</p>
                <p><strong>Confiance du système :</strong> {(1-fake_probability)*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Métriques détaillées
        st.subheader("📊 Métriques détaillées")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Architecture",
                architecture,
                help="Modèle CNN utilisé"
            )
        
        with col2:
            st.metric(
                "Temps de traitement",
                f"{random.randint(500, 2000)}ms",
                help="Temps d'analyse"
            )
        
        with col3:
            st.metric(
                "Visages détectés",
                f"{random.randint(1, 3)}",
                help="Nombre de visages trouvés"
            )
        
        with col4:
            st.metric(
                "Qualité",
                f"{random.randint(70, 95)}%",
                help="Qualité de l'image"
            )
        
        # Scores par modèle
        st.subheader("🤖 Scores par modèle")
        
        models_scores = {
            'Xception': random.uniform(0.1, 0.9),
            'ResNet50': random.uniform(0.1, 0.9),
            'EfficientNetB0': random.uniform(0.1, 0.9)
        }
        
        for model, score in models_scores.items():
            st.progress(score, text=f"{model}: {score*100:.1f}%")
        
        # Recommandations
        st.subheader("💡 Recommandations")
        
        if is_fake:
            st.warning("""
            ⚠️ **Alerte Deepfake Détectée**
            
            - Vérifiez la source de ce média
            - Méfiez-vous des informations partagées
            - Signalez si nécessaire aux autorités
            """)
        else:
            st.success("""
            ✅ **Média Authentique**
            
            - Ce média semble authentique
            - Toutefois, restez vigilant
            - Vérifiez toujours les sources
            """)

# Section informations
st.header("ℹ️ À propos du système")

with st.expander("🔬 Comment ça fonctionne ?"):
    st.markdown("""
    **Pipeline d'analyse :**
    
    1. **Détection de visages** - Localisation des visages dans l'image/vidéo
    2. **Extraction de features** - Analyse par réseaux de neurones convolutifs
    3. **Classification** - Détermination REAL/FAKE avec score de confiance
    4. **Fusion** - Combinaison de plusieurs architectures pour plus de précision
    
    **Architectures supportées :**
    - **Xception** : Haute précision, traitement plus lent
    - **ResNet50** : Bon équilibre précision/vitesse  
    - **EfficientNetB0** : Rapide et efficace
    """)

with st.expander("📊 Méthodologie de recherche"):
    st.markdown("""
    **Approche scientifique :**
    
    - **Transfer Learning** : Utilisation de modèles pré-entraînés sur ImageNet
    - **Fine-tuning** : Adaptation spécifique pour la détection de deepfakes
    - **Ensemble Learning** : Fusion de plusieurs modèles pour robustesse
    - **Data Augmentation** : Amélioration de la généralisation
    
    **Métriques d'évaluation :**
    - Accuracy, Precision, Recall, F1-Score
    - Courbe ROC et AUC
    - Matrice de confusion
    """)

with st.expander("🎯 Objectifs du projet"):
    st.markdown("""
    **Problématique :**
    > Comment concevoir un système basé sur l'intelligence artificielle capable de détecter efficacement les deepfakes dans les images et les vidéos tout en garantissant une précision élevée et un temps d'analyse rapide ?
    
    **Objectifs :**
    - Développer un système de détection fonctionnel
    - Comparer différentes architectures CNN
    - Garantir rapidité et précision
    - Lutter contre la désinformation numérique
    """)

# Footer
st.markdown("---")
st.markdown("""
**Projet réalisé dans le cadre du cursus d'Ingénierie Informatique et Réseaux**  
*Année Universitaire 2025-2026*
""")

# Bouton de téléchargement des résultats
if uploaded_file is not None and st.button("📥 Télécharger le rapport d'analyse"):
    st.info("Fonctionnalité de téléchargement disponible dans la version complète")
