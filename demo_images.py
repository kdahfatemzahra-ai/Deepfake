#!/usr/bin/env python3
"""
Script de démonstration avec images exemples
Pour présentation rapide du projet
"""

import os
import sys
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Ajouter le chemin src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import User, Media, Preprocessor, CNNModel, Classifier

def create_demo_image(label, size=(300, 300)):
    """Crée une image de démonstration avec un visage simple"""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Dessiner un visage simple
    w, h = size
    # Face
    draw.ellipse([w*0.2, h*0.15, w*0.8, h*0.85], fill='peachpuff', outline='black', width=2)
    # Eyes
    draw.ellipse([w*0.3, h*0.3, w*0.4, h*0.4], fill='white', outline='black', width=2)
    draw.ellipse([w*0.6, h*0.3, w*0.7, h*0.4], fill='white', outline='black', width=2)
    draw.ellipse([w*0.33, h*0.33, w*0.37, h*0.37], fill='black')
    draw.ellipse([w*0.63, h*0.33, w*0.67, h*0.37], fill='black')
    # Mouth
    draw.arc([w*0.35, h*0.55, w*0.65, h*0.75], 0, 180, fill='black', width=2)
    # Nose
    draw.polygon([(w*0.5, h*0.45), (w*0.45, h*0.55), (w*0.55, h*0.55)], fill='peachpuff', outline='black')
    
    # Ajouter le label
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), label, fill='red' if label == "FAKE" else 'green', font=font)
    
    return img

def run_demo():
    """Lance la démonstration complète"""
    print("🔍 DÉMONSTRATION - Deepfake Detection System")
    print("=" * 50)
    
    # Créer les images de démo
    print("📸 Création des images de démonstration...")
    
    # Image authentique
    real_img = create_demo_image("REAL")
    real_path = "demo_real.jpg"
    real_img.save(real_path)
    print(f"✅ Image réelle créée : {real_path}")
    
    # Image deepfake
    fake_img = create_demo_image("FAKE")
    fake_path = "demo_fake.jpg"
    fake_img.save(fake_path)
    print(f"✅ Image fake créée : {fake_path}")
    
    # Initialiser le système
    print("\n🤖 Initialisation du système...")
    try:
        preprocessor = Preprocessor()
        model = CNNModel(architecture="Xception")
        classifier = Classifier()
        user = User(id=1, name="Démonstration")
        print("✅ Système initialisé avec succès")
    except Exception as e:
        print(f"❌ Erreur d'initialisation : {e}")
        return
    
    # Analyser l'image réelle
    print("\n🔬 Analyse de l'image authentique...")
    try:
        media_real = Media(type='image', path=real_path, format='jpg')
        processed_real = preprocessor.preprocess_media(media_real)
        
        if processed_real.frames:
            prediction_real = model.predict(np.array([processed_real.frames[0].image]))
            result_real = classifier.classify(prediction_real[0])
            print(f"📊 Résultat : {user.view_result(result_real)}")
        else:
            print("❌ Aucun visage détecté dans l'image réelle")
    except Exception as e:
        print(f"❌ Erreur analyse image réelle : {e}")
    
    # Analyser l'image fake
    print("\n🔬 Analyse de l'image deepfake...")
    try:
        media_fake = Media(type='image', path=fake_path, format='jpg')
        processed_fake = preprocessor.preprocess_media(media_fake)
        
        if processed_fake.frames:
            prediction_fake = model.predict(np.array([processed_fake.frames[0].image]))
            result_fake = classifier.classify(prediction_fake[0])
            print(f"📊 Résultat : {user.view_result(result_fake)}")
        else:
            print("❌ Aucun visage détecté dans l'image fake")
    except Exception as e:
        print(f"❌ Erreur analyse image fake : {e}")
    
    # Comparaison des architectures
    print("\n🏗️ Comparaison des architectures...")
    architectures = ["Xception", "ResNet50", "EfficientNetB0"]
    
    for arch in architectures:
        try:
            print(f"\n🤖 Test avec {arch}...")
            model_test = CNNModel(architecture=arch)
            
            # Test sur une image
            if processed_real.frames:
                pred = model_test.predict(np.array([processed_real.frames[0].image]))
                result = classifier.classify(pred[0])
                print(f"   📊 {arch}: {result.label} ({result.confidence:.1f}%)")
            
        except Exception as e:
            print(f"   ❌ Erreur avec {arch}: {e}")
    
    # Nettoyage
    try:
        os.remove(real_path)
        os.remove(fake_path)
        print("\n🧹 Nettoyage des fichiers temporaires")
    except:
        pass
    
    print("\n🎉 Démonstration terminée !")
    print("\n💡 Pour lancer l'interface web :")
    print("   python app/app.py")
    print("\n🎯 Pour lancer l'interface Streamlit :")
    print("   streamlit run streamlit_demo.py")

if __name__ == "__main__":
    run_demo()
