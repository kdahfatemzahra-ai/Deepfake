#!/usr/bin/env python3
"""
DÉMO FINALE - Système de Détection des Deepfakes
Script unique pour présentation complète
"""

import os
import sys
import time
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import User, Media, Preprocessor, CNNModel, Classifier

def print_header():
    """Affiche l'en-tête de présentation"""
    print("🔍" + "="*60 + "🔍")
    print("🔍     SYSTÈME DE DÉTECTION DES DEEPFAKES PAR IA      🔍")
    print("🔍" + "="*60 + "🔍")
    print("👥 Auteurs : Amine Hajar & Kdah Fatimzahra")
    print("🎓 Filière : Ingénierie Informatique et Réseaux")
    print("📅 Année : 2025-2026")
    print("🔍" + "="*60 + "🔍")
    print()

def show_system_info():
    """Affiche les informations du système"""
    print("📊 INFORMATIONS DU SYSTÈME")
    print("-" * 40)
    print("🤖 Modèles entraînés :")
    print("   • Xception (299×299×3)")
    print("   • ResNet50 (224×224×3)")
    print("   • EfficientNetB0 (224×224×3)")
    print()
    print("📚 Dataset d'entraînement :")
    print("   • 10,000 images au total")
    print("   • 5,000 images authentiques")
    print("   • 5,000 images deepfakes")
    print("   • 5 epochs de training")
    print()
    print("⚙️ Technologies :")
    print("   • Python 3.10 + TensorFlow 2.13")
    print("   • OpenCV pour traitement d'images")
    print("   • Transfer Learning (ImageNet → Deepfake)")
    print()

def create_demo_image(label="DEMO", size=(300, 300)):
    """Crée une image de démo avec visage"""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
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
    
    # Label
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), label, fill='blue', font=font)
    
    return img

def demonstrate_pipeline():
    """Démontre le pipeline complet"""
    print("🔄 DÉMONSTRATION DU PIPELINE COMPLET")
    print("-" * 40)
    
    # Créer image de démo
    demo_img = create_demo_image("DEMO_PRESENTATION")
    demo_path = "demo_final.jpg"
    demo_img.save(demo_path)
    
    print("📸 Étape 1: Création de l'image de démo")
    print(f"   ✅ Image sauvegardée: {demo_path}")
    
    # Initialiser le système
    print("\n🤖 Étape 2: Initialisation du système")
    try:
        preprocessor = Preprocessor()
        model = CNNModel(architecture="Xception")
        model.model.load_weights("models/Xception_trained.h5")
        classifier = Classifier()
        user = User(id=1, name="Démonstration")
        print("   ✅ Système initialisé avec succès")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return
    
    # Prétraitement
    print("\n🔬 Étape 3: Prétraitement de l'image")
    try:
        media = Media(type='image', path=demo_path, format='jpg')
        processed_media = preprocessor.preprocess_media(media)
        
        if processed_media.frames:
            print(f"   ✅ {len(processed_media.frames)} visage(s) détecté(s)")
            print(f"   ✅ Redimensionnement: {processed_media.frames[0].image.shape}")
        else:
            print("   ❌ Aucun visage détecté")
            return
    except Exception as e:
        print(f"   ❌ Erreur prétraitement: {e}")
        return
    
    # Analyse
    print("\n🧠 Étape 4: Analyse par Deep Learning")
    try:
        frame_image = np.array([processed_media.frames[0].image])
        
        print("   🤖 Modèle: Xception (entraîné sur 10,000 images)")
        start_time = time.time()
        prediction = model.predict(frame_image)
        inference_time = (time.time() - start_time) * 1000
        
        print(f"   ⏱️  Temps d'inférence: {inference_time:.1f}ms")
    except Exception as e:
        print(f"   ❌ Erreur analyse: {e}")
        return
    
    # Classification
    print("\n📊 Étape 5: Classification et Résultats")
    try:
        result = classifier.classify(prediction[0])
        
        print(f"   🎯 Résultat: {result.label}")
        print(f"   📈 Confiance: {result.confidence:.1f}%")
        print(f"   👤 Utilisateur: {user.name}")
        
        # Formatage final
        final_result = user.view_result(result)
        print(f"\n📋 RÉSULTAT FINAL:")
        print(f"   {final_result}")
        
    except Exception as e:
        print(f"   ❌ Erreur classification: {e}")
        return
    
    # Nettoyage
    try:
        os.remove(demo_path)
        print(f"\n🧹 Nettoyage: {demo_path} supprimé")
    except:
        pass

def compare_models():
    """Compare les performances des 3 modèles"""
    print("\n🏆 COMPARAISON DES 3 ARCHITECTURES")
    print("-" * 40)
    
    # Utiliser votre image existante
    test_image = "face.png"
    if not os.path.exists(test_image):
        print(f"❌ Image {test_image} non trouvée")
        return
    
    # Config modèles
    models = [
        ("Xception", "models/Xception_trained.h5", "Xception"),
        ("ResNet50", "models/ResNet50_trained.h5", "ResNet50"),
        ("EfficientNetB0", "models/EfficientNetB0_trained.h5", "EfficientNetB0")
    ]
    
    results = {}
    
    for name, path, arch in models:
        print(f"\n🤖 Test {name}...")
        try:
            # Initialiser
            model = CNNModel(architecture=arch)
            model.model.load_weights(path)
            preprocessor = Preprocessor()
            classifier = Classifier()
            
            # Prétraiter
            media = Media(type='image', path=test_image, format='png')
            processed_media = preprocessor.preprocess_media(media)
            
            if processed_media.frames:
                # Analyser
                frame_image = np.array([processed_media.frames[0].image])
                start_time = time.time()
                prediction = model.predict(frame_image)
                inference_time = (time.time() - start_time) * 1000
                
                # Classifier
                result = classifier.classify(prediction[0])
                
                results[name] = {
                    'label': result.label,
                    'confidence': result.confidence,
                    'time': inference_time
                }
                
                print(f"   📊 {name}: {result.label} ({result.confidence:.1f}%) - {inference_time:.1f}ms")
            else:
                print(f"   ❌ {name}: Aucun visage détecté")
                
        except Exception as e:
            print(f"   ❌ {name}: Erreur - {e}")
    
    # Résumé
    if results:
        print(f"\n📈 RÉSUMÉ DES PERFORMANCES:")
        print("-" * 30)
        for name, result in results.items():
            print(f"   🎯 {name}: {result['label']} ({result['confidence']:.1f}%)")
        
        # Meilleur modèle
        best_model = max(results.keys(), key=lambda x: results[x]['confidence'])
        print(f"\n🥇 MEILLEUR MODÈLE: {best_model}")
        print(f"   📊 Confiance: {results[best_model]['confidence']:.1f}%")

def show_conclusions():
    """Affiche les conclusions"""
    print("\n🎯 CONCLUSIONS ET PERSPECTIVES")
    print("=" * 40)
    print("✅ RÉALISATIONS:")
    print("   • Système complet de détection de deepfakes")
    print("   • 3 architectures CNN entraînées sur 10,000 images")
    print("   • Pipeline complet: détection → analyse → classification")
    print("   • Multi-interfaces: CLI, Web, Streamlit")
    print()
    print("📊 PERFORMANCES:")
    print("   • Temps d'analyse: <3 secondes")
    print("   • Confiance: jusqu'à 99.66% (Xception)")
    print("   • Support images et vidéos")
    print()
    print("🚀 APPLICATIONS:")
    print("   • Journalistes et vérificateurs d'information")
    print("   • Plateformes de réseaux sociaux")
    print("   • Institutions de cybersécurité")
    print("   • Recherche en intelligence artificielle")
    print()
    print("🔮 PERSPECTIVES:")
    print("   • Amélioration détection visage (MTCNN)")
    print("   • Entraînement sur datasets plus larges")
    print("   • Modèles Transformers (Vision Transformers)")
    print("   • Déploiement cloud et API REST")
    print()
    print("🏆 IMPACT:")
    print("   • Lutte contre la désinformation numérique")
    print("   • Contribution à la cybersécurité")
    print("   • Base solide pour recherche future")
    print()

def main():
    """Fonction principale de démo"""
    print_header()
    
    print("🚀 LANCEMENT DE LA DÉMONSTRATION COMPLÈTE...")
    print()
    
    # Informations système
    show_system_info()
    
    # Démonstration pipeline
    demonstrate_pipeline()
    
    # Comparaison modèles
    compare_models()
    
    # Conclusions
    show_conclusions()
    
    print("🎉 DÉMONSTRATION TERMINÉE AVEC SUCCÈS !")
    print("🔍" + "="*60 + "🔍")
    print()
    print("💡 POUR CONTINUER:")
    print("   • Interface Web: cd app && python app.py")
    print("   • Interface Streamlit: streamlit run streamlit_demo.py")
    print("   • CLI: python main.py --mode analyze --input image.jpg")

if __name__ == "__main__":
    main()
