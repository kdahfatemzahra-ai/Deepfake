#!/usr/bin/env python3
"""
Test complet de tous les modèles entraînés sur plusieurs images
"""

import os
import sys
import time
from PIL import Image, ImageDraw
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import User, Media, Preprocessor, CNNModel, Classifier

def create_test_images():
    """Crée des images de test variées"""
    images = {}
    
    # Image 1: Visage simple
    img1 = Image.new('RGB', (300, 300), color='white')
    draw = ImageDraw.Draw(img1)
    w, h = 300, 300
    draw.ellipse([w*0.2, h*0.15, w*0.8, h*0.85], fill='peachpuff', outline='black', width=2)
    draw.ellipse([w*0.3, h*0.3, w*0.4, h*0.4], fill='white', outline='black', width=2)
    draw.ellipse([w*0.6, h*0.3, w*0.7, h*0.4], fill='white', outline='black', width=2)
    draw.ellipse([w*0.33, h*0.33, w*0.37, h*0.37], fill='black')
    draw.ellipse([w*0.63, h*0.33, w*0.67, h*0.37], fill='black')
    draw.arc([w*0.35, h*0.55, w*0.65, h*0.75], 0, 180, fill='black', width=2)
    images['simple_face'] = img1
    
    # Image 2: Visage stylisé
    img2 = Image.new('RGB', (300, 300), color='lightblue')
    draw = ImageDraw.Draw(img2)
    w, h = 300, 300
    draw.ellipse([w*0.15, h*0.1, w*0.85, h*0.9], fill='lightgreen', outline='darkgreen', width=3)
    draw.ellipse([w*0.25, h*0.25, w*0.45, h*0.45], fill='yellow', outline='orange', width=2)
    draw.ellipse([w*0.55, h*0.25, w*0.75, h*0.45], fill='yellow', outline='orange', width=2)
    draw.ellipse([w*0.3, h*0.3, w*0.4, h*0.4], fill='blue')
    draw.ellipse([w*0.6, h*0.3, w*0.7, h*0.4], fill='blue')
    draw.arc([w*0.3, h*0.6, w*0.7, h*0.8], 0, 180, fill='red', width=3)
    images['stylized_face'] = img2
    
    # Image 3: Visage abstrait
    img3 = Image.new('RGB', (300, 300), color='purple')
    draw = ImageDraw.Draw(img3)
    w, h = 300, 300
    # Face abstraite
    draw.polygon([(w*0.5, h*0.1), (w*0.2, h*0.3), (w*0.2, h*0.7), (w*0.5, h*0.9), (w*0.8, h*0.7), (w*0.8, h*0.3)], fill='pink', outline='black', width=2)
    # Yeux
    draw.ellipse([w*0.3, h*0.35, w*0.4, h*0.45], fill='cyan', outline='black', width=2)
    draw.ellipse([w*0.6, h*0.35, w*0.7, h*0.45], fill='cyan', outline='black', width=2)
    draw.ellipse([w*0.33, h*0.38, w*0.37, h*0.42], fill='black')
    draw.ellipse([w*0.63, h*0.38, w*0.67, h*0.42], fill='black')
    # Bouche
    draw.ellipse([w*0.4, h*0.65, w*0.6, h*0.75], fill='orange', outline='black', width=2)
    images['abstract_face'] = img3
    
    return images

def test_all_models():
    """Test tous les modèles entraînés"""
    print("🧪 TEST COMPLET DES MODÈLES ENTRAÎNÉS")
    print("=" * 50)
    
    # Créer images de test
    test_images = create_test_images()
    
    # Modèles à tester
    models_config = [
        ('Xception', 'models/Xception_trained.h5'),
        ('ResNet50', 'models/ResNet50_trained.h5'),
        ('EfficientNetB0', 'models/EfficientNetB0_trained.h5')
    ]
    
    # Initialiser composants
    preprocessor = Preprocessor()
    classifier = Classifier()
    user = User(id=1, name="Test")
    
    results = {}
    
    for img_name, img in test_images.items():
        print(f"\n📸 Test avec image: {img_name}")
        print("-" * 30)
        
        # Sauvegarder l'image temporairement
        temp_path = f"temp_{img_name}.jpg"
        img.save(temp_path)
        
        # Créer Media et prétraiter
        media = Media(type='image', path=temp_path, format='jpg')
        processed_media = preprocessor.preprocess_media(media)
        
        if not processed_media.frames:
            print(f"❌ Aucun visage détecté dans {img_name}")
            os.remove(temp_path)
            continue
        
        frame_image = np.array([processed_media.frames[0].image])
        results[img_name] = {}
        
        for model_name, model_path in models_config:
            try:
                print(f"\n🤖 Test {model_name}...")
                
                # Charger le modèle
                model = CNNModel(architecture=model_name)
                model.model.load_weights(model_path)
                
                # Prédire
                start_time = time.time()
                prediction = model.predict(frame_image)
                inference_time = (time.time() - start_time) * 1000
                
                # Classifier
                result = classifier.classify(prediction[0])
                
                # Stocker résultats
                results[img_name][model_name] = {
                    'label': result.label,
                    'confidence': result.confidence,
                    'inference_time_ms': round(inference_time, 1)
                }
                
                print(f"   📊 {model_name}: {result.label} ({result.confidence:.1f}%) - {inference_time*1000:.1f}ms")
                
            except Exception as e:
                print(f"   ❌ Erreur {model_name}: {e}")
                results[img_name][model_name] = {'error': str(e)}
        
        # Nettoyer
        os.remove(temp_path)
    
    # Résumé des résultats
    print("\n📋 RÉSUMÉ COMPLET DES PERFORMANCES")
    print("=" * 50)
    
    for img_name, img_results in results.items():
        print(f"\n🎯 {img_name.upper()}:")
        for model_name, result in img_results.items():
            if 'error' in result:
                print(f"   ❌ {model_name}: ERREUR - {result['error']}")
            else:
                print(f"   ✅ {model_name}: {result['label']} ({result['confidence']:.1f}%) - {result['inference_time_ms']}ms")
    
    # Meilleures performances
    print("\n🏆 PERFORMANCES PAR MODÈLE:")
    print("-" * 30)
    
    model_stats = {}
    for model_name, _ in models_config:
        confidences = []
        times = []
        for img_results in results.values():
            if model_name in img_results and 'error' not in img_results[model_name]:
                confidences.append(img_results[model_name]['confidence'])
                times.append(img_results[model_name]['inference_time_ms'])
        
        if confidences:
            avg_conf = np.mean(confidences)
            avg_time = np.mean(times)
            model_stats[model_name] = {'avg_confidence': avg_conf, 'avg_time': avg_time}
            print(f"   📊 {model_name}: Confiance moyenne {avg_conf:.1f}% - Temps moyen {avg_time:.1f}ms")
    
    # Meilleur modèle
    if model_stats:
        best_model = max(model_stats.keys(), key=lambda x: model_stats[x]['avg_confidence'])
        print(f"\n🥇 MEILLEUR MODÈLE: {best_model} ({model_stats[best_model]['avg_confidence']:.1f}% confiance moyenne)")
    
    return results

if __name__ == "__main__":
    test_all_models()
