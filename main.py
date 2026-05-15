#!/usr/bin/env python3
"""
Système de Détection des Deepfakes par Intelligence Artificielle
Main Application Entry Point
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import (
    User, Media, Preprocessor, DatasetManager, 
    CNNModel, Classifier, VideoClassifier, Dataset
)

class DeepfakeDetectionSystem:
    def __init__(self, model_path: str = None, architecture: str = "Xception"):
        self.preprocessor = Preprocessor()
        self.classifier = Classifier()
        self.video_classifier = VideoClassifier(self.classifier)
        
        # Initialize model
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path, architecture)
        else:
            self.model = CNNModel(architecture=architecture)
            print(f"Model initialized with {architecture} architecture")
    
    def _load_model(self, model_path: str, architecture: str):
        """Load a pre-trained model"""
        model = CNNModel(architecture=architecture)
        model.model.load_weights(model_path)
        print(f"Model loaded from {model_path}")
        return model
    
    def analyze_media(self, media_path: str, user: User) -> str:
        """Analyze a single media file (image or video)"""
        try:
            # Determine media type
            if media_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                media_type = 'image'
            elif media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                media_type = 'video'
            else:
                return "Error: Unsupported file format"
            
            # Create Media object
            media = Media(
                type=media_type,
                path=media_path,
                format=media_path.split('.')[-1]
            )
            
            # Preprocess media
            processed_media = self.preprocessor.preprocess_media(media)
            
            if not processed_media.frames:
                return "Error: No faces detected in the media"
            
            # Prepare data for prediction
            frame_images = np.array([frame.image for frame in processed_media.frames])
            
            # Make predictions
            predictions = self.model.predict(frame_images)
            
            # Classify
            if media_type == 'image':
                result = self.classifier.classify(predictions[0])
            else:
                result = self.video_classifier.classify_video(predictions)
            
            # Return formatted result
            return user.view_result(result)
            
        except Exception as e:
            return f"Error during analysis: {str(e)}"
    
    def train_model(self, dataset_path: str, epochs: int = 10):
        """Train the model on a dataset"""
        try:
            # Initialize dataset manager
            dataset_manager = DatasetManager(dataset_path)
            
            # Load dataset (example paths - adjust according to your dataset structure)
            real_path = os.path.join(dataset_path, 'real')
            fake_path = os.path.join(dataset_path, 'fake')
            
            if not os.path.exists(real_path) or not os.path.exists(fake_path):
                print("Dataset structure not found. Expected 'real' and 'fake' subdirectories.")
                return
            
            # Load data paths and labels just for statistics
            image_paths, labels = dataset_manager.load_faceforensics_dataset(real_path, fake_path)
            
            # Print dataset statistics
            stats = dataset_manager.get_data_statistics(labels)
            print("Dataset Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # Train model using data generators (avoids loading everything into RAM)
            print(f"Training model on {len(image_paths)} images using data generators...")
            history = self.model.train_from_directory(dataset_path, epochs=epochs)
            
            print("Training complete! Model saved.")
            
            # Save model
            model_save_path = f"models/{self.model.architecture}_trained.h5"
            os.makedirs("models", exist_ok=True)
            self.model.model.save(model_save_path)
            print(f"Model saved to {model_save_path}")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection System")
    parser.add_argument('--mode', choices=['analyze', 'train'], required=True,
                       help='Mode: analyze or train')
    parser.add_argument('--input', required=True,
                       help='Input file path (for analyze) or dataset path (for train)')
    parser.add_argument('--model', default=None,
                       help='Path to pre-trained model')
    parser.add_argument('--architecture', default='Xception',
                       choices=['Xception', 'ResNet50', 'EfficientNetB0'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--user-name', default='User',
                       help='User name for results')
    
    args = parser.parse_args()
    
    # Initialize system
    system = DeepfakeDetectionSystem(model_path=args.model, architecture=args.architecture)
    
    # Create user
    user = User(id=1, name=args.user_name)
    
    if args.mode == 'analyze':
        if not os.path.exists(args.input):
            print(f"Error: File {args.input} not found")
            return
        
        result = system.analyze_media(args.input, user)
        print(result)
    
    elif args.mode == 'train':
        if not os.path.exists(args.input):
            print(f"Error: Dataset path {args.input} not found")
            return
        
        system.train_model(args.input, epochs=args.epochs)

if __name__ == "__main__":
    main()
