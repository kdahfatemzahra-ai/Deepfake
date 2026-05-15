import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from .models import Dataset

class DatasetManager:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data = []
        self.labels = []
    
    def load_faceforensics_dataset(self, real_path: str, fake_path: str) -> Tuple[np.ndarray, np.ndarray]:
        real_images = []
        fake_images = []
        
        # Load real images
        for filename in os.listdir(real_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(real_path, filename)
                real_images.append(img_path)
        
        # Load fake images
        for filename in os.listdir(fake_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(fake_path, filename)
                fake_images.append(img_path)
        
        # Create labels (0 for real, 1 for fake)
        image_paths = real_images + fake_images
        labels = [0] * len(real_images) + [1] * len(fake_images)
        
        return np.array(image_paths), np.array(labels)
    
    def load_celebdf_dataset(self, metadata_path: str, videos_path: str) -> Tuple[np.ndarray, np.ndarray]:
        # Load Celeb-DF metadata
        metadata = pd.read_csv(metadata_path)
        
        image_paths = []
        labels = []
        
        for _, row in metadata.iterrows():
            video_path = os.path.join(videos_path, row['video_name'])
            if os.path.exists(video_path):
                image_paths.append(video_path)
                labels.append(row['label'])  # 0 for real, 1 for fake
        
        return np.array(image_paths), np.array(labels)
    
    def create_dataset(self, name: str, data: np.ndarray, labels: np.ndarray, dataset_type: str) -> Dataset:
        return Dataset(
            name=name,
            size=len(data),
            type=dataset_type
        )
    
    def split_data(self, data: np.ndarray, labels: np.ndarray, 
                   train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple:
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            data, labels, test_size=1-train_ratio-val_ratio, random_state=42, stratify=labels
        )
        
        # Second split: separate train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio/(train_ratio+val_ratio), 
            random_state=42, stratify=y_temp
        )
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    
    def get_data_statistics(self, labels: np.ndarray) -> dict:
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        stats = {
            'total_samples': total,
            'real_samples': counts[0] if 0 in unique else 0,
            'fake_samples': counts[1] if 1 in unique else 0,
            'real_percentage': (counts[0] / total * 100) if 0 in unique else 0,
            'fake_percentage': (counts[1] / total * 100) if 1 in unique else 0
        }
        
        return stats
