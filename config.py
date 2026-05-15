"""
Configuration file for Deepfake Detection System
"""

# Model configurations
MODEL_CONFIGS = {
    'Xception': {
        'input_shape': (299, 299, 3),
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'dropout_rate': 0.2
    },
    'ResNet50': {
        'input_shape': (224, 224, 3),
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'dropout_rate': 0.2
    },
    'EfficientNetB0': {
        'input_shape': (224, 224, 3),
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'dropout_rate': 0.2
    }
}

# Preprocessing configurations
PREPROCESSING_CONFIG = {
    'target_size': (224, 224),
    'max_frames': 30,
    'face_detection_confidence': 0.9,
    'normalization_range': (0, 1)
}

# Classification configurations
CLASSIFICATION_CONFIG = {
    'threshold': 0.5,
    'video_strategy': 'majority',  # 'majority', 'average', 'weighted'
    'confidence_threshold': 0.7
}

# Dataset configurations
DATASET_CONFIG = {
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'supported_formats': ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.mkv'],
    'max_image_size': 10 * 1024 * 1024,  # 10MB
    'max_video_size': 100 * 1024 * 1024   # 100MB
}

# Training configurations
TRAINING_CONFIG = {
    'early_stopping_patience': 5,
    'reduce_lr_patience': 3,
    'checkpoint_dir': 'models/checkpoints',
    'logs_dir': 'logs',
    'fine_tune_epochs': 5,
    'fine_tune_learning_rate': 1e-5
}

# Application configurations
APP_CONFIG = {
    'debug': False,
    'host': 'localhost',
    'port': 5000,
    'upload_folder': 'uploads',
    'max_content_length': 16 * 1024 * 1024,  # 16MB
    'allowed_extensions': {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/deepfake_detection.log'
}
