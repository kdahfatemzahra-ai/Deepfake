import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from .models import Dataset

class Model(ABC):
    def __init__(self):
        self.accuracy = 0.0
        self.model = None
    
    @abstractmethod
    def train(self, dataset: Dataset) -> None:
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def evaluate(self) -> float:
        pass

class CNNModel(Model):
    def __init__(self, architecture: str = "Xception", input_shape: Tuple[int, int, int] = (224, 224, 3)):
        super().__init__()
        self.architecture = architecture
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        if self.architecture == "Xception":
            base_model = keras.applications.Xception(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.architecture == "ResNet50":
            base_model = keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.architecture == "EfficientNetB0":
            base_model = keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        
        # Freeze the base model
        base_model.trainable = False
        
        # Add custom layers
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def train(self, dataset: Dataset, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, epochs: int = 10, 
              batch_size: int = 32) -> keras.callbacks.History:
        
        # Data augmentation
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        # Create training pipeline
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3),
            keras.callbacks.ModelCheckpoint(
                f'models/{self.architecture}_best.h5',
                save_best_only=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def train_from_directory(self, dataset_path: str, epochs: int = 10, 
                             batch_size: int = 32) -> keras.callbacks.History:
        
        # Data augmentation and rescaling
        data_augmentation = keras.Sequential([
            layers.Rescaling(1./255),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        # Create training pipeline from directory
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=self.input_shape[:2],
            batch_size=batch_size,
            label_mode='binary'
        )
        
        val_dataset = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=self.input_shape[:2],
            batch_size=batch_size,
            label_mode='binary'
        )
        
        # Apply data augmentation
        train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Apply only rescaling to validation
        rescale = keras.Sequential([layers.Rescaling(1./255)])
        val_dataset = val_dataset.map(lambda x, y: (rescale(x, training=False), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3),
            keras.callbacks.ModelCheckpoint(
                f'models/{self.architecture}_best.h5',
                save_best_only=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        predictions = self.model.predict(data)
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float]:
        results = self.model.evaluate(X_test, y_test, verbose=0)
        self.accuracy = results[1]  # accuracy is at index 1
        return self.accuracy, results[2], results[3]  # accuracy, precision, recall
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        # Extract features from the second to last layer
        feature_extractor = keras.Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[-2].output
        )
        features = feature_extractor.predict(data)
        return features
    
    def fine_tune(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray, epochs: int = 5):
        # Unfreeze the top layers of the model
        for layer in self.model.layers[-20:]:
            layer.trainable = True
        
        # Recompile with a lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
        )
        
        # Continue training
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16
        )
        
        return history

    def generate_gradcam(self, img_array: np.ndarray) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap for the given image
        """
        try:
            # Find the last convolutional layer
            last_conv_layer_name = None
            target_model = None
            for layer in reversed(self.model.layers):
                if isinstance(layer, keras.Model): # This is our base_model
                    for sub_layer in reversed(layer.layers):
                        if 'conv' in sub_layer.name.lower() or 'act' in sub_layer.name.lower():
                            last_conv_layer_name = sub_layer.name
                            target_model = layer
                            break
                    if last_conv_layer_name: break
            
            if not last_conv_layer_name:
                last_conv_layer_name = self.model.layers[-4].name 
                target_model = self.model

            grad_model = keras.Model(
                inputs=[target_model.inputs],
                outputs=[target_model.get_layer(last_conv_layer_name).output, target_model.output]
            )

            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = grad_model(img_array)
                class_channel = preds[:, 0]

            grads = tape.gradient(class_channel, last_conv_layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
            return heatmap.numpy()
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            return np.zeros((7, 7)) # Return empty heatmap on failure

class EnsembleModel(Model):
    def __init__(self, models: list):
        super().__init__()
        self.models = models
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        predictions = []
        for model in self.models:
            pred = model.predict(data)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def train(self, dataset: Dataset) -> None:
        # Train each model individually
        pass
    
    def evaluate(self) -> float:
        # Evaluate ensemble performance
        pass
