import numpy as np
from typing import Tuple
from .models import Result
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

class Classifier:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def classify(self, features: np.ndarray) -> Result:
        """
        Classify based on features and return a Result object
        """
        # For binary classification, features should be a single probability value
        if len(features.shape) == 1:
            probability = float(features[0]) if len(features) > 1 else float(features)
        else:
            probability = float(features[0][0]) if len(features[0]) > 0 else 0.5
        
        # Convert probability to percentage
        confidence = probability * 100
        
        # Determine label based on threshold
        if probability >= self.threshold:
            label = "FAKE"
            confidence = confidence
        else:
            label = "REAL"
            confidence = 100 - confidence
        
        return Result(label=label, confidence=confidence)
    
    def get_confidence_score(self, predictions: np.ndarray, true_labels: np.ndarray) -> dict:
        """
        Calculate confidence scores and metrics for model evaluation
        """
        # Convert predictions to binary labels
        predicted_labels = (predictions >= self.threshold).astype(int)
        
        # Calculate metrics
        accuracy = np.mean(predicted_labels == true_labels)
        
        # Calculate ROC AUC
        try:
            auc = roc_auc_score(true_labels, predictions)
        except:
            auc = 0.5
        
        # Generate classification report
        report = classification_report(
            true_labels, 
            predicted_labels, 
            target_names=['REAL', 'FAKE'],
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'precision_real': report['REAL']['precision'],
            'precision_fake': report['FAKE']['precision'],
            'recall_real': report['REAL']['recall'],
            'recall_fake': report['FAKE']['recall'],
            'f1_real': report['REAL']['f1-score'],
            'f1_fake': report['FAKE']['f1-score'],
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def batch_classify(self, batch_predictions: np.ndarray) -> list:
        """
        Classify a batch of predictions
        """
        results = []
        for prediction in batch_predictions:
            result = self.classify(prediction)
            results.append(result)
        return results
    
    def set_threshold(self, threshold: float):
        """
        Update the classification threshold
        """
        if 0 <= threshold <= 1:
            self.threshold = threshold
        else:
            raise ValueError("Threshold must be between 0 and 1")
    
    def get_optimal_threshold(self, predictions: np.ndarray, true_labels: np.ndarray) -> float:
        """
        Find optimal threshold using Youden's J statistic
        """
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold

class VideoClassifier:
    def __init__(self, frame_classifier: Classifier, strategy: str = 'majority'):
        self.frame_classifier = frame_classifier
        self.strategy = strategy  # 'majority', 'average', 'weighted'
    
    def classify_video(self, frame_predictions: np.ndarray) -> Result:
        """
        Classify video based on frame predictions
        """
        if self.strategy == 'majority':
            return self._majority_voting(frame_predictions)
        elif self.strategy == 'average':
            return self._average_confidence(frame_predictions)
        elif self.strategy == 'weighted':
            return self._weighted_average(frame_predictions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _majority_voting(self, frame_predictions: np.ndarray) -> Result:
        """
        Classify based on majority vote of frames
        """
        frame_results = self.frame_classifier.batch_classify(frame_predictions)
        
        fake_votes = sum(1 for result in frame_results if result.label == "FAKE")
        real_votes = len(frame_results) - fake_votes
        
        if fake_votes > real_votes:
            label = "FAKE"
            confidence = (fake_votes / len(frame_results)) * 100
        else:
            label = "REAL"
            confidence = (real_votes / len(frame_results)) * 100
        
        return Result(label=label, confidence=confidence)
    
    def _average_confidence(self, frame_predictions: np.ndarray) -> Result:
        """
        Classify based on average confidence across frames
        """
        avg_prediction = np.mean(frame_predictions)
        return self.frame_classifier.classify(avg_prediction)
    
    def _weighted_average(self, frame_predictions: np.ndarray, weights: np.ndarray = None) -> Result:
        """
        Classify based on weighted average of frame predictions
        """
        if weights is None:
            # Use linear weights giving more importance to middle frames
            n_frames = len(frame_predictions)
            weights = np.array([min(i, n_frames-i) for i in range(n_frames)])
            weights = weights / np.sum(weights)
        
        weighted_prediction = np.average(frame_predictions, weights=weights)
        return self.frame_classifier.classify(weighted_prediction)

class FusionClassifier:
    def __init__(self, weights: dict = None):
        """
        Fusion strategy to combine outputs of multiple deepfake models
        """
        # Default weights for Xception, EfficientNetB0, ResNet50
        self.weights = weights or {
            'Xception': 0.5,
            'EfficientNetB0': 0.3,
            'ResNet50': 0.2
        }
    
    def fuse(self, model_predictions: dict) -> Result:
        """
        Combines predictions from multiple models using weighted averaging
        model_predictions: dict mapping architecture name to its probability output
        """
        weighted_sum = 0
        total_weight = 0
        
        for arch, prob in model_predictions.items():
            if arch in self.weights:
                weighted_sum += prob * self.weights[arch]
                total_weight += self.weights[arch]
        
        if total_weight == 0:
            # Fallback to simple average if no recognized architectures found
            fused_prob = np.mean(list(model_predictions.values()))
        else:
            fused_prob = weighted_sum / total_weight
            
        # Standard classification threshold
        is_fake = fused_prob > 0.5
        label = "FAKE" if is_fake else "REAL"
        
        # Calculate confidence based on probability
        # (if 0.9 fake -> 90% confidence, if 0.1 fake -> 90% confidence for real)
        confidence = (fused_prob if is_fake else 1 - fused_prob) * 100
        
        return Result(
            label=label, 
            confidence=round(float(confidence), 2),
            # Store the raw probability for potential heatmap weighting later
            metadata={'fake_probability': round(float(fused_prob), 4)}
        )
