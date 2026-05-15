from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from PIL import Image

@dataclass
class User:
    id: int
    name: str
    
    def upload_media(self, media_path: str) -> 'Media':
        return Media(media_path)
    
    def view_result(self, result: 'Result') -> str:
        return f"Résultat pour {self.name}: {result.label} (Confiance: {result.confidence:.2f}%)"

@dataclass
class Frame:
    id: int
    image: np.ndarray
    timestamp: float

@dataclass
class Media:
    type: str  # 'image' or 'video'
    path: str
    format: str
    duration: Optional[float] = None
    frames: Optional[List[Frame]] = None
    
    def __post_init__(self):
        if self.frames is None:
            self.frames = []

@dataclass
class Result:
    label: str  # 'FAKE' or 'REAL'
    confidence: float
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __str__(self):
        return f"Label: {self.label}, Confidence: {self.confidence:.2f}%"

@dataclass
class Dataset:
    name: str
    size: int
    type: str  # 'training', 'validation', 'test'
    
    def load_data(self) -> tuple:
        pass
    
    def split_data(self, train_ratio: float = 0.8) -> tuple:
        pass
