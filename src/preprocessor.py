import cv2
import numpy as np
from typing import List, Tuple, Optional
from .models import Media, Frame

class Preprocessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.target_size)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0
    
    def extract_frames(self, media: Media, max_frames: int = 30) -> List[Frame]:
        frames = []
        
        if media.type == 'image':
            image = cv2.imread(media.path)
            if image is not None:
                resized_image = self.resize(image)
                normalized_image = self.normalize(resized_image)
                frame = Frame(id=0, image=normalized_image, timestamp=0.0)
                frames.append(frame)
        
        elif media.type == 'video':
            cap = cv2.VideoCapture(media.path)
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval to get max_frames evenly distributed
            if total_frames > max_frames:
                interval = total_frames // max_frames
            else:
                interval = 1
            
            while cap.isOpened() and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    resized_frame = self.resize(frame)
                    normalized_frame = self.normalize(resized_frame)
                    timestamp = frame_count / fps
                    frame_obj = Frame(
                        id=len(frames), 
                        image=normalized_frame, 
                        timestamp=timestamp
                    )
                    frames.append(frame_obj)
                    
                    if len(frames) >= max_frames:
                        break
                
                frame_count += 1
            
            cap.close()
        
        return frames
    
    def detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        # Use OpenCV's built-in face detection (Haar Cascades)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the first face found
            x, y, w, h = faces[0]
            face_image = image[y:y+h, x:x+w]
            # Resize face to target size
            face_resized = cv2.resize(face_image, self.target_size)
            return face_resized
        
        return None
    
    def preprocess_media(self, media: Media) -> Media:
        # Extract frames
        frames = self.extract_frames(media)
        
        # Detect faces in each frame
        processed_frames = []
        for frame in frames:
            # Convert back to BGR for face detection
            bgr_image = (frame.image * 255).astype(np.uint8)
            face = self.detect_face(bgr_image)
            
            if face is not None:
                # Convert face back to normalized format
                face_normalized = face.astype(np.float32) / 255.0
                processed_frame = Frame(
                    id=frame.id,
                    image=face_normalized,
                    timestamp=frame.timestamp
                )
                processed_frames.append(processed_frame)
        
        # Update media with processed frames
        media.frames = processed_frames
        return media
