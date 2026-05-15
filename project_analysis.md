# 🔍 Deepfake Detection System — Complete Project Analysis

> **Authors:** Amine Hajar & Kdah Fatimzahra
> **Version:** 1.0.0
> **Language:** Python 3.8+
> **Root:** `deepfake_detection/`

---

## 📁 Project Structure

```
deepfake_detection/
├── src/                          ← Core source code (Python package)
│   ├── __init__.py               ← Public API exports
│   ├── models.py                 ← Data classes (entities)
│   ├── preprocessor.py           ← Image/video preprocessing
│   ├── dataset_manager.py        ← Dataset loading and splitting
│   ├── deep_learning_models.py   ← Neural network architectures
│   └── classifier.py             ← Prediction classification logic
├── app/
│   ├── app.py/                   ← Flask app (empty/placeholder)
│   └── templates/
│       └── index.html            ← Web UI (empty/placeholder)
├── Model/
│   └── deepfake_model.h5/        ← Saved Keras model (directory)
├── models/
│   └── checkpoints/              ← Model training checkpoints
├── data/
│   ├── real/                     ← Real face images (training)
│   └── fake/                     ← Fake/deepfake images (training)
├── Dataset/                      ← Raw dataset storage
├── uploads/                      ← User-uploaded media files
├── cropped_images/               ← Cropped face outputs
├── logs/                         ← Application & training logs
├── main.py                       ← CLI entry point
├── config.py                     ← All configuration constants
├── requirements.txt              ← Python dependencies
├── test.py                       ← Test file (empty)
└── README.md                     ← Project documentation
```

---

## 🧩 Module: `src/models.py` — Data Classes (Entities)

These are **pure data containers** using Python's `@dataclass` decorator. No database, no ORM — they live in memory.

### 1. `User`
| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Unique user identifier |
| `name` | `str` | User's display name |

**Methods:**
- `upload_media(media_path: str) → Media` — Creates a Media object from a file path
- `view_result(result: Result) → str` — Formats and returns a human-readable result string

---

### 2. `Frame`
| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Frame index number |
| `image` | `np.ndarray` | Normalized pixel array (float32, 0–1 range) |
| `timestamp` | `float` | Time position in video (seconds) |

> Used for both image (single frame) and video (multiple frames) media.

---

### 3. `Media`
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | — | `'image'` or `'video'` |
| `path` | `str` | — | Absolute/relative path to file |
| `format` | `str` | — | File extension (`'jpg'`, `'mp4'`, etc.) |
| `duration` | `Optional[float]` | `None` | Video duration in seconds |
| `frames` | `Optional[List[Frame]]` | `[]` | Extracted frames (populated by Preprocessor) |

> `__post_init__` ensures `frames` is never `None`, always an empty list.

---

### 4. `Result`
| Field | Type | Description |
|-------|------|-------------|
| `label` | `str` | `'FAKE'` or `'REAL'` |
| `confidence` | `float` | Confidence percentage (0–100) |

**Methods:**
- `__str__()` — Returns `"Label: FAKE, Confidence: 97.32%"` format

---

### 5. `Dataset`
| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Dataset name (e.g. `"Deepfake Dataset"`) |
| `size` | `int` | Number of samples |
| `type` | `str` | `'training'`, `'validation'`, or `'test'` |

**Methods (stubs — not implemented):**
- `load_data() → tuple`
- `split_data(train_ratio: float = 0.8) → tuple`

> Acts as a **metadata wrapper** passed to model training methods.

---

## 🧩 Module: `src/deep_learning_models.py` — Neural Network Models

### Abstract Base: `Model` (ABC)
| Attribute | Type | Description |
|-----------|------|-------------|
| `accuracy` | `float` | Stored accuracy after evaluation (default 0.0) |
| `model` | `keras.Model` or `None` | The underlying Keras model |

**Abstract Methods (must be implemented by subclasses):**
- `train(dataset: Dataset) → None`
- `predict(data: np.ndarray) → np.ndarray`
- `evaluate() → float`

---

### Concrete Class: `CNNModel(Model)`
**Type:** Convolutional Neural Network with Transfer Learning

| Attribute | Type | Description |
|-----------|------|-------------|
| `architecture` | `str` | `'Xception'`, `'ResNet50'`, or `'EfficientNetB0'` |
| `input_shape` | `Tuple[int,int,int]` | Image input dimensions |
| `model` | `keras.Model` | Built Keras functional model |

#### Supported Backbone Architectures

| Architecture | Input Size | Pre-trained On | Characteristics |
|-------------|-----------|----------------|-----------------|
| **Xception** | 299×299×3 | ImageNet | Highest accuracy, slower inference |
| **ResNet50** | 224×224×3 | ImageNet | Good balance speed/accuracy |
| **EfficientNetB0** | 224×224×3 | ImageNet | Fastest, most lightweight |

#### Model Architecture (custom head added on top of backbone):
```
Input (224×224×3 or 299×299×3)
    ↓
[Frozen Backbone: Xception / ResNet50 / EfficientNetB0]
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.2)
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.1)
    ↓
Dense(1, activation='sigmoid')   ← Binary output: 0=REAL, 1=FAKE
```

#### Compilation Settings:
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Cross-Entropy |
| Metrics | Accuracy, Precision, Recall |

#### Methods:
| Method | Signature | Description |
|--------|-----------|-------------|
| `_build_model` | `() → keras.Model` | Constructs the Keras functional model |
| `train` | `(dataset, X_train, y_train, X_val, y_val, epochs, batch_size) → History` | Trains with augmentation + callbacks |
| `predict` | `(data: np.ndarray) → np.ndarray` | Runs inference, returns probabilities |
| `evaluate` | `(X_test, y_test) → Tuple[float, float, float]` | Returns (accuracy, precision, recall) |
| `extract_features` | `(data) → np.ndarray` | Extracts features from second-to-last layer |
| `fine_tune` | `(X_train, y_train, X_val, y_val, epochs) → History` | Unfreezes top 20 layers, retrains at lr=1e-5 |

#### Training Callbacks:
- **EarlyStopping** — patience=5, restores best weights
- **ReduceLROnPlateau** — factor=0.1, patience=3
- **ModelCheckpoint** — saves best model to `models/{arch}_best.h5`

#### Data Augmentation (applied during training):
- Random horizontal flip
- Random rotation (±10%)
- Random zoom (±10%)

---

### Concrete Class: `EnsembleModel(Model)`
**Type:** Model averaging ensemble

| Attribute | Type | Description |
|-----------|------|-------------|
| `models` | `list` | List of individual `Model` instances |

**Methods:**
| Method | Description |
|--------|-------------|
| `predict(data) → np.ndarray` | Averages predictions across all member models |
| `train(dataset) → None` | Stub (not yet implemented) |
| `evaluate() → float` | Stub (not yet implemented) |

> The ensemble combines multiple `CNNModel` instances by averaging their sigmoid outputs — a simple but effective strategy.

---

## 🧩 Module: `src/classifier.py` — Decision Layer

### Class: `Classifier`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.5` | Decision boundary (probability ≥ threshold → FAKE) |

**Methods:**
| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `classify` | `np.ndarray` (single prediction) | `Result` | Converts probability → REAL/FAKE label with confidence |
| `batch_classify` | `np.ndarray` (batch) | `List[Result]` | Classifies multiple predictions |
| `set_threshold` | `float` | `None` | Updates decision threshold (validates 0–1 range) |
| `get_confidence_score` | `predictions, true_labels` | `dict` | Full evaluation metrics dictionary |
| `get_optimal_threshold` | `predictions, true_labels` | `float` | Finds best threshold via Youden's J statistic |

#### Classification Logic:
```python
if probability >= threshold:  # default 0.5
    label = "FAKE"
    confidence = probability * 100
else:
    label = "REAL"
    confidence = (1 - probability) * 100
```

#### Metrics returned by `get_confidence_score()`:
| Metric | Description |
|--------|-------------|
| `accuracy` | Overall accuracy |
| `auc` | Area Under ROC Curve |
| `precision_real` / `precision_fake` | Per-class precision |
| `recall_real` / `recall_fake` | Per-class recall |
| `f1_real` / `f1_fake` | Per-class F1 score |
| `confusion_matrix` | 2×2 matrix as list |

---

### Class: `VideoClassifier`
**Purpose:** Aggregates per-frame predictions into a single video-level decision

| Attribute | Type | Description |
|-----------|------|-------------|
| `frame_classifier` | `Classifier` | Used to classify individual frames |
| `strategy` | `str` | Aggregation strategy |

**Three Aggregation Strategies:**

| Strategy | Method | Description |
|----------|--------|-------------|
| `'majority'` | `_majority_voting()` | Each frame votes REAL/FAKE → winner wins |
| `'average'` | `_average_confidence()` | Average all frame probabilities → single classify |
| `'weighted'` | `_weighted_average()` | Center frames weighted more heavily |

---

## 🧩 Module: `src/preprocessor.py` — Media Preprocessing

### Class: `Preprocessor`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_size` | `Tuple[int,int]` | `(224, 224)` | Output image dimensions |

**Methods:**
| Method | Description |
|--------|-------------|
| `resize(image)` | Resizes image using OpenCV to `target_size` |
| `normalize(image)` | Divides pixels by 255.0 → float32 in [0,1] |
| `extract_frames(media, max_frames=30)` | Extracts frames from image or video |
| `detect_face(image)` | Detects first face using Haar Cascades, crops and resizes it |
| `preprocess_media(media)` | Full pipeline: extract → detect face → normalize |

#### Preprocessing Pipeline:
```
Input Media (image or video file)
        ↓
extract_frames()   ← reads file with OpenCV, samples up to 30 frames
        ↓
detect_face()      ← Haar Cascade frontal face detector
        ↓
crop + resize      ← to (224, 224)
        ↓
normalize          ← pixel values ÷ 255.0
        ↓
List[Frame]        ← attached to Media.frames
```

> If **no face is detected** in a frame, that frame is **discarded**. If no frames remain, analysis returns an error.

---

## 🧩 Module: `src/dataset_manager.py` — Data Management

### Class: `DatasetManager`

| Attribute | Type | Description |
|-----------|------|-------------|
| `dataset_path` | `str` | Root path to dataset folder |
| `data` | `list` | Raw data accumulator |
| `labels` | `list` | Label accumulator |

**Methods:**
| Method | Description |
|--------|-------------|
| `load_faceforensics_dataset(real_path, fake_path)` | Loads image paths from `real/` and `fake/` folders. Labels: 0=real, 1=fake |
| `load_celebdf_dataset(metadata_path, videos_path)` | Loads Celeb-DF from CSV metadata file |
| `create_dataset(name, data, labels, type)` | Returns a `Dataset` metadata object |
| `split_data(data, labels, train=0.8, val=0.1)` | Stratified 80/10/10 train/val/test split |
| `get_data_statistics(labels)` | Returns count and percentage of real vs fake samples |

#### Supported Datasets:
| Dataset | Loading Method | Format |
|---------|---------------|--------|
| **FaceForensics++** | `load_faceforensics_dataset` | Folder-based (`real/`, `fake/`) |
| **Celeb-DF** | `load_celebdf_dataset` | CSV metadata + video files |

---

## 🧩 `main.py` — Application Entry Point & Orchestrator

### Class: `DeepfakeDetectionSystem`
The top-level orchestrator that wires all modules together.

| Attribute | Type | Description |
|-----------|------|-------------|
| `preprocessor` | `Preprocessor` | Image/video processing |
| `classifier` | `Classifier` | Single-frame decision logic |
| `video_classifier` | `VideoClassifier` | Multi-frame video decision |
| `model` | `CNNModel` | Active neural network |

**Methods:**
| Method | Description |
|--------|-------------|
| `_load_model(path, arch)` | Loads weights from `.h5` file into a fresh `CNNModel` |
| `analyze_media(path, user)` | Full analysis pipeline for one file, returns formatted string |
| `train_model(dataset_path, epochs)` | Full training pipeline including preprocessing, splitting, training, evaluation, saving |

### CLI Usage:
```bash
# Analyze a single image or video
python main.py --mode analyze --input path/to/file.jpg --user-name "Alice"

# Train with a specific architecture
python main.py --mode train --input data/ --epochs 20 --architecture Xception

# Use a pre-trained model
python main.py --mode analyze --input video.mp4 --model models/Xception_trained.h5
```

---

## ⚙️ `config.py` — Configuration Reference

### Model Configs (`MODEL_CONFIGS`)
| Architecture | Input Shape | LR | Batch | Epochs | Dropout |
|-------------|------------|-----|-------|--------|---------|
| Xception | 299×299×3 | 0.001 | 32 | 10 | 0.2 |
| ResNet50 | 224×224×3 | 0.001 | 32 | 10 | 0.2 |
| EfficientNetB0 | 224×224×3 | 0.001 | 32 | 10 | 0.2 |

### Preprocessing Config
| Parameter | Value |
|-----------|-------|
| Target Size | 224×224 |
| Max Frames (video) | 30 |
| Face Detection Confidence | 0.9 |
| Normalization Range | [0, 1] |

### Classification Config
| Parameter | Value |
|-----------|-------|
| Decision Threshold | 0.5 |
| Video Strategy | `'majority'` |
| Confidence Threshold | 0.7 |

### Dataset Config
| Parameter | Value |
|-----------|-------|
| Train / Val / Test Split | 80% / 10% / 10% |
| Supported Image Formats | `.jpg`, `.jpeg`, `.png` |
| Supported Video Formats | `.mp4`, `.avi`, `.mov`, `.mkv` |
| Max Image Size | 10 MB |
| Max Video Size | 100 MB |

### Training Config
| Parameter | Value |
|-----------|-------|
| Early Stopping Patience | 5 epochs |
| ReduceLR Patience | 3 epochs |
| Checkpoint Dir | `models/checkpoints` |
| Fine-tune Epochs | 5 |
| Fine-tune LR | 1e-5 |

### App Config (Flask)
| Parameter | Value |
|-----------|-------|
| Host | `localhost` |
| Port | `5000` |
| Upload Folder | `uploads/` |
| Max Upload Size | 16 MB |
| Debug | `False` |

---

## 🔄 Full Data Flow Diagram

```
User (CLI or future Web UI)
        │
        ▼
DeepfakeDetectionSystem.analyze_media()
        │
        ▼
Media(type, path, format)
        │
        ▼
Preprocessor.preprocess_media()
        ├── extract_frames()       ← OpenCV VideoCapture / imread
        └── detect_face()          ← Haar Cascade → crop → normalize
        │
        ▼
List[Frame] (max 30 frames, face-only, 224×224, float32 [0–1])
        │
        ▼
CNNModel.predict(frame_images)
        └── Backbone (Xception/ResNet50/EfficientNetB0) → sigmoid → probability
        │
        ▼
Classifier.classify()    (image)
  OR
VideoClassifier.classify_video()   (video: majority / average / weighted)
        │
        ▼
Result(label='FAKE'|'REAL', confidence=float)
        │
        ▼
User.view_result(result) → "Résultat pour Alice: FAKE (Confiance: 97.32%)"
```

---

## 📦 Dependencies (`requirements.txt`)

| Library | Version | Role |
|---------|---------|------|
| `tensorflow` | 2.13.0 | Deep learning framework (Keras included) |
| `opencv-python` | 4.8.1.78 | Image/video reading, face detection |
| `numpy` | 1.24.3 | Array operations |
| `pandas` | 2.0.3 | Dataset metadata (CSV for Celeb-DF) |
| `scikit-learn` | 1.3.0 | Train/test split, metrics, ROC |
| `matplotlib` | 3.7.2 | Plotting training curves |
| `seaborn` | 0.12.2 | Visualization |
| `Pillow` | 9.5.0 | Image I/O |
| `tqdm` | 4.66.1 | Progress bars |
| `albumentations` | 1.3.1 | Advanced image augmentation |
| `flask` | 2.3.3 | Web framework (planned) |
| `streamlit` | 1.25.0 | Interactive UI (planned) |

---

## 🚦 What's Implemented vs. Planned

| Feature | Status |
|---------|--------|
| Face detection (Haar Cascade) | ✅ Done |
| Video frame extraction | ✅ Done |
| Image normalization & resizing | ✅ Done |
| CNN training (3 architectures) | ✅ Done |
| Binary classification (REAL/FAKE) | ✅ Done |
| Performance evaluation metrics | ✅ Done |
| Multi-architecture support | ✅ Done |
| Transfer learning + fine-tuning | ✅ Done |
| Ensemble model | ✅ Implemented (partial) |
| FaceForensics++ dataset support | ✅ Done |
| Celeb-DF dataset support | ✅ Done |
| Flask web interface | ✅ Logic Done (Real Model Integration) |
| Streamlit UI | 🔄 Planned |
| Batch processing | 🔄 Planned |
| REST API | 🔄 Planned |
| Results export | 🔄 Planned |
