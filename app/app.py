#!/usr/bin/env python3
"""
Deepfake Detection — Flask Backend
"""

import os, sys, uuid, time, sqlite3
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import traceback
import cv2

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# TensorFlow and the neural-network layers are optional.
# If not installed, the system runs on forensics-only mode.
try:
    import tensorflow as tf
    from tensorflow.keras.applications import xception, resnet50, efficientnet
    from src import User, Media, Preprocessor, CNNModel, Classifier, FusionClassifier
    _HAS_TF = True
except ImportError:
    _HAS_TF = False
    # Stub FusionClassifier so the rest of the file compiles without TF
    class FusionClassifier:
        def fuse(self, preds):
            class _R:
                label = "REAL"
                confidence = 50.0
                metadata = {'fake_probability': 0.5}
            return _R()

app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'jfif', 'mp4', 'avi', 'mov', 'mkv', 'webm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'heatmaps')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.secret_key = 'deepshield_secret_key_change_me'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)

# ── Database Setup ──────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        # User table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Analyses table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                media_type TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                fake_probability REAL NOT NULL,
                heatmap_path TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()

init_db()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_media_type(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    return 'image' if ext in {'png', 'jpg', 'jpeg', 'gif', 'webp', 'jfif'} else 'video'


import numpy as np
import tempfile

# Load Face Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Per-architecture input sizes (standard sizes used during ImageNet pre-training)
ARCH_INPUT_SIZES = {
    'Xception':       (299, 299, 3),
    'ResNet50':       (224, 224, 3),
    'EfficientNetB0': (224, 224, 3),
}

def build_models():
    """Builds and loads the deepfake detection models. Returns (models_dict, weights_loaded_flag)."""
    if not _HAS_TF:
        print("[!] TensorFlow not installed — running in forensics-only mode.")
        return None, False

    models = {}
    weights_loaded = False
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    for arch, shape in ARCH_INPUT_SIZES.items():
        try:
            print(f"[*] Initializing {arch} model ({shape[:2]})...")
            cnn_m = CNNModel(architecture=arch, input_shape=shape)

            weight_paths = [
                os.path.join(base_dir, 'models', f'{arch}_trained.h5'),
                os.path.join(base_dir, 'models', f'{arch}_best.h5'),
                os.path.join(base_dir, 'Model', 'deepfake_model.h5'),
            ]

            for path in weight_paths:
                if os.path.exists(path):
                    try:
                        if os.path.isdir(path):
                            cnn_m.model = tf.keras.models.load_model(path)
                        else:
                            cnn_m.model.load_weights(path)
                        weights_loaded = True
                        print(f"    [+] Loaded weights for {arch} from {path}")
                        break
                    except Exception as le:
                        print(f"    [!] Failed to load {path}: {le}")
            else:
                print(f"    [!] No weights for {arch} — using untrained ImageNet backbone.")

            models[arch] = cnn_m
        except Exception as e:
            print(f"    [!] Could not initialise {arch}: {e}")

    return (models if models else None), weights_loaded

real_models, _weights_loaded = build_models()


def preprocess_for_model(img_path, architecture='Xception'):
    """Return a preprocessed image array using the correct normalization for each architecture."""
    shape = ARCH_INPUT_SIZES.get(architecture, (224, 224, 3))
    target_size = shape[:2]

    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        pad = int(max(w, h) * 0.1)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        img = img[y1:y2, x1:x2]

    img = cv2.resize(img, target_size)
    arr = np.expand_dims(img.astype('float32'), axis=0)

    # Apply the correct model-specific normalisation (critical for valid predictions)
    if _HAS_TF:
        if architecture == 'Xception':
            arr = xception.preprocess_input(arr)
        elif architecture == 'ResNet50':
            arr = resnet50.preprocess_input(arr)
        elif architecture == 'EfficientNetB0':
            arr = efficientnet.preprocess_input(arr)
        else:
            arr = arr / 255.0
    else:
        arr = arr / 255.0

    return arr


# Keep a simple wrapper for legacy callers (face detection only, no model-specific norm)
def preprocess_image(img_path, target_size=(299, 299)):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        img = img[y:y+h, x:x+w]
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


def forensic_score(filepath):
    """
    Heuristic deepfake / GAN-synthesis detection using image forensics.
    Returns a fake-probability in [0, 1] without requiring trained weights.

    Each signal has an individual weight. Signals tuned for face-swap deepfakes
    (ELA, boundary) get lower weight; signals tuned for fully-synthesised GAN
    images (FFT smoothness, skin texture, EXIF) get higher weight.

      1. ELA              — re-compression artefacts reveal editing/splicing.
      2. Face-boundary    — blending seam from face-swap deepfakes.
      3. FFT mid-freq     — GAN upsampling artefacts elevate mid-band energy.
      4. FFT HF smoothness— synthesised images lack natural camera sensor noise.
      5. Face local var   — GAN skin is unnaturally uniform vs real skin texture.
      6. EXIF metadata    — real camera photos always have rich EXIF; GAN images do not.
    """
    try:
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            return 0.5
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        weighted = []  # list of (score 0-1, weight)

        # ── 1. ELA (weight 0.8 — good for splice/edit, weak for pure synthesis) ──
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                tmp_path = f.name
            cv2.imwrite(tmp_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            reloaded = cv2.imread(tmp_path)
            os.unlink(tmp_path)
            if reloaded is not None and reloaded.shape == img_bgr.shape:
                ela = np.abs(img_bgr.astype(np.float32) - reloaded.astype(np.float32))
                ela_mean = float(np.mean(ela))
                ela_signal = float(np.clip((ela_mean - 8.0) / 22.0, 0.0, 1.0))
                weighted.append((ela_signal, 0.8))
        except Exception:
            pass

        # ── 2. Face-boundary gradient (weight 0.8 — face-swap seam detector) ──
        try:
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                border = 6
                x1, y1 = max(0, x - border), max(0, y - border)
                x2, y2 = min(gray.shape[1], x + w + border), min(gray.shape[0], y + h + border)
                roi = gray[y1:y2, x1:x2].astype(np.float32)
                gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
                grad = np.sqrt(gx**2 + gy**2)
                inner = grad[border:-border, border:-border] if grad.shape[0] > 2*border and grad.shape[1] > 2*border else grad
                outer = np.concatenate([
                    grad[:border, :].flatten(), grad[-border:, :].flatten(),
                    grad[:, :border].flatten(), grad[:, -border:].flatten()
                ])
                if inner.size > 0 and outer.size > 0:
                    ratio = float(np.mean(outer)) / (float(np.mean(inner)) + 1e-6)
                    boundary_signal = float(np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))
                    weighted.append((boundary_signal, 0.8))
        except Exception:
            pass

        # ── 3 & 4. FFT frequency analysis ──────────────────────────────────────
        try:
            gray_f = gray.astype(np.float32) / 255.0
            fft = np.fft.fftshift(np.fft.fft2(gray_f))
            mag = np.log1p(np.abs(fft))
            h_i, w_i = mag.shape
            Y, X = np.ogrid[:h_i, :w_i]
            dist = np.sqrt((X - w_i // 2) ** 2 + (Y - h_i // 2) ** 2)
            r = min(h_i, w_i) / 2.0

            low_e  = float(np.mean(mag[dist < r * 0.12]))
            mid_e  = float(np.mean(mag[(dist >= r * 0.12) & (dist < r * 0.45)]))
            high_e = float(np.mean(mag[dist >= r * 0.45]))

            # Signal 3 (weight 1.0): mid-freq elevation from GAN upsampling / compression
            if low_e > 0:
                freq_signal = float(np.clip((mid_e / low_e - 0.35) / 0.55, 0.0, 1.0))
                weighted.append((freq_signal, 1.0))

            # Signal 4 (weight 2.0): high-freq smoothness
            # Real camera images: high_e/mid_e ≈ 0.7-1.0 (sensor noise fills high freqs)
            # GAN synthesised (e.g. StyleGAN3): high_e/mid_e ≈ 0.4-0.7 — "too clean"
            if mid_e > 0:
                hf_ratio = high_e / mid_e
                smoothness_signal = float(np.clip((0.75 - hf_ratio) / 0.35, 0.0, 1.0))
                weighted.append((smoothness_signal, 2.0))
        except Exception:
            pass

        # ── 5. Face-region local variance (weight 1.5) ─────────────────────────
        # GAN skin is rendered without natural pore/texture detail; 16×16 block
        # median variance is consistently lower than real skin.
        try:
            det_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(det_faces) > 0:
                fx, fy, fw, fh = det_faces[0]
                face_roi = gray[fy:fy+fh, fx:fx+fw].astype(np.float32)
                if face_roi.size > 1024:
                    block = 16
                    local_vars = []
                    h_r, w_r = face_roi.shape
                    for by in range(0, h_r - block, block):
                        for bx in range(0, w_r - block, block):
                            local_vars.append(float(np.var(face_roi[by:by+block, bx:bx+block])))
                    if local_vars:
                        median_lv = float(np.median(local_vars))
                        # Real face photos: median_lv ≈ 80-500 (skin texture + sensor noise)
                        # GAN face images: median_lv ≈ 15-100 (unnaturally smooth)
                        skin_signal = float(np.clip((100.0 - median_lv) / 100.0, 0.0, 1.0))
                        weighted.append((skin_signal, 1.5))
        except Exception:
            pass

        # ── 6. EXIF metadata (weight 3.0 — strongest single signal for GAN images) ─
        # Real camera photos always carry camera make/model, shutter, aperture, etc.
        # GAN-generated images have no EXIF or only bare dimension metadata.
        # Caveat: social-media platforms strip EXIF from real photos too, so absence
        # alone is not conclusive — but it is still the strongest available signal.
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(filepath) as pil_img:
                exif = pil_img.getexif()
            CAMERA_TAGS = {271, 272, 306, 36867, 36868, 37377, 37378, 37386}
            camera_fields = sum(1 for t in CAMERA_TAGS if t in exif)
            if camera_fields >= 3:
                exif_signal = 0.0   # rich camera EXIF → very likely real
            elif camera_fields >= 1:
                exif_signal = 0.25  # partial camera metadata
            elif exif:
                exif_signal = 0.65  # EXIF present but no camera tags
            else:
                exif_signal = 0.90  # no EXIF at all — typical of GAN outputs
            weighted.append((exif_signal, 3.0))
        except Exception:
            pass

        if not weighted:
            return 0.5

        total_w = sum(w for _, w in weighted)
        return float(sum(s * w for s, w in weighted) / total_w)

    except Exception as e:
        print(f"Forensic analysis error: {e}")
        return 0.5

def _generate_ela_heatmap(filepath, heatmap_path):
    """Generate an Error Level Analysis overlay and save it to heatmap_path."""
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        return False
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        tmp_path = f.name
    try:
        cv2.imwrite(tmp_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        reloaded = cv2.imread(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    if reloaded is None or reloaded.shape != img_bgr.shape:
        cv2.imwrite(heatmap_path, img_bgr)
        return True
    ela = np.abs(img_bgr.astype(np.float32) - reloaded.astype(np.float32))
    ela_scaled = np.clip(ela * 10, 0, 255).astype(np.uint8)
    ela_gray = cv2.cvtColor(ela_scaled, cv2.COLOR_BGR2GRAY)
    ela_colored = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)
    combined = cv2.addWeighted(img_bgr, 0.55, ela_colored, 0.45, 0)
    cv2.imwrite(heatmap_path, combined)
    return True


def run_video_analysis(filepath, user_id=None):
    """Extract frames from a video and aggregate forensic scores."""
    start_time = time.time()
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise Exception("Could not open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise Exception("Video file appears to have no frames.")

    n_samples = min(16, total_frames)
    frame_indices = [int(i * total_frames / n_samples) for i in range(n_samples)]

    frame_scores = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        try:
            cv2.imwrite(tmp_path, frame)
            score = forensic_score(tmp_path)
            frame_scores.append(score)
        except Exception as e:
            print(f"Frame {idx} analysis error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    cap.release()

    if not frame_scores:
        raise Exception("Could not extract usable frames from video.")

    blended_prob = float(np.mean(frame_scores))
    is_fake = blended_prob > 0.5
    label = "FAKE" if is_fake else "REAL"
    confidence = round((blended_prob if is_fake else 1.0 - blended_prob) * 100, 2)
    processing_time = round((time.time() - start_time) * 1000, 0)

    result = {
        "label": label,
        "fake_probability": round(blended_prob * 100, 1),
        "confidence": confidence,
        "media_type": "video",
        "model": "Forensic Frame Analysis",
        "processing_ms": processing_time,
        "heatmap_url": None,
        "individual_scores": {"Forensics (avg)": round(blended_prob * 100, 1)},
        "frames_analyzed": len(frame_scores),
    }

    if user_id:
        try:
            with get_db_connection() as conn:
                conn.execute(
                    'INSERT INTO analyses (user_id, filename, media_type, verdict, confidence, fake_probability, heatmap_path) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (user_id, os.path.basename(filepath), 'video', label, confidence, round(blended_prob * 100, 1), None)
                )
                conn.commit()
        except Exception as db_err:
            print(f"DB save error: {db_err}")

    return result


def run_analysis(filepath, media_type, user_id=None):
    """Real analysis pipeline with face detection, model inference, and forensic analysis."""
    if media_type == 'video':
        return run_video_analysis(filepath, user_id)

    start_time = time.time()

    # Verify image is readable
    if cv2.imread(filepath) is None:
        raise Exception("Could not read image file.")

    # 1. Neural-network predictions (only when trained weights are available)
    nn_predictions = {}
    if real_models and _weights_loaded:
        for name, cnn_model in real_models.items():
            try:
                img_for_model = preprocess_for_model(filepath, architecture=name)
                if img_for_model is not None:
                    pred = cnn_model.predict(img_for_model)
                    nn_predictions[name] = float(pred[0][0])
            except Exception as e:
                print(f"Inference error for {name}: {e}")

    # 2. Image forensics score (always computed — architecture-independent signal)
    f_score = forensic_score(filepath)

    # 3. Blend predictions
    if _weights_loaded and nn_predictions:
        # Trained models: trust them 80%, sanity-check with forensics 20%
        fusion = FusionClassifier()
        nn_result = fusion.fuse(nn_predictions)
        nn_prob = nn_result.metadata['fake_probability']
        blended_prob = 0.80 * nn_prob + 0.20 * f_score
        predictions_display = dict(nn_predictions)
    else:
        # No trained weights — untrained CNN output is noise; use pure forensics
        blended_prob = f_score
        predictions_display = {}

    predictions_display['Forensics'] = f_score

    # 4. Final classification
    is_fake = blended_prob > 0.5
    label = "FAKE" if is_fake else "REAL"
    confidence = round((blended_prob if is_fake else 1.0 - blended_prob) * 100, 2)
    fake_prob = blended_prob * 100

    # 5. ELA heatmap (genuine forensic visualisation)
    heatmap_filename = None
    try:
        heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
        heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
        if not _generate_ela_heatmap(filepath, heatmap_path):
            heatmap_filename = None
    except Exception as e:
        print(f"Heatmap error: {e}")
        heatmap_filename = None

    processing_time = round((time.time() - start_time) * 1000, 0)

    result = {
        "label": label,
        "fake_probability": round(fake_prob, 1),
        "confidence": round(confidence, 1),
        "media_type": media_type,
        "model": "Fusion Ensemble + Forensics" if _weights_loaded else "Forensic Analysis (ELA + FFT + Gradients)",
        "processing_ms": processing_time,
        "heatmap_url": f"/static/heatmaps/{heatmap_filename}" if heatmap_filename else None,
        "individual_scores": {k: round(v * 100, 1) for k, v in predictions_display.items()}
    }

    if user_id:
        try:
            with get_db_connection() as conn:
                conn.execute(
                    'INSERT INTO analyses (user_id, filename, media_type, verdict, confidence, fake_probability, heatmap_path) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (user_id, os.path.basename(filepath), media_type, result['label'], result['confidence'], result['fake_probability'], heatmap_filename)
                )
                conn.commit()
        except Exception as db_err:
            print(f"DB save error: {db_err}")

    return result


@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))


@app.route('/history')
def get_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    with get_db_connection() as conn:
        rows = conn.execute('SELECT * FROM analyses WHERE user_id = ? ORDER BY timestamp DESC', (session['user_id'],)).fetchall()
    
    history = [dict(row) for row in rows]
    return jsonify({'ok': True, 'history': history})


@app.route('/delete_analysis/<int:analysis_id>', methods=['POST'])
def delete_analysis(analysis_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    with get_db_connection() as conn:
        conn.execute('DELETE FROM analyses WHERE id = ? AND user_id = ?', (analysis_id, session['user_id']))
        conn.commit()
    
    return jsonify({'ok': True})


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        with get_db_connection() as conn:
            user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('index'))
            
        flash('Invalid username or password')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('All fields are required')
            return render_template('register.html')
            
        hashed_pw = generate_password_hash(password)
        
        try:
            with get_db_connection() as conn:
                conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
                conn.commit()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists')
            
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file attached'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'Empty file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported format. Use JPG, PNG, WEBP, MP4, etc.'}), 400

    filename = secure_filename(file.filename)
    uid_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], uid_name)
    file.save(filepath)

    try:
        media_type = get_media_type(filename)
        result = run_analysis(filepath, media_type, user_id=session.get('user_id'))
        result['filename'] = filename
        return jsonify({'ok': True, 'result': result})
    except Exception as e:
        print("ERROR DURING ANALYSIS:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        try: os.remove(filepath)
        except: pass


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
