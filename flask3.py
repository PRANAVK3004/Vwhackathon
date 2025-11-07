"""
Accurate Driver Monitoring System using dlib
Works on Render.com - Downloads shape predictor automatically
"""

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
import dlib
import time
from collections import defaultdict, deque
import warnings
import urllib.request
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Store session data with sliding windows for accurate detection
sessions = defaultdict(lambda: {
    'metrics': {
        'ear': 0.0,
        'mar': 0.0,
        'perclos': 0.0,
        'blink_count': 0,
        'yawn_count': 0,
        'total_frames': 0,
        'closed_eyes_frames': 0,
        'is_drowsy': False,
        'is_yawning': False,
        'head_pose': 'centered',
        'attention_score': 100.0
    },
    'ear_history': deque(maxlen=30),
    'mar_history': deque(maxlen=30),
    'blink_state': False,
    'yawn_state': False,
    'consecutive_drowsy': 0,
    'consecutive_yawn': 0,
    'last_update': time.time()
})


def download_shape_predictor():
    """Download dlib shape predictor if not exists"""
    model_path = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.exists(model_path):
        print("Downloading shape predictor model...")
        url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
        
        try:
            urllib.request.urlretrieve(url, "shape_predictor.dat.bz2")
            
            import bz2
            with bz2.open("shape_predictor.dat.bz2", 'rb') as f_in:
                with open(model_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            os.remove("shape_predictor.dat.bz2")
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    
    return model_path


class AccurateDriverMonitor:
    """Accurate monitoring using dlib facial landmarks"""
    
    def __init__(self):
        print("Initializing detector...")
        
        # Initialize dlib detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        
        model_path = download_shape_predictor()
        self.predictor = dlib.shape_predictor(model_path)
        
        # Facial landmark indices
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))
        
        # Thresholds (calibrated for accuracy)
        self.EAR_THRESHOLD = 0.21
        self.EAR_CONSEC_FRAMES = 15  # ~0.5 seconds at 30fps
        self.MAR_THRESHOLD = 0.65
        self.MAR_CONSEC_FRAMES = 20  # ~0.67 seconds
        self.PERCLOS_THRESHOLD = 20.0  # PERCLOS > 20% = drowsy
        
        print("Detector ready!")
    
    def shape_to_np(self, shape):
        """Convert dlib shape to numpy array"""
        coords = np.zeros((68, 2), dtype=int)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    
    def eye_aspect_ratio(self, eye):
        """Calculate Eye Aspect Ratio (EAR)"""
        # Vertical eye landmarks
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        
        # Horizontal eye landmark
        C = np.linalg.norm(eye[0] - eye[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth):
        """Calculate Mouth Aspect Ratio (MAR)"""
        # Vertical mouth landmarks
        A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[8])   # 53, 57
        
        # Horizontal mouth landmark
        C = np.linalg.norm(mouth[0] - mouth[6])   # 49, 55
        
        # MAR formula
        mar = (A + B) / (2.0 * C)
        return mar
    
    def calculate_head_pose(self, landmarks, frame_shape):
        """Estimate head pose/direction"""
        h, w = frame_shape[:2]
        
        # Get key points
        nose_tip = landmarks[30]
        left_eye = np.mean(landmarks[self.LEFT_EYE], axis=0)
        right_eye = np.mean(landmarks[self.RIGHT_EYE], axis=0)
        
        # Calculate center
        eye_center = (left_eye + right_eye) / 2
        
        # Horizontal deviation
        horizontal_ratio = (nose_tip[0] - w/2) / (w/2)
        
        # Vertical deviation
        vertical_ratio = (nose_tip[1] - h/2) / (h/2)
        
        if abs(horizontal_ratio) > 0.15:
            if horizontal_ratio > 0:
                return "looking_right"
            else:
                return "looking_left"
        elif vertical_ratio > 0.15:
            return "looking_down"
        elif vertical_ratio < -0.15:
            return "looking_up"
        else:
            return "centered"
    
    def process_frame(self, frame, session_data):
        """Process frame with accurate detection"""
        h, w = frame.shape[:2]
        session_data['metrics']['total_frames'] += 1
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return {
                'status': 'no_face',
                'message': 'No face detected. Please position your face in camera view.'
            }
        
        # Use first detected face
        face = faces[0]
        
        # Get facial landmarks
        shape = self.predictor(gray, face)
        landmarks = self.shape_to_np(shape)
        
        # Extract eye coordinates
        left_eye = landmarks[self.LEFT_EYE]
        right_eye = landmarks[self.RIGHT_EYE]
        
        # Calculate EAR for both eyes
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Extract mouth coordinates
        mouth = landmarks[self.MOUTH]
        mar = self.mouth_aspect_ratio(mouth)
        
        # Calculate head pose
        head_pose = self.calculate_head_pose(landmarks, frame.shape)
        
        # Add to history
        session_data['ear_history'].append(avg_ear)
        session_data['mar_history'].append(mar)
        
        # Update metrics
        session_data['metrics']['ear'] = float(avg_ear)
        session_data['metrics']['mar'] = float(mar)
        session_data['metrics']['head_pose'] = head_pose
        
        # === BLINK DETECTION ===
        # Blink = quick close and open (EAR drops then recovers)
        if avg_ear < self.EAR_THRESHOLD:
            if not session_data['blink_state']:
                session_data['blink_state'] = True
        else:
            if session_data['blink_state']:
                session_data['blink_state'] = False
                session_data['metrics']['blink_count'] += 1
        
        # === DROWSINESS DETECTION ===
        if avg_ear < self.EAR_THRESHOLD:
            session_data['consecutive_drowsy'] += 1
            session_data['metrics']['closed_eyes_frames'] += 1
            
            # Eyes closed for extended period = drowsy
            if session_data['consecutive_drowsy'] >= self.EAR_CONSEC_FRAMES:
                session_data['metrics']['is_drowsy'] = True
        else:
            session_data['consecutive_drowsy'] = 0
            session_data['metrics']['is_drowsy'] = False
        
        # === YAWN DETECTION ===
        if mar > self.MAR_THRESHOLD:
            session_data['consecutive_yawn'] += 1
            
            # Mouth open for extended period = yawn
            if session_data['consecutive_yawn'] >= self.MAR_CONSEC_FRAMES:
                if not session_data['yawn_state']:
                    session_data['yawn_state'] = True
                    session_data['metrics']['yawn_count'] += 1
                session_data['metrics']['is_yawning'] = True
        else:
            session_data['consecutive_yawn'] = 0
            session_data['yawn_state'] = False
            session_data['metrics']['is_yawning'] = False
        
        # === PERCLOS (Percentage of Eye Closure) ===
        total = session_data['metrics']['total_frames']
        closed = session_data['metrics']['closed_eyes_frames']
        perclos = (closed / total * 100) if total > 0 else 0.0
        session_data['metrics']['perclos'] = float(perclos)
        
        # === ATTENTION SCORE ===
        # Based on multiple factors
        attention_score = 100.0
        
        if session_data['metrics']['is_drowsy']:
            attention_score -= 40
        elif perclos > self.PERCLOS_THRESHOLD:
            attention_score -= 25
        
        if session_data['metrics']['is_yawning']:
            attention_score -= 15
        
        if head_pose != "centered":
            attention_score -= 10
        
        # Use EAR history for smoothing
        if len(session_data['ear_history']) >= 30:
            avg_ear_30 = np.mean(list(session_data['ear_history']))
            if avg_ear_30 < 0.25:
                attention_score -= 10
        
        attention_score = max(0.0, min(100.0, attention_score))
        session_data['metrics']['attention_score'] = float(attention_score)
        
        session_data['last_update'] = time.time()
        
        return {
            'status': 'success',
            'metrics': session_data['metrics'].copy()
        }


# Initialize detector (will download model on first run)
try:
    detector = AccurateDriverMonitor()
except Exception as e:
    print(f"Error initializing detector: {e}")
    detector = None


@app.route('/')
def home():
    """Serve enhanced web interface"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Accurate Driver Monitoring System</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container { max-width: 1400px; margin: 0 auto; }
            .header {
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }
            .header h1 { 
                font-size: 2.8em; 
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .header p { font-size: 1.2em; opacity: 0.9; }
            
            .main-grid {
                display: grid;
                grid-template-columns: 1.2fr 1fr;
                gap: 25px;
                margin-bottom: 25px;
            }
            
            .card {
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 15px 40px rgba(0,0,0,0.4);
            }
            
            .video-section { text-align: center; }
            
            #video {
                width: 100%;
                max-width: 640px;
                border-radius: 15px;
                background: #000;
                box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            }
            
            .controls {
                margin-top: 25px;
                display: flex;
                gap: 12px;
                justify-content: center;
                flex-wrap: wrap;
            }
            
            button {
                padding: 14px 35px;
                font-size: 17px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-weight: bold;
                transition: all 0.3s;
                box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            }
            
            .btn-start { background: #4CAF50; color: white; }
            .btn-start:hover { background: #45a049; transform: translateY(-2px); }
            .btn-stop { background: #f44336; color: white; }
            .btn-stop:hover { background: #da190b; transform: translateY(-2px); }
            .btn-reset { background: #ff9800; color: white; }
            .btn-reset:hover { background: #e68900; transform: translateY(-2px); }
            button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-top: 20px;
            }
            
            .metric {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 18px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }
            
            .metric-label {
                font-size: 0.95em;
                color: #555;
                margin-bottom: 8px;
                font-weight: 600;
            }
            
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #1e3c72;
            }
            
            .attention-gauge {
                margin: 20px 0;
                text-align: center;
            }
            
            .gauge-container {
                width: 100%;
                height: 30px;
                background: #e0e0e0;
                border-radius: 15px;
                overflow: hidden;
                margin-top: 10px;
            }
            
            .gauge-fill {
                height: 100%;
                background: linear-gradient(90deg, #f44336, #ff9800, #4CAF50);
                transition: width 0.5s ease;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 10px;
                color: white;
                font-weight: bold;
            }
            
            .alert-box {
                padding: 25px;
                border-radius: 15px;
                margin-top: 25px;
                text-align: center;
                font-size: 1.3em;
                font-weight: bold;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            
            .alert-normal { background: #4CAF50; color: white; }
            .alert-warning { background: #ff9800; color: white; }
            .alert-critical {
                background: #f44336;
                color: white;
                animation: pulse 1s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.8; transform: scale(1.02); }
            }
            
            .status-badge {
                display: inline-block;
                padding: 8px 20px;
                border-radius: 25px;
                font-size: 1em;
                margin-top: 15px;
                font-weight: bold;
            }
            
            .status-active { background: #4CAF50; color: white; }
            .status-inactive { background: #999; color: white; }
            
            .info-box {
                background: #e3f2fd;
                border-left: 5px solid #2196F3;
                padding: 18px;
                margin-top: 25px;
                border-radius: 8px;
                font-size: 0.95em;
            }
            
            .head-pose {
                background: #fff3e0;
                border-left: 5px solid #ff9800;
                padding: 15px;
                margin-top: 15px;
                border-radius: 8px;
                text-align: left;
            }
            
            @media (max-width: 768px) {
                .main-grid { grid-template-columns: 1fr; }
                .metrics-grid { grid-template-columns: 1fr; }
                .header h1 { font-size: 2em; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚗 Accurate Driver Monitoring System</h1>
                <p>Advanced drowsiness detection with facial landmark analysis</p>
            </div>
            
            <div class="main-grid">
                <div class="card video-section">
                    <h2>📹 Live Camera Feed</h2>
                    <video id="video" autoplay playsinline></video>
                    <canvas id="canvas" style="display:none;"></canvas>
                    
                    <div class="controls">
                        <button id="startBtn" class="btn-start">▶ Start Monitoring</button>
                        <button id="stopBtn" class="btn-stop" disabled>⏹ Stop</button>
                        <button id="resetBtn" class="btn-reset" disabled>🔄 Reset Stats</button>
                    </div>
                    
                    <div id="status">
                        <span class="status-badge status-inactive">⚫ Inactive</span>
                    </div>
                    
                    <div class="head-pose">
                        <strong>👤 Head Position:</strong> 
                        <span id="headPose">Not detected</span>
                    </div>
                    
                    <div class="info-box">
                        <strong>💡 System Features:</strong>
                        <ul style="text-align: left; margin-top: 10px; padding-left: 20px;">
                            <li>68-point facial landmark detection</li>
                            <li>Accurate blink counting</li>
                            <li>Drowsiness detection (PERCLOS)</li>
                            <li>Yawn detection with counter</li>
                            <li>Head pose estimation</li>
                            <li>Real-time attention scoring</li>
                        </ul>
                    </div>
                </div>
                
                <div class="card">
                    <h2>📊 Real-time Metrics</h2>
                    
                    <div class="attention-gauge">
                        <div style="font-size: 1.1em; font-weight: bold; margin-bottom: 10px;">
                            🎯 Attention Score
                        </div>
                        <div class="gauge-container">
                            <div id="attentionGauge" class="gauge-fill" style="width: 100%;">
                                100%
                            </div>
                        </div>
                    </div>
                    
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-label">👁️ EAR (Eye Aspect)</div>
                            <div class="metric-value" id="ear">0.000</div>
                            <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                                Normal: > 0.21
                            </div>
                        </div>
                        
                        <div class="metric">
                            <div class="metric-label">😮 MAR (Mouth Aspect)</div>
                            <div class="metric-value" id="mar">0.000</div>
                            <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                                Yawn: > 0.65
                            </div>
                        </div>
                        
                        <div class="metric">
                            <div class="metric-label">💤 PERCLOS</div>
                            <div class="metric-value" id="perclos">0.0%</div>
                            <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                                Alert: < 20%
                            </div>
                        </div>
                        
                        <div class="metric">
                            <div class="metric-label">👁️ Blink Count</div>
                            <div class="metric-value" id="blinks">0</div>
                            <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                                Normal: 15-20/min
                            </div>
                        </div>
                        
                        <div class="metric">
                            <div class="metric-label">🥱 Yawn Count</div>
                            <div class="metric-value" id="yawns">0</div>
                            <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                                Fatigue indicator
                            </div>
                        </div>
                        
                        <div class="metric">
                            <div class="metric-label">📹 Frames</div>
                            <div class="metric-value" id="frames">0</div>
                            <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                                Processed
                            </div>
                        </div>
                        
                        <div class="metric" style="grid-column: 1 / -1;">
                            <div class="metric-label">😴 Drowsiness Status</div>
                            <div class="metric-value" id="drowsy" style="font-size: 1.5em;">
                                ALERT ✓
                            </div>
                        </div>
                    </div>
                    
                    <div id="alertBox" class="alert-box alert-normal">
                        ✅ NORMAL - Driver is Alert
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const resetBtn = document.getElementById('resetBtn');
            const statusBadge = document.querySelector('.status-badge');
            
            let sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            let isMonitoring = false;
            let stream = null;
            
            startBtn.addEventListener('click', async () => {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { 
                            width: { ideal: 640 },
                            height: { ideal: 480 },
                            facingMode: 'user'
                        } 
                    });
                    video.srcObject = stream;
                    
                    isMonitoring = true;
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    resetBtn.disabled = false;
                    statusBadge.textContent = '🟢 Active';
                    statusBadge.className = 'status-badge status-active';
                    
                    video.onloadedmetadata = () => {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        processFrames();
                    };
                } catch (err) {
                    alert('Error accessing webcam: ' + err.message + '\\n\\nPlease allow camera access and reload.');
                }
            });
            
            stopBtn.addEventListener('click', () => {
                isMonitoring = false;
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
                video.srcObject = null;
                
                startBtn.disabled = false;
                stopBtn.disabled = true;
                statusBadge.textContent = '⚫ Inactive';
                statusBadge.className = 'status-badge status-inactive';
            });
            
            resetBtn.addEventListener('click', async () => {
                if (!confirm('Reset all statistics?')) return;
                
                try {
                    const response = await fetch('/api/reset', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ session_id: sessionId })
                    });
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        alert('✓ Statistics reset successfully!');
                    }
                } catch (err) {
                    alert('Error resetting: ' + err.message);
                }
            });
            
            async function processFrames() {
                if (!isMonitoring) return;
                
                try {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    canvas.toBlob(async (blob) => {
                        if (!blob || !isMonitoring) return;
                        
                        const formData = new FormData();
                        formData.append('image', blob, 'frame.jpg');
                        formData.append('session_id', sessionId);
                        
                        try {
                            const response = await fetch('/api/process_frame', {
                                method: 'POST',
                                body: formData
                            });
                            
                            if (!response.ok) throw new Error('Server error');
                            
                            const data = await response.json();
                            
                            if (data.status === 'success') {
                                updateMetrics(data.metrics);
                            } else if (data.status === 'no_face') {
                                console.log(data.message);
                            }
                        } catch (err) {
                            console.error('Processing error:', err);
                        }
                        
                        if (isMonitoring) {
                            setTimeout(processFrames, 100);
                        }
                    }, 'image/jpeg', 0.8);
                } catch (err) {
                    console.error('Frame capture error:', err);
                    if (isMonitoring) {
                        setTimeout(processFrames, 100);
                    }
                }
            }
            
            function updateMetrics(metrics) {
                document.getElementById('ear').textContent = metrics.ear.toFixed(3);
                document.getElementById('mar').textContent = metrics.mar.toFixed(3);
                document.getElementById('perclos').textContent = metrics.perclos.toFixed(1) + '%';
                document.getElementById('blinks').textContent = metrics.blink_count;
                document.getElementById('yawns').textContent = metrics.yawn_count;
                document.getElementById('frames').textContent = metrics.total_frames;
                
                const headPoseText = metrics.head_pose.replace('_', ' ').toUpperCase();
                document.getElementById('headPose').textContent = headPoseText;
                
                const attentionScore = metrics.attention_score;
                const gauge = document.getElementById('attentionGauge');
                gauge.style.width = attentionScore + '%';
                gauge.textContent = attentionScore.toFixed(0) + '%';
                
                const drowsyEl = document.getElementById('drowsy');
                if (metrics.is_drowsy) {
                    drowsyEl.textContent = 'DROWSY ⚠️';
                    drowsyEl.style.color = '#f44336';
                } else {
                    drowsyEl.textContent = 'ALERT ✓';
                    drowsyEl.style.color = '#4CAF50';
                }
                
                const alertBox = document.getElementById('alertBox');
                if (metrics.is_drowsy) {
                    alertBox.textContent = '🚨 CRITICAL ALERT - DROWSINESS DETECTED!';
                    alertBox.className = 'alert-box alert-critical';
                } else if (metrics.perclos > 20) {
                    alertBox.textContent = '⚠️ WARNING - High Eye Closure Rate (PERCLOS > 20%)';
                    alertBox.className = 'alert-box alert-warning';
                } else if (metrics.is_yawning) {
                    alertBox.textContent = '⚠️ CAUTION - Yawning Detected (Fatigue Sign)';
                    alertBox.className = 'alert-box alert-warning';
                } else if (attentionScore < 70) {
                    alertBox.textContent = '⚠️ ATTENTION - Low Attention Score';
                    alertBox.className = 'alert-box alert-warning';
                } else {
                    alertBox.textContent = '✅ NORMAL - Driver is Alert and Attentive';
                    alertBox.className = 'alert-box alert-normal';
                }
            }
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)


@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Process frame from browser"""
    if detector is None:
        return jsonify({
            'status': 'error',
            'message': 'Detector not initialized'
        }), 500
    
    try:
        session_id = request.form.get('session_id', 'default')
        image_file = request.files.get('image')
        
        if not image_file:
            return jsonify({'status': 'error', 'message': 'No image provided'}), 400
        
        # Read image
        image_bytes = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400
        
        # Process frame
        session_data = sessions[session_id]
        result = detector.process_frame(frame, session_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get current session metrics"""
    session_id = request.args.get('session_id', 'default')
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404
    
    return jsonify({
        'status': 'success',
        'data': sessions[session_id]['metrics']
    }), 200


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset session counters"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in sessions:
        sessions[session_id]['metrics'] = {
            'ear': 0.0,
            'mar': 0.0,
            'perclos': 0.0,
            'blink_count': 0,
            'yawn_count': 0,
            'total_frames': 0,
            'closed_eyes_frames': 0,
            'is_drowsy': False,
            'is_yawning': False,
            'head_pose': 'centered',
            'attention_score': 100.0
        }
        sessions[session_id]['ear_history'].clear()
        sessions[session_id]['mar_history'].clear()
        sessions[session_id]['consecutive_drowsy'] = 0
        sessions[session_id]['consecutive_yawn'] = 0
        sessions[session_id]['blink_state'] = False
        sessions[session_id]['yawn_state'] = False
    
    return jsonify({'status': 'success', 'message': 'Session reset successfully'}), 200


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Accurate Driver Monitoring System',
        'detector': 'dlib' if detector else 'not initialized'
    }), 200


@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    active_sessions = []
    current_time = time.time()
    
    for session_id, data in sessions.items():
        if current_time - data['last_update'] < 300:  # Active in last 5 minutes
            active_sessions.append({
                'session_id': session_id,
                'metrics': data['metrics'],
                'last_update': data['last_update']
            })
    
    return jsonify({
        'status': 'success',
        'count': len(active_sessions),
        'sessions': active_sessions
    }), 200


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    
    print("=" * 60)
    print("🚗 Accurate Driver Monitoring System")
    print("=" * 60)
    print(f"Starting server on port {port}...")
    print("Features:")
    print("  ✓ 68-point facial landmark detection (dlib)")
    print("  ✓ Accurate EAR calculation")
    print("  ✓ Accurate MAR calculation")
    print("  ✓ Blink counting")
    print("  ✓ Yawn detection & counting")
    print("  ✓ PERCLOS measurement")
    print("  ✓ Head pose estimation")
    print("  ✓ Attention scoring")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)
