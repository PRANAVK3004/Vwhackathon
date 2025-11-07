"""
Browser-Based Driver Monitoring - NO SCIPY VERSION
Works on Render.com free tier
"""

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
import time
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Store session data
sessions = defaultdict(lambda: {
    'metrics': {
        'ear': 0.0,
        'mar': 0.0,
        'perclos': 0.0,
        'blink_count': 0,
        'total_frames': 0,
        'closed_eyes_frames': 0,
        'is_drowsy': False,
        'is_yawning': False,
        'ear_counter': 0,
        'mar_counter': 0,
        'blink_counter': 0
    },
    'last_update': time.time()
})


def euclidean_distance(point1, point2):
    """Calculate distance between two points without scipy"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


class SimpleFaceDetector:
    """Simple face detection using OpenCV only"""
    
    def __init__(self):
        # Load Haar Cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        self.EAR_THRESHOLD = 0.25
        self.EAR_CONSEC_FRAMES = 20
        self.MAR_THRESHOLD = 0.6
        self.MAR_CONSEC_FRAMES = 15
    
    def calculate_ear_from_eye_region(self, eye_region):
        """Estimate EAR from eye region"""
        if eye_region.size == 0:
            return 0.3
        
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
        mean_intensity = np.mean(gray)
        
        # Normalize to EAR-like values
        ear = (mean_intensity / 255.0) * 0.2 + 0.15
        return ear
    
    def detect_mouth_opening(self, face_region):
        """Detect mouth opening"""
        if face_region.size == 0:
            return 0.0
        
        h, w = face_region.shape[:2]
        lower_face = face_region[int(h*0.6):h, :]
        
        if lower_face.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY) if len(lower_face.shape) == 3 else lower_face
        
        # Detect dark regions (open mouth)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        dark_pixels = np.sum(thresh == 255)
        total_pixels = thresh.size
        
        mar = (dark_pixels / total_pixels) * 10 if total_pixels > 0 else 0.0
        return mar
    
    def process_frame(self, frame, session_data):
        """Process frame"""
        h, w = frame.shape[:2]
        session_data['metrics']['total_frames'] += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        if len(faces) == 0:
            return {
                'status': 'no_face',
                'message': 'No face detected. Please position your face in front of the camera.'
            }
        
        # Use the largest face
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w_face, h_face) = faces_sorted[0]
        
        face_roi = frame[y:y+h_face, x:x+w_face]
        face_gray = gray[y:y+h_face, x:x+w_face]
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10)
        
        # Calculate EAR
        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
            
            ear_values = []
            for (ex, ey, ew, eh) in eyes_sorted:
                eye_region = face_roi[ey:ey+eh, ex:ex+ew]
                if eye_region.size > 0:
                    ear = self.calculate_ear_from_eye_region(eye_region)
                    ear_values.append(ear)
            
            avg_ear = np.mean(ear_values) if ear_values else 0.3
        else:
            # Estimate from upper face
            upper_face = face_roi[:int(h_face*0.5), :]
            avg_ear = self.calculate_ear_from_eye_region(upper_face)
        
        # Detect mouth/yawning
        mar = self.detect_mouth_opening(face_roi)
        
        # Update metrics
        session_data['metrics']['ear'] = float(avg_ear)
        session_data['metrics']['mar'] = float(mar)
        
        # Drowsiness detection
        if avg_ear < self.EAR_THRESHOLD:
            session_data['metrics']['ear_counter'] += 1
            session_data['metrics']['closed_eyes_frames'] += 1
            
            if session_data['metrics']['ear_counter'] >= self.EAR_CONSEC_FRAMES:
                session_data['metrics']['is_drowsy'] = True
        else:
            session_data['metrics']['ear_counter'] = 0
            session_data['metrics']['is_drowsy'] = False
        
        # Yawn detection
        if mar > self.MAR_THRESHOLD:
            session_data['metrics']['mar_counter'] += 1
            if session_data['metrics']['mar_counter'] >= self.MAR_CONSEC_FRAMES:
                session_data['metrics']['is_yawning'] = True
        else:
            session_data['metrics']['mar_counter'] = 0
            session_data['metrics']['is_yawning'] = False
        
        # Blink detection
        if avg_ear < 0.20:
            session_data['metrics']['blink_counter'] += 1
        else:
            if 2 <= session_data['metrics']['blink_counter'] <= 5:
                session_data['metrics']['blink_count'] += 1
            session_data['metrics']['blink_counter'] = 0
        
        # Calculate PERCLOS
        perclos = (session_data['metrics']['closed_eyes_frames'] / 
                  session_data['metrics']['total_frames']) * 100
        session_data['metrics']['perclos'] = float(perclos)
        
        session_data['last_update'] = time.time()
        
        return {
            'status': 'success',
            'metrics': session_data['metrics'].copy()
        }


# Initialize detector
detector = SimpleFaceDetector()


@app.route('/')
def home():
    """Serve web interface"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Driver Monitoring System</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            .header {
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { font-size: 1.2em; opacity: 0.9; }
            
            .main-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }
            
            .card {
                background: white;
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            
            .video-section { text-align: center; }
            
            #video {
                width: 100%;
                max-width: 640px;
                border-radius: 10px;
                background: #000;
            }
            
            .controls {
                margin-top: 20px;
                display: flex;
                gap: 10px;
                justify-content: center;
                flex-wrap: wrap;
            }
            
            button {
                padding: 12px 30px;
                font-size: 16px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                transition: all 0.3s;
            }
            
            .btn-start { background: #4CAF50; color: white; }
            .btn-start:hover { background: #45a049; }
            .btn-stop { background: #f44336; color: white; }
            .btn-stop:hover { background: #da190b; }
            .btn-reset { background: #ff9800; color: white; }
            .btn-reset:hover { background: #e68900; }
            button:disabled { opacity: 0.5; cursor: not-allowed; }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
            }
            
            .metric {
                background: #f5f5f5;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            }
            
            .metric-label {
                font-size: 0.9em;
                color: #666;
                margin-bottom: 5px;
            }
            
            .metric-value {
                font-size: 1.8em;
                font-weight: bold;
                color: #333;
            }
            
            .alert-box {
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
                text-align: center;
                font-size: 1.2em;
                font-weight: bold;
            }
            
            .alert-normal { background: #4CAF50; color: white; }
            .alert-warning { background: #ff9800; color: white; }
            .alert-critical {
                background: #f44336;
                color: white;
                animation: pulse 1s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            
            .status-badge {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9em;
                margin-top: 10px;
            }
            
            .status-active { background: #4CAF50; color: white; }
            .status-inactive { background: #999; color: white; }
            
            .info-box {
                background: #e3f2fd;
                border-left: 4px solid #2196F3;
                padding: 15px;
                margin-top: 20px;
                border-radius: 5px;
                font-size: 0.9em;
            }
            
            @media (max-width: 768px) {
                .main-grid { grid-template-columns: 1fr; }
                .metrics-grid { grid-template-columns: 1fr; }
                .header h1 { font-size: 1.8em; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚗 Driver Monitoring System</h1>
                <p>Real-time drowsiness detection using your webcam</p>
            </div>
            
            <div class="main-grid">
                <div class="card video-section">
                    <h2>📹 Live Camera Feed</h2>
                    <video id="video" autoplay playsinline></video>
                    <canvas id="canvas" style="display:none;"></canvas>
                    
                    <div class="controls">
                        <button id="startBtn" class="btn-start">▶ Start Monitoring</button>
                        <button id="stopBtn" class="btn-stop" disabled>⏹ Stop</button>
                        <button id="resetBtn" class="btn-reset" disabled>🔄 Reset</button>
                    </div>
                    
                    <div id="status">
                        <span class="status-badge status-inactive">⚫ Inactive</span>
                    </div>
                    
                    <div class="info-box">
                        <strong>💡 Tips:</strong>
                        <ul style="text-align: left; margin-top: 10px; padding-left: 20px;">
                            <li>Position your face in front of camera</li>
                            <li>Ensure good lighting</li>
                            <li>Look directly at camera</li>
                            <li>First load may take 30-60 seconds</li>
                        </ul>
                    </div>
                </div>
                
                <div class="card">
                    <h2>📊 Real-time Metrics</h2>
                    
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-label">👁️ EAR (Eye Aspect Ratio)</div>
                            <div class="metric-value" id="ear">0.000</div>
                        </div>
                        
                        <div class="metric">
                            <div class="metric-label">😮 MAR (Mouth Aspect)</div>
                            <div class="metric-value" id="mar">0.000</div>
                        </div>
                        
                        <div class="metric">
                            <div class="metric-label">💤 PERCLOS</div>
                            <div class="metric-value" id="perclos">0.0%</div>
                        </div>
                        
                        <div class="metric">
                            <div class="metric-label">👁️ Blinks</div>
                            <div class="metric-value" id="blinks">0</div>
                        </div>
                        
                        <div class="metric">
                            <div class="metric-label">📹 Frames Processed</div>
                            <div class="metric-value" id="frames">0</div>
                        </div>
                        
                        <div class="metric">
                            <div class="metric-label">😴 Drowsiness</div>
                            <div class="metric-value" id="drowsy">NO</div>
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
            
            let sessionId = 'session_' + Date.now();
            let isMonitoring = false;
            let stream = null;
            
            startBtn.addEventListener('click', async () => {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { 
                            width: 640, 
                            height: 480,
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
                    
                    // Wait for video to be ready
                    video.onloadedmetadata = () => {
                        processFrames();
                    };
                } catch (err) {
                    alert('Error accessing webcam: ' + err.message + '. Please allow camera access.');
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
                resetBtn.disabled = true;
                statusBadge.textContent = '⚫ Inactive';
                statusBadge.className = 'status-badge status-inactive';
            });
            
            resetBtn.addEventListener('click', async () => {
                try {
                    const response = await fetch('/api/reset', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ session_id: sessionId })
                    });
                    const data = await response.json();
                    alert(data.message);
                } catch (err) {
                    alert('Error resetting: ' + err.message);
                }
            });
            
            async function processFrames() {
                if (!isMonitoring) return;
                
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);
                
                canvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('image', blob);
                    formData.append('session_id', sessionId);
                    
                    try {
                        const response = await fetch('/api/process_frame', {
                            method: 'POST',
                            body: formData
                        });
                        const data = await response.json();
                        
                        if (data.status === 'success') {
                            updateMetrics(data.metrics);
                        } else if (data.status === 'no_face') {
                            console.log(data.message);
                        }
                    } catch (err) {
                        console.error('Processing error:', err);
                    }
                    
                    setTimeout(processFrames, 150);
                }, 'image/jpeg', 0.7);
            }
            
            function updateMetrics(metrics) {
                document.getElementById('ear').textContent = metrics.ear.toFixed(3);
                document.getElementById('mar').textContent = metrics.mar.toFixed(3);
                document.getElementById('perclos').textContent = metrics.perclos.toFixed(1) + '%';
                document.getElementById('blinks').textContent = metrics.blink_count;
                document.getElementById('frames').textContent = metrics.total_frames;
                document.getElementById('drowsy').textContent = metrics.is_drowsy ? 'YES ⚠️' : 'NO ✓';
                
                const alertBox = document.getElementById('alertBox');
                if (metrics.is_drowsy) {
                    alertBox.textContent = '🚨 CRITICAL - DROWSINESS DETECTED!';
                    alertBox.className = 'alert-box alert-critical';
                } else if (metrics.is_yawning) {
                    alertBox.textContent = '⚠️ WARNING - Yawning Detected';
                    alertBox.className = 'alert-box alert-warning';
                } else if (metrics.perclos > 20) {
                    alertBox.textContent = '⚠️ CAUTION - High Eye Closure Rate';
                    alertBox.className = 'alert-box alert-warning';
                } else {
                    alertBox.textContent = '✅ NORMAL - Driver is Alert';
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
    try:
        session_id = request.form.get('session_id', 'default')
        image_file = request.files.get('image')
        
        if not image_file:
            return jsonify({'status': 'error', 'message': 'No image'}), 400
        
        image_bytes = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Invalid image'}), 400
        
        session_data = sessions[session_id]
        result = detector.process_frame(frame, session_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get session metrics"""
    session_id = request.args.get('session_id', 'default')
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404
    
    return jsonify({'status': 'success', 'data': sessions[session_id]['metrics']}), 200


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset counters"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in sessions:
        sessions[session_id]['metrics'] = {
            'ear': 0.0, 'mar': 0.0, 'perclos': 0.0,
            'blink_count': 0, 'total_frames': 0, 'closed_eyes_frames': 0,
            'is_drowsy': False, 'is_yawning': False,
            'ear_counter': 0, 'mar_counter': 0, 'blink_counter': 0
        }
    
    return jsonify({'status': 'success', 'message': 'Reset successfully'}), 200


@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'service': 'Driver Monitoring'}), 200


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
