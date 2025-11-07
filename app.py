# from flask import Flask, jsonify, request, render_template_string
# from flask_cors import CORS
# import cv2
# import numpy as np
# import time
# from collections import defaultdict, deque
# import warnings
# import urllib.request
# import os

# warnings.filterwarnings('ignore')

# app = Flask(__name__)
# CORS(app)

# # Store session data
# sessions = defaultdict(lambda: {
#     'metrics': {
#         'ear': 0.0,
#         'mar': 0.0,
#         'perclos': 0.0,
#         'blink_count': 0,
#         'yawn_count': 0,
#         'total_frames': 0,
#         'closed_eyes_frames': 0,
#         'is_drowsy': False,
#         'is_yawning': False,
#         'head_pose': 'centered',
#         'attention_score': 100.0
#     },
#     'ear_history': deque(maxlen=30),
#     'mar_history': deque(maxlen=30),
#     'blink_state': False,
#     'yawn_state': False,
#     'consecutive_drowsy': 0,
#     'consecutive_yawn': 0,
#     'last_blink_frame': 0,
#     'last_update': time.time()
# })


# def download_models():
#     """Download OpenCV face detection models"""
#     models = {
#         'face_detector': {
#             'prototxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
#             'caffemodel': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
#         }
#     }
    
#     model_dir = 'models'
#     os.makedirs(model_dir, exist_ok=True)
    
#     files = {
#         'deploy.prototxt': models['face_detector']['prototxt'],
#         'res10_300x300_ssd_iter_140000.caffemodel': models['face_detector']['caffemodel']
#     }
    
#     for filename, url in files.items():
#         filepath = os.path.join(model_dir, filename)
#         if not os.path.exists(filepath):
#             print(f"Downloading {filename}...")
#             try:
#                 urllib.request.urlretrieve(url, filepath)
#                 print(f"Downloaded {filename}")
#             except Exception as e:
#                 print(f"Error downloading {filename}: {e}")
    
#     return model_dir


# class AccurateDriverMonitor:
#     """Accurate monitoring using OpenCV DNN and geometry"""
    
#     def __init__(self):
#         print("Initializing detector...")
        
#         # Download and load models
#         model_dir = download_models()
        
#         prototxt = os.path.join(model_dir, 'deploy.prototxt')
#         caffemodel = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
        
#         # Load DNN face detector (more accurate than Haar)
#         self.face_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        
#         # Load eye and mouth cascades (backup)
#         self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#         self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
#         # Thresholds
#         self.EAR_THRESHOLD = 0.21
#         self.EAR_CONSEC_FRAMES = 12
#         self.MAR_THRESHOLD = 0.6
#         self.MAR_CONSEC_FRAMES = 15
#         self.PERCLOS_THRESHOLD = 20.0
        
#         print("Detector ready!")
    
#     def detect_face_dnn(self, frame):
#         """Detect face using DNN (more accurate)"""
#         h, w = frame.shape[:2]
#         blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
#         self.face_net.setInput(blob)
#         detections = self.face_net.forward()
        
#         # Find best detection
#         best_confidence = 0
#         best_box = None
        
#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
            
#             if confidence > 0.5 and confidence > best_confidence:
#                 best_confidence = confidence
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 best_box = box.astype(int)
        
#         return best_box
    
#     def calculate_ear_from_contours(self, eye_region):
#         """Calculate EAR using contour analysis"""
#         if eye_region.size == 0:
#             return 0.3
        
#         gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
        
#         # Apply adaptive threshold
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
#         # Find contours
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         if not contours:
#             return 0.25
        
#         # Get largest contour (pupil/iris)
#         largest_contour = max(contours, key=cv2.contourArea)
        
#         # Calculate bounding ellipse
#         if len(largest_contour) >= 5:
#             ellipse = cv2.fitEllipse(largest_contour)
#             (x, y), (MA, ma), angle = ellipse
            
#             # EAR approximation from ellipse axes
#             if MA > 0:
#                 ear = ma / MA
#                 return np.clip(ear, 0.0, 0.5)
        
#         # Fallback to intensity-based
#         mean_intensity = np.mean(gray)
#         ear = (mean_intensity / 255.0) * 0.3 + 0.15
#         return ear
    
#     def calculate_mar_from_contours(self, mouth_region):
#         """Calculate MAR using contour analysis"""
#         if mouth_region.size == 0:
#             return 0.3
        
#         gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY) if len(mouth_region.shape) == 3 else mouth_region
        
#         # Detect dark regions (open mouth)
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)
        
#         # Calculate opening ratio
#         h, w = thresh.shape
#         dark_pixels = np.sum(thresh == 255)
#         total_pixels = h * w
        
#         if total_pixels == 0:
#             return 0.3
        
#         # MAR based on dark pixel ratio and shape
#         dark_ratio = dark_pixels / total_pixels
        
#         # Find contours for shape analysis
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         if contours:
#             largest = max(contours, key=cv2.contourArea)
#             if len(largest) >= 5:
#                 ellipse = cv2.fitEllipse(largest)
#                 (x, y), (MA, ma), angle = ellipse
#                 aspect_ratio = ma / MA if MA > 0 else 1.0
#                 mar = dark_ratio * aspect_ratio * 5.0
#                 return np.clip(mar, 0.0, 2.0)
        
#         mar = dark_ratio * 3.0
#         return np.clip(mar, 0.0, 2.0)
    
#     def detect_eyes_and_mouth(self, face_roi, face_gray):
#         """Detect eyes and mouth in face region"""
#         h, w = face_roi.shape[:2]
        
#         # Eye detection in upper half
#         upper_half = face_gray[:int(h*0.6), :]
#         eyes = self.eye_cascade.detectMultiScale(upper_half, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20))
        
#         # Mouth detection in lower half
#         lower_half_y = int(h*0.5)
#         lower_half = face_roi[lower_half_y:, :]
        
#         # Calculate EAR
#         if len(eyes) >= 2:
#             # Sort by size and take 2 largest
#             eyes_sorted = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
            
#             ear_values = []
#             for (ex, ey, ew, eh) in eyes_sorted:
#                 eye_region = face_roi[ey:ey+eh, ex:ex+ew]
#                 if eye_region.size > 0:
#                     ear = self.calculate_ear_from_contours(eye_region)
#                     ear_values.append(ear)
            
#             avg_ear = np.mean(ear_values) if ear_values else 0.25
#         else:
#             # Estimate from upper face region
#             upper_face_region = face_roi[:int(h*0.5), :]
#             avg_ear = self.calculate_ear_from_contours(upper_face_region)
        
#         # Calculate MAR
#         mar = self.calculate_mar_from_contours(lower_half)
        
#         return avg_ear, mar
    
#     def estimate_head_pose(self, face_box, frame_shape):
#         """Estimate head pose from face position"""
#         if face_box is None:
#             return "centered"
        
#         h, w = frame_shape[:2]
#         x1, y1, x2, y2 = face_box
        
#         face_center_x = (x1 + x2) / 2
#         face_center_y = (y1 + y2) / 2
        
#         # Horizontal deviation
#         h_deviation = (face_center_x - w/2) / (w/2)
#         v_deviation = (face_center_y - h/2) / (h/2)
        
#         if abs(h_deviation) > 0.2:
#             return "looking_right" if h_deviation > 0 else "looking_left"
#         elif v_deviation > 0.2:
#             return "looking_down"
#         elif v_deviation < -0.2:
#             return "looking_up"
#         else:
#             return "centered"
    
#     def process_frame(self, frame, session_data):
#         """Process frame with accurate detection"""
#         h, w = frame.shape[:2]
#         session_data['metrics']['total_frames'] += 1
#         total_frames = session_data['metrics']['total_frames']
        
#         # Detect face using DNN
#         face_box = self.detect_face_dnn(frame)
        
#         if face_box is None:
#             return {
#                 'status': 'no_face',
#                 'message': 'No face detected. Please position your face clearly in camera view.'
#             }
        
#         x1, y1, x2, y2 = face_box
#         face_roi = frame[y1:y2, x1:x2]
#         face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
#         # Detect eyes and mouth
#         avg_ear, mar = self.detect_eyes_and_mouth(face_roi, face_gray)
        
#         # Add to history
#         session_data['ear_history'].append(avg_ear)
#         session_data['mar_history'].append(mar)
        
#         # Update metrics
#         session_data['metrics']['ear'] = float(avg_ear)
#         session_data['metrics']['mar'] = float(mar)
        
#         # Head pose
#         head_pose = self.estimate_head_pose(face_box, frame.shape)
#         session_data['metrics']['head_pose'] = head_pose
        
#         # === BLINK DETECTION ===
#         if avg_ear < 0.20:
#             if not session_data['blink_state']:
#                 session_data['blink_state'] = True
#                 session_data['last_blink_frame'] = total_frames
#         else:
#             if session_data['blink_state']:
#                 # Blink duration check (2-6 frames = natural blink)
#                 blink_duration = total_frames - session_data['last_blink_frame']
#                 if 2 <= blink_duration <= 10:
#                     session_data['metrics']['blink_count'] += 1
#                 session_data['blink_state'] = False
        
#         # === DROWSINESS DETECTION ===
#         if avg_ear < self.EAR_THRESHOLD:
#             session_data['consecutive_drowsy'] += 1
#             session_data['metrics']['closed_eyes_frames'] += 1
            
#             if session_data['consecutive_drowsy'] >= self.EAR_CONSEC_FRAMES:
#                 session_data['metrics']['is_drowsy'] = True
#         else:
#             session_data['consecutive_drowsy'] = 0
#             session_data['metrics']['is_drowsy'] = False
        
#         # === YAWN DETECTION ===
#         if mar > self.MAR_THRESHOLD:
#             session_data['consecutive_yawn'] += 1
            
#             if session_data['consecutive_yawn'] >= self.MAR_CONSEC_FRAMES:
#                 if not session_data['yawn_state']:
#                     session_data['yawn_state'] = True
#                     session_data['metrics']['yawn_count'] += 1
#                 session_data['metrics']['is_yawning'] = True
#         else:
#             session_data['consecutive_yawn'] = 0
#             session_data['yawn_state'] = False
#             session_data['metrics']['is_yawning'] = False
        
#         # === PERCLOS ===
#         closed = session_data['metrics']['closed_eyes_frames']
#         perclos = (closed / total_frames * 100) if total_frames > 0 else 0.0
#         session_data['metrics']['perclos'] = float(perclos)
        
#         # === ATTENTION SCORE ===
#         attention_score = 100.0
        
#         if session_data['metrics']['is_drowsy']:
#             attention_score -= 40
#         elif perclos > self.PERCLOS_THRESHOLD:
#             attention_score -= 25
        
#         if session_data['metrics']['is_yawning']:
#             attention_score -= 15
        
#         if head_pose != "centered":
#             attention_score -= 10
        
#         # Use EAR history
#         if len(session_data['ear_history']) >= 20:
#             avg_ear_recent = np.mean(list(session_data['ear_history']))
#             if avg_ear_recent < 0.23:
#                 attention_score -= 10
        
#         attention_score = max(0.0, min(100.0, attention_score))
#         session_data['metrics']['attention_score'] = float(attention_score)
        
#         session_data['last_update'] = time.time()
        
#         return {
#             'status': 'success',
#             'metrics': session_data['metrics'].copy()
#         }


# # Initialize detector
# try:
#     detector = AccurateDriverMonitor()
# except Exception as e:
#     print(f"Error initializing detector: {e}")
#     detector = None


# @app.route('/')
# def home():
#     """Serve web interface"""
#     html = '''
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>Driver Monitoring System</title>
#         <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         <style>
#             * { margin: 0; padding: 0; box-sizing: border-box; }
#             body {
#                 font-family: 'Segoe UI', Arial, sans-serif;
#                 background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
#                 min-height: 100vh;
#                 padding: 20px;
#             }
#             .container { max-width: 1400px; margin: 0 auto; }
#             .header {
#                 text-align: center;
#                 color: white;
#                 margin-bottom: 30px;
#             }
#             .header h1 { 
#                 font-size: 2.5em; 
#                 margin-bottom: 10px;
#                 text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
#             }
#             .main-grid {
#                 display: grid;
#                 grid-template-columns: 1.2fr 1fr;
#                 gap: 20px;
#             }
#             .card {
#                 background: white;
#                 border-radius: 15px;
#                 padding: 25px;
#                 box-shadow: 0 10px 30px rgba(0,0,0,0.3);
#             }
#             #video {
#                 width: 100%;
#                 max-width: 640px;
#                 border-radius: 10px;
#                 background: #000;
#             }
#             .controls {
#                 margin-top: 20px;
#                 display: flex;
#                 gap: 10px;
#                 justify-content: center;
#             }
#             button {
#                 padding: 12px 30px;
#                 font-size: 16px;
#                 border: none;
#                 border-radius: 8px;
#                 cursor: pointer;
#                 font-weight: bold;
#                 transition: all 0.3s;
#             }
#             .btn-start { background: #4CAF50; color: white; }
#             .btn-stop { background: #f44336; color: white; }
#             .btn-reset { background: #ff9800; color: white; }
#             button:disabled { opacity: 0.5; cursor: not-allowed; }
#             .metrics-grid {
#                 display: grid;
#                 grid-template-columns: 1fr 1fr;
#                 gap: 15px;
#                 margin-top: 20px;
#             }
#             .metric {
#                 background: #f5f5f5;
#                 padding: 15px;
#                 border-radius: 10px;
#                 text-align: center;
#             }
#             .metric-label {
#                 font-size: 0.9em;
#                 color: #666;
#                 margin-bottom: 5px;
#             }
#             .metric-value {
#                 font-size: 1.8em;
#                 font-weight: bold;
#                 color: #333;
#             }
#             .alert-box {
#                 padding: 20px;
#                 border-radius: 10px;
#                 margin-top: 20px;
#                 text-align: center;
#                 font-size: 1.2em;
#                 font-weight: bold;
#             }
#             .alert-normal { background: #4CAF50; color: white; }
#             .alert-warning { background: #ff9800; color: white; }
#             .alert-critical {
#                 background: #f44336;
#                 color: white;
#                 animation: pulse 1s infinite;
#             }
#             @keyframes pulse {
#                 0%, 100% { opacity: 1; }
#                 50% { opacity: 0.7; }
#             }
#             .gauge-container {
#                 width: 100%;
#                 height: 30px;
#                 background: #e0e0e0;
#                 border-radius: 15px;
#                 overflow: hidden;
#                 margin: 15px 0;
#             }
#             .gauge-fill {
#                 height: 100%;
#                 background: linear-gradient(90deg, #f44336, #ff9800, #4CAF50);
#                 transition: width 0.5s;
#                 display: flex;
#                 align-items: center;
#                 justify-content: flex-end;
#                 padding-right: 10px;
#                 color: white;
#                 font-weight: bold;
#             }
#             @media (max-width: 768px) {
#                 .main-grid { grid-template-columns: 1fr; }
#             }
#         </style>
#     </head>
#     <body>
#         <div class="container">
#             <div class="header">
#                 <h1>🚗 Driver Monitoring System</h1>
#                 <p>Accurate drowsiness detection with DNN face detection</p>
#             </div>
            
#             <div class="main-grid">
#                 <div class="card">
#                     <h2>📹 Live Camera</h2>
#                     <video id="video" autoplay playsinline></video>
#                     <canvas id="canvas" style="display:none;"></canvas>
                    
#                     <div class="controls">
#                         <button id="startBtn" class="btn-start">▶ Start</button>
#                         <button id="stopBtn" class="btn-stop" disabled>⏹ Stop</button>
#                         <button id="resetBtn" class="btn-reset" disabled>🔄 Reset</button>
#                     </div>
#                 </div>
                
#                 <div class="card">
#                     <h2>📊 Metrics</h2>
                    
#                     <div style="text-align: center; margin: 15px 0;">
#                         <strong>🎯 Attention Score</strong>
#                         <div class="gauge-container">
#                             <div id="gauge" class="gauge-fill" style="width: 100%;">100%</div>
#                         </div>
#                     </div>
                    
#                     <div class="metrics-grid">
#                         <div class="metric">
#                             <div class="metric-label">👁️ EAR</div>
#                             <div class="metric-value" id="ear">0.000</div>
#                         </div>
#                         <div class="metric">
#                             <div class="metric-label">😮 MAR</div>
#                             <div class="metric-value" id="mar">0.000</div>
#                         </div>
#                         <div class="metric">
#                             <div class="metric-label">💤 PERCLOS</div>
#                             <div class="metric-value" id="perclos">0%</div>
#                         </div>
#                         <div class="metric">
#                             <div class="metric-label">👁️ Blinks</div>
#                             <div class="metric-value" id="blinks">0</div>
#                         </div>
#                         <div class="metric">
#                             <div class="metric-label">🥱 Yawns</div>
#                             <div class="metric-value" id="yawns">0</div>
#                         </div>
#                         <div class="metric">
#                             <div class="metric-label">📹 Frames</div>
#                             <div class="metric-value" id="frames">0</div>
#                         </div>
#                     </div>
                    
#                     <div id="alertBox" class="alert-box alert-normal">
#                         ✅ NORMAL - Driver Alert
#                     </div>
#                 </div>
#             </div>
#         </div>
        
#         <script>
#             const video = document.getElementById('video');
#             const canvas = document.getElementById('canvas');
#             const ctx = canvas.getContext('2d');
            
#             let sessionId = 'session_' + Date.now();
#             let isMonitoring = false;
#             let stream = null;
            
#             document.getElementById('startBtn').addEventListener('click', async () => {
#                 try {
#                     stream = await navigator.mediaDevices.getUserMedia({ 
#                         video: { width: 640, height: 480, facingMode: 'user' } 
#                     });
#                     video.srcObject = stream;
#                     isMonitoring = true;
                    
#                     document.getElementById('startBtn').disabled = true;
#                     document.getElementById('stopBtn').disabled = false;
#                     document.getElementById('resetBtn').disabled = false;
                    
#                     video.onloadedmetadata = () => {
#                         canvas.width = video.videoWidth;
#                         canvas.height = video.videoHeight;
#                         processFrames();
#                     };
#                 } catch (err) {
#                     alert('Camera error: ' + err.message);
#                 }
#             });
            
#             document.getElementById('stopBtn').addEventListener('click', () => {
#                 isMonitoring = false;
#                 if (stream) stream.getTracks().forEach(t => t.stop());
#                 video.srcObject = null;
#                 document.getElementById('startBtn').disabled = false;
#                 document.getElementById('stopBtn').disabled = true;
#             });
            
#             document.getElementById('resetBtn').addEventListener('click', async () => {
#                 await fetch('/api/reset', {
#                     method: 'POST',
#                     headers: { 'Content-Type': 'application/json' },
#                     body: JSON.stringify({ session_id: sessionId })
#                 });
#             });
            
#             async function processFrames() {
#                 if (!isMonitoring) return;
                
#                 ctx.drawImage(video, 0, 0);
#                 canvas.toBlob(async (blob) => {
#                     const formData = new FormData();
#                     formData.append('image', blob);
#                     formData.append('session_id', sessionId);
                    
#                     try {
#                         const res = await fetch('/api/process_frame', {
#                             method: 'POST',
#                             body: formData
#                         });
#                         const data = await res.json();
#                         if (data.status === 'success') updateMetrics(data.metrics);
#                     } catch (err) {
#                         console.error(err);
#                     }
                    
#                     setTimeout(processFrames, 100);
#                 }, 'image/jpeg', 0.7);
#             }
            
#             function updateMetrics(m) {
#                 document.getElementById('ear').textContent = m.ear.toFixed(3);
#                 document.getElementById('mar').textContent = m.mar.toFixed(3);
#                 document.getElementById('perclos').textContent = m.perclos.toFixed(1) + '%';
#                 document.getElementById('blinks').textContent = m.blink_count;
#                 document.getElementById('yawns').textContent = m.yawn_count;
#                 document.getElementById('frames').textContent = m.total_frames;
                
#                 const gauge = document.getElementById('gauge');
#                 gauge.style.width = m.attention_score + '%';
#                 gauge.textContent = m.attention_score.toFixed(0) + '%';
                
#                 const alert = document.getElementById('alertBox');
#                 if (m.is_drowsy) {
#                     alert.textContent = '🚨 DROWSINESS DETECTED!';
#                     alert.className = 'alert-box alert-critical';
#                 } else if (m.is_yawning) {
#                     alert.textContent = '⚠️ Yawning Detected';
#                     alert.className = 'alert-box alert-warning';
#                 } else if (m.perclos > 20) {
#                     alert.textContent = '⚠️ High Eye Closure';
#                     alert.className = 'alert-box alert-warning';
#                 } else {
#                     alert.textContent = '✅ NORMAL - Driver Alert';
#                     alert.className = 'alert-box alert-normal';
#                 }
#             }
#         </script>
#     </body>
#     </html>
#     '''
#     return render_template_string(html)


# @app.route('/api/process_frame', methods=['POST'])
# def process_frame():
#     """Process frame"""
#     if detector is None:
#         return jsonify({'status': 'error', 'message': 'Detector not initialized'}), 500
    
#     try:
#         session_id = request.form.get('session_id', 'default')
#         image_file = request.files.get('image')
        
#         if not image_file:
#             return jsonify({'status': 'error', 'message': 'No image'}), 400
        
#         image_bytes = np.frombuffer(image_file.read(), np.uint8)
#         frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
#         if frame is None:
#             return jsonify({'status': 'error', 'message': 'Invalid image'}), 400
        
#         session_data = sessions[session_id]
#         result = detector.process_frame(frame, session_data)
        
#         return jsonify(result), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500


# @app.route('/api/reset', methods=['POST'])
# def reset_session():
#     """Reset session"""
#     data = request.json
#     session_id = data.get('session_id', 'default')
    
#     if session_id in sessions:
#         sessions[session_id]['metrics'] = {
#             'ear': 0.0, 'mar': 0.0, 'perclos': 0.0,
#             'blink_count': 0, 'yawn_count': 0,
#             'total_frames': 0, 'closed_eyes_frames': 0,
#             'is_drowsy': False, 'is_yawning': False,
#             'head_pose': 'centered', 'attention_score': 100.0
#         }
#         sessions[session_id]['ear_history'].clear()
#         sessions[session_id]['mar_history'].clear()
    
#     return jsonify({'status': 'success'}), 200


# @app.route('/api/health', methods=['GET'])
# def health():
#     """Health check"""
#     return jsonify({
#         'status': 'healthy',
#         'service': 'Driver Monitoring System',
#         'detector': 'OpenCV DNN'
#     }), 200


# if __name__ == '__main__':
#     import os
#     port = int(os.environ.get('PORT', 5001))
#     print(f"Starting on port {port}...")
#     app.run(host='0.0.0.0', port=port, debug=False)
"""
Headless Driver Monitoring System - Flask API Only
No GUI, runs completely in background, exposes metrics via REST API
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
import time
from collections import deque
import threading
import warnings
from flask import Flask, jsonify
from flask_cors import CORS
import math

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)


class BackgroundSteeringAnalyzer:
    """Analyzes steering wheel behavior without GUI"""
    
    def __init__(self):
        self.steering_angle = 0.0
        self.steering_history = deque(maxlen=300)
        self.correction_count = 0
        self.last_correction_time = 0
        self.manual_input = 0
        self.steering_velocity = 0.0
        self.lane_deviation_score = 0
        self.micro_correction_rate = 0
        self.steering_smoothness = 100.0
        self.steering_alert_level = "NORMAL"
        self.road_offset = 0
        self.speed = 60
        self.road_position = 0
        
    def simulate_steering_input(self, head_rotation=0):
        """Simulate steering without visualization"""
        if self.manual_input != 0:
            self.steering_velocity += self.manual_input * 1.5
        
        self.steering_velocity *= 0.92
        self.steering_angle += self.steering_velocity
        self.steering_angle = np.clip(self.steering_angle, -90, 90)
        
        self.road_offset += self.steering_angle * 0.003
        self.road_offset = np.clip(self.road_offset, -150, 150)
        
        self.road_position += self.speed * 0.15
        if self.road_position > 1000:
            self.road_position = 0
        
        self.steering_history.append({
            'angle': self.steering_angle,
            'offset': self.road_offset,
            'timestamp': time.time()
        })
    
    def analyze_steering_patterns(self):
        """Analyze steering behavior for drowsiness indicators"""
        if len(self.steering_history) < 60:
            return
        
        angles = [s['angle'] for s in self.steering_history]
        offsets = [s['offset'] for s in self.steering_history]
        
        avg_offset = np.mean(np.abs(offsets[-90:]))
        self.lane_deviation_score = min(100, avg_offset * 0.8)
        
        sign_changes = 0
        for i in range(1, min(60, len(angles))):
            if np.sign(angles[-i]) != np.sign(angles[-(i+1)]):
                sign_changes += 1
        self.micro_correction_rate = sign_changes
        
        angle_variance = np.var(angles[-60:])
        self.steering_smoothness = max(0, 100 - angle_variance * 2)
        
        if self.lane_deviation_score > 60 or self.micro_correction_rate > 15:
            self.steering_alert_level = "CRITICAL"
        elif self.lane_deviation_score > 40 or self.micro_correction_rate > 10:
            self.steering_alert_level = "WARNING"
        elif self.lane_deviation_score > 25 or self.micro_correction_rate > 7:
            self.steering_alert_level = "CAUTION"
        else:
            self.steering_alert_level = "NORMAL"
    
    def get_metrics(self):
        """Return current steering metrics"""
        return {
            'steering_angle': round(self.steering_angle, 2),
            'lane_deviation_score': round(self.lane_deviation_score, 2),
            'micro_correction_rate': self.micro_correction_rate,
            'steering_smoothness': round(self.steering_smoothness, 2),
            'steering_alert_level': self.steering_alert_level,
            'road_offset': round(self.road_offset, 2)
        }


class HeadlessDrowsinessDetector:
    """Background drowsiness detection without GUI"""
    
    def __init__(self, camera_index=0):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.steering = BackgroundSteeringAnalyzer()
        
        # Thresholds
        self.EAR_THRESHOLD = 0.21
        self.MAR_THRESHOLD = 0.6
        self.PERCLOS_THRESHOLD = 0.2
        self.EAR_CONSEC_FRAMES = 20
        self.MAR_CONSEC_FRAMES = 15
        
        # Counters
        self.ear_counter = 0
        self.mar_counter = 0
        self.total_blinks = 0
        self.blink_counter = 0
        self.closed_eyes_frames = 0
        self.total_frames = 0
        
        # Alert state
        self.is_drowsy = False
        self.is_yawning = False
        self.camera_status = "alert"
        
        # Current metrics
        self.current_ear = 0.0
        self.current_mar = 0.0
        self.current_perclos = 0.0
        self.head_rotation = 0.0
        
        # Performance
        self.fps = 0.0
        self.fps_values = deque(maxlen=30)
        
        # Camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Cannot open webcam")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Threading
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # MediaPipe landmark indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH = [61, 291, 0, 17, 269, 405]
        
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2.0 * C) if C > 0 else 0
        return ear
    
    def calculate_mar(self, mouth_landmarks):
        """Calculate Mouth Aspect Ratio"""
        if len(mouth_landmarks) < 6:
            return 0
        vertical = distance.euclidean(mouth_landmarks[1], mouth_landmarks[5])
        horizontal = distance.euclidean(mouth_landmarks[0], mouth_landmarks[3])
        mar = vertical / horizontal if horizontal > 0 else 0
        return mar
    
    def estimate_head_pose(self, landmarks, frame_width, frame_height):
        """Estimate head rotation"""
        nose_tip = landmarks[1]
        nose_x = nose_tip.x * frame_width
        center_x = frame_width / 2
        rotation = (nose_x - center_x) / (frame_width / 2)
        return rotation * 30
    
    def process_frame(self):
        """Process single frame without display"""
        start_time = time.time()
        
        ret, frame = self.cap.read()
        if not ret:
            return False
        
        with self.lock:
            self.total_frames += 1
            h, w = frame.shape[:2]
            
            # Detect face using MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            # If no face detected
            if not results.multi_face_landmarks:
                self.current_ear = 0.0
                self.current_mar = 0.0
                return True
            
            # Get facial landmarks
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            
            # Extract coordinates
            left_eye_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] 
                                        for i in self.LEFT_EYE])
            right_eye_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] 
                                         for i in self.RIGHT_EYE])
            mouth_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] 
                                    for i in self.MOUTH])
            
            # Estimate head pose
            self.head_rotation = self.estimate_head_pose(landmarks, w, h)
            
            # Calculate metrics
            left_ear = self.calculate_ear(left_eye_points)
            right_ear = self.calculate_ear(right_eye_points)
            ear = (left_ear + right_ear) / 2.0
            mar = self.calculate_mar(mouth_points)
            
            self.current_ear = ear
            self.current_mar = mar
            
            # DROWSINESS DETECTION
            if ear < self.EAR_THRESHOLD:
                self.ear_counter += 1
                self.closed_eyes_frames += 1
                
                if self.ear_counter >= self.EAR_CONSEC_FRAMES:
                    self.is_drowsy = True
                    self.camera_status = "drowsy"
            else:
                self.ear_counter = 0
                self.is_drowsy = False
                if not self.is_yawning:
                    self.camera_status = "alert"
            
            # YAWN DETECTION
            if mar > self.MAR_THRESHOLD:
                self.mar_counter += 1
                
                if self.mar_counter >= self.MAR_CONSEC_FRAMES:
                    self.is_yawning = True
            else:
                self.mar_counter = 0
                self.is_yawning = False
            
            # BLINK DETECTION
            if ear < 0.18:
                self.blink_counter += 1
            else:
                if 2 <= self.blink_counter <= 5:
                    self.total_blinks += 1
                self.blink_counter = 0
            
            # PERCLOS
            self.current_perclos = (self.closed_eyes_frames / self.total_frames) * 100 if self.total_frames > 0 else 0
            
            # Update steering
            self.steering.simulate_steering_input(self.head_rotation)
            self.steering.analyze_steering_patterns()
            
            # FPS
            frame_time = time.time() - start_time
            self.fps_values.append(1.0 / frame_time if frame_time > 0 else 0)
            self.fps = np.mean(self.fps_values)
        
        return True
    
    def detection_loop(self):
        """Main detection loop running in background"""
        print("🟢 Detection system started in background...")
        while self.running:
            try:
                if not self.process_frame():
                    print("⚠️  Failed to process frame")
                    time.sleep(0.1)
            except Exception as e:
                print(f"❌ Error in detection loop: {e}")
                time.sleep(0.1)
        
        print("🛑 Detection system stopped")
    
    def start(self):
        """Start background detection"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.thread.start()
            print("✅ Background detection thread started")
    
    def stop(self):
        """Stop background detection"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.cap.release()
        print("✅ Detection system cleaned up")
    
    def get_metrics(self):
        """Get current metrics (thread-safe)"""
        with self.lock:
            return {
                'camera_metrics': {
                    'ear': round(self.current_ear, 4),
                    'mar': round(self.current_mar, 4),
                    'perclos': round(self.current_perclos, 2),
                    'blink_count': self.total_blinks,
                    'total_frames': self.total_frames,
                    'closed_eyes_frames': self.closed_eyes_frames,
                    'is_drowsy': self.is_drowsy,
                    'is_yawning': self.is_yawning,
                    'camera_status': self.camera_status,
                    'head_rotation': round(self.head_rotation, 2)
                },
                'steering_metrics': self.steering.get_metrics(),
                'performance': {
                    'fps': round(self.fps, 2)
                },
                'timestamp': time.time()
            }
    
    def get_assessment(self):
        """Get overall driver assessment"""
        with self.lock:
            risk_score = 0
            risk_level = "LOW"
            action_needed = "Continue monitoring"
            risk_factors = []
            
            # Camera risk assessment
            if self.is_drowsy:
                risk_score += 40
                risk_factors.append("Drowsiness detected")
            
            if self.current_perclos > 20:
                risk_score += 30
                risk_factors.append(f"High eye closure rate ({self.current_perclos:.1f}%)")
            elif self.current_perclos > 10:
                risk_score += 15
                risk_factors.append(f"Moderate eye closure rate ({self.current_perclos:.1f}%)")
            
            if self.is_yawning:
                risk_score += 10
                risk_factors.append("Yawning detected")
            
            # Steering risk assessment
            if self.steering.steering_alert_level == "CRITICAL":
                risk_score += 35
                risk_factors.append("Critical steering behavior")
            elif self.steering.steering_alert_level == "WARNING":
                risk_score += 20
                risk_factors.append("Warning steering behavior")
            elif self.steering.steering_alert_level == "CAUTION":
                risk_score += 10
                risk_factors.append("Cautionary steering behavior")
            
            if self.steering.lane_deviation_score > 60:
                risk_score += 15
                risk_factors.append(f"High lane deviation ({self.steering.lane_deviation_score:.0f})")
            
            # Determine risk level
            if risk_score >= 70:
                risk_level = "CRITICAL"
                action_needed = "PULL OVER IMMEDIATELY"
            elif risk_score >= 50:
                risk_level = "HIGH"
                action_needed = "URGENT ACTION NEEDED"
            elif risk_score >= 30:
                risk_level = "MODERATE"
                action_needed = "Take precautions"
            
            return {
                'risk_score': min(100, risk_score),
                'risk_level': risk_level,
                'action_needed': action_needed,
                'risk_factors': risk_factors,
                'camera_status': self.camera_status,
                'steering_status': self.steering.steering_alert_level,
                'timestamp': time.time()
            }
    
    def reset_counters(self):
        """Reset detection counters"""
        with self.lock:
            self.total_blinks = 0
            self.closed_eyes_frames = 0
            self.total_frames = 0
            self.steering.steering_history.clear()
            print("🔄 Counters reset")


# Global detector instance
detector = None


@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start the detection system"""
    global detector
    try:
        if detector is None:
            detector = HeadlessDrowsinessDetector(camera_index=0)
            detector.start()
            return jsonify({
                'status': 'success',
                'message': 'Detection system started'
            }), 200
        else:
            return jsonify({
                'status': 'info',
                'message': 'Detection system already running'
            }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop the detection system"""
    global detector
    try:
        if detector is not None:
            detector.stop()
            detector = None
            return jsonify({
                'status': 'success',
                'message': 'Detection system stopped'
            }), 200
        else:
            return jsonify({
                'status': 'info',
                'message': 'Detection system not running'
            }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get current detection metrics"""
    global detector
    if detector is None or not detector.running:
        return jsonify({
            'status': 'error',
            'message': 'Detection system not running'
        }), 400
    
    try:
        metrics = detector.get_metrics()
        return jsonify({
            'status': 'success',
            'data': metrics
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/assessment', methods=['GET'])
def get_assessment():
    """Get overall driver assessment"""
    global detector
    if detector is None or not detector.running:
        return jsonify({
            'status': 'error',
            'message': 'Detection system not running'
        }), 400
    
    try:
        assessment = detector.get_assessment()
        return jsonify({
            'status': 'success',
            'data': assessment
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/reset', methods=['POST'])
def reset_counters():
    """Reset detection counters"""
    global detector
    if detector is None or not detector.running:
        return jsonify({
            'status': 'error',
            'message': 'Detection system not running'
        }), 400
    
    try:
        detector.reset_counters()
        return jsonify({
            'status': 'success',
            'message': 'Counters reset successfully'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    global detector
    return jsonify({
        'status': 'success',
        'data': {
            'running': detector is not None and detector.running,
            'timestamp': time.time()
        }
    }), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Driver Monitoring System',
        'timestamp': time.time()
    }), 200


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚗 HEADLESS DRIVER MONITORING SYSTEM - API SERVER")
    print("="*80)
    print("\n📡 API Endpoints:")
    print("   POST   /api/start       - Start detection system")
    print("   POST   /api/stop        - Stop detection system")
    print("   GET    /api/metrics     - Get current metrics")
    print("   GET    /api/assessment  - Get driver assessment")
    print("   POST   /api/reset       - Reset counters")
    print("   GET    /api/status      - Get system status")
    print("   GET    /api/health      - Health check")
    print("\n💡 Usage:")
    print("   1. Start server: python app.py")
    print("   2. Start detection: curl -X POST http://localhost:5001/api/start")
    print("   3. Get metrics: curl http://localhost:5001/api/metrics")
    print("   4. Stop detection: curl -X POST http://localhost:5001/api/stop")
    print("\n" + "="*80 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down server...")
        if detector is not None:
            detector.stop()
        print("✅ Server stopped successfully")
