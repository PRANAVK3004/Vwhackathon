"""
Headless Driver Monitoring System - Flask API
Receives frames from a client, processes them, and returns metrics.
This single file also serves the minimal HTML/JS client.
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
import time
from collections import deque
import threading
import warnings
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
import math
import base64
import io
from PIL import Image
import os # <-- IMPORTED OS

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)


# --- HTML/JavaScript Client ---
# This is the "front end" that will run in the user's browser.
# We embed it here as a string to keep everything in one file.

HTML_CLIENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Driver Monitoring System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'brand-dark': '#1a202c',
                        'brand-light': '#2d3748',
                        'brand-accent': '#4299e1',
                        'status-normal': '#48bb78',
                        'status-caution': '#f6e05e',
                        'status-warning': '#f56565',
                        'status-critical': '#e53e3e',
                    },
                },
            },
        }
    </script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        pre { white-space: pre-wrap; word-wrap: break-word; }
        .video-container { position: relative; width: 100%; max-width: 640px; margin: auto; }
        .video-overlay {
            position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            display: flex; justify-content: center; align-items: center;
            background-color: rgba(0,0,0,0.5); color: white; font-size: 1.25rem;
            border-radius: 0.5rem; opacity: 0; transition: opacity 0.3s ease;
        }
        .video-container:hover .video-overlay { opacity: 1; }
    </style>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body class="bg-brand-dark text-gray-200 min-h-screen flex flex-col items-center p-4">

    <div class="w-full max-w-5xl mx-auto">
        <h1 class="text-3xl font-bold text-center text-brand-accent mb-6">
            🚗 Driver Monitoring System
        </h1>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            <!-- Left Column: Camera and Controls -->
            <div class="bg-brand-light p-6 rounded-lg shadow-xl">
                <h2 class="text-2xl font-semibold mb-4 border-b border-gray-600 pb-2">Live Feed</h2>
                
                <div class="video-container bg-gray-900 rounded-lg overflow-hidden">
                    <video id="video" width="640" height="480" autoplay muted playsinline class="w-full h-auto rounded-lg" style="transform: scaleX(-1);"></video>
                    <div class="video-overlay">Your Webcam</div>
                </div>
                <canvas id="canvas" width="640" height="480" class="hidden"></canvas>

                <div class="flex space-x-4 mt-6">
                    <button id="startButton" class="flex-1 bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 shadow-md">
                        Start Camera
                    </button>
                    <button id="stopButton" class="flex-1 bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 shadow-md" disabled>
                        Stop Camera
                    </button>
                </div>
                <button id="resetButton" class="w-full mt-4 bg-yellow-600 hover:bg-yellow-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 shadow-md">
                    Reset Counters
                </button>
            </div>

            <!-- Right Column: Analysis -->
            <div class="bg-brand-light p-6 rounded-lg shadow-xl">
                <h2 class="text-2xl font-semibold mb-4 border-b border-gray-600 pb-2">Live Analysis</h2>
                
                <div id="status-card" class="p-6 rounded-lg bg-brand-dark mb-4 transition-all duration-300">
                    <h3 class="text-lg font-semibold mb-2">Overall Risk</h3>
                    <div id="risk-level" class="text-4xl font-bold text-status-normal">LOW</div>
                    <div id="action-needed" class="text-lg mt-2 text-gray-300">Continue monitoring</div>
                </div>

                <div class="grid grid-cols-2 gap-4 text-center">
                    <div class="bg-brand-dark p-4 rounded-lg">
                        <div class="text-sm text-gray-400">Camera Status</div>
                        <div id="camera-status" class="text-xl font-semibold text-gray-200">N/A</div>
                    </div>
                    <div class="bg-brand-dark p-4 rounded-lg">
                        <div class="text-sm text-gray-400">Steering Status</div>
                        <div id="steering-status" class="text-xl font-semibold text-gray-200">N/A</div>
                    </div>
                </div>

                <h3 class="text-lg font-semibold mt-6 mb-2">Risk Factors</h3>
                <ul id="risk-factors" class="list-disc list-inside text-gray-300 bg-brand-dark p-4 rounded-lg min-h-[100px]">
                    <li>No risks detected.</li>
                </ul>

                <h3 class="text-lg font-semibold mt-6 mb-2">Raw Metrics</h3>
                <pre id="results" class="bg-gray-900 text-sm text-gray-300 p-4 rounded-lg overflow-auto h-64">
Waiting for data...
                </pre>
            </div>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const context = canvas.getContext('2d');
        const startButton = document.getElementById('startButton');
        const stopButton = document.getElementById('stopButton');
        const resetButton = document.getElementById('resetButton');
        const resultsDiv = document.getElementById('results');
        
        const statusCard = document.getElementById('status-card');
        const riskLevelDiv = document.getElementById('risk-level');
        const actionNeededDiv = document.getElementById('action-needed');
        const cameraStatusDiv = document.getElementById('camera-status');
        const steeringStatusDiv = document.getElementById('steering-status');
        const riskFactorsList = document.getElementById('risk-factors');

        let stream = null;
        let processing = false;
        let animationFrameId = null;

        const API_URL = '/api/process';
        const RESET_URL = '/api/reset';
        
        const statusColors = {
            "LOW": "text-status-normal", "MODERATE": "text-status-caution",
            "HIGH": "text-status-warning", "CRITICAL": "text-status-critical"
        };
        const bgStatusColors = {
            "LOW": "bg-status-normal", "MODERATE": "bg-status-caution",
            "HIGH": "bg-status-warning", "CRITICAL": "bg-status-critical"
        };
        const oldStatusColors = Object.values(statusColors);
        const oldBgStatusColors = Object.values(bgStatusColors);

        startButton.addEventListener('click', startCamera);
        stopButton.addEventListener('click', stopCamera);
        resetButton.addEventListener('click', resetMetrics);

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }, 
                    audio: false 
                });
                video.srcObject = stream;
                video.play();
                startButton.disabled = true;
                stopButton.disabled = false;
                processing = true;
                animationFrameId = requestAnimationFrame(processLoop);
                console.log("Camera started");
            } catch (err) {
                console.error("Error accessing webcam:", err);
                resultsDiv.textContent = "Error: Could not access webcam. " + err.message;
            }
        }

        function stopCamera() {
            if (stream) { stream.getTracks().forEach(track => track.stop()); }
            video.srcObject = null;
            stream = null;
            startButton.disabled = false;
            stopButton.disabled = true;
            processing = false;
            if (animationFrameId) { cancelAnimationFrame(animationFrameId); animationFrameId = null; }
            console.log("Camera stopped");
        }

        async function processLoop() {
            if (!processing || video.readyState < video.HAVE_METADATA) {
                if(processing) animationFrameId = requestAnimationFrame(processLoop);
                return;
            }

            context.clearRect(0, 0, canvas.width, canvas.height);
            context.save();
            context.scale(-1, 1);
            context.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
            context.restore();
            
            const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

            try {
                const response = await fetch(API_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: dataUrl })
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Server error: ${response.status} ${errorText}`);
                }

                const result = await response.json();
                
                if (result.status === 'success') {
                    updateUI(result.data);
                    resultsDiv.textContent = JSON.stringify(result.data, null, 2);
                } else {
                    resultsDiv.textContent = `Error: ${result.message}`;
                }

            } catch (err) {
                console.error("Error processing frame:", err);
                resultsDiv.textContent = `Error: ${err.message}. Is the server running?`;
                stopCamera();
            }

            if(processing) animationFrameId = requestAnimationFrame(processLoop);
        }

        function updateUI(data) {
            const riskLevel = data.risk_level;
            riskLevelDiv.textContent = riskLevel;
            actionNeededDiv.textContent = data.action_needed;

            riskLevelDiv.classList.remove(...oldStatusColors);
            riskLevelDiv.classList.add(statusColors[riskLevel] || 'text-status-normal');
            
            statusCard.classList.remove(...oldBgStatusColors.map(c => c.replace('bg-', 'border-')));
            statusCard.style.borderColor = tailwind.config.theme.extend.colors[`status-${riskLevel.toLowerCase()}`] || '#48bb78';
            statusCard.style.borderWidth = '2px';

            cameraStatusDiv.textContent = data.camera_status.replace('_', ' ').toUpperCase();
            steeringStatusDiv.textContent = data.steering_status.toUpperCase();
            
            if (data.risk_factors && data.risk_factors.length > 0) {
                riskFactorsList.innerHTML = '';
                data.risk_factors.forEach(factor => {
                    const li = document.createElement('li');
                    li.textContent = factor;
                    riskFactorsList.appendChild(li);
                });
            } else {
                riskFactorsList.innerHTML = '<li>No risks detected.</li>';
            }
        }
        
        async function resetMetrics() {
             try {
                const response = await fetch(RESET_URL, { method: 'POST' });
                const result = await response.json();
                
                if (result.status === 'success') {
                    console.log("Counters reset");
                    resultsDiv.textContent = "Counters reset. Waiting for new data...";
                    riskFactorsList.innerHTML = '<li>Counters reset.</li>';
                    riskLevelDiv.textContent = "LOW";
                    actionNeededDiv.textContent = "Continue monitoring";
                    riskLevelDiv.classList.remove(...oldStatusColors);
                    riskLevelDiv.classList.add('text-status-normal');
                    statusCard.style.borderColor = tailwind.config.theme.extend.colors['status-normal'];
                } else {
                    console.error("Failed to reset:", result.message);
                }
             } catch (err) {
                console.error("Error resetting counters:", err);
             }
        }
    </script>
</body>
</html>
"""

# --- Python Classes (Your Backend Code) ---

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
    """Processes image frames and detects drowsiness"""
    
    def __init__(self):
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
        
        # Thread-safe lock for updating state
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
    
    def process_frame(self, frame):
        """Process single frame received from client"""
        start_time = time.time()
        
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
                self.camera_status = "no_face"
                return
            
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
                    self.camera_status = "yawning"
            else:
                self.mar_counter = 0
                self.is_yawning = False
                if not self.is_drowsy:
                     self.camera_status = "alert"
            
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


# --- Global Detector Instance ---
try:
    detector = HeadlessDrowsinessDetector()
    print("🟢 Detection system initialized")
except Exception as e:
    detector = None
    print(f"❌ Failed to initialize detector: {e}")


# --- Helper Function ---
def b64_to_cv_image(b64_string):
    """Convert base64 string to an OpenCV image (numpy array)"""
    if "," in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    pil_image = Image.open(io.BytesIO(img_bytes))
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return cv_image


# --- API Endpoints ---

@app.route('/')
def index():
    """Serve the client-side HTML page from the string"""
    return render_template_string(HTML_CLIENT)


@app.route('/api/process', methods=['POST'])
def process_frame_endpoint():
    """Receives a frame, processes it, and returns assessment"""
    global detector
    if detector is None:
        return jsonify({
            'status': 'error',
            'message': 'Detection system not initialized'
        }), 500
        
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No image data in request'
            }), 400

        frame = b64_to_cv_image(data['image'])
        detector.process_frame(frame)
        assessment = detector.get_assessment()
        
        return jsonify({
            'status': 'success',
            'data': assessment
        }), 200

    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get current detection metrics"""
    global detector
    if detector is None:
        return jsonify({
            'status': 'error',
            'message': 'Detection system not initialized'
        }), 500
    
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
    if detector is None:
        return jsonify({
            'status': 'error',
            'message': 'Detection system not initialized'
        }), 500
    
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
    if detector is None:
        return jsonify({
            'status': 'error',
            'message': 'Detection system not initialized'
        }), 500
    
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
            'running': detector is not None,
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
    print("🚗 DRIVER MONITORING SYSTEM - API SERVER (Single-File Mode)")
    print("="*80)
    print("\n📡 API Endpoints:")
    print("   GET     /                - Serves the client-side webpage")
    print("   POST    /api/process     - Receives frame, returns assessment")
    print("   GET     /api/metrics     - Get current metrics")
    print("   GET     /api/assessment  - Get driver assessment")
    print("   POST    /api/reset       - Reset counters")
    print("   GET     /api/status      - Get system status")
    print("   GET     /api/health      - Health check")
    print("\n💡 Usage:")
    print("   1. Start this server: python app.py")
    print("   2. Open your browser and go to: http://localhost:5001")
    print("   3. Click 'Start Camera' on the webpage.")
    print("\n" + "="*80 + "\n")
    
    # Render will set the PORT environment variable
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
