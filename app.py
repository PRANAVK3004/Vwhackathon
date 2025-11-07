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