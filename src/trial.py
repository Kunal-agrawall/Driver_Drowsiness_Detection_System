import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
import os
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project configuration (update these paths as needed)
PROJECT_PATH = "D:\my desk\Coding\Projects\Driver_Drowsiness_detection(CNN multithreading)"
EYE_MODEL_PATH = os.path.join(PROJECT_PATH, "models", "eye_model.h5")
YAWN_MODEL_PATH = os.path.join(PROJECT_PATH, "models", "yawn_detection_model.h5")
EYE_TFLITE_PATH = os.path.join(PROJECT_PATH, "models", "eye_model.tflite")
YAWN_TFLITE_PATH = os.path.join(PROJECT_PATH, "models", "yawn_detection_model.tflite")

# Paths to Haar cascade files
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'
MOUTH_CASCADE_PATH = os.path.join(PROJECT_PATH, "src", "haarcascade_smile.xml")

# Check cascade file existence
if not os.path.exists(FACE_CASCADE_PATH):
    logger.error("Face cascade file not found")
    exit(1)
if not os.path.exists(EYE_CASCADE_PATH):
    logger.error("Eye cascade file not found")
    exit(1)
if not os.path.exists(MOUTH_CASCADE_PATH):
    logger.error("Mouth cascade file not found")
    exit(1)

# Initialize Haar cascades
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
mouth_cascade = cv2.CascadeClassifier(MOUTH_CASCADE_PATH)

# Stabilization parameters for eye detection
EYE_SMOOTHING_WINDOW = 7
EYE_OPEN_THRESHOLD = 0.6
EYE_CLOSED_THRESHOLD = 0.4
EYE_PERSISTENCE_FRAMES = 3

# Stabilization parameters for yawn detection
YAWN_SMOOTHING_WINDOW = 5
YAWN_THRESHOLD = 0.5
YAWN_PERSISTENCE_FRAMES = 2
YAWN_CONSECUTIVE_FRAMES = 3

# Global configuration alerts (optional)
CONFIG = {
    "show_fps": True,
    "show_confidence": True,
    "enable_alerts": False,  # Set to True if you integrate an alert sound
    "alert_sound": "alert.wav"
}

class RealTimeDetector:
    def __init__(self):
        # Load models (Keras models converted to TFLite)
        self.eye_model = self.load_model(EYE_MODEL_PATH, EYE_TFLITE_PATH)
        self.yawn_model = self.load_model(YAWN_MODEL_PATH, YAWN_TFLITE_PATH)
        
        # Buffers for temporal smoothing
        self.eye_state_buffer = deque(maxlen=EYE_SMOOTHING_WINDOW)
        self.yawn_state_buffer = deque(maxlen=YAWN_SMOOTHING_WINDOW)
        
        # State tracking for eye detection
        self.eye_current_state = "Open"
        self.eye_state_counter = 0
        self.closed_eye_counter = 0
        self.drowsy_alert = False
        
        # State tracking for yawn detection
        self.yawn_current_state = "No Yawn"
        self.yawn_state_counter = 0
        self.consecutive_yawns = 0
        self.yawn_alert = False
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Alert sound setup (if needed)
        self.alert_sound = None
        # If you have alert sound functionality, you can initialize it here.
    
    def load_model(self, model_path, tflite_path):
        try:
            # Load Keras model and convert it to TFLite for faster inference
            model = load_model(model_path)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            return {
                "interpreter": interpreter,
                "input_details": interpreter.get_input_details(),
                "output_details": interpreter.get_output_details()
            }
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            exit(1)
    
    def preprocess_eye(self, eye_roi):
        try:
            eye_roi = cv2.resize(eye_roi, (26, 36))
            eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
            eye_roi = cv2.GaussianBlur(eye_roi, (3, 3), 0)
            eye_roi = eye_roi.reshape(1, 36, 26, 1).astype(np.float32) / 255.0
            return eye_roi
        except Exception as e:
            logger.error(f"Error preprocessing eye: {e}")
            return None
        
    def preprocess_mouth(self, mouth_roi):
        try:
            # Resize and preprocess mouth ROI for the yawn model
            mouth_roi = cv2.resize(mouth_roi, (64, 32))
            mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            mouth_roi = cv2.GaussianBlur(mouth_roi, (3, 3), 0)
            mouth_roi = mouth_roi.reshape(1, 64, 32, 1).astype(np.float32) / 255.0
            return mouth_roi
        except Exception as e:
            logger.error(f"Error preprocessing mouth: {e}")
            return None

    def detect_eye_status(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
                for (ex, ey, ew, eh) in eyes:
                    eye_roi = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
                    if eye_roi.size == 0:
                        continue
                    processed_eye = self.preprocess_eye(eye_roi)
                    if processed_eye is None:
                        continue
                    # Inference with TFLite eye model
                    self.eye_model["interpreter"].set_tensor(
                        self.eye_model["input_details"][0]['index'], processed_eye)
                    self.eye_model["interpreter"].invoke()
                    prediction = self.eye_model["interpreter"].get_tensor(
                        self.eye_model["output_details"][0]['index'])[0][0]
                    
                    # Append prediction for smoothing
                    self.eye_state_buffer.append(prediction)
                    avg_prediction = sum(self.eye_state_buffer) / len(self.eye_state_buffer)
                    
                    # Determine state based on thresholds
                    if avg_prediction < EYE_OPEN_THRESHOLD:
                        new_state = "Open"
                    elif avg_prediction > EYE_CLOSED_THRESHOLD:
                        new_state = "Closed"
                    else:
                        new_state = self.eye_current_state
                        
                    # State persistence for stable prediction
                    if new_state != self.eye_current_state:
                        self.eye_state_counter += 1
                        if self.eye_state_counter >= EYE_PERSISTENCE_FRAMES:
                            self.eye_current_state = new_state
                            self.eye_state_counter = 0
                    else:
                        self.eye_state_counter = 0
                    
                    # Track consecutive closed frames
                    if self.eye_current_state == "Closed":
                        self.closed_eye_counter += 1
                    else:
                        self.closed_eye_counter = 0
                    
                    color = (0, 255, 0) if self.eye_current_state == "Open" else (0, 0, 255)
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, 1)
                    cv2.putText(frame, self.eye_current_state, (x+ex, y+ey-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    if self.closed_eye_counter >= 10:
                        self.drowsy_alert = True
                        # Uncomment below if using alert sound
                        # if CONFIG["enable_alerts"] and self.alert_sound:
                        #     self.alert_sound.play()
                    else:
                        self.drowsy_alert = False
            return frame
        except Exception as e:
            logger.error(f"Error in eye detection: {e}")
            return frame

    def detect_yawn_status(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                # Adjusted parameters for mouth detection
                mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
                for (mx, my, mw, mh) in mouths:
                    mouth_roi = frame[y+my:y+my+mh, x+mx:x+mx+mw]
                    if mouth_roi.size == 0:
                        continue
                    processed_mouth = self.preprocess_mouth(mouth_roi)
                    if processed_mouth is None:
                        continue
                    # Inference with TFLite yawn model
                    self.yawn_model["interpreter"].set_tensor(
                        self.yawn_model["input_details"][0]['index'], processed_mouth)
                    self.yawn_model["interpreter"].invoke()
                    prediction = self.yawn_model["interpreter"].get_tensor(
                        self.yawn_model["output_details"][0]['index'])[0][0]
                    
                    # Append prediction for smoothing
                    self.yawn_state_buffer.append(prediction)
                    avg_prediction = sum(self.yawn_state_buffer) / len(self.yawn_state_buffer)
                    
                    if avg_prediction > YAWN_THRESHOLD:
                        new_state = "Yawn"
                    else:
                        new_state = "No Yawn"
                    
                    if new_state != self.yawn_current_state:
                        self.yawn_state_counter += 1
                        if self.yawn_state_counter >= YAWN_PERSISTENCE_FRAMES:
                            self.yawn_current_state = new_state
                            self.yawn_state_counter = 0
                    else:
                        self.yawn_state_counter = 0
                    
                    if self.yawn_current_state == "Yawn":
                        self.consecutive_yawns += 1
                    else:
                        if self.consecutive_yawns >= YAWN_CONSECUTIVE_FRAMES:
                            self.yawn_alert = True
                            # Uncomment below if using alert sound
                            # if CONFIG["enable_alerts"] and self.alert_sound:
                            #     self.alert_sound.play()
                        self.consecutive_yawns = 0
                    
                    color = (0, 0, 255) if self.yawn_current_state == "Yawn" else (0, 255, 0)
                    cv2.rectangle(frame, (x+mx, y+my), (x+mx+mw, y+my+mh), color, 1)
                    cv2.putText(frame, self.yawn_current_state, (x+mx, y+my-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            return frame
        except Exception as e:
            logger.error(f"Error in yawn detection: {e}")
            return frame

    def calculate_fps(self):
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            end_time = time.time()
            fps = self.frame_count / (end_time - self.start_time)
            self.start_time = end_time
            self.frame_count = 0
            return f"FPS: {fps:.2f}"
        return ""

    def display_alerts(self, frame):
        height, width, _ = frame.shape
        if self.drowsy_alert:
            cv2.putText(frame, "DROWSINESS DETECTED!", (10, height-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if self.yawn_alert:
            cv2.putText(frame, "YAWN DETECTED!", (10, height-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    def main(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self.start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame (using the original resolution for detection)
            frame = self.detect_eye_status(frame)
            frame = self.detect_yawn_status(frame)
            frame = self.display_alerts(frame)
            
            fps_text = self.calculate_fps()
            if fps_text and CONFIG["show_fps"]:
                cv2.putText(frame, fps_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow('Eye and Yawn Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = RealTimeDetector()
    detector.main()
