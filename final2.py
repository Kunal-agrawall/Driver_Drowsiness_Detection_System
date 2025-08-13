import cv2
import mediapipe as mp
import dlib
import numpy as np
import tensorflow as tf
import time
import os
import threading
import queue
from collections import deque
from scipy.spatial import distance as dist
import simpleaudio as sa
from imutils import face_utils

# ------------------------ Configuration ------------------------
PROJECT_PATH = "D:\my desk\Coding\Projects\Driver_Drowsiness_detection(CNN multithreading)"
EYE_MODEL_PATH = os.path.join(PROJECT_PATH, "models", "eye_model.h5")
YAWN_MODEL_PATH = os.path.join(PROJECT_PATH, "models", "yawn_detection_model.h5")
EYE_TFLITE_PATH = os.path.join(PROJECT_PATH, "models", "eye_model.tflite")
YAWN_TFLITE_PATH = os.path.join(PROJECT_PATH, "models", "yawn_detection_model.tflite")
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'
LANDMARK_PREDICTOR_PATH = os.path.join(PROJECT_PATH, "landmark-predictor", "shape_predictor_68_face_landmarks.dat")

HIGH_BLINK_RATE = 28    # blinks per minute
LOW_BLINK_RATE = 5      # blinks per minute
PROLONGED_CLOSURE_DURATION = 2  # seconds to trigger alert from TFLite results
BEEP_INTERVAL = 0.5       # seconds between beeps

# Use a single threshold for TFLite eye detection
EYE_THRESHOLD = 0.5

EYE_SMOOTHING_WINDOW = 7
YAWN_SMOOTHING_WINDOW = 5
YAWN_THRESHOLD = 0.5
YAWN_PERSISTENCE_FRAMES = 2
YAWN_CONSECUTIVE_FRAMES = 3

# ------------------------ Audio Alert Functions ------------------------
def play_beep():
    frequency = 440  # Hz
    duration = 0.5   # seconds
    fs = 44100       # samples per second
    t = np.linspace(0, duration, int(fs * duration), False)
    note = np.sin(frequency * 2 * np.pi * t) * 0.3
    audio = (note * 32767).astype(np.int16)
    sa.play_buffer(audio, 1, 2, fs)

def play_double_beep():
    play_beep()
    time.sleep(0.2)
    play_beep()

# ------------------------ Blink Detection (Mediapipe-based from File1) ------------------------
class BlinkDetection:
    def __init__(self):
        self.total_blinks = 0
        self.start_time = time.time()
        self.closed_start_time = None  # For Mediapipe-based blink detection

    def eye_aspect_ratio(self, eye_landmarks):
        left_point = np.array([eye_landmarks[0].x, eye_landmarks[0].y])
        right_point = np.array([eye_landmarks[3].x, eye_landmarks[3].y])
        top_mid = np.mean([[eye_landmarks[1].x, eye_landmarks[1].y],
                           [eye_landmarks[2].x, eye_landmarks[2].y]], axis=0)
        bottom_mid = np.mean([[eye_landmarks[4].x, eye_landmarks[4].y],
                              [eye_landmarks[5].x, eye_landmarks[5].y]], axis=0)
        hor_dist = dist.euclidean(left_point, right_point)
        ver_dist = dist.euclidean(top_mid, bottom_mid)
        ear = ver_dist / hor_dist
        return ear

    def detect_blink(self, ear, threshold=0.22):
        if ear < threshold:
            if self.closed_start_time is None:
                self.closed_start_time = time.time()
        else:
            if self.closed_start_time is not None:
                self.total_blinks += 1
            self.closed_start_time = None

    def get_blink_rate(self):
        elapsed = time.time() - self.start_time
        return (self.total_blinks / elapsed) * 60 if elapsed > 0 else 0

    def get_closed_duration(self):
        if self.closed_start_time is not None:
            return time.time() - self.closed_start_time
        else:
            return 0

    def head_tilt_angle(self, landmarks):
        left_eye = np.array([landmarks[33].x, landmarks[33].y])
        right_eye = np.array([landmarks[263].x, landmarks[263].y])
        nose_tip = np.array([landmarks[1].x, landmarks[1].y])
        eye_mid = (left_eye + right_eye) / 2
        tilt = np.arctan2(nose_tip[1] - eye_mid[1], nose_tip[0] - eye_mid[0])
        return np.degrees(tilt)

# ------------------------ TFLite-Based Eye & Yawn Detector (from File2) ------------------------
class TFLiteDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(LANDMARK_PREDICTOR_PATH)

        self.eye_model = self.load_model(EYE_MODEL_PATH, EYE_TFLITE_PATH)
        self.yawn_model = self.load_model(YAWN_MODEL_PATH, YAWN_TFLITE_PATH)

        self.eye_state_buffer = deque(maxlen=EYE_SMOOTHING_WINDOW)
        self.yawn_state_buffer = deque(maxlen=YAWN_SMOOTHING_WINDOW)

        self.eye_current_state = "Open"
        self.yawn_current_state = "No Yawn"
        self.consecutive_yawns = 0

        self.eye_closed_start = None

    def load_model(self, model_path, tflite_path):
        try:
            model = tf.keras.models.load_model(model_path)
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
            print(f"Error loading model: {e}")
            exit(1)

    def preprocess_eye(self, eye_roi):
        try:
            eye_roi = cv2.resize(eye_roi, (26, 36))
            eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
            eye_roi = cv2.GaussianBlur(eye_roi, (3, 3), 0)
            eye_roi = eye_roi.reshape(1, 36, 26, 1).astype(np.float32) / 255.0
            return eye_roi
        except Exception as e:
            print(f"Error preprocessing eye: {e}")
            return None

    def preprocess_mouth(self, mouth_roi):
        try:
            mouth_roi = cv2.resize(mouth_roi, (64, 64))
            mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            mouth_roi = cv2.GaussianBlur(mouth_roi, (3, 3), 0)
            mouth_roi = mouth_roi.reshape(1, 64, 64, 1).astype(np.float32) / 255.0
            return mouth_roi
        except Exception as e:
            print(f"Error preprocessing mouth: {e}")
            return None

    def extract_mouth_dlib(self, gray, face):
        landmarks = self.landmark_predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        mouth_points = landmarks[48:68]
        x, y, w, h = cv2.boundingRect(mouth_points)
        mouth_roi = gray[y:y+h, x:x+w]
        return mouth_roi, mouth_points

    def detect_eye_status(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        detected = False
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes:
                eye_roi = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
                if eye_roi.size == 0:
                    continue
                processed_eye = self.preprocess_eye(eye_roi)
                if processed_eye is None:
                    continue
                interpreter = self.eye_model["interpreter"]
                interpreter.set_tensor(self.eye_model["input_details"][0]['index'], processed_eye)
                interpreter.invoke()
                prediction = interpreter.get_tensor(self.eye_model["output_details"][0]['index'])[0][0]
                self.eye_state_buffer.append(prediction)
                avg_prediction = sum(self.eye_state_buffer) / len(self.eye_state_buffer)
                new_state = "Closed" if avg_prediction > EYE_THRESHOLD else "Open"
                if new_state == "Closed":
                    if self.eye_closed_start is None:
                        self.eye_closed_start = time.time()
                else:
                    self.eye_closed_start = None
                self.eye_current_state = new_state
                detected = True
                return  # Process only one eye for performance
        if not detected:
            self.eye_current_state = "Open"
            self.eye_closed_start = None

    def detect_yawn_status(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.dlib_detector(gray, 0)
        detected = False
        for face in faces:
            mouth_roi, mouth_points = self.extract_mouth_dlib(gray, face)
            if mouth_roi.size == 0:
                continue
            processed_mouth = self.preprocess_mouth(cv2.cvtColor(mouth_roi, cv2.COLOR_GRAY2BGR))
            if processed_mouth is None:
                continue
            interpreter = self.yawn_model["interpreter"]
            interpreter.set_tensor(self.yawn_model["input_details"][0]['index'], processed_mouth)
            interpreter.invoke()
            prediction = interpreter.get_tensor(self.yawn_model["output_details"][0]['index'])[0][0]
            self.yawn_state_buffer.append(prediction)
            avg_prediction = sum(self.yawn_state_buffer) / len(self.yawn_state_buffer)
            new_state = "Yawn" if avg_prediction > YAWN_THRESHOLD else "No Yawn"
            if new_state == "Yawn":
                self.consecutive_yawns += 1
            else:
                self.consecutive_yawns = 0
            self.yawn_current_state = new_state
            detected = True
            return  # Process only one face for performance
        if not detected:
            self.yawn_current_state = "No Yawn"
            self.consecutive_yawns = 0

    def process_frame(self, frame):
        self.detect_eye_status(frame)
        self.detect_yawn_status(frame)

# ------------------------ Shared Global Data ------------------------
tflite_result = {
    "eye_state": "Open",
    "eye_closed_duration": 0,
    "yawn_state": "No Yawn",
    "timestamp": time.time()
}
tflite_lock = threading.Lock()
# Reduced queue size to ensure more recent frame processing
detection_queue = queue.Queue(maxsize=3)

# ------------------------ TFLite Worker Thread ------------------------
def tflite_worker():
    detector = TFLiteDetector()
    while True:
        try:
            frame = detection_queue.get(timeout=1)
        except queue.Empty:
            continue
        detector.process_frame(frame)
        with tflite_lock:
            tflite_result["eye_state"] = detector.eye_current_state
            if detector.eye_closed_start is not None:
                tflite_result["eye_closed_duration"] = time.time() - detector.eye_closed_start
            else:
                tflite_result["eye_closed_duration"] = 0
            tflite_result["yawn_state"] = detector.yawn_current_state
            tflite_result["timestamp"] = time.time()
        detection_queue.task_done()
        # Reduced sleep time to allow faster processing
        time.sleep(0.001)

threading.Thread(target=tflite_worker, daemon=True).start()

# ------------------------ Mediapipe Setup (For Blink/Head Tilt) ------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# ------------------------ Main Process: Blink Detection, Head Tilt, and Alerts ------------------------
blink_detector = BlinkDetection()
last_beep_time = 0
prev_yawn_state = "No Yawn"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit(1)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    drowsiness_alert = ""

    frame_count += 1
    # Enqueue only every 2nd frame to reduce load on TFLite processing
    if frame_count % 2 == 0 and detection_queue.qsize() < detection_queue.maxsize:
        detection_queue.put(frame.copy())

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            left_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            left_ear = blink_detector.eye_aspect_ratio(left_eye_landmarks)
            right_ear = blink_detector.eye_aspect_ratio(right_eye_landmarks)
            avg_ear = (left_ear + right_ear) / 2
            mediapipe_status = "Closed" if avg_ear < 0.25 else "Open"
            cv2.putText(frame, f"Eyes (Mediapipe): {mediapipe_status}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            blink_detector.detect_blink(avg_ear)
            blink_rate = blink_detector.get_blink_rate()
            head_tilt = blink_detector.head_tilt_angle(face_landmarks.landmark)

            with tflite_lock:
                current_eye_state = tflite_result["eye_state"]
                tflite_eye_duration = tflite_result["eye_closed_duration"]
                current_yawn_state = tflite_result["yawn_state"]

            cv2.putText(frame, f"Eyes (TFLite): {current_eye_state}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if current_yawn_state == "Yawn":
                drowsiness_alert = "Yawn Detected!"
                if prev_yawn_state != "Yawn" and (time.time() - last_beep_time) > BEEP_INTERVAL:
                    play_beep()
                    last_beep_time = time.time()
            elif current_eye_state == "Closed" and tflite_eye_duration >= PROLONGED_CLOSURE_DURATION:
                drowsiness_alert = "Drowsy: Sleep Detected!"
                if (time.time() - last_beep_time) > BEEP_INTERVAL:
                    play_beep()
                    last_beep_time = time.time()
            elif blink_rate > HIGH_BLINK_RATE or blink_rate < LOW_BLINK_RATE:
                drowsiness_alert = "Drowsy: Blink Rate Alert!"
                if (time.time() - last_beep_time) > BEEP_INTERVAL:
                    play_double_beep()
                    last_beep_time = time.time()

            prev_yawn_state = current_yawn_state

            cv2.putText(frame, f'Blinks: {blink_detector.total_blinks}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Blink Rate: {int(blink_rate)} BPM', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Head Tilt: {head_tilt:.2f} deg', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if drowsiness_alert:
                cv2.putText(frame, drowsiness_alert, (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=0),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=0)
            )
            break

    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
