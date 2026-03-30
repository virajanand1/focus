import cv2, json, os, time # cv2: read webcam, json/os: save/load calibration data, time: countdown and calibration duration
import mediapipe as mp # mediapipe: face mesh detection and head pose estimation
import numpy as np 

from focus.detector import get_head_angles, build_camera_matrix

CALIBRATION_FILE = "data/calibration.json"
COUNTDOWN = 3
CALIB_TIME = 10

# save dictionary with neutral head pose angles to json file
def save_calibration_data(calibration_data):
    os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(calibration_data, f)

# load calibration data from json file, return None if file doesn't exist
def load_calibration_data():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    return None

# run calibration process: capture webcam video, detect face and head pose, calculate average neutral yaw and pitch angles, return them as a dictionary
def run_calibration():
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    h, w = frame.shape[:2]
    cam_mat = build_camera_matrix(w, h)

    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        for i in range(COUNTDOWN, 0, -1):
            print(f"Look at your screen in {i} seconds...")
            time.sleep(1)
        
        print("Calibrating...")

        yaws, pitches = [], []
        start_time = time.time()

        while time.time() - start_time < CALIB_TIME:
            ok, frame = cap.read()
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                pitch, yaw = get_head_angles(results.multi_face_landmarks[0], w, h, cam_mat)
                yaws.append(yaw)
                pitches.append(pitch)
        
        if not yaws or not pitches:
            cap.release()
            raise RuntimeError("Calibration failed: No face detected.")
        
        neutral_yaw = float(np.mean(yaws))
        neutral_pitch = float(np.mean(pitches))

        cap.release()
        print("neutral_yaw:", neutral_yaw, "neutral_pitch:", neutral_pitch)
        return neutral_yaw, neutral_pitch
            

        

        
