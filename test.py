# test.py: test to make sure calibration and detection are working correctly

import os
import cv2
import mediapipe as mp
from calibration import run_calibration, save_calibration_data, load_calibration_data
from detector import build_camera_matrix, get_head_angles, classify, build_landmarker

cal = load_calibration_data() #try to load existing calibration data

if cal is None: # calibrate if no existing file
    print("No calibration found, running calibration...")
    neutral_yaw, neutral_pitch = run_calibration()
    save_calibration_data({"neutral_yaw": neutral_yaw, "neutral_pitch": neutral_pitch})
else:
    neutral_yaw   = cal["neutral_yaw"]
    neutral_pitch = cal["neutral_pitch"]
    print(f"Loaded calibration — neutral_yaw={neutral_yaw:.1f}  neutral_pitch={neutral_pitch:.1f}\n")

cap = cv2.VideoCapture(0) #open default webcam
_, frame = cap.read() #read frame to get cam dimensions
h, w = frame.shape[:2] #get height and width for matrix
cam_mat = build_camera_matrix(w, h) #build camera matrix
landmarker = build_landmarker()

print("Running detection — press q to quit\n")

try:
    while cap.isOpened():
        ok, frame = cap.read() #read next webcam frame
        if not ok:
            break

        # convert frame to RGB and classify results
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect(mp_image)

        if not results.face_landmarks:
            print("NO_FACE")
        else:
            pitch, yaw = get_head_angles(results.face_landmarks[0], w, h, cam_mat)
            status = classify(pitch, yaw, neutral_pitch, neutral_yaw)
            print(f"{status:<14}  yaw={yaw:+.1f}  pitch={pitch:+.1f}")

        cv2.imshow("Test — press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    #delete calibration data file after test
    if os.path.exists("data/calibration.json"):
        os.remove("data/calibration.json")