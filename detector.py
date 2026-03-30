import cv2
import mediapipe as mp
import numpy as np

FACE_3D = np.array([
    [0.0,    0.0,    0.0],    # nose tip
    [0.0,   -63.6, -12.5],   # chin
    [-43.3,  32.7, -26.0],   # left eye outer corner
    [43.3,   32.7, -26.0],   # right eye outer corner
    [-28.9, -28.9, -24.1],   # left mouth corner
    [28.9,  -28.9, -24.1],   # right mouth corner
    ], dtype=np.float64)

FACE_2D = [1,152,33,263,61,291] #mediapipe face landmarks for the above 3D points

YAW_MAX = 20
PITCH_MAX = 20

def build_camera_matrix(w, h):
    f = float(w)
    return np.array([
        [f,   0.0, w / 2.0], #f: [0,0],[1,1]: focal length, [0,2],[1,2]: principal point
        [0.0, f,   h / 2.0], #assume principal point is in center of camera
        [0.0, 0.0, 1.0    ],
    ], dtype=np.float64)

def get_head_angles(face_landmarks, w, h, cam_mat):
    lm = face_landmarks.landmark #list of 478 detected points on the face (with x,y,z)
    face_2d = np.array([[lm[i].x * w, lm[i].y * h] for i in FACE_2D], #multiply landmarks by width and height to get pixel coordinates
                       dtype=np.float64)
    
    dist = np.zeros((4, 1), dtype=np.float64) #assume no lens distortion

    ok, rot_vec, _ = cv2.solvePnP( #solvePnP: find rotation of head
        FACE_3D, face_2d, cam_mat, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0 #set pitch and yaw to 0 if solvePnP fails, helps normalize classification 
    
    rmat, _ = cv2.Rodrigues(rot_vec) #Rodrigues: convert rotation vector to rotation matrix
    angles, *_ = cv2.RQDecomp3x3(rmat) #RQDecomp3x3: decompose rotation matrix into readable angles (pitch, yaw, roll)
    return angles[0], angles[1]   # pitch, yaw

def classify(pitch, yaw, neutral_pitch, neutral_yaw):
    if abs(yaw - neutral_yaw > YAW_MAX):
        return "LOOKING_AWAY"
    if abs(pitch - neutral_pitch > PITCH_MAX):
        return "LOOKING_AWAY"
    return "FOCUSED"
