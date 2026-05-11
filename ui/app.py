import cv2
import mediapipe as mp
import tkinter as tk
import threading
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageTk  # for displaying webcam feed in Tkinter
from calibration import run_calibration, save_calibration_data, load_calibration_data
from detector import build_camera_matrix, get_head_angles, classify, build_landmarker
from tracker import FocusTracker

class App(tk.Tk):
    def __init__(self): 
        super().__init__()
        self.title("Focus Tracker")
        self.geometry("800x600")
        self.configure(bg="#1a1a1a")
        self.tracker = FocusTracker()
        self.current_frame = None
        self._running = False
        self._thread = None
        self._build_ui()
        self._tick()

    def _build_ui(self):
        self.video_canvas = tk.Canvas(self, width=480, height=360, bg="#000000")
        self.video_canvas.grid(row=0, column=0, padx=20, pady=20)

        self.timeline_canvas = tk.Canvas(self, width=480, height=50, bg="#333333")
        self.timeline_canvas.grid(row=1, column=0, padx=20, pady=(0, 20))

        right_frame = tk.Frame(self, bg="#1a1a1a")
        right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="n")

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(right_frame, textvariable=self._status_var, bg="#1a1a1a", fg="#ffffff", font=("Arial", 16)).pack(pady=(0, 30))

        self._distracted_time_var = tk.StringVar(value="Distracted Time: 0.0s")
        tk.Label(right_frame, textvariable=self._distracted_time_var, bg="#1a1a1a", fg="#ffffff", font=("Arial", 16)).pack(pady=(0, 30))

        self._distracted_count_var = tk.StringVar(value="Distracted Count: 0")
        tk.Label(right_frame, textvariable=self._distracted_count_var, bg="#1a1a1a", fg="#ffffff", font=("Arial", 16)).pack(pady=(0, 30))

        self._start_button = tk.Button(right_frame, text="Start", command=self._start, bg="#4CAF50", fg="white", font=("Arial", 14), width=10)
        self._start_button.pack(pady=(5))

        self._stop_button = tk.Button(right_frame, text="Stop", command=self._stop, bg="#f44336", fg="white", font=("Arial", 14), width=10)
        self._stop_button.pack(pady=(5))

        self._recalibrate_button = tk.Button(right_frame, text="Recalibrate", command=self._recalibrate, bg="#2196F3", fg="white", font=("Arial", 14), width=10)
        self._recalibrate_button.pack(pady=(5))

        self._status_label = tk.Label(right_frame, text="", bg="#1a1a1a", fg="#ffffff", font=("Arial", 14))
        self._status_label.pack(pady=(20, 0))

    def _start(self):
        if self._running:
            return
        
        cal = load_calibration_data()
        if cal is None:
            self._recalibrate()
            cal = load_calibration_data()
        
        self.neutral_yaw, self.neutral_pitch = cal["neutral_yaw"], cal["neutral_pitch"]
        self.tracker.reset()
        self._running = True

        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()
    
    def _stop(self):
        self._running = False
        self._status_var.set("Stopped")
    
    def _recalibrate(self):
        if os.path.exists("data/calibration.json"):
            os.remove("data/calibration.json")
        
        self.neutral_yaw, self.neutral_pitch = run_calibration()
        save_calibration_data({"neutral_yaw": self.neutral_yaw, "neutral_pitch": self.neutral_pitch})

    
    def _detect_loop(self):
        cap = cv2.VideoCapture(0) #open default webcam
        _, frame = cap.read() #read frame to get cam dimensions
        h, w = frame.shape[:2] #get height and width for matrix
        cam_mat = build_camera_matrix(w, h) #build camera matrix
        landmarker = build_landmarker()

        while self._running:
            ok, frame = cap.read() #read next webcam frame
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect(mp_image)

            if not results.face_landmarks:
                status = "NO_FACE"
            else:
                pitch, yaw = get_head_angles(results.face_landmarks[0], w, h, cam_mat)
                status = classify(pitch, yaw, self.neutral_pitch, self.neutral_yaw)

            self.tracker.update(status)
            self.current_frame = frame
        
        cap.release()
        landmarker.close()
    
    def _tick(self):
        if self._running:
            if self.tracker.current_state == "FOCUSED":
                self._status_var.set("FOCUSED")
                self._status_label.config(fg="#4CAF50")  # green
            else:
                self._status_var.set("LOOKING AWAY")
                self._status_label.config(fg="#f44336")  # red
            
            self._distracted_time_var.set(f"Distracted Time: {self.tracker.distracted_time:.1f}s")
            self._distracted_count_var.set(f"Distracted Count: {self.tracker.distracted_count}")

            if self.current_frame is not None:
                img = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((480, 360))
                self._photo = ImageTk.PhotoImage(img)
                self.video_canvas.create_image(0, 0, anchor="nw", image=self._photo)

        self.after(200, self._tick)  # update every 200ms    

if __name__ == "__main__":
    App().mainloop()