
import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import time
from pathlib import Path
from PIL import Image, ImageTk
import pyautogui


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

VIDEO_WIDTH = 660
VIDEO_HEIGHT = 300


# ==========================================
# 2. MEDIAPIPE CONFIGURATION
# ==========================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode


FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]


# ==========================================
# 3. GESTURE RECOGNITION
# ==========================================

def recognize_gesture(landmarks):

    thumb_up = landmarks[4].x < landmarks[3].x

    fingers = []

    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):

        fingers.append(
            landmarks[tip].y < landmarks[pip].y
        )

    if all(fingers) and thumb_up:

        return "OPEN PALM"

    elif not any(fingers) and not thumb_up:

        return "FIST"

    elif fingers[0] and not any(fingers[1:]) and not thumb_up:

        return "INDEX FINGER"

    else:

        return "UNKNOWN GESTURE"


# ==========================================
# 4. MAIN DASHBOARD
# ==========================================

class GestureController:

    def __init__(self, root):

        self.root = root

        self.cap = None
        self.detector = None

        self.running = False

        self.last_timestamp = 0

        self.last_action_time = 0
        self.last_command_gesture = None
        self.cooldown = 1.2
        # Window configuration

        root.title("AI Gesture Presentation Controller")

        root.geometry("900x700")

        root.resizable(True, True)

        root.configure(bg="#101827")

        # ==================================
        # HEADING
        # ==================================

        heading = tk.Label(
            root,
            text="AI GESTURE PRESENTATION CONTROLLER",
            font=("Arial", 21, "bold"),
            bg="#101827",
            fg="white"
        )

        heading.pack(pady=(15, 5))

        subtitle = tk.Label(
            root,
            text="Control your presentation using hand gestures",
            font=("Arial", 12),
            bg="#101827",
            fg="#AAB7CF"
        )

        subtitle.pack(pady=5)

        # ==================================
        # CAMERA PREVIEW FRAME
        # ==================================

        camera_frame = tk.Frame(
            root,
            bg="#26344F",
            width=680,
            height=320
        )

        camera_frame.pack(pady=15)

        # Prevent frame from shrinking

        camera_frame.pack_propagate(False)

        self.video_label = tk.Label(
            camera_frame,
            text="CAMERA IS CURRENTLY OFF",
            font=("Arial", 17, "bold"),
            bg="#26344F",
            fg="white"
        )

        self.video_label.pack(
            fill="both",
            expand=True
        )

        # ==================================
        # GESTURE STATUS
        # ==================================

        self.gesture_label = tk.Label(
            root,
            text="Detected Gesture: None",
            font=("Arial", 16, "bold"),
            bg="#101827",
            fg="#38D9A9"
        )

        self.gesture_label.pack(pady=6)

        # ==================================
        # CAMERA STATUS
        # ==================================

        self.status_label = tk.Label(
            root,
            text="Camera Status: OFF",
            font=("Arial", 12, "bold"),
            bg="#101827",
            fg="#FF7777"
        )

        self.status_label.pack(pady=4)

        # ==================================
        # CONTROL BUTTONS
        # ==================================

        button_frame = tk.Frame(
            root,
            bg="#101827"
        )

        button_frame.pack(pady=12)

        self.start_button = tk.Button(
            button_frame,
            text="START CAMERA",
            font=("Arial", 12, "bold"),
            bg="#16A085",
            fg="white",
            width=18,
            height=2,
            command=self.start_camera,
            cursor="hand2"
        )

        self.start_button.grid(
            row=0,
            column=0,
            padx=12
        )

        self.stop_button = tk.Button(
            button_frame,
            text="STOP CAMERA",
            font=("Arial", 12, "bold"),
            bg="#C0392B",
            fg="white",
            width=18,
            height=2,
            command=self.stop_camera,
            state=tk.DISABLED,
            cursor="hand2"
        )

        self.stop_button.grid(
            row=0,
            column=1,
            padx=12
        )

        # ==================================
        # GESTURE INSTRUCTIONS
        # ==================================

        instructions = tk.Label(
            root,
            text=(
                "OPEN PALM = NEXT SLIDE     |     "
                "FIST = PREVIOUS SLIDE     |     "
                "INDEX FINGER = PAUSE"
            ),
            font=("Arial", 10, "bold"),
            bg="#101827",
            fg="white"
        )

        instructions.pack(pady=10)

        # ==================================
        # CLOSE WINDOW EVENT
        # ==================================

        root.protocol(
            "WM_DELETE_WINDOW",
            self.close_app
        )

    # ======================================
    # 5. START CAMERA
    # ======================================

    def start_camera(self):

        if self.running:
            return

        if not MODEL_PATH.exists():

            messagebox.showerror(
                "Model Missing",
                "hand_landmarker.task was not found in the models folder."
            )

            return

        try:

            self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():

                raise RuntimeError(
                    "Could not open webcam. Check camera permissions."
                )

            options = HandLandmarkerOptions(

                base_options=BaseOptions(
                    model_asset_path=str(MODEL_PATH)
                ),

                running_mode=VisionRunningMode.VIDEO,

                num_hands=1,

                min_hand_detection_confidence=0.7,

                min_hand_presence_confidence=0.7,

                min_tracking_confidence=0.7
            )

            self.detector = (
                HandLandmarker.create_from_options(options)
            )

            self.running = True

            self.last_timestamp = 0

            self.start_button.config(
                state=tk.DISABLED
            )

            self.stop_button.config(
                state=tk.NORMAL
            )

            self.status_label.config(
                text="Camera Status: ACTIVE",
                fg="#38D9A9"
            )

            self.update_camera()

        except Exception as error:

            self.stop_camera()

            messagebox.showerror(
                "Camera Error",
                str(error)
            )

    # ======================================
    # 6. CAMERA UPDATE AND HAND DETECTION
    # ======================================

    def update_camera(self):

        if not self.running:
            return

        success, frame = self.cap.read()

        if not success:

            self.stop_camera()

            messagebox.showerror(
                "Camera Error",
                "Unable to read webcam."
            )

            return

        # Mirror camera

        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert to MediaPipe image

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Generate timestamp

        now = int(time.monotonic() * 1000)

        self.last_timestamp = max(
            now,
            self.last_timestamp + 1
        )

        # Detect hands

        results = self.detector.detect_for_video(
            mp_image,
            self.last_timestamp
        )

        gesture = "NO HAND DETECTED"

        if results.hand_landmarks:

            for hand_landmarks in results.hand_landmarks:

                gesture = recognize_gesture(
                    hand_landmarks
                )

                height, width, _ = frame.shape

                # Draw green hand landmarks

                for point in hand_landmarks:

                    x = int(point.x * width)

                    y = int(point.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

        # Show gesture on video

        cv2.putText(
            frame,
            gesture,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # ==================================
        # POWERPOINT GESTURE CONTROL
        # ==================================

        current_time = time.monotonic()

        if gesture in (
            "NO HAND DETECTED",
            "UNKNOWN GESTURE",
            "INDEX FINGER"
        ):

            self.last_command_gesture = None

        elif (
            gesture != self.last_command_gesture
            and current_time - self.last_action_time >= self.cooldown
        ):

            if gesture == "OPEN PALM":

                pyautogui.press("right")

                print("NEXT SLIDE")

            elif gesture == "FIST":

                pyautogui.press("left")

                print("PREVIOUS SLIDE")

            self.last_action_time = current_time

            self.last_command_gesture = gesture


        # Update gesture status

        self.gesture_label.config(
            text="Detected Gesture: " + gesture
        )

        # ==================================
        # LARGE CAMERA DISPLAY
        # ==================================

        display_frame = cv2.resize(
            frame,
            (VIDEO_WIDTH, VIDEO_HEIGHT)
        )

        image = Image.fromarray(
            cv2.cvtColor(
                display_frame,
                cv2.COLOR_BGR2RGB
            )
        )

        photo = ImageTk.PhotoImage(
            image=image
        )

        self.video_label.config(
            image=photo,
            text=""
        )

        # Keep image reference

        self.video_label.image = photo

        # Update every 20 milliseconds

        self.root.after(
            20,
            self.update_camera
        )

    # ======================================
    # 7. STOP CAMERA
    # ======================================

    def stop_camera(self):

        self.running = False

        if self.cap is not None:

            self.cap.release()

            self.cap = None

        if self.detector is not None:

            self.detector.close()

            self.detector = None

        self.video_label.config(
            image="",
            text="CAMERA IS CURRENTLY OFF"
        )

        self.video_label.image = None

        self.gesture_label.config(
            text="Detected Gesture: None"
        )

        self.status_label.config(
            text="Camera Status: OFF",
            fg="#FF7777"
        )

        self.start_button.config(
            state=tk.NORMAL
        )

        self.stop_button.config(
            state=tk.DISABLED
        )

    # ======================================
    # 8. CLOSE APPLICATION
    # ======================================

    def close_app(self):

        self.stop_camera()

        self.root.destroy()


# ==========================================
# 9. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    root = tk.Tk()

    app = GestureController(root)

    root.mainloop()
