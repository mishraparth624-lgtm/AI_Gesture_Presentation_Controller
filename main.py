
import cv2
import mediapipe as mp
import pyautogui
import time

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]

def recognize_gesture(landmarks):

    thumb_up = landmarks[4].x < landmarks[3].x

    fingers = []

    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        fingers.append(landmarks[tip].y < landmarks[pip].y)

    if all(fingers) and thumb_up:
        return "OPEN PALM"

    elif not any(fingers) and not thumb_up:
        return "FIST"

    elif fingers[0] and not any(fingers[1:]) and not thumb_up:
        return "INDEX FINGER"

    else:
        return "UNKNOWN"


# Start webcam
cap = cv2.VideoCapture(0)

last_action_time = 0
cooldown = 1.5

with HandLandmarker.create_from_options(options) as detector:

    timestamp = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp += 33

        results = detector.detect_for_video(
            mp_image,
            timestamp
        )

        gesture = "NO HAND DETECTED"

        if results.hand_landmarks:

            for hand_landmarks in results.hand_landmarks:

                gesture = recognize_gesture(hand_landmarks)

        current_time = time.time()

        # Control PowerPoint
        if current_time - last_action_time > cooldown:

            if gesture == "OPEN PALM":
                pyautogui.press("right")
                last_action_time = current_time

            elif gesture == "FIST":
                pyautogui.press("left")
                last_action_time = current_time

        # Display gesture
        cv2.putText(
            frame,
            gesture,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Q = EXIT",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow("AI PowerPoint Controller", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("Presentation controller stopped.")
