import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import queue
import sys
sys.path.append("handwash_live")
from utils import speak
import time
from collections import defaultdict
from termcolor import colored
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



labels_dict = {
    0: "Rub both wrists in rotating manner",
    1: "Rub your palms together",
    2: "Rub the back of your fingers and hands",
    3: "Rub your hands by interlocking your fingers",
    4: "Interlock fingers and rub the back of fingers of both hands",
    5: "Rub the area between index finger and thumb",
    6: "Rub fingertips on palm of both hands in circular manner",
}

# Load your handwash classification model
handwash_model = tf.keras.models.load_model(
    "models/best_model_epoch_018_val_acc_0.9558.keras",
    {"preprocess_input": tf.keras.applications.mobilenet_v3.preprocess_input},
)

def predict_frame(model, frame_queue, result_queue):
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Preprocess and predict
            prediction = model.predict(preprocess_frame(frame), verbose=0)[0]
            result_queue.put(prediction)

# Queues for frames and predictions
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

# Start the prediction thread
threading.Thread(
    target=predict_frame, args=(handwash_model, frame_queue, result_queue), daemon=True
).start()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture
source = int(sys.argv[1]) if sys.argv[1].isdigit() else sys.argv[1]
cap = cv2.VideoCapture(source)
# cap = cv2.VideoCapture('http://192.168.43.135:4747/video')
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("handwash_live/sample_videos/recorded videos/VID_20250117_132117.mp4")

# Helper function to preprocess frame for the CNN model
def preprocess_frame(frame):
    frame_resized = tf.keras.layers.Resizing(224, 224, crop_to_aspect_ratio=True)(
        frame
    )  # Adjust to your model's input size
    return np.expand_dims(frame_resized, axis=0)

# Parameters for text overlay
predicted_class = "initial text1"
predicted_conf = "initial text2"

# Parameters for handwash session management
SESSION_TIMEOUT = 5  # seconds
STEP_0_DUR = 10
DELAY_COMPENSATION = 5
STEP_0_MIN_THRESHOLD = .5  # fraction of frames detected as step 0
last_hand_detected_time = time.time()
is_session_initiated = True
freq_dict = defaultdict(int)
total_frames_processed = 0
are_session_vars_initialized = False


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Flip frame for a mirrored view (optional)
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB (Mediapipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks or time.time() - last_hand_detected_time <= SESSION_TIMEOUT:
            # Hand detected
            # if results.multi_hand_landmarks:
            #     logger.debug(colored("Hands detected!", "red", attrs=["bold"]))

            if results.multi_hand_landmarks:
                last_hand_detected_time = time.time()


            # Perform operations within current session
            if is_session_initiated:

                if not are_session_vars_initialized:
                    speak("Session initiated.")
                    speak(f"{labels_dict[1]} for {STEP_0_DUR} seconds.", 160)
                    are_session_vars_initialized = True
                    step_0_start_time = time.time()
                    freq_dict = defaultdict(int)
                    total_frames_processed = 0

                if not frame_queue.full():
                    frame_queue.put(rgb_frame)

                if not result_queue.empty():
                    prediction = result_queue.get()
                    predicted_label = np.argmax(prediction)
                    predicted_class = labels_dict[predicted_label]
                    predicted_conf = f"{prediction[predicted_label]:.2f}"

                    freq_dict[predicted_label] += 1
                    total_frames_processed += 1
                    if time.time()-step_0_start_time > (STEP_0_DUR+DELAY_COMPENSATION):
                        if freq_dict[1]/total_frames_processed >= STEP_0_MIN_THRESHOLD:
                            speak("Great job.", 155)
                        else:
                            speak("Try again.", 155)
                            logger.debug(colored(freq_dict, "blue", attrs=["bold"]))
                            step_0_start_time = time.time()
                        freq_dict = defaultdict(int)
                        total_frames_processed = 0

                if results.multi_hand_landmarks:
                    # Draw hand landmarks on the frame
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        else:
            # enable speak after `SESSION_TIMEOUT` for the next session
            if time.time() - last_hand_detected_time > SESSION_TIMEOUT:
                is_session_initiated = True

        cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, predicted_conf, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Handwash Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    print(f"{freq_dict=}")