# Import required libraries
import cv2
import sys
import mediapipe as mp
import numpy as np
import threading
import queue
import logging
from termcolor import colored
import time
from collections import defaultdict
from isolated_code import mp_hand_detect_n_draw, predict_frame, speak, handwash_model, labels_dict



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



# Queues for frames and predictions
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

# Start the prediction thread
threading.Thread(
    target=predict_frame, args=(handwash_model, frame_queue, result_queue), daemon=True
).start()

# Parameters for text overlay
predicted_class = "initial text1"
predicted_conf = "initial text2"

# Parameters for handwash session management
SESSION_TIMEOUT = 5  # seconds
STEP_DUR = 10
DELAY_COMPENSATION = 5
STEP_0_MIN_THRESHOLD = .5  # fraction of frames detected as step 0
curr_step_lbl = 0
is_session_active = False
last_hand_detected_time = None
freq_dict = defaultdict(int)
total_frames_processed = 0


# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils



# Capture camera feed
source = int(sys.argv[1]) if sys.argv[1].isdigit() else sys.argv[1]
cap = cv2.VideoCapture(source)
while True:
    ret, frame = cap.read()
    if not ret: break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Mediapipe requires rgb images so the handwash model does

    # Mediapipe hand landmarks detection and drawing on the frame
    frame, results = mp_hand_detect_n_draw(rgb_frame, hands, mp_draw, mp_hands)

    if results.multi_hand_landmarks:
        if not is_session_active:
            speak("Session initiated.")
            speak(f"{labels_dict[curr_step_lbl]} for {STEP_DUR} seconds.", 160)
            curr_step_start_time = time.time()
            is_session_active = True
        last_hand_detected_time = time.time()

    # A handwash session is initiated as soon as the mediapipe detects a hand.
    # A handwash session is terminated if no hands were detected within the session timeout period. 
    # Following is the check whether a session is active.
    if (last_hand_detected_time and ((time.time() - last_hand_detected_time) <= SESSION_TIMEOUT)):

        if not frame_queue.full():
            frame_queue.put(rgb_frame)

        if not result_queue.empty():
            prediction = result_queue.get()
            predicted_label = np.argmax(prediction)
            predicted_class = labels_dict[predicted_label]
            predicted_conf = f"{prediction[predicted_label]:.2f}"

            freq_dict[predicted_label] += 1
            total_frames_processed += 1
            if time.time()-curr_step_start_time > (STEP_DUR+DELAY_COMPENSATION):
                if freq_dict[curr_step_lbl]/total_frames_processed >= STEP_0_MIN_THRESHOLD:
                    speak("Great job.", 155)
                    curr_step_lbl += 1
                    if curr_step_lbl > 2:
                        speak("Congratulations! You have successfully completed all the steps.")
                        exit()
                    speak(f"{labels_dict[curr_step_lbl]} for {STEP_DUR} seconds.", 160)
                else:
                    speak("Try again.", 155)
                    logger.debug(colored(freq_dict, "blue", attrs=["bold"]))
                curr_step_start_time = time.time()
                freq_dict = defaultdict(int)
                total_frames_processed = 0

        # Placing predicted handwash step name and confidence score on the frame
        cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, predicted_conf, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif last_hand_detected_time and ((time.time() - last_hand_detected_time) > SESSION_TIMEOUT):
        # Terminating session if no hands were detected within the session timeout period. 
        is_session_active = False

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release()
cv2.destroyAllWindows()