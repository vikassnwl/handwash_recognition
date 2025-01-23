import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import queue
import pyttsx3
import time
from collections import defaultdict
import mlu_tools.utils as mlutils


# Speech Engine
engine = pyttsx3.init()

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
    "models/best_model_epoch_010_val_acc_0.9591.keras",
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
# cap = cv2.VideoCapture('http://192.168.43.110:4747/video')
# cap = cv2.VideoCapture('http://192.168.43.135:4747/video')
cap = cv2.VideoCapture(0)
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
should_speak = True
freq_dict = defaultdict(int)
total_frames_processed = 0


# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
# Create VideoWriter object
save_as = "collected_data/"+mlutils.get_datetime_str()+".mp4"
out = cv2.VideoWriter(save_as, fourcc, fps, frame_size)


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

        if results.multi_hand_landmarks:
            # Hand detected

            last_hand_detected_time = time.time()

            if should_speak:
                engine.say("Session initiated.")
                engine.runAndWait()
                engine.setProperty("rate", 170)
                engine.say(f"Rub both wrists in rotating manner for {STEP_0_DUR} seconds.")
                engine.runAndWait()
                should_speak = False
                step_0_start_time = time.time()
                freq_dict = defaultdict(int)
                total_frames_processed = 0

            if not frame_queue.full():
                frame_queue.put(rgb_frame)

            if not result_queue.empty():
                prediction = result_queue.get()
                predicted_label = np.argmax(prediction)

                freq_dict[predicted_label] += 1
                total_frames_processed += 1
                if time.time()-step_0_start_time > (STEP_0_DUR+DELAY_COMPENSATION):
                    if freq_dict[0]/total_frames_processed >= STEP_0_MIN_THRESHOLD:
                        engine.setProperty("rate", 155)
                        engine.say("Great job.")
                        engine.runAndWait()
                    else:
                        engine.setProperty("rate", 155)
                        engine.say("Try again.")
                        engine.runAndWait()

                    engine.setProperty("rate", 170)
                    engine.say(f"Rub both wrists in rotating manner for {STEP_0_DUR} seconds.")
                    engine.runAndWait()

                    # Get video properties
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_size = (width, height)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
                    # Create VideoWriter object
                    save_as = "collected_data/"+mlutils.get_datetime_str()+".mp4"
                    out = cv2.VideoWriter(save_as, fourcc, fps, frame_size)

                    step_0_start_time = time.time()

                predicted_class = labels_dict[predicted_label]
                predicted_conf = f"{prediction[predicted_label]:.2f}"

            # Draw hand landmarks on the frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        else:
            # enable speak after `SESSION_TIMEOUT` for the next session
            if time.time() - last_hand_detected_time > SESSION_TIMEOUT:
                should_speak = True

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