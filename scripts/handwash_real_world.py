import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import queue
import pyttsx3
import time


# Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

labels_dict = {0: "Rub both wrists in rotating manner",
            1: "Rub your palms together",
            2: "Rub the back of your fingers and hands",
            3: "Rub your hands by interlocking your fingers",
            4: "Interlock fingers and rub the back of fingers of both hands",
            5: "Rub the area between index finger and thumb",
            6: "Rub fingertips on palm of both hands in circular manner"}

# Load your handwash classification model
handwash_model = tf.keras.models.load_model(
    'models/best_model_epoch_010_val_acc_0.9591.keras',
    {"preprocess_input": tf.keras.applications.mobilenet_v3.preprocess_input}
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
threading.Thread(target=predict_frame, args=(handwash_model, frame_queue, result_queue), daemon=True).start()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Parameters for hand detection timeout
HAND_DETECTION_TIMEOUT = 5  # seconds
last_hand_detected_time = None

# Initialize video capture
# cap = cv2.VideoCapture('http://192.168.43.130:4747/video')
# cap = cv2.VideoCapture('http://192.168.43.135:4747/video')
cap = cv2.VideoCapture(0)

# Helper function to preprocess frame for the CNN model
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Adjust to your model's input size
    frame_normalized = frame_resized / 255.0  # Normalize pixel values
    return np.expand_dims(frame_normalized, axis=0)

# Variables
predicted_class = "initial text1"
predicted_conf = "initial text2"
should_speak_hand_detected = True
speak_patience = 3  # Seconds
last_hand_detection_time = time.time()

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

            last_hand_detection_time = time.time()

            if should_speak_hand_detected:
                engine.say("Hand detected")
                # engine.say("Hand detected. Rub both wrists in rotating manner.")
                engine.runAndWait()
                should_speak_hand_detected = False

            if not frame_queue.full():
                frame_queue.put(frame)

            if not result_queue.empty():
                prediction = result_queue.get()
                predicted_label = np.argmax(prediction)
                predicted_class = labels_dict[predicted_label]
                predicted_conf = f"{prediction[predicted_label]:.2f}"

            cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, predicted_conf, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks on the frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        else:
            # enable speak after `speak_patience` seconds of non-hand detection
            if time.time()-last_hand_detection_time > speak_patience:
                should_speak_hand_detected = True

        # Display the frame
        cv2.imshow('Handwash Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()