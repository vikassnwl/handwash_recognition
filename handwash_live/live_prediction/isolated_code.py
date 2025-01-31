import tensorflow as tf
import numpy as np
import cv2
import logging
from termcolor import colored
import pyttsx3



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



def mp_hand_detect_n_draw(frame, hands, mp_draw, mp_hands):
    results = hands.process(frame)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        # Draw hand landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame_bgr, results



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
logger.debug(colored("Loading handwash model...", "red", attrs=["bold"]))
handwash_model = tf.keras.models.load_model(
    "models/best_model_epoch_018_val_acc_0.9558.keras",
    {"preprocess_input": tf.keras.applications.mobilenet_v3.preprocess_input},
)
logger.debug(colored("Handwash model loaded successfully!", "red", attrs=["bold"]))

def predict_frame(model, frame_queue, result_queue):
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Preprocess and predict
            prediction = model.predict(preprocess_frame(frame), verbose=0)[0]
            result_queue.put(prediction)

# Helper function to preprocess frame for the CNN model
def preprocess_frame(frame):
    frame_resized = tf.keras.layers.Resizing(224, 224, crop_to_aspect_ratio=True)(
        frame
    )  # Adjust to your model's input size
    return np.expand_dims(frame_resized, axis=0)




logger.debug(colored("Initializing speech engine...", "blue", attrs=["bold"]))
engine = pyttsx3.init()
logger.debug(colored("Speech engine initialized successfully!", "blue", attrs=["bold"]))

def speak(sentence, speech_rate=200):
    engine.setProperty("rate", speech_rate)
    engine.say(sentence)
    engine.runAndWait()