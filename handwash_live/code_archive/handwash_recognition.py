import cv2
import numpy as np
import tensorflow as tf
import threading
import queue


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

# Initialize video capture
cap = cv2.VideoCapture(0)

# Helper function to preprocess frame for the CNN model
def preprocess_frame(frame):
    frame_resized = tf.keras.layers.Resizing(224, 224, crop_to_aspect_ratio=True)(
        frame
    )  # Adjust to your model's input size
    return np.expand_dims(frame_resized, axis=0)


# Variables
predicted_class = "initial text1"
predicted_conf = "initial text2"

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

        if not frame_queue.full():
            frame_queue.put(rgb_frame)

        if not result_queue.empty():
            prediction = result_queue.get()
            predicted_label = np.argmax(prediction)
            predicted_class = labels_dict[predicted_label]
            predicted_conf = f"{prediction[predicted_label]:.2f}"

        cv2.putText(
            frame,
            predicted_class,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            predicted_conf,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Display the frame
        cv2.imshow("Handwash Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
