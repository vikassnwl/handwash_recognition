from flask import Flask
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2


app = Flask(__name__)
CORS(app)

def load_model():
    print("Loading the model...")
    global model
    model = tf.keras.models.load_model("models/downloaded_model.keras")
    print("model loaded successfully!")

load_model()

@app.route('/predict', methods=["GET"])
def predict():
    vid_pth = "uploaded_videos/test_video.mp4"
    vidcap = cv2.VideoCapture(vid_pth)

    is_success, image = vidcap.read()
    frame_number = 0

    freq_dict = dict()

    while is_success:
        if frame_number%20 == 0:
            img = image[..., ::-1] / 255.
            img = np.expand_dims(cv2.resize(img, (256, 256)), axis=0)
            probas = model.predict(img, verbose=0)
            pred = probas.argmax()
            if pred in freq_dict:
                freq_dict[pred] += 1
            else:
                freq_dict[pred] = 1
        is_success, image = vidcap.read()
        frame_number += 1

    labels_dict = {0: "Rub both wrists in rotating manner",
                1: "Rub your palms together",
                2: "Rub the back of your fingers and hands",
                3: "Rub your hands by interlocking your fingers",
                4: "Interlock fingers and rub the back of fingers of both hands",
                5: "Rub the area between index finger and thumb",
                6: "Rub fingertips on palm of both hands in circular manner"}

    max_freq_label = max(freq_dict.items(), key=lambda x: x[1])[0]

    return f"{labels_dict[max_freq_label]}"


if __name__ == '__main__':
    app.run(debug=True)
