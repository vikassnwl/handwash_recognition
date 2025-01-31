import mlu_tools.utils as mlutils
import cv2
import pyttsx3
import time
import os
import sys


def main(source):
    engine = pyttsx3.init()

    def speak(rate, text):
        engine.setProperty("rate", rate)
        engine.say(text)
        engine.runAndWait()

    collected_data_dir = "handwash_live/data_collection_n_preparation/dataset/collected_data"
    os.makedirs(collected_data_dir, exist_ok=True)

    SAVED_VIDEO_DURATION = 5
    RECORDING_DELAY = 5

    cap = cv2.VideoCapture(source)
    # cap = cv2.VideoCapture(0)
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4

    # Create VideoWriter object
    save_as = f"{collected_data_dir}/{mlutils.get_datetime_str()}.mp4"
    out = cv2.VideoWriter(save_as, fourcc, fps, frame_size)

    speak(rate=195, text=f"Recording starting in {RECORDING_DELAY} seconds.")
    st = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        cv2.imshow("Frame", frame)

        if (time.time()-st) > RECORDING_DELAY:
            if (time.time()-st) <= (RECORDING_DELAY+SAVED_VIDEO_DURATION):
                out.write(frame)
            else:
                out.release()
                # Create VideoWriter object
                save_as = f"{collected_data_dir}/{mlutils.get_datetime_str()}.mp4"
                out = cv2.VideoWriter(save_as, fourcc, fps, frame_size)
                speak(rate=195, text=f"Recording starting in {RECORDING_DELAY} seconds.")
                st = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


source = int(sys.argv[1]) if sys.argv[1].isdigit() else sys.argv[1]
if len(sys.argv) > 1 and sys.argv[1] == "testing":
    cap = cv2.VideoCapture(source)
    while True:
        ret, frame = cap.read()
        if not ret: break

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release()
    cv2.destroyAllWindows()
else:
    main(source)