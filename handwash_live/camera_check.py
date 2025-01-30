import cv2
import sys


source = int(sys.argv[1]) if sys.argv[1].isdigit() else sys.argv[1]
cap = cv2.VideoCapture(source)
while True:
    ret, frame = cap.read()
    if not ret: break

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release()
cv2.destroyAllWindows()