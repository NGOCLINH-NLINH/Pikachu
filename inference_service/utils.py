import cv2
import numpy as np

pts = []


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        x_real = int(x / scale)
        y_real = int(y / scale)
        pts.append((x_real, y_real))
        print(f"Point: ({x_real}, {y_real})")


cap = cv2.VideoCapture("data/vehicles.mp4")
ret, frame = cap.read()
if not ret:
    print("Không thể đọc frame, kiểm tra đường dẫn video.")
    exit()

scale = 0.5
frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)

cv2.imshow("frame", frame_resized)
cv2.setMouseCallback("frame", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
print("SOURCE =", np.array(pts))
