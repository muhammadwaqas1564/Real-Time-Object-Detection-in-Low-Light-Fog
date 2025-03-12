import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cap = cv2.VideoCapture('fogy_video.mp4')

# Create a named window that allows for resizing
cv2.namedWindow('Real-time Object Detection', cv2.WINDOW_NORMAL)

# Set the desired window size (e.g., 640x480 pixels)
cv2.resizeWindow('Real-time Object Detection', 640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    img_result = results.render()[0]
    img_bgr = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)

    # Show the real-time object detection
    cv2.imshow('Real-time Object Detection', img_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
