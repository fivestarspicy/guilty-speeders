import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("/Users/coryslater/Dropbox/Programming/speed-cam-app/guilty-speeders/src/yolov4-tiny.weights", "/Users/coryslater/Dropbox/Programming/speed-cam-app/guilty-speeders/src/yolov4-tiny.cfg")
layer_names = net.getLayerNames()

# Correct approach for getting output layers
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO class labels
with open("/Users/coryslater/Dropbox/Programming/speed-cam-app/guilty-speeders/src/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video
cap = cv2.VideoCapture('/Users/coryslater/Dropbox/Programming/speed-cam-app/guilty-speeders/videos/test_video.mov')


while True:
    _, frame = cap.read()
    height, width, _ = frame.shape

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "car":
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
