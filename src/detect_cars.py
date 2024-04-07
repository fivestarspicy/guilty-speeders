import cv2

# Path to the Haar Cascade for car detection
cascade_src = 'src/cascades/cars.xml'

# Create a CascadeClassifier object for car detection
car_cascade = cv2.CascadeClassifier(cascade_src)

# Function to detect cars in a frame
def detect_cars(frame):
    cars = car_cascade.detectMultiScale(frame, 1.1, 2)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def main():
    # Path to your video file or replace with 0 for webcam
    cap = cv2.VideoCapture('videos/test_video.mov')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect cars in the frame
        frame_with_detections = detect_cars(frame)
        cv2.imshow('frame', frame_with_detections)

        if cv2.waitKey(33) == 27:  # Press Esc to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
