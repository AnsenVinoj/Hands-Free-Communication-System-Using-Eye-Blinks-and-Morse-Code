import cv2

def test_webcam():
    # 1. Initialize the webcam (0 is usually the default built-in camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    while True:
        # 2. Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        # 3. Display the resulting frame
        cv2.imshow('Webcam Test - Press Q to Exit', frame)

        # 4. Stop the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 5. Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()