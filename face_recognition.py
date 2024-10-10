import cv2

# Function for Face and Eye Detection
def detect_and_draw(img, face_cascade, eye_cascade, scale=1.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale for face detection
    gray = cv2.equalizeHist(gray)  # Histogram equalization to improve contrast

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces and search for eyes only within each face
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Define the region of interest (ROI) for eyes, which is inside the face region
        face_roi_gray = gray[y:y + h, x:x + w]
        face_roi_color = img[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

        for (ex, ey, ew, eh) in eyes:
            # Draw circles around detected eyes within the face
            center = (x + ex + ew // 2, y + ey + eh // 2)
            radius = int((ew + eh) * 0.25)
            cv2.circle(face_roi_color, center, radius, (0, 255, 0), 2)

    # Display the processed image with detected faces and eyes
    cv2.imshow("Face and Eye Detection", img)

def main():
    # Load pre-trained classifiers from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Capture video from the webcam (change '0' to a video file path if needed)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Face Detection Started.... Press 'q' to quit.")

    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        detect_and_draw(frame, face_cascade, eye_cascade)  # Detect faces and draw

        # Break loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
