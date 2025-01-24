import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from tensorflow.keras.preprocessing import image

# Load the saved model
loaded_model = load_model("my_model.keras")

# Initialize the MTCNN detector
mtcnn = MTCNN()

# Function to process video and predict
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    frame_count = 0  # Initialize frame count

    while cap.isOpened() and frame_count < 4:  # Process only first four frames
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame using MTCNN
        faces = mtcnn.detect_faces(frame)

        # Process each detected face
        for face in faces:
            # Extract face bounding box
            x, y, w, h = face['box']
            x1, y1 = x + w, y + h

            # Crop and align the face region
            face_img = frame[y:y1, x:x1]
            face_img = cv2.resize(face_img, (64, 64))  # Resize to match model input size

            # Preprocess the cropped face image
            img_array = image.img_to_array(face_img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Rescale to [0, 1]

            # Make prediction
            prediction = np.argmax(loaded_model.predict(img_array))
            predictions.append(prediction)

            # Display box around face with prediction result
            #if prediction == 0:
            if final_prediction == "Real":
                cv2.putText(frame, "Fake", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)  # Red box for "Fake"
            else:
                cv2.putText(frame, "Real", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)  # Green box for "Real"

        frame_count += 1  # Increment frame count

    cap.release()
    cv2.destroyAllWindows()

    # Display final result
    final_prediction = "Real" if predictions.count(1) > predictions.count(0) else "Fake"
    print(f"Final Prediction: {final_prediction}")

    # Display final image
    cv2.imshow("Final Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






# Process the video and predict
video_path = r"C:\Users\91911\Desktop\sam.mp4"
# video_path = r"C:\Users\91911\Desktop\sejith.mp4"
process_video(video_path)
