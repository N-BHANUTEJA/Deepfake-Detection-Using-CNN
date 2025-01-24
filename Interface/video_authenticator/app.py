from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from tensorflow.keras.preprocessing import image
import base64
import os


app = Flask(__name__)

# Load the saved model
loaded_model = load_model("C:/Users/BHANUTEJA/Desktop/DeepfakeDetection1/DeepfakeDetection/Interface/video_authenticator/my_model.keras")

# Initialize the MTCNN detector
mtcnn = MTCNN()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    frames = [] # To store split frame images
    cropped_faces = [] # To store cropped face images

    frame_count = 0 # Initialize frame count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count > 40: # Skip frames if not every 40th frame
            break

        # Store the original frame for display
        frames.append(frame)

        # Detect faces in the frame using MTCNN
        faces = mtcnn.detect_faces(frame)

        # Process each detected face
        for face in faces:
            # Extract face bounding box
            x, y, w, h = face['box']
            x1, y1 = x + w, y + h

            # Crop and align the face region
            face_img = frame[y:y1, x:x1]
            face_img = cv2.resize(face_img, (64, 64)) # Resize to match model input size

            cropped_faces.append(face_img) # Store cropped face image

            # Preprocess the cropped face image
            img_array = image.img_to_array(face_img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0 # Rescale to [0, 1]

            # Make prediction
            prediction = np.argmax(loaded_model.predict(img_array))
            predictions.append(prediction)

    cap.release()

    # Display final result
    final_prediction = "Real" if predictions.count(1) > predictions.count(0) else "Fake"
    # Select the last frame for display
    final_frame = frames[0]

    # Draw a box around the face in the final frame
    for face in mtcnn.detect_faces(final_frame):
        x, y, w, h = face['box']
        x1, y1 = x + w, y + h
        if final_prediction == "Real": # Real face
            cv2.rectangle(final_frame, (x, y), (x1, y1), (0, 255, 0), 2) # Green box
        else: # Fake face
            cv2.rectangle(final_frame, (x, y), (x1, y1), (0, 0, 255), 2) # Red box

    return final_prediction, final_frame, frames, cropped_faces

# Custom filter for base64 encoding
def b64encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Register the custom filter in Jinja environment
app.jinja_env.filters['b64encode'] = b64encode_image

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No file uploaded', 400

    video = request.files['video']
    video.save('uploaded_video.mp4') # Save the uploaded video to disk
    prediction, final_frame, frames, cropped_faces = process_video('uploaded_video.mp4')
    return render_template('output.html', prediction=prediction, final_frame=final_frame, frames=frames, cropped_faces=cropped_faces)

if __name__ == '__main__':
    app.run(debug=True)
