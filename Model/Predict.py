import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("my_model.keras")

# Function to preprocess a single frame
def preprocess_frame(frame):
    # Resize the frame to match the input size of the model
    resized_frame = cv2.resize(frame, (64, 64))
    # Normalize pixel values
    normalized_frame = resized_frame / 255.0
    # Expand dimensions to create a batch of size 1
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0)
    return preprocessed_frame

# Function to perform prediction on a video
def predict_video(video_path):
    # Open the video capture object
    video_capture = cv2.VideoCapture(video_path)
    
    # Initialize an empty list to store predictions for each frame
    predictions = []
    
    # Read frames from the video
    while True:
        # Read a frame
        ret, frame = video_capture.read()
        
        # Break the loop if no frame is read
        if not ret:
            break
        
        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)
        
        # Perform prediction using the model
        prediction = model.predict(preprocessed_frame)
        
        # Append the prediction to the list
        predictions.append(prediction[0])
    
    # Release the video capture object
    video_capture.release()
    
    return predictions

# Function to determine whether the video is real or fake
def classify_video(predictions, threshold=0.5):
    # Convert predictions list to numpy array
    predictions_array = np.array(predictions)
    print(predictions_array)
    # Calculate the mean prediction score
    mean_prediction_score0 = np.mean(predictions_array[:,0])
    mean_prediction_score1 = np.mean(predictions_array[:,1])
    # print(mean_prediction_score0)
    # print(mean_prediction_score1)
    # Determine the class based on the threshold
    if mean_prediction_score1 < mean_prediction_score0:
        return "Fake"
    else:
        return "Real"

# Specify the path to the video
video_path = r"D:\DFDC\aducalsrif.mp4"

# Perform prediction on the video
video_predictions = predict_video(video_path)

# Determine whether the video is real or fake
classification = classify_video(video_predictions)

# Print the classification
print("Video classification:", classification)
