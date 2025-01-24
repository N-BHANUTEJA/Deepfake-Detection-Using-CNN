import os
import cv2
import csv
from mtcnn import MTCNN

# Function to extract frames from videos and organize data
def preprocess_videos(video_folder, csv_file, output_folder, frames_per_video):
    # Read CSV file to get video names and labels
    video_data = []
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        for row in csvreader:
            video_name, label = row
            video_data.append((video_name, label))

    # Create output folders for real and fake videos
    os.makedirs(os.path.join(output_folder, 'real', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'fake', 'images'), exist_ok=True)

    # Indicate the start of the preprocessing process
    print("Preprocessing videos...")

    # Count the total number of videos
    total_videos = len(video_data)

    # Initialize counter for extracted videos
    extracted_videos = 0

    # Create MTCNN detector instance
    detector = MTCNN()

    # Iterate over video data and extract frames
    for video_name, label in video_data:
        video_path = os.path.join(video_folder, video_name)
        label_folder = 'real' if label.lower() == 'real' else 'fake'
        output_path = os.path.join(output_folder, label_folder, 'images')

        # Extract frames from video
        extract_frames(video_path, output_path, frames_per_video, detector)

        # Increment the counter for extracted videos
        extracted_videos += 1

        # Calculate progress percentage
        progress_percent = (extracted_videos / total_videos) * 100

        # Print progress bar
        print(f"Progress: [{'#' * int(progress_percent / 10):<10}] {progress_percent:.2f}%", end='\r')

    # Indicate the end of the preprocessing process
    print("\nPreprocessing completed.")

# Function to extract frames from a video
def extract_frames(video_path, output_folder, frames_per_video, detector):
    # Open video capture object
    video_capture = cv2.VideoCapture(video_path)

    # Get total number of frames in video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine frame interval
    interval = max(total_frames // frames_per_video, 1)

    # Initialize frame count
    frame_count = 0

    # Read frames from the video
    while True:
        # Read a frame
        ret, frame = video_capture.read()

        # Break the loop if no frame is read
        if not ret:
            break

        # Increment frame count
        frame_count += 1

        # Extract frames at specified interval
        if frame_count % interval == 0:
            # Detect faces in the frame
            faces = detector.detect_faces(frame)

            # Filter faces based on confidence score
            for face in faces:
                if face['confidence'] > 0.5:  # Adjust confidence threshold as needed
                    x, y, width, height = face['box']
                    cropped_face = frame[y:y+height, x:x+width]
                    # Save the cropped face to the output folder
                    cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count}_face_{x}_{y}.jpg"), cropped_face)

    # Release the video capture object
    video_capture.release()

# Main function
if __name__ == "__main__":
    # Specify paths and parameters
    video_folder = r"C:\Users\91911\Desktop\H\02_Academics\04_Project\01_Project_Deepfake\02_Data\test"
    csv_file = r"C:\Users\91911\Desktop\H\02_Academics\04_Project\01_Project_Deepfake\02_Data\test_metadata.csv"
    output_folder = r"C:\Users\91911\Desktop\H\02_Academics\04_Project\01_Project_Deepfake\02_Data\testimgs"
    frames_per_video = 4  # Number of frames to extract per video

    # Preprocess videos and organize data
    preprocess_videos(video_folder, csv_file, output_folder, frames_per_video)
