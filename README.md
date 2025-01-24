# Deepfake Detection Project

## Overview 🧠
This project focuses on detecting deepfake media using Convolutional Neural Networks (CNN). 
With the increasing use of AI-generated content, the project aims to build an effective solution 
for identifying manipulated videos and images. Additionally, a web interface is designed using Flask 
to upload videos and determine whether the content is real or fake.

---

## Features ✨
- **Deep Learning Model**: Built using CNN for detecting deepfake content.
- **Video Upload Interface**: A user-friendly web interface to upload videos for analysis.
- **Automated Workflow**: From data preprocessing to model evaluation.

---

## Technologies Used 🔧
- **Python** 🐍
- **OpenCV** (cv2) for image and video processing.
- **TensorFlow** for deep learning model creation.
- **Flask** for building the web application.
- **NumPy & Pandas** for data manipulation and preprocessing.
- **Matplotlib & Seaborn** for visualizing results.

---

## Workflow 🛠️

### 1. Data Collection 📊
- Gathered a dataset of real and deepfake videos/images from reliable sources.
- Ensured the dataset includes diverse scenarios to improve model robustness.

### 2. Data Preprocessing 🧹
- **Frame Extraction**: Extracted frames from videos using OpenCV.
- **Image Resizing**: Standardized all frames to the required input size for CNN.
- **Normalization**: Scaled pixel values to enhance model performance.
- **Label Encoding**: Assigned labels (Real/Fake) to the dataset.

### 🧠 Model Architecture & Training 🏗️
- **CNN Architecture**: The model uses multiple convolutional layers to extract features from video frames and images, 
    followed by pooling layers and fully connected layers for classification.

- **Activation Functions & Loss**: ReLU is used in convolutional layers for non-linearity,
   and Softmax in the output layer for classification. Categorical Crossentropy is used as the loss function.

- **Optimizer & Training**: Adam optimizer is used for efficient training, with accuracy as the evaluation metric.

### 4. Model Training ⚙️
- Split the dataset into training, validation, and testing sets.
- Used TensorFlow and Keras for implementing the CNN.
- **Augmentation**: Applied techniques like flipping, rotation, and zoom to expand the training dataset.

### 5. Model Testing 🧪
- Tested the model on unseen data to evaluate its accuracy and reliability.
- Used metrics like **accuracy**, **precision**, **recall**, and **F1-score** for performance evaluation.

### 6. Model Evaluation 📈
- Visualized training and validation loss/accuracy trends.
- Fine-tuned hyperparameters to improve the model’s accuracy.

### 7. Web Interface 🌐
- Built using **Flask** to provide a user-friendly platform.
- **Features**:
  - Video upload option.
  - Real-time detection of deepfake content.
  - Displays the analysis result (Real/Fake).

---

## How to Run 🚀

### Prerequisites:
- Python 3.x
- Install required libraries:
  ```bash
  pip install -r requirements.txt
  ```

### Steps:
1. Clone this repository:
   ```bash
   git clone https://github.com/N-Bhanuteja/deepfake-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd deepfake-detection
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Open the application in your browser at:
   `http://127.0.0.1:5000`

## Results 🏆
- Achieved high accuracy in detecting deepfake content.
- Successfully deployed a working interface for real-time detection.
- Improved detection speed and accuracy through model optimization.

## Contact 📬
For any queries or feedback, feel free to reach out:
- **Email**: bhanubhanuteja83@gmail.com
- **GitHub**: https://github.com/N-BHANUTEJA

