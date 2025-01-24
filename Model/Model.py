import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to build the model
def build_model(input_shape=(64, 64, 3), num_classes=2):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, train_dir, validation_dir, epochs=100, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary')
    print(type(train_generator))
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary')
    
    # Correctly calculate steps_per_epoch and validation_steps
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size
    
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps)

    return history






# Build the model
model = build_model()

# Define the paths to your training and validation datasets
train_dir = r'C:\Users\91911\Desktop\H\02_Academics\04_Project\01_Project_Deepfake\02_Data\train'
validation_dir = r'C:\Users\91911\Desktop\H\02_Academics\04_Project\01_Project_Deepfake\02_Data\validation'

# Train the model
history = train_model(model, train_dir, validation_dir)


# Print the training history
# print(history.history)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Function to load and preprocess the test data
def load_test_data(test_dir, target_size=(64, 64), batch_size=32):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False) # Shuffle is set to False for evaluation
    return test_generator

# Function to evaluate the model on the test dataset
def evaluate_model(model, test_dir, target_size=(64, 64), batch_size=32):
    # Load the test data
    test_generator = load_test_data(test_dir, target_size, batch_size)
    
    # Evaluate the model
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get the true labels
    true_classes = test_generator.classes
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(true_classes, predicted_classes))
    
    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(true_classes, predicted_classes))

# Define the path to your test dataset
test_dir = r"C:\Users\91911\Desktop\H\02_Academics\04_Project\01_Project_Deepfake\02_Data\test"

# Evaluate the model
evaluate_model(model, test_dir)

# Save the model in the native Keras format
model.save("my_model.keras") 