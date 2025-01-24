import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(real_folder, fake_folder, test_size=0.2, validation_size=0.25, random_state=42):
    real_images = os.listdir(real_folder)
    fake_images = os.listdir(fake_folder)

    print("Number of real images:", len(real_images))
    print("Number of fake images:", len(fake_images))

    # Combine real and fake images
    all_images = real_images + fake_images
    all_labels = [0] * len(real_images) + [1] * len(fake_images) # 0 for real, 1 for fake

    # Split into training, validation, and testing sets
    train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=test_size, random_state=random_state)
    train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=validation_size / (1 - test_size), random_state=random_state)

    # Create directories for the split data
    base_path = r'C:\Users\91911\Desktop\H\02_Academics\04_Project\01_Project_Deepfake\02_Data'
    folders = ['train', 'validation', 'test']
    labels = ['real', 'fake']
    for folder in folders:
        for label in labels:
            os.makedirs(os.path.join(base_path, folder, label), exist_ok=True)

    # Move images to the appropriate directories
    move_images(train_images, train_labels, real_folder, fake_folder, base_path, 'train')
    move_images(validation_images, validation_labels, real_folder, fake_folder, base_path, 'validation')
    move_images(test_images, test_labels, real_folder, fake_folder, base_path, 'test')

def move_images(images, labels, real_folder, fake_folder, base_path, folder):
    for img, label in zip(images, labels):
        source_folder = real_folder if label == 0 else fake_folder
        destination_folder = os.path.join(base_path, folder, 'real' if label == 0 else 'fake')
        source_path = os.path.join(source_folder, img)
        destination_path = os.path.join(destination_folder, img)
        try:
            shutil.move(source_path, destination_path)
        except shutil.Error:
            # Handle if file already exists in the destination folder
            new_filename = img.split('.')[0] + '_1.' + img.split('.')[1]  # Appending '_1' before the extension
            shutil.move(source_path, os.path.join(destination_folder, new_filename))

# Example usage
real_folder = r'C:\Users\91911\Desktop\H\02_Academics\04_Project\01_Project_Deepfake\02_Data\CroppedFaces\real'
fake_folder = r'C:\Users\91911\Desktop\H\02_Academics\04_Project\01_Project_Deepfake\02_Data\CroppedFaces\fake'

split_dataset(real_folder, fake_folder)
