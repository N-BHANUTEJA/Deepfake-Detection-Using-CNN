import os
from tensorflow.keras.models import load_model

# Print the absolute path to the model file
model_file = "my_model.keras"
absolute_path = os.path.abspath(model_file)
print("Absolute path to model file:", absolute_path)

# Load the model using the absolute path
loaded_model = load_model(model_file)
