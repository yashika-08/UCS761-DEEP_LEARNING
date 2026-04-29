import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/resnet50_defect.h5")  # change if needed

IMG_SIZE = (224, 224)

# Class labels (IMPORTANT: must match training order)
CLASS_NAMES = [
    'crazing',
    'inclusion',
    'patches',
    'pitted_surface',
    'rolled-in_scale',
    'scratches'
]

# LOAD MODEL
model = load_model(MODEL_PATH)

# PREDICT FUNCTION
def predict(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds)

    return CLASS_NAMES[class_idx]

# RUN
if __name__ == "__main__":
    img_path = input("Enter image path: ")
    result = predict(img_path)
    print("Predicted Class:", result)