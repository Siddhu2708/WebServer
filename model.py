from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np

# Load your trained model
model = load_model('uploads/tomato_disease_model.h5')

def predict_disease(image_path):
    img = image.load_img(image_path, target_size=(128, 128))  # Adjust size according to your model
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    
    # Map class_idx to a human-readable label
    labels = {0: 'Healthy', 1: 'Early Blight', 2: 'Late Blight'}
    return labels[class_idx]
