import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the trained model
MODEL_PATH = 'tomato_disease_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Define the class labels (Ensure this matches your model's classes)
CLASS_LABELS = {
    0: "Healthy",
    1: "Bacterial Spot",
    2: "Early Blight",
    3: "Late Blight",
    4: "Leaf Mold",
    5: "Septoria Leaf Spot",
    6: "Spider Mites (Two-Spotted Spider Mite)",
    7: "Target Spot",
    8: "Tomato Yellow Leaf Curl Virus",
    9: "Tomato Mosaic Virus",
    10: "Powdery Mildew"
}

# Configure Flask app
app = Flask(__name__)

# Upload folder configuration
UPLOAD_FOLDER = 'static/uploads'  # Directory for saving uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

# Ensure the uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess and predict the disease
def predict_disease(image_path):
    try:
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128))  # Ensure this matches model input size
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the model
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)  # Get highest probability class
        predicted_label = CLASS_LABELS.get(class_index, "Unknown Disease")

        return predicted_label
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Home page (Opens `index.html` when the user starts the app)
@app.route('/')
def home():
    return render_template('index.html')

# About page (Opens `about.html`)
@app.route('/about')
def about():
    return render_template('about.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('about.html', error="No file uploaded.")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('about.html', error="No selected file.")
    
    if not allowed_file(file.filename):
        return render_template('about.html', error="Invalid file format. Upload PNG, JPG, or JPEG.")

    # Secure the filename and save it
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Predict disease
    predicted_disease = predict_disease(filepath)

    return render_template('about.html', predicted_disease=predicted_disease, image_filename=filename)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
