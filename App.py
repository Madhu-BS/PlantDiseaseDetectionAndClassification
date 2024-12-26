# Updated App.py
import os
import time
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from threading import Thread

# Initialize Flask app
app = Flask(__name__)

# Define constants
IMAGE_SIZE = 224  # Adjust based on your model's input size
ALLOWED_MODELS = ['sugarcane', 'paddy', 'areca', 'kaggle']
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model_by_name(model_name):
    model_paths = {
        'sugarcane': './models/Sugarcane.keras',
        'paddy': './models/Paddy.keras',
        'areca': './models/Areca.keras',
        'kaggle': './models/Kaggle.keras'
    }
    model_path = model_paths.get(model_name)
    if model_path and os.path.exists(model_path):
        return load_model(model_path)
    else:
        raise ValueError("Model not found")

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def simulate_long_task(progress_callback):
    for i in range(1, 101):
        time.sleep(0.05)  # Simulating a task
        progress_callback(i)  # Update progress

@app.route('/')
def index():
    return render_template('index.html', models=ALLOWED_MODELS)

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form.get('model_name')
    if model_name not in ALLOWED_MODELS:
        return jsonify({"error": "Invalid model selected"}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            model = load_model_by_name(model_name)
        except ValueError as e:
            return jsonify({"error": str(e)}), 500

        img_array = prepare_image(filepath)

        # Prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)

        class_labels = {
            'sugarcane': ["Healthy", "Unhealthy"],
            'paddy': ["Healthy", "Unhealthy"],
            'areca': ["Healthy", "Unhealthy"],
            'kaggle': ["Healthy", "Rust", "Scab"]
        }

        predicted_label = class_labels[model_name][predicted_class[0]]

        return jsonify({
            'redirect_url': url_for('result', label=predicted_label, image_file=filename, model_name=model_name)
        })

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/result')
def result():
    label = request.args.get('label')
    image_file = request.args.get('image_file')
    model_name = request.args.get('model_name')

    return render_template('result.html', label=label, image_file=os.path.join('uploads/', image_file), model_name=model_name)


if __name__ == '__main__':
    app.run(debug=True)
