from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Constants
UPLOAD_FOLDER = './static/uploads/'
MODEL_PATH = './models/best_model.h5'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
model = load_model(MODEL_PATH)

def extract_frames(video_path, output_dir):
    """Extract frames from the video."""
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(5)  # Frame rate
    count = 0
    while cap.isOpened():
        frame_id = cap.get(1)  # Current frame number
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % math.floor(frame_rate) == 0:
            scale_ratio = determine_scale_ratio(frame.shape[1])
            resized_frame = resize_frame(frame, scale_ratio)
            save_frame(resized_frame, output_dir, count)
            count += 1
    cap.release()

def detect_faces(image_dir, faces_dir):
    """Detect faces in images and crop them."""
    detector = MTCNN()
    for frame in os.listdir(image_dir):
        image_path = os.path.join(image_dir, frame)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)
        for i, result in enumerate(results):
            if len(results) < 2 or result['confidence'] > 0.95:
                crop_and_save_face(result, image, faces_dir, frame, i)

def preprocess_image(image_path, input_size=128):
    """Preprocess the image for prediction."""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(input_size, input_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        # Save the uploaded file
        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(video_path)

        # Extract frames from the video
        frames_dir = os.path.join(UPLOAD_FOLDER, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        extract_frames(video_path, frames_dir)

        # Detect faces in the frames
        faces_dir = os.path.join(UPLOAD_FOLDER, 'faces')
        os.makedirs(faces_dir, exist_ok=True)
        detect_faces(frames_dir, faces_dir)

        # Predict using the model
        predictions = []
        for face in os.listdir(faces_dir):
            face_path = os.path.join(faces_dir, face)
            processed_image = preprocess_image(face_path)
            pred = model.predict(processed_image)
            predictions.append(pred[0][0])

        # Determine the final result (average prediction)
        avg_prediction = np.mean(predictions)
        result = "Real" if avg_prediction < 0.5 else "Fake"

        # Return the result
        return jsonify({
            "result": result,
            "confidence": float(avg_prediction)
        })

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)