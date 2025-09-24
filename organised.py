import json
import os
import cv2
import math
from mtcnn import MTCNN
import numpy as np
import splitfolders
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd

# Constants
BASE_PATH = '.\\train_sample_videos\\'
DATASET_PATH = '.\\prepared_dataset\\'
TMP_FAKE_PATH = '.\\tmp_fake_faces'
CHECKPOINT_PATH = '.\\tmp_checkpoint'

def get_filename_only(file_path):
    """Extracts the filename without extension."""
    return os.path.basename(file_path).split('.')[0]

def create_directory(path):
    """Creates a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    print(f'Creating Directory: {path}')

def load_metadata(base_path):
    """Loads metadata from JSON file."""
    with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
        return json.load(metadata_json)

def extract_frames(video_file, output_dir):
    """Extracts frames from video and saves them as images."""
    cap = cv2.VideoCapture(video_file)
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

def determine_scale_ratio(width):
    """Determines the scale ratio based on frame width."""
    if width < 300:
        return 2
    elif width > 1900:
        return 0.33
    elif 1000 < width <= 1900:
        return 0.5
    else:
        return 1

def resize_frame(frame, scale_ratio):
    """Resizes the frame based on scale ratio."""
    width = int(frame.shape[1] * scale_ratio)
    height = int(frame.shape[0] * scale_ratio)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def save_frame(frame, output_dir, count):
    """Saves the frame as an image file."""
    new_filename = f'{output_dir}-{count:03d}.png'
    cv2.imwrite(new_filename, frame)

def detect_faces(image_dir, faces_dir):
    """Detects faces in images and crops them."""
    detector = MTCNN()
    for frame in os.listdir(image_dir):
        image_path = os.path.join(image_dir, frame)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)
        for i, result in enumerate(results):
            if len(results) < 2 or result['confidence'] > 0.95:
                crop_and_save_face(result, image, faces_dir, frame, i)

def crop_and_save_face(result, image, faces_dir, frame, count):
    """Crops the face from the image and saves it."""
    bounding_box = result['box']
    margin_x = bounding_box[2] * 0.3
    margin_y = bounding_box[3] * 0.3
    x1, y1, x2, y2 = calculate_bounding_box(bounding_box, margin_x, margin_y, image.shape)
    cropped_image = image[y1:y2, x1:x2]
    new_filename = f'{os.path.join(faces_dir, get_filename_only(frame))}-{count:02d}.png'
    cv2.imwrite(new_filename, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

def calculate_bounding_box(box, margin_x, margin_y, shape):
    """Calculates the bounding box coordinates."""
    x1 = max(0, int(box[0] - margin_x))
    x2 = min(shape[1], int(box[0] + box[2] + margin_x))
    y1 = max(0, int(box[1] - margin_y))
    y2 = min(shape[0], int(box[1] + box[3] + margin_y))
    return x1, y1, x2, y2

def prepare_dataset(metadata, base_path, dataset_path, tmp_fake_path):
    """Prepares the dataset by copying real and fake faces."""
    real_path = os.path.join(dataset_path, 'real')
    fake_path = os.path.join(dataset_path, 'fake')
    create_directory(real_path)
    create_directory(fake_path)
    
    for filename, info in metadata.items():
        tmp_path = os.path.join(base_path, get_filename_only(filename), 'faces')
        if os.path.exists(tmp_path):
            if info['label'] == 'REAL':
                copy_tree(tmp_path, real_path)
            elif info['label'] == 'FAKE':
                copy_tree(tmp_path, tmp_fake_path)

def downsample_fake_faces(real_path, tmp_fake_path, fake_path):
    """Downsamples fake faces to match the number of real faces."""
    all_real_faces = [f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]
    all_fake_faces = [f for f in os.listdir(tmp_fake_path) if os.path.isfile(os.path.join(tmp_fake_path, f))]
    random_faces = np.random.choice(all_fake_faces, len(all_real_faces), replace=False)
    
    for fname in random_faces:
        shutil.copyfile(os.path.join(tmp_fake_path, fname), os.path.join(fake_path, fname))

def split_dataset(dataset_path):
    """Splits the dataset into train, validation, and test sets."""
    splitfolders.ratio(dataset_path, output='split_dataset', seed=1377, ratio=(.8, .1, .1))

def create_generators(input_size, batch_size, dataset_path):
    """Creates data generators for training, validation, and testing."""
    train_datagen = ImageDataGenerator(rescale=1/255, rotation_range=10, width_shift_range=0.1,
                                       height_shift_range=0.1, shear_range=0.2, zoom_range=0.1,
                                       horizontal_flip=True, fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale=1/255)
    test_datagen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_datagen.flow_from_directory(directory=os.path.join(dataset_path, 'train'),
                                                        target_size=(input_size, input_size),
                                                        color_mode="rgb", class_mode="binary",
                                                        batch_size=batch_size, shuffle=True)
    val_generator = val_datagen.flow_from_directory(directory=os.path.join(dataset_path, 'val'),
                                                    target_size=(input_size, input_size),
                                                    color_mode="rgb", class_mode="binary",
                                                    batch_size=batch_size, shuffle=True)
    test_generator = test_datagen.flow_from_directory(directory=os.path.join(dataset_path, 'test'),
                                                      classes=['real', 'fake'], target_size=(input_size, input_size),
                                                      color_mode="rgb", class_mode=None, batch_size=1, shuffle=False)
    return train_generator, val_generator, test_generator

def build_model(input_size):
    """Builds the CNN model using EfficientNetB0."""
    efficient_net = EfficientNetB0(weights='imagenet', input_shape=(input_size, input_size, 3),
                                   include_top=False, pooling='max')
    model = Sequential([
        efficient_net,
        Dense(units=512, activation='relu'),
        Dropout(0.5),
        Dense(units=128, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, val_generator, num_epochs, callbacks):
    """Trains the model."""
    history = model.fit(train_generator, epochs=num_epochs, steps_per_epoch=len(train_generator),
                        validation_data=val_generator, validation_steps=len(val_generator), callbacks=callbacks)
    return history

def main():
    # Load metadata
    metadata = load_metadata(BASE_PATH)
    
    # Create necessary directories
    create_directory(DATASET_PATH)
    create_directory(TMP_FAKE_PATH)
    create_directory(CHECKPOINT_PATH)
    
    # Process videos and extract frames
    for filename in metadata.keys():
        if filename.endswith(".mp4"):
            tmp_path = os.path.join(BASE_PATH, get_filename_only(filename))
            create_directory(tmp_path)
            extract_frames(os.path.join(BASE_PATH, filename), tmp_path)
    
    # Detect faces in frames
    for filename in metadata.keys():
        tmp_path = os.path.join(BASE_PATH, get_filename_only(filename))
        faces_path = os.path.join(tmp_path, 'faces')
        create_directory(faces_path)
        detect_faces(tmp_path, faces_path)
    
    # Prepare dataset
    prepare_dataset(metadata, BASE_PATH, DATASET_PATH, TMP_FAKE_PATH)
    
    # Downsample fake faces
    real_path = os.path.join(DATASET_PATH, 'real')
    fake_path = os.path.join(DATASET_PATH, 'fake')
    downsample_fake_faces(real_path, TMP_FAKE_PATH, fake_path)
    
    # Split dataset
    split_dataset(DATASET_PATH)
    
    # Create data generators
    input_size = 128
    batch_size = 32
    train_generator, val_generator, test_generator = create_generators(input_size, batch_size, DATASET_PATH)
    
    # Build and train model
    model = build_model(input_size)
    callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1),
        ModelCheckpoint(filepath=os.path.join(CHECKPOINT_PATH, 'best_model.h5'), monitor='val_loss',
                        mode='min', verbose=1, save_best_only=True)
    ]
    history = train_model(model, train_generator, val_generator, num_epochs=20, callbacks=callbacks)
    
    # Load best model and generate predictions
    best_model = tf.keras.models.load_model(os.path.join(CHECKPOINT_PATH, 'best_model.h5'))
    test_generator.reset()
    preds = best_model.predict(test_generator, verbose=1)
    test_results = pd.DataFrame({"Filename": test_generator.filenames, "Prediction": preds.flatten()})
    print(test_results)

if __name__ == "__main__":
    main()
