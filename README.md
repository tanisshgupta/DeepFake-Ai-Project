# Veridion: AI-Powered Deepfake Detection Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project aims to guide developers in training a deep learning-based deepfake detection model from scratch using **Python**, **Keras**, and **TensorFlow**. The proposed deepfake detector is based on the state-of-the-art **EfficientNet** architecture with customizations on the network layers. The model was trained on a massive and comprehensive set of deepfake datasets, making it effective in identifying potential deepfake content.

The model is also integrated into a web-based interface at [DF-Detect](https://deepfake-detect.com/) to assist both general internet users and digital media providers in identifying deepfake content. This tool helps raise awareness about fake content and ultimately contributes to combating the rise of deepfakes.

---

## Acknowledgments & Credits

This project is heavily inspired by and adapted from the work of:

- **Aaron Chong** ([@aaronchong888](https://github.com/aaronchong888)) - Initial work.
- **Hugo Ng** ([@hugoclong](https://github.com/hugoclong)) - Contributions to the initial development.

We would like to express our gratitude to these contributors for their foundational work. You can find the original repository [here](https://github.com/aaronchong888/DeepFake-Detect).

---

## Features
- **EfficientNet Backbone**: Utilizes EfficientNet-B0 as the base model for feature extraction.
- **Binary Classification**: Detects whether an image is real or a deepfake.
- **Facial Detection**: Uses **MTCNN** or **Azure Computer Vision API** to extract faces from video frames.
- **Dataset Preparation**: Includes scripts to balance and split datasets into training, validation, and testing sets.
- **Web Interface**: Provides a user-friendly web-based interface for detecting deepfakes.
---
## Deepfake Datasets
To achieve promising results, we used a combination of the following deepfake datasets:
- **DeepFake-TIMIT**
- **FaceForensics++**
- **Google Deep Fake Detection (DFD)**
- **Celeb-DF**
- **Facebook Deepfake Detection Challenge (DFDC)**
Combining all datasets provides a total of **134,446 videos** with approximately **1,140 unique identities** and around **20 deepfake synthesis methods**.
---

## Getting Started

Combining all datasets provides a total of **134,446 videos** with approximately

### Installation

```bash
pip install -r requirements.txt
```
---
## Usage
### Step 0: Convert Video Frames to Individual Images
Extract all video frames from the acquired deepfake datasets and save them as individual images.
```bash
python 00-convert_video_to_image.py
```

Image resizing strategies:
- **2x resize** for videos with width less than 300 pixels.
- **1x resize** for videos with width between 300 and 1000 pixels.
- **0.5x resize** for videos with width between 1000 and 1900 pixels.
- **0.33x resize** for videos with width greater than 1900 pixels.
---

### Step 1: Extract Faces from Images with MTCNN

Process frame images to crop out facial parts, allowing the neural network to focus on capturing facial manipulation artifacts.
```bash
python 01a-crop_faces_with_mtcnn.py
```
#### Optional: Use Azure Computer Vision API
If you don't have sufficient hardware to run MTCNN, or want faster execution, use Azure Computer Vision API instead.
```bash
python 01b-crop_faces_with_azure-vision-api.py
```

Replace the missing parts (API Name & API Key) before running.
---

### Step 2: Balance and Split Datasets

Balance the dataset by downsampling the fake dataset based on the number of real crops. Then, split the dataset into training, validation, and testing sets (e.g., 80:10:10 ratio).

```bash
python 02-prepare_fake_real_dataset.py
```
---
### Step 3: Model Training
Train the deepfake detection model using EfficientNet-B0 as the backbone.
```bash
python 03-train_cnn.py
```

The model uses:
- Input size: **128x128** with a depth of 3.
- Global max pooling layer.
- Two additional fully connected layers with ReLU activations.
- Final output layer with Sigmoid activation for binary classification.
---

## Model Architecture

| Layer (type)              | Output Shape     | Param #   |
|---------------------------|------------------|-----------|
| EfficientNet-B0           | (None, 1280)     | 4,049,564 |
| Dense                     | (None, 512)      | 655,872   |
| Dropout                   | (None, 512)      | 0         |
| Dense                     | (None, 128)      | 65,664    |
| Dense                     | (None, 1)        | 129       |

**Total params:** **4,771,229**

---
## Common Errors and Troubleshooting
### Error 1: GPU Device Not Found (`IndexError: list index out of range`)
If you encounter the following error while running `01a-crop_faces_with_mtcnn.py`:
```bash
IndexError: list index out of range
```
This happens because the script attempts to access the first GPU device using `physical_devices[0]`, but no GPU was detected. If you are not using a GPU or do not have one available, you can safely bypass this part of the code. Replace the GPU configuration block with the following:
```python
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU found:", physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
```
---
### Error 2: SSL Certificate Verification Failed (`Exception: URL fetch failure`)
When running `03-train_cnn.py`, you might encounter the following error:
```bash
Exception: URL fetch failure on https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5: None -- [SSL: CERTIFICATE_VERIFY_FAILED]
```

This error occurs due to SSL certificate verification issues when downloading the pre-trained weights for EfficientNet. To resolve this, you can try the following solutions:
1. **Upgrade `certifi` package**:
   ```bash
   pip install --upgrade certifi
   ```
2. **Manually download the weights**:
   - Download the weights file from the URL provided in the error message.
   - Place the downloaded `.h5` file in the appropriate directory and modify the script to load the weights locally.
---
### Error 3: Filepath Format Issue (`ValueError: The filepath provided must end in .keras`)
If you encounter the following error:

```bash
ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=.\tmp_checkpoint/best_model.h5
```
This error occurs because TensorFlow 2.x expects the model checkpoint file to have the `.keras` extension instead of `.h5`. Update the `ModelCheckpoint` callback to use the correct file extension:
```python
ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath, 'best_model.keras'),
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True
)
```

---

### Error 4: `fit_generator` Deprecation (`AttributeError: 'Sequential' object has no attribute 'fit_generator'`)

If you encounter the following error:

```bash
AttributeError: 'Sequential' object has no attribute 'fit_generator'
```
This is because `fit_generator` has been deprecated in TensorFlow 2.x. Replace `fit_generator` with `fit`:
```python
history = model.fit(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=custom_callbacks
)
```

---
### Error 5: Data Generator Issues (`TypeError: generator yielded an element that did not match the expected structure`)
If you encounter errors related to the data generator, such as:
```bash
TypeError: `generator` yielded an element that did not match the expected structure.
```

This typically happens when the data generator is not producing the expected data format. Ensure that your dataset is correctly structured and that the `ImageDataGenerator` is configured properly. You may also need to use the `.repeat()` function when building your dataset.

---

## Authors

- **AYUSH YADAV**
- **OM KUMAR SINGH**
- **MOHD. FARZAN KHAN**
- **PUNEET**
- **Tanish Gupta**
See also the list who participated in this project.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
---
## Resources
- **Original Repository**: [DeepFake-Detect](https://github.com/aaronchong888/DeepFake-Detect)
- **Demo Site**: [DF-Detect](https://deepfake-detect.com/)
- **MTCNN GitHub Repo**: [ipazc/mtcnn](https://github.com/ipazc/mtcnn)
---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any bugs or feature requests.
