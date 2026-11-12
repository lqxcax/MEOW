#  Deep Colorization Network

### *An End-to-End Convolutional Autoencoder for Image Colorization in Keras*

---

## 📖 Overview

This repository implements a **Convolutional Encoder-Decoder (Autoencoder)** architecture built using **TensorFlow/Keras** designed to solve the image colorization problem. Given a single-channel grayscale image (e.g., landscape photography), the model learns spatial representations and reconstructs a vivid, 3-channel RGB representation.

The architecture features **skip connections** inspired by U-Net models to retain high-resolution spatial details from the encoding layers, combined with **Batch Normalization** and **Huber Loss** for stable training dynamics.

---

## 🛠️ Architecture & Pipeline

The pipeline is split into modular components for clean data loading, model definition, and training execution:

* **`load_dataset.py`**: Handles batch reading of grayscale and target color directories, image resizing, normalization, and numpy array conversion.
* **`model.py`**: Defines the custom convolutional neural network utilizing encoder layers (`Conv2D`, `MaxPool2D`, `BatchNormalization`), transposed convolutions (`Conv2DTranspose`), and skip-connection concatenation (`Concatenate`).
* **`main.py`**: Orchestrates data splitting, model compilation (using Adam optimizer and Huber loss), callback configuration (`EarlyStopping`, `ReduceLROnPlateau`), and performance evaluation via Mean Absolute Error (MAE).

---

## 📂 Project Structure

```text
├── load_dataset.py     # Dataset loader and image preprocessing utility
├── model.py            # Encoder-Decoder network architecture definition
├── main.py             # Training script, hyperparameter setup, and evaluation
└── README.md           # Project documentation

```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed along with the required deep learning dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pillow

```

### Dataset Organization

Organize your dataset directory structure locally (e.g., named `landscape Images`) matching the expected paths:

```text
landscape Images/
├── gray/
│   ├── image1.jpg
│   └── image2.jpg
└── color/
    ├── image1.jpg
    └── image2.jpg

```

### Running the Model

Execute the training script from your terminal:

```bash
python main.py

```

---

## ⚙️ Model Architecture Details

1. **Encoder Path**: Extracts features using stacked 2D convolutions (`32`, `64`, and `128` filters), spatial batch normalization, and max pooling downsampling.
2. **Decoder Path**: Up-samples feature maps via transposed convolutions and combines them with encoder feature maps using tensor concatenation to preserve sharp edge details.
3. **Output Layer**: Employs a 1x1 convolution with a `sigmoid` activation function to scale pixel color channels smoothly between `[0, 1]`.

---

## 📊 Training Strategy

* **Loss Function**: `Huber Loss` — robust to outliers compared to Mean Squared Error (MSE).
* **Optimizer**: `Adam` (Initial Learning Rate: `0.001`).
* **Callbacks**:
* `EarlyStopping` (patience = 10 epochs) to prevent overfitting and restore best weights.
* `ReduceLROnPlateau` (factor = 0.1, patience = 3 epochs) to dynamically decay learning rate when validation loss plateaus.



---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.