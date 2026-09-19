# Cat & Dog Image Classifier

A CNN-based image classifier built with Python and PyTorch to distinguish between cat and dog images.

## Features

- Image classification using a Convolutional Neural Network (CNN)
- Cat and dog image classification
- Image preprocessing with Resize and ToTensor
- Dataset organization using PyTorch ImageFolder
- Training and validation using PyTorch
- GPU acceleration with CUDA
- Model prediction on new images

## Technologies

- Python
- PyTorch
- Convolutional Neural Networks (CNN)
- CUDA
- Jupyter Notebook
- Git & GitHub

## Project Structure

```text
cat-dog-classifier/
├── .gitignore
├── dataset_test.py
├── image_test.py
├── model.py
├── predict.py
├── test.ipynb
├── test.py
├── test_Pytorch.py
├── train.py
└── README.md


بعد از این ساختار، یک توضیح کوتاه هم می‌ذاریم:

```markdown
### Main Files

- `model.py` — Defines the CNN model architecture.
- `train.py` — Trains the model using the prepared dataset.
- `predict.py` — Uses the trained model to make predictions on images.
- `dataset_test.py` — Tests and explores the dataset structure.
- `image_test.py` — Tests image loading and processing.
- `test.py` — Contains testing code used during development.
- `test_Pytorch.py` — Contains PyTorch-related tests and experiments.
- `test.ipynb` — Jupyter Notebook used for experimentation and testing.
```

## Dataset

The model was trained on a dataset of cat and dog images organized into two classes:

- `Cat`
- `Dog`

The images were loaded using PyTorch `ImageFolder` and resized to `224 × 224` pixels before being converted to tensors.

The dataset was divided into training, validation, and test sets.

## Model

The project uses a Convolutional Neural Network (CNN) designed for binary image classification.

The CNN includes:

- Convolutional layers
- ReLU activation functions
- Pooling layers
- Fully connected layers

The model was trained using PyTorch with GPU acceleration through CUDA.


## How to Run

### 1. Clone the repository

Clone the project from GitHub and move into the project directory.

```bash
git clone https://github.com/maryamhessampour-art/cat-dog-classifier.git
cd cat-dog-classifier
```

### 2. Create a virtual environment

Create a Python virtual environment to keep the project's dependencies isolated from other Python projects.

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On Windows, activate the virtual environment using:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

Install the required PyTorch packages:

```bash
pip install torch torchvision
```

### 5. Prepare the dataset

Place the cat and dog image dataset in the `dataset/` directory.

The dataset should contain two classes:

```text
dataset/
├── Cat/
└── Dog/
```

### 6. Train the model

Run the training script to train the CNN model:

```bash
python train.py
```

### 7. Make predictions


After training, use the prediction script to classify new images:

```bash
python predict.py
```

## Notes

- The dataset is not included in this repository because of its size.
- The trained model file is also excluded from the repository.
- A CUDA-compatible GPU can be used to accelerate model training.

