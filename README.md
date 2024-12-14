# Traffic Sign Recognition

## Overview
This project develops a Traffic Sign Recognition system using Convolutional Neural Networks (CNN) to classify images of traffic signs. 
The goal is to create an automated system capable of accurately identifying and categorizing different traffic signs, which will contribute to the improvement of advanced driver-assistance systems (ADAS) and autonomous vehicles.  

## Problem Statement
Traffic signs play a vital role in ensuring road safety as they provide crucial information to drivers. 
An automated recognition system can enhance both safety and efficiency on the roads. For example, a system that identifies a "Stop" sign and notifies the driver to halt the vehicle can significantly reduce the risk of accidents.

## Examples Images
<div style="display: flex;">
  <img src="data/Meta/0.png" alt="Stop Sign" width="200" />
  <img src="data/Meta/14.png" alt="Yield Sign" width="200" />
  <img src="data/Meta/40.png" alt="Speed Limit Sign" width="200" />
</div>

## Datasets
The dataset employed in this project is the German Traffic Sign Recognition Benchmark (GTSRB), which consists of over 50,000 images divided into 43 categories (classes) of traffic signs.

### Dataset Structure
``` 
  data/ 
      ├── Train/ # Contains train images organized by class
      ├── Test/ # Contains test images organized by class
```

#### CSV Files
- `Train.csv` : Contains paths and labels for the training set.
- `Test.csv` : Contains paths and labels for the test set.

## Project Structure
```
    Traffic-Signs-Recognition-using-CNN-Keras/
    │
    ├── data/
    │   ├── Meta/
    │   ├── Test/
    │   ├── Train/
    │   Meta.csv
    │   Test.csv
    │   Train.csv 
    ├── model/ # Contains the model
    ├── performance/ # Contains model performance statistics (loss, accuracy graphs, confusion matrix)
    ├── Generate.ipynb
    ├── main.py
    ├── requirements.txt
    └── README.md
```

## Setup and Installation
```
    git clone <repository-url>
    cd Traffic-Signs-Recognition-using-CNN-Keras
     python -m venv env
     source env/bin/activate  # On Windows use `env\Scripts\activate`
     pip install -r requirements.txt
```

## Usage
### 1. Image Loading and Preprocessing:
   - **What it does**: The code loads images from the `./data/train/` directory for each class (from 0 to 42), resizes them to 30x30 pixels, and converts them into NumPy arrays for further processing.
   - **Description**: 
     The dataset is loaded from the `./data/train/` folder, and each image is resized to 30x30 pixels. These images are then converted into NumPy arrays to be ready for model training. The labels for each image are assigned based on the folder it belongs to, representing the class of the traffic sign.

### 2. Dataset Splitting into Training and Testing:
   - **What it does**: The dataset is split into training (80%) and test (20%) sets using `train_test_split`.
   - **Description**: 
     The dataset is divided into two sets: one for training (80%) and one for testing (20%). This separation allows the model to be trained on the training data and evaluated on unseen test data, ensuring that the model generalizes well to new examples.

### 3. One-Hot Encoding of Labels:
   - **What it does**: The labels (representing the classes) are converted into one-hot encoding using `to_categorical`.
   - **Description**: 
     The labels, which indicate the class of each traffic sign, are converted into one-hot encoding. This is required for multi-class classification, where each label is represented as a binary vector with a '1' at the correct class index and '0' elsewhere.

### 4. Building the CNN Model:
   - **What it does**: A Convolutional Neural Network (CNN) is built with convolutional layers, pooling layers, dropout layers, and fully connected layers.
   - **Description**: 
     The CNN model consists of several layers: convolutional layers to extract features (such as edges and shapes) from the images, pooling layers to reduce dimensionality and improve efficiency, and dropout layers to prevent overfitting. The final fully connected layers use a softmax activation function to output class probabilities.

### 5. Model Compilation:
   - **What it does**: The model is compiled using the `categorical_crossentropy` loss function, the `adam` optimizer, and accuracy as the evaluation metric.
   - **Description**: 
     The model is compiled with the `categorical_crossentropy` loss function, which is suitable for multi-class classification, and the `adam` optimizer, which is highly effective for training complex models. Accuracy is used as the evaluation metric.

### 6. Model Training:
   - **What it does**: The model is trained on the training data for a predefined number of epochs, and validation is performed on the test set.
   - **Description**: 
     The model is trained for 35 epochs on the training data, with validation performed on the test data after each epoch. During training, both accuracy and loss are monitored on the training and validation sets.

### 7. Model Performance Evaluation:
   - **What it does**: It visualizes accuracy and loss graphs for both training and validation data.
   - **Description**: 
     After training, graphs of accuracy and loss are displayed to observe how well the model performed during training and validation. This helps in identifying issues like overfitting or underfitting.

### 8. Testing on the Test Dataset:
   - **What it does**: It computes the accuracy of the model on a separate test dataset.
   - **Description**: 
     The model is tested on a separate test dataset to calculate the accuracy, which indicates how well the model classifies traffic signs that it has not seen during training.

### 9. Saving and Loading the Model:
   - **What it does**: The trained model is saved to disk for future predictions and can be reloaded for testing on new images.
   - **Description**: 
     The trained model is saved to disk for later use, allowing it to be reloaded and tested on new images without needing to retrain. This makes it easy to deploy the model for inference on new data.

### 10. Confusion Matrix and Classification Report:
   - **What it does**: It generates a confusion matrix and a classification report to assess the model's performance in more detail.
   - **Description**: 
     To provide a more thorough evaluation, a confusion matrix is generated to show the number of correct and incorrect predictions for each class. Additionally, a classification report is produced, which includes precision, recall, and F1-score for each class.
## Streamlit Application
To launch the Streamlit application for interactive traffic sign recognition:
```
    streamlit run main.py
```

