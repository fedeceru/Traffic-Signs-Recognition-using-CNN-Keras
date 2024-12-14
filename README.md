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
### 1. **Data Preprocessing**
First, the dataset needs to be preprocessed. This step involves loading images from the `./data/train/` directory, resizing them to a consistent size (30x30 pixels), and assigning labels based on the class of each image. This process is essential for preparing the dataset for training.

### 2. **Dataset Splitting**
After preprocessing the data, the dataset is split into training and test sets. Typically, 80% of the data is used for training, while the remaining 20% is used for testing. This split allows you to train the model on a majority of the data and evaluate its performance on unseen test data.

### 3. **One-Hot Encoding of Labels**
Since the model outputs class probabilities, labels are converted into one-hot encoding format. This step ensures that each label is represented as a binary vector, where only the index corresponding to the correct class is set to 1, and all others are set to 0. One-hot encoding is necessary for categorical classification tasks, like traffic sign recognition.

### 4. **Building the CNN Model**
At this stage, the Convolutional Neural Network (CNN) model is built. The model consists of several layers:
- **Convolutional Layers (Conv2D)**: These layers extract features from the images, such as edges and shapes.
- **Pooling Layers (MaxPool2D)**: These layers downsample the feature maps, reducing the dimensionality and making the model more efficient.
- **Dropout Layers**: These layers randomly deactivate some neurons during training, which helps prevent overfitting.
- **Dense Layers**: These fully connected layers output the final class predictions. The final layer uses a softmax activation function to output class probabilities.
The model is then compiled using the Adam optimizer and categorical cross-entropy loss, which are standard choices for multi-class classification tasks.

### 5. **Training the Model**
With the model defined, training begins. The model is trained on the training dataset for a specified number of epochs, with the test dataset used for validation. During training, the model's performance is monitored on both the training and validation sets to ensure it is learning effectively.

### 6. **Evaluating the Model**
Once training is complete, the model's performance is evaluated on the test dataset. This evaluation gives insights into how well the model generalizes to unseen data. The accuracy score is typically used to measure the model's performance.

### 7. **Model Prediction**
To use the trained model for prediction, new images can be input to the model. The model will output the predicted class for each image. The predictions are generated using the `predict` method, and the class with the highest probability is selected as the model's output.

### 8. **Saving and Loading the Model**
After training, the model is saved to disk. This allows you to reuse the model later without retraining. The model can be loaded from disk using the `load_model` method, which facilitates inference on new images or further fine-tuning.

### 9. **Confusion Matrix and Classification Report**
To assess the model's performance more thoroughly, a confusion matrix is generated, which shows the number of correct and incorrect predictions for each class. Additionally, a classification report is created, which provides metrics such as precision, recall, and F1-score for each class, offering a deeper understanding of the model's strengths and weaknesses.

## Streamlit Application
To launch the Streamlit application for interactive traffic sign recognition:
```
    streamlit run main.py
```

