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


