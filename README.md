# Machine Learning Algorithms Implemented From Scratch
## Introduction:
This repository contains implementations of fundamental machine learning algorithms, built entirely from scratch. The project aims to provide a comprehensive resource for educational and research purposes, offering in-depth codebase insights for each algorithm.
## Motivation:
The repository aims to offer a granular look at the internal workings of machine learning algorithms, providing a resource for those interested in the foundational principles of the field. The implementations are designed to reveal the intricacies often obscured by higher-level libraries.
## Features:
__In-depth Algorithmic Understanding__: Each algorithm is implemented from the ground up, providing a comprehensive view into its mechanics.
__Comprehensive Documentation__: Accompanying each algorithm is a detailed architecture diagram and thorough testing procedures.
__Modular Design__: The codebase is structured to allow for easy integration into various machine learning pipelines.
## Technology Stack:
This project is implemented using Python 3.x. For testing and demonstration purposes, Jupyter Notebooks are utilized.
## File Structure:
The repository is organized into two main directories:
- __models__: Contains the Python scripts for each implemented machine learning algorithm.
- __tests__: Includes Jupyter Notebooks and Python scripts that rigorously test the algorithms.
```commandline
├── models
│   ├── AdaBoost.py
│   ├── DecisionTree.py
│   ├── KNN.py
│   ├── LinearRegression.py
│   ├── LogisticRegression.py
│   ├── NaiveBayes.py
│   ├── PCA.py
│   ├── RandomForest.py
│   └── SVM.py
├── tests
│   ├── AdaBoost_test.ipynb
│   ├── DecisionTree_test.ipynb
│   ├── KNN_comprehensive_test.ipynb
│   ├── KNN_test.py
│   ├── LinearRegression_test.ipynb
│   ├── LogisticRegression_test.ipynb
│   ├── MultiClassSVM_test.ipynb
│   ├── MultiClassSVM_test_2.ipynb
│   ├── NaiveBayes_test.ipynb
│   ├── PCA_test.ipynb
│   └── random_forest_test.ipynb
```
## Getting Started:
To utilize the algorithms in this repository, clone the project to your local machine. Navigate to the models directory to access the Python scripts for each algorithm.
```commandline
git clone https://github.com/nagarx/Implementing_ML_Algorithms_From_Scratch.git
cd ML_Algorithms_From_Scratch/models
```

## Usage:
Import the desired algorithm into your Python script or Jupyter Notebook for usage. Below is an example of how to import the K-Nearest Neighbors (KNN) algorithm:
```python
from models import KNN
```