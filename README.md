# Diabetes Classifier using K-Nearest Neighbors (KNN)

## Overview
This project implements a simple machine learning system that predicts whether a person has diabetes and classifies the type of diabetes (Type 1 or Type 2) using the K-Nearest Neighbors (KNN) algorithm.

The system takes medical input values from the user, processes them with models trained on historical diabetes data, and outputs the diagnosis.

## Features
- Detects if a person is diabetic or not.
- Classifies diabetes type if diagnosed positive.
- Uses real-world diabetes dataset from Plotly.
- User-friendly command-line interface to enter medical data.

## Dataset
The dataset is sourced from [Plotly's Diabetes Dataset](https://github.com/plotly/datasets/blob/master/diabetes.csv).  
**Note:** Diabetes type labels are simulated for demo purposes based on age and insulin levels.

## Dependencies
- pandas
- numpy
- scikit-learn

## Requirements
All the dependencies needed to run this project are listed in the requirements.txt file.

To install them, use the following command:
pip install -r requirements.txt