
## 📌 Credit Card Fraud Detection
🔍 Project Overview

This project aims to detect fraudulent credit card transactions using Machine Learning techniques. The model is trained on highly imbalanced transaction data to classify whether a transaction is Fraud (1) or Normal (0).

Fraud detection is a real-world problem used by banks and financial institutions to prevent financial losses.

## Dataset Information

Dataset Name: Credit Card Fraud Detection Dataset

Total Transactions: 284,807

Fraud Cases: 492

Features: 30 numerical features (V1–V28, Time, Amount)

Target Variable: Class (0 = Normal, 1 = Fraud)

Dataset Source:

## Kaggle Dataset Link:\

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


## Technologies Used :

Python

Pandas

NumPy

Scikit-Learn

Matplotlib

Seaborn

XGBoost (if used)

## Project Workflow:

Data Ingestion

Data Cleaning

Exploratory Data Analysis (EDA)

Feature Scaling

Handling Imbalanced Data (SMOTE / Scale_Pos_Weight)

Model Training

Model Evaluation

Model Saving (Pickle)

## Model Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

ROC-AUC Score

Confusion Matrix

Fraud detection mainly focuses on Recall and Precision because missing fraud transactions is very costly.