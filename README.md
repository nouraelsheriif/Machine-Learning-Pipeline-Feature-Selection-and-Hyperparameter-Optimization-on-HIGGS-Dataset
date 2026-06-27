# Machine-Learning-Pipeline-Feature-Selection-and-Hyperparameter-Optimization-on-HIGGS-Dataset
Machine Learning Pipeline: Feature Selection and Hyperparameter Optimization on HIGGS Dataset

https://img.shields.io/badge/Python-3.12-blue.svg
https://img.shields.io/badge/License-MIT-green.svg
https://img.shields.io/badge/Scikit--learn-1.6.1-orange.svg
https://img.shields.io/badge/XGBoost-3.3.0-red.svg

 Project Overview

This project implements a comprehensive machine learning pipeline for binary classification on the HIGGS Dataset from the UCI Machine Learning Repository. The dataset contains simulated particle collision events used to distinguish between Higgs boson signal events and background events.

Key Objectives

 Outlier analysis using Interquartile Range (IQR) method
 Feature scaling using MinMaxScaler [0,1]
 Filter-based feature selection using ANOVA F-score
 Nested cross-validation (5-fold outer, 3-fold inner)
 Hyperparameter optimization with GridSearchCV
 Comparison of four machine learning algorithms:
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Multi-Layer Perceptron (MLP)
XGBoost
 Model evaluation using multiple performance metrics
 ROC curves with One-vs-All (OVA) approach
 Dataset Information

Property	Value
Dataset	HIGGS Dataset
Samples Used	100,000 (from 11 million)
Features	28 numerical features
Target	Binary (0 = background, 1 = signal)
Class Distribution	47.17% Class 0, 52.83% Class 1
Source	UCI Machine Learning Repository
🏆 Best Results

Best Performing Model: XGBoost

Metric	Score
Accuracy	0.7127 ± 0.0028
Precision	0.7288 ± 0.0034
Recall	0.7266 ± 0.0048
F1-Score	0.7277 ± 0.0028
ROC-AUC	0.7883 ± 0.0042
Selected Features (Top 15)

Rank	Feature	ANOVA F-Score
1	feature_25	1503.5703
2	feature_27	1452.5176
3	feature_3	531.2680
4	feature_26	313.6290
5	feature_5	242.0314
6	feature_12	182.5386
7	feature_17	111.1480
8	feature_0	105.0935
9	feature_22	67.1628
10	feature_16	55.1363
11	feature_9	30.9327
12	feature_20	20.5560
13	feature_21	15.7272
14	feature_23	11.2063
15	feature_13	9.7601
🚀 Installation

Prerequisites

Python 3.12 or higher
Git
Step 1: Clone the Repository

bash
git clone https://github.com/your-username/higgs-ml-pipeline.git
cd higgs-ml-pipeline
Step 2: Install Required Libraries

bash
pip install -r requirements.txt
Or install manually:

bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
Step 3: Download the Dataset

Download the HIGGS dataset from UCI Repository
Place the file as HIGGS.csv in the project directory
Note: The dataset is large (8GB). The code automatically loads only 100,000 samples.

 Project Structure

text
higgs-ml-pipeline/
│
├── final_project.py          # Main implementation with nested CV
├── quick_plots.py             # Fast plotting script (no retraining)
├── extract_dataset.py         # Extract HIGGS.csv.gz file
│
├── HIGGS.csv                  # Dataset (not included in repo)
│
├── outputs/
│   ├── feature_importance.png # Feature importance visualization
│   ├── roc_curves.png         # ROC curves for all models
│   ├── roc_curves_ova.png     # ROC curves (OVA approach)
│   ├── confusion_matrices.png # Confusion matrices
│   └── model_comparison.png   # Model comparison bar chart
│
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LICENSE                    # MIT License
 Usage

Option 1: Full Pipeline (with Nested CV)

Run the complete machine learning pipeline with nested cross-validation:

bash
python3 final_project.py
Expected Runtime: ~14-15 minutes (depending on hardware)

What it does:

Loads and preprocesses the HIGGS dataset
Performs outlier analysis (IQR method)
Scales features using MinMaxScaler
Selects top 15 features using ANOVA F-score
Runs nested cross-validation (5-fold outer, 3-fold inner)
Performs hyperparameter tuning with GridSearchCV
Trains and evaluates KNN, SVM, MLP, and XGBoost
Generates all visualization plots
Option 2: Quick Plots Only

Generate visualizations without retraining the full pipeline:

bash
python3 quick_plots.py
Expected Runtime: ~1-2 minutes

What it does:

Loads preprocessed data
Trains only final models (no nested CV)
Generates all visualization plots
 Visualization Outputs

File	Description
feature_importance.png	Bar chart of ANOVA F-scores for all features
roc_curves.png	ROC curves comparing all four models
roc_curves_ova.png	ROC curves using One-vs-All approach
confusion_matrices.png	Confusion matrices for each model
model_comparison.png	Bar chart comparing all performance metrics
 Hyperparameter Grids

Model	Hyperparameters
KNN	n_neighbors: [3, 5, 7, 9, 11]
SVM	C: [0.1, 1.0, 10.0], kernel: ['linear', 'rbf']
MLP	hidden_layer_sizes: [(50,), (100,)], activation: ['relu', 'tanh']
XGBoost	n_estimators: [50, 100], max_depth: [3, 5], learning_rate: [0.01, 0.1]
 Model Performance Comparison

Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC	Time/Fold
KNN	0.6630±0.0065	0.6655±0.0048	0.7280±0.0088	0.6954±0.0066	0.7179±0.0064	~19s
SVM	0.5038±0.0095	0.5765±0.0190	0.2275±0.0234	0.3258±0.0269	0.5545±0.0139	~85s
MLP	0.7100±0.0034	0.7227±0.0116	0.7332±0.0299	0.7273±0.0100	0.7857±0.0035	~65s
XGBoost	0.7127±0.0028	0.7288±0.0034	0.7266±0.0048	0.7277±0.0028	0.7883±0.0042	~2s
 Key Findings

1. Best Performing Model

XGBoost achieved the highest ROC-AUC score of 0.7883, outperforming MLP (0.7857), KNN (0.7179), and SVM (0.5545).

2. Most Important Features

Features 25 and 27 showed the highest discriminative power with ANOVA scores exceeding 1400.

3. Outlier Analysis

A significant 47.20% of samples contained outliers, which were clipped using the IQR method.

4. Nested Cross-Validation

The 5-fold outer × 3-fold inner CV provided unbiased performance estimates while effectively tuning hyperparameters.

Requirements

Python Dependencies

text
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
Hardware Requirements





