# Machine Learning Pipeline: Feature Selection and Hyperparameter Optimization on HIGGS Dataset

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-red.svg)](https://xgboost.ai/)

---

##  Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Information](#-dataset-information)
- [Best Results](#-best-results)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Visualization Outputs](#-visualization-outputs)
- [Hyperparameter Grids](#-hyperparameter-grids)
- [Model Performance Comparison](#-model-performance-comparison)
- [Key Findings](#-key-findings)
- [Requirements](#-requirements)
- [Academic Context](#-academic-context)
- [License](#-license)
- [References](#-references)
- [Contact](#-contact)

---

## 📋 Project Overview

This project implements a comprehensive machine learning pipeline for binary classification on the **HIGGS Dataset** from the UCI Machine Learning Repository. The dataset contains simulated particle collision events used to distinguish between Higgs boson signal events and background events.

### 🎯 Key Objectives

| # | Objective |
|---|-----------|
| 1 | Outlier analysis using Interquartile Range (IQR) method |
| 2 | Feature scaling using MinMaxScaler [0,1] |
| 3 | Filter-based feature selection using ANOVA F-score |
| 4 | Nested cross-validation (5-fold outer, 3-fold inner) |
| 5 | Hyperparameter optimization with GridSearchCV |
| 6 | Comparison of four machine learning algorithms: KNN, SVM, MLP, XGBoost |
| 7 | Model evaluation using multiple performance metrics |
| 8 | ROC curves with One-vs-All (OVA) approach |

---

## 📊 Dataset Information

| Property | Value |
|----------|-------|
| **Dataset** | HIGGS Dataset |
| **Total Samples** | 11,000,000 |
| **Samples Used** | 100,000 (random sample) |
| **Features** | 28 numerical features |
| **Target Variable** | Binary (0 = background, 1 = signal) |
| **Class Distribution** | 47.17% Class 0, 52.83% Class 1 |
| **Data Type** | High-dimensional particle physics data |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HIGGS) |

---

## 🏆 Best Results

### Best Performing Model: **XGBoost**

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.7127 ± 0.0028 |
| **Precision** | 0.7288 ± 0.0034 |
| **Recall** | 0.7266 ± 0.0048 |
| **F1-Score** | 0.7277 ± 0.0028 |
| **ROC-AUC** | **0.7883 ± 0.0042** |

### Selected Features (Top 15 by ANOVA F-Score)

| Rank | Feature | ANOVA F-Score |
|:----:|---------|:-------------:|
| 1 | feature_25 | **1503.5703** |
| 2 | feature_27 | **1452.5176** |
| 3 | feature_3 | 531.2680 |
| 4 | feature_26 | 313.6290 |
| 5 | feature_5 | 242.0314 |
| 6 | feature_12 | 182.5386 |
| 7 | feature_17 | 111.1480 |
| 8 | feature_0 | 105.0935 |
| 9 | feature_22 | 67.1628 |
| 10 | feature_16 | 55.1363 |
| 11 | feature_9 | 30.9327 |
| 12 | feature_20 | 20.5560 |
| 13 | feature_21 | 15.7272 |
| 14 | feature_23 | 11.2063 |
| 15 | feature_13 | 9.7601 |

---

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- Git
- 10GB free storage (for dataset)

