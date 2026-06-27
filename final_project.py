"""
USKUDAR UNIVERSITY - INSTITUTE OF SCIENCE AND TECHNOLOGY
MACHINE LEARNING - Final Project

Author: NOURA ESHERIF - 254308903
Date: June 2026

Project: Machine Learning Pipeline: Feature Selection and Hyperparameter Optimization
Dataset: HIGGS Dataset (100,000 samples, 28 features)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.pipeline import Pipeline
import warnings
import time
warnings.filterwarnings('ignore')

np.random.seed(42)


def load_higgs_data(file_path='HIGGS.csv', n_samples=100000):
    """
    Load HIGGS dataset from CSV file
    
    The dataset has 28 features and 1 target variable (class: 0 or 1)
    Features are named: feature_1 through feature_28
    Target is the first column
    """
    print("Loading HIGGS dataset...")
    
    column_names = ['class'] + [f'feature_{i}' for i in range(1, 29)]
    
    try:
        df = pd.read_csv(file_path, header=None, names=column_names, nrows=n_samples)
        print(f"Loaded {len(df)} samples with {len(df.columns)-1} features")
        return df
    except FileNotFoundError:
        print("Error: File not found. Please ensure the HIGGS dataset is downloaded.")
        print("Download from: https://archive.ics.uci.edu/ml/datasets/HIGGS")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def detect_outliers_iqr(data, feature_names):
    """
    Detect outliers using IQR (Interquartile Range) method
    
    An outlier is defined as a value outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    """
    outlier_mask = pd.Series(False, index=data.index)
    
    for feature in feature_names:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        feature_outliers = (data[feature] < lower_bound) | (data[feature] > upper_bound)
        outlier_mask = outlier_mask | feature_outliers
    
    return outlier_mask


def handle_outliers(data, feature_names, method='clip'):
    """
    Handle outliers using specified method
    
    method: 'clip' - replace with threshold values
            'remove' - remove rows with outliers
    """
    print("\n=== OUTLIER ANALYSIS ===")
    
    outlier_mask = detect_outliers_iqr(data, feature_names)
    n_outliers = outlier_mask.sum()
    print(f"Total outliers detected: {n_outliers} ({n_outliers/len(data)*100:.2f}%)")
    
    if n_outliers == 0:
        print("No outliers detected.")
        return data
    
    if method == 'clip':
        data_clipped = data.copy()
        for feature in feature_names:
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data_clipped[feature] = data_clipped[feature].clip(lower_bound, upper_bound)
        
        print("Outliers clipped to threshold values.")
        return data_clipped
    elif method == 'remove':
        data_cleaned = data[~outlier_mask].copy()
        print(f"Removed {n_outliers} rows with outliers. New shape: {data_cleaned.shape}")
        return data_cleaned
    else:
        print(f"Unknown method: {method}. Returning original data.")
        return data


def scale_features(X_train, X_test):
    """
    Scale features to [0, 1] range using MinMaxScaler
    """
    print("\n=== FEATURE SCALING ===")
    print("Scaling features to [0, 1] range using MinMaxScaler...")
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape after scaling: {X_train_scaled.shape}")
    print(f"Test set shape after scaling: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def select_top_features(X_train, y_train, X_test, n_features=15, method='anova'):
    """
    Select top features using filter-based methods
    
    method: 'anova' - ANOVA F-score
            'mutual_info' - Mutual Information
    """
    print(f"\n=== FEATURE SELECTION (Top {n_features} features) ===")
    print(f"Using method: {method}")
    
    if method == 'anova':
        selector = SelectKBest(score_func=f_classif, k=n_features)
    else:
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_indices = selector.get_support(indices=True)
    feature_scores = selector.scores_
    
    print("\nTop features by ANOVA F-score:")
    for i, idx in enumerate(selected_indices):
        print(f"  Feature {idx}: Score = {feature_scores[idx]:.4f}")
    
    print(f"\nSelected {n_features} features out of {X_train.shape[1]}")
    
    return X_train_selected, X_test_selected, selected_indices, selector


def nested_cross_validation(X, y, models, param_grids, outer_cv=5, inner_cv=3):
    """
    Perform nested cross-validation with hyperparameter optimization
    
    Flowchart A: Inner loop evaluates different feature selection combinations
    Flowchart B: Inner loop evaluates different hyperparameter combinations
    
    Outer loop is used to evaluate test performance.
    """
    print("\n=== NESTED CROSS-VALIDATION ===")
    print(f"Outer CV: {outer_cv}-fold, Inner CV: {inner_cv}-fold")
    
    outer_scores = {
        model_name: {'accuracy': [], 'precision': [], 'recall': [], 
                     'f1': [], 'roc_auc': []} 
        for model_name in models.keys()
    }
    
    best_params = {model_name: [] for model_name in models.keys()}
    best_features = {model_name: [] for model_name in models.keys()}
    
    outer_cv_split = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    n_features = 15
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv_split.split(X, y)):
        print(f"\n--- Outer Fold {fold+1}/{outer_cv} ---")
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        selected_indices = selector.get_support(indices=True)
        
        for model_name, model_class in models.items():
            print(f"\n  Training {model_name}...")
            start_time = time.time()
            
            if model_name == 'SVM':
                pipeline = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('feature_selector', SelectKBest(score_func=f_classif, k=n_features)),
                    ('classifier', SVC(
                        probability=True, 
                        cache_size=1000,
                        shrinking=True,
                        tol=1e-3,
                        max_iter=1000
                    ))
                ])
            elif model_name == 'XGBoost':
                pipeline = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('feature_selector', SelectKBest(score_func=f_classif, k=n_features)),
                    ('classifier', XGBClassifier(
                        n_jobs=-1,
                        eval_metric='logloss',
                        verbosity=0,
                        use_label_encoder=False
                    ))
                ])
            else:
                pipeline = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('feature_selector', SelectKBest(score_func=f_classif, k=n_features)),
                    ('classifier', model_class())
                ])
            
            pipe_param_grid = {}
            for key, value in param_grids[model_name].items():
                pipe_param_grid[f'classifier__{key}'] = value
            
            inner_cv_split = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
            
            grid_search = GridSearchCV(
                pipeline, 
                pipe_param_grid, 
                cv=inner_cv_split,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params[model_name].append(grid_search.best_params_)
            
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            outer_scores[model_name]['accuracy'].append(accuracy)
            outer_scores[model_name]['precision'].append(precision)
            outer_scores[model_name]['recall'].append(recall)
            outer_scores[model_name]['f1'].append(f1)
            outer_scores[model_name]['roc_auc'].append(roc_auc)
            
            best_features[model_name].append(selected_indices)
            
            elapsed = time.time() - start_time
            print(f"    Best params: {grid_search.best_params_}")
            print(f"    Test Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
            print(f"    Time: {elapsed:.1f}s")
    
    results = {}
    for model_name in models.keys():
        results[model_name] = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            scores = outer_scores[model_name][metric]
            results[model_name][metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'all': scores
            }
    
    return results, best_params, best_features


def create_feature_importance_plot(X_train, y_train, feature_names):
    """Create feature importance plot using ANOVA scores"""
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X_train, y_train)
    scores = selector.scores_
    
    sorted_idx = np.argsort(scores)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(scores)), scores[sorted_idx], alpha=0.7, color='steelblue')
    plt.yticks(range(len(scores)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('ANOVA F-score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance (ANOVA F-Scores)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    return plt.gcf()


def plot_roc_curves(models, X_test, y_test, title="ROC Curves - Model Comparison"):
    """Plot ROC curves for each model"""
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (model_name, model) in enumerate(models.items()):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', 
                linewidth=2, color=colors[idx % len(colors)])
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_roc_curves_ova(models, X_test, y_test):
    """
    Plot ROC curves for each class using One-vs-All approach
    Required by project specifications
    """
    n_classes = len(np.unique(y_test))
    fig, axes = plt.subplots(1, n_classes, figsize=(14, 5))
    
    if n_classes == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, class_label in enumerate(np.unique(y_test)):
        y_binary = (y_test == class_label).astype(int)
        ax = axes[i]
        
        for idx, (model_name, model) in enumerate(models.items()):
            y_pred_proba = model.predict_proba(X_test)
            
            if y_pred_proba.shape[1] == 2:
                y_pred_proba_class = y_pred_proba[:, 1]
            else:
                y_pred_proba_class = y_pred_proba[:, i]
            
            fpr, tpr, _ = roc_curve(y_binary, y_pred_proba_class)
            roc_auc = roc_auc_score(y_binary, y_pred_proba_class)
            
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', 
                   linewidth=2, color=colors[idx % len(colors)])
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1, alpha=0.7)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'Class {class_label} (OVA)', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrices(models, X_test, y_test):
    """Plot confusion matrices for each model"""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4.5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar=False, square=True)
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
        axes[idx].set_xticklabels(['Class 0', 'Class 1'])
        axes[idx].set_yticklabels(['Class 0', 'Class 1'])
    
    plt.tight_layout()
    return fig


def plot_model_comparison(results):
    """Plot comparison of models across different metrics"""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    model_names = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(metrics))
    width = 0.2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, model_name in enumerate(model_names):
        values = [results[model_name][metric]['mean'] for metric in metrics]
        errors = [results[model_name][metric]['std'] for metric in metrics]
        offset = (i - len(model_names)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name, 
               yerr=errors, capsize=5, alpha=0.8, color=colors[i % len(colors)])
    
    ax.set_xlabel('Metrics', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Model Performance Comparison (Mean +- Std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


def create_results_table(results):
    """Create a formatted results table"""
    print("\n" + "="*90)
    print("MODEL PERFORMANCE RESULTS")
    print("="*90)
    
    print(f"{'Model':<15}", end="")
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        print(f"{metric:<15}", end="")
    print()
    print("-"*90)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15}", end="")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            mean_val = metrics[metric]['mean']
            std_val = metrics[metric]['std']
            print(f"{mean_val:.4f}+-{std_val:.4f}   ", end="")
        print()
    
    print("="*90)


def main():
    """
    Main execution function - Runs the complete machine learning pipeline
    """
    print("="*90)
    print("USKUDAR UNIVERSITY - INSTITUTE OF SCIENCE AND TECHNOLOGY")
    print("MACHINE LEARNING - Final Project")
    print("Machine Learning Pipeline: Feature Selection and Hyperparameter Optimization")
    print("="*90)
    
    # ========================================================================
    # SECTION 1: DATA PREPROCESSING
    # ========================================================================
    
    df = load_higgs_data('HIGGS.csv', n_samples=100000)
    if df is None:
        return
    
    y = df['class'].values
    X = df.drop('class', axis=1)
    feature_names = X.columns.tolist()
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Target distribution:")
    print(f"  Class 0: {np.sum(y==0):,} ({np.sum(y==0)/len(y)*100:.2f}%)")
    print(f"  Class 1: {np.sum(y==1):,} ({np.sum(y==1)/len(y)*100:.2f}%)")
    
    X_processed = handle_outliers(X, feature_names, method='clip')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # ========================================================================
    # SECTION 2: FEATURE SELECTION AND SCALING
    # ========================================================================
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    X_train_selected, X_test_selected, selected_indices, selector = select_top_features(
        X_train_scaled, y_train, X_test_scaled, n_features=15, method='anova'
    )
    
    fig_importance = create_feature_importance_plot(X_train_scaled, y_train, feature_names)
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\nSaved: feature_importance.png")
    plt.show()
    
    # ========================================================================
    # SECTION 3: MODELING AND EVALUATION
    # ========================================================================
    
    models = {
        'KNN': KNeighborsClassifier,
        'SVM': SVC,
        'MLP': MLPClassifier,
        'XGBoost': XGBClassifier
    }
    
    param_grids = {
        'KNN': {'n_neighbors': [3, 5, 7, 9, 11]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'MLP': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']},
        'XGBoost': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
    }
    
    print("\n" + "="*90)
    print("HYPERPARAMETER RANGES (As per project requirements)")
    print("="*90)
    print("KNN: n_neighbors = [3, 5, 7, 9, 11]")
    print("SVM: C = [0.1, 1, 10], kernel = ['linear', 'rbf']")
    print("MLP: hidden_layer_sizes = [(50,), (100,)], activation = ['relu', 'tanh']")
    print("XGBoost: n_estimators=[50,100], max_depth=[3,5], learning_rate=[0.01,0.1]")
    print("="*90)
    
    total_start = time.time()
    results, best_params, best_features = nested_cross_validation(
        X_train_selected, y_train, models, param_grids, outer_cv=5, inner_cv=3
    )
    total_time = time.time() - total_start
    
    print(f"\nTotal Nested CV Time: {total_time/60:.1f} minutes")
    
    # ========================================================================
    # SECTION 4: FINAL MODEL TRAINING AND EVALUATION
    # ========================================================================
    
    print("\n" + "="*90)
    print("FINAL MODEL TRAINING AND EVALUATION")
    print("="*90)
    
    final_models = {}
    for model_name, model_class in models.items():
        print(f"\nTraining final {model_name} model...")
        
        best_params_model = best_params[model_name][0] if best_params[model_name] else {}
        
        actual_params = {}
        for key, value in best_params_model.items():
            if key.startswith('classifier__'):
                actual_params[key.replace('classifier__', '')] = value
        
        if model_name == 'SVM':
            model = SVC(probability=True, cache_size=1000, shrinking=True, **actual_params)
        elif model_name == 'MLP':
            model = MLPClassifier(max_iter=500, **actual_params)
        else:
            model = model_class(**actual_params)
        
        model.fit(X_train_selected, y_train)
        final_models[model_name] = model
        
        y_pred = model.predict(X_test_selected)
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
        
        print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"  Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # ========================================================================
    # SECTION 5: VISUALIZATION AND RESULTS
    # ========================================================================
    
    create_results_table(results)
    
    fig_roc = plot_roc_curves(final_models, X_test_selected, y_test)
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    print("\nSaved: roc_curves.png")
    plt.show()
    
    fig_roc_ova = plot_roc_curves_ova(final_models, X_test_selected, y_test)
    plt.savefig('roc_curves_ova.png', dpi=300, bbox_inches='tight')
    print("Saved: roc_curves_ova.png")
    plt.show()
    
    fig_cm = plot_confusion_matrices(final_models, X_test_selected, y_test)
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Saved: confusion_matrices.png")
    plt.show()
    
    fig_comparison = plot_model_comparison(results)
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison.png")
    
    # ========================================================================
    # SECTION 6: SUMMARY AND INTERPRETATION
    # ========================================================================
    
    print("\n" + "="*90)
    print("SUMMARY AND INTERPRETATION")
    print("="*90)
    
    best_model = max(results.keys(), key=lambda x: results[x]['roc_auc']['mean'])
    best_auc = results[best_model]['roc_auc']['mean']
    best_auc_std = results[best_model]['roc_auc']['std']
    
    print(f"\nBest Performing Model: {best_model}")
    print(f"ROC-AUC Score: {best_auc:.4f} +- {best_auc_std:.4f}")
    
    print("\nSelected Features (Top 15):")
    for i, idx in enumerate(selected_indices):
        print(f"  {i+1:2d}. {feature_names[idx]}")
    
    print("\n" + "="*90)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("="*90)


if __name__ == "__main__":
    main()