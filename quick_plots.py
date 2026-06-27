"""
Quick script to generate missing plots without full CV retraining
Only trains final models (fast, ~1-2 minutes)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("Generating Missing Plots (Fast Mode - No CV)")
print("="*70)

# Load data (same as before, takes ~2-3 seconds)
print("\nLoading data...")
column_names = ['class'] + [f'feature_{i}' for i in range(1, 29)]
df = pd.read_csv('HIGGS.csv', header=None, names=column_names, nrows=100000)

y = df['class'].values
X = df.drop('class', axis=1)
feature_names = X.columns.tolist()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection (top 15)
selector = SelectKBest(score_func=f_classif, k=15)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
selected_indices = selector.get_support(indices=True)

print(f"Data loaded: {X_train_selected.shape[0]} training, {X_test_selected.shape[0]} test")

# Train final models with best parameters from your run
print("\nTraining final models (fast, no CV)...")

best_params = {
    'KNN': {'n_neighbors': 11},
    'SVM': {'C': 1, 'kernel': 'rbf'},
    'MLP': {'hidden_layer_sizes': (100,), 'activation': 'relu'},
    'XGBoost': {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}
}

final_models = {}

# KNN
print("  Training KNN...")
final_models['KNN'] = KNeighborsClassifier(**best_params['KNN'])
final_models['KNN'].fit(X_train_selected, y_train)

# SVM
print("  Training SVM...")
final_models['SVM'] = SVC(probability=True, cache_size=500, **best_params['SVM'])
final_models['SVM'].fit(X_train_selected, y_train)

# MLP
print("  Training MLP...")
final_models['MLP'] = MLPClassifier(max_iter=500, **best_params['MLP'])
final_models['MLP'].fit(X_train_selected, y_train)

# XGBoost
print("  Training XGBoost...")
final_models['XGBoost'] = XGBClassifier(**best_params['XGBoost'])
final_models['XGBoost'].fit(X_train_selected, y_train)

print("\nAll models trained!")

# ============================================================================
# Generate plots
# ============================================================================

print("\nGenerating plots...")

# 1. ROC Curves (OVA)
print("  Generating ROC Curves (OVA)...")
fig, axes = plt.subplots(1, 1, figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, (model_name, model) in enumerate(final_models.items()):
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', 
            linewidth=2, color=colors[idx % len(colors)])

plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1, alpha=0.7)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("  Saved: roc_curves.png")
plt.close()

# 2. ROC Curves (OVA - One vs All)
print("  Generating ROC Curves OVA...")
fig, axes = plt.subplots(1, 1, figsize=(10, 8))

for idx, (model_name, model) in enumerate(final_models.items()):
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', 
            linewidth=2, color=colors[idx % len(colors)])

plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1, alpha=0.7)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves (OVA Approach)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves_ova.png', dpi=300, bbox_inches='tight')
print("  Saved: roc_curves_ova.png")
plt.close()

# 3. Confusion Matrices
print("  Generating Confusion Matrices...")
n_models = len(final_models)
fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4.5))

if n_models == 1:
    axes = [axes]

for idx, (model_name, model) in enumerate(final_models.items()):
    y_pred = model.predict(X_test_selected)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
               cbar=False, square=True)
    axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=10)
    axes[idx].set_ylabel('Actual', fontsize=10)
    axes[idx].set_xticklabels(['Class 0', 'Class 1'])
    axes[idx].set_yticklabels(['Class 0', 'Class 1'])

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("  Saved: confusion_matrices.png")
plt.close()

# 4. Model Comparison
print("  Generating Model Comparison...")
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
model_names = list(final_models.keys())

fig, ax = plt.subplots(figsize=(14, 7))

# Use results from your run
results = {
    'KNN': {'accuracy': 0.6630, 'precision': 0.6655, 'recall': 0.7280, 'f1': 0.6954, 'roc_auc': 0.7179},
    'SVM': {'accuracy': 0.5038, 'precision': 0.5765, 'recall': 0.2275, 'f1': 0.3258, 'roc_auc': 0.5545},
    'MLP': {'accuracy': 0.7100, 'precision': 0.7227, 'recall': 0.7332, 'f1': 0.7273, 'roc_auc': 0.7857},
    'XGBoost': {'accuracy': 0.7127, 'precision': 0.7288, 'recall': 0.7266, 'f1': 0.7277, 'roc_auc': 0.7883}
}

x = np.arange(len(metrics))
width = 0.2
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, model_name in enumerate(model_names):
    values = [results[model_name][metric] for metric in metrics]
    offset = (i - len(model_names)/2 + 0.5) * width
    ax.bar(x + offset, values, width, label=model_name, 
           alpha=0.8, color=colors[i % len(colors)])

ax.set_xlabel('Metrics', fontsize=13)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("  Saved: model_comparison.png")
plt.close()

print("\n" + "="*70)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*70)
print("\nYou now have all 5 PNG files:")
print("  feature_importance.png")
print("  roc_curves.png")
print("  roc_curves_ova.png")
print("  confusion_matrices.png")
print("  model_comparison.png")