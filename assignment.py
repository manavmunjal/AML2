import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
from xgboost import XGBClassifier, plot_importance
from sklearn.neural_network import MLPClassifier

from ucimlrepo import fetch_ucirepo

# Set up directories
os.makedirs('plots', exist_ok=True)
np.random.seed(42)

# ==========================================
# 1. Data Preparation
# ==========================================
print("Fetching Bank Marketing dataset (id=222)...")
bank_marketing = fetch_ucirepo(id=222)

# data (as pandas dataframes)
X = bank_marketing.data.features
y = bank_marketing.data.targets

# Clean up target: 'yes' -> 1, 'no' -> 0
y = (y['y'] == 'yes').astype(int)

# Handle categorical variables
print("Preprocessing features...")
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# One-hot encoding for categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split data into train / validation / test sets (70 / 15 / 15)
# First split 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.30, random_state=42, stratify=y)
# Split temp into 50% val, 50% test (each 15% of total)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Feature scaling for MLP
scaler = StandardScaler()
# Fit only on training set!
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert to pandas for easier handling later
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# ==========================================
# 2. Gradient Boosted Tree (GBDT) Implementation
# ==========================================
print("\n--- Training GBDT ---")
start_time = time.time()

# Hyperparameter search space
gbdt_param_dist = {
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 10]
}

xgb = XGBClassifier(eval_metric='logloss', random_state=42)
gbdt_search = RandomizedSearchCV(xgb, param_distributions=gbdt_param_dist, n_iter=10, 
                                 scoring='f1', cv=3, random_state=42, n_jobs=-1, verbose=1)
gbdt_search.fit(X_train, y_train)

best_gbdt = gbdt_search.best_estimator_
print(f"Best GBDT parameters: {gbdt_search.best_params_}")

# Now retrain best model WITH early stopping purely for visualization of training vs val loss
best_params = gbdt_search.best_params_.copy()
best_params['n_estimators'] = 1000 # Large value for early stopping

print("Training final GBDT model with early stopping to extract curves...")
final_gbdt = XGBClassifier(**best_params, random_state=42, 
                           early_stopping_rounds=20, eval_metric='logloss') 
eval_set = [(X_train, y_train), (X_val, y_val)]
final_gbdt.fit(X_train, y_train, eval_set=eval_set, verbose=False)
gbdt_time = time.time() - start_time
print(f"GBDT Training & Tuning Time: {gbdt_time:.2f} seconds")

# Visualize Training vs validation loss
results = final_gbdt.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Validation')
plt.legend()
plt.title('GBDT Training vs Validation Loss')
plt.ylabel('Log Loss')
plt.xlabel('Trees (n_estimators)')
plt.savefig('plots/gbdt_loss_curve.png')
plt.close()

# Feature importance
fig, ax = plt.subplots(figsize=(10, 8))
plot_importance(final_gbdt, max_num_features=15, title="GBDT Feature Importance (Top 15)", ax=ax)
plt.savefig('plots/gbdt_feature_importance.png', bbox_inches='tight')
plt.close()

# Effect of learning rate
print("Evaluating effect of learning rate on GBDT...")
lrs = [0.01, 0.1, 0.3]
lr_val_losses = []
for lr in lrs:
    model = XGBClassifier(learning_rate=lr, n_estimators=200, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    lr_val_losses.append(model.evals_result()['validation_0']['logloss'])

plt.figure(figsize=(10, 6))
for i, lr in enumerate(lrs):
    plt.plot(range(len(lr_val_losses[i])), lr_val_losses[i], label=f'LR: {lr}')
plt.legend()
plt.title('Effect of Learning Rate on GBDT Validation Loss')
plt.ylabel('Log Loss')
plt.xlabel('Trees')
plt.savefig('plots/gbdt_lr_effect.png')
plt.close()

# ==========================================
# 3. Multi-Layer Perceptron (MLP) Implementation
# ==========================================
print("\n--- Training MLP ---")
start_mlp_time = time.time()

mlp_param_dist = {
    'hidden_layer_sizes': [(64,), (128, 64), (256, 128, 64)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [300]
}

mlp = MLPClassifier(random_state=42, early_stopping=False)
mlp_search = RandomizedSearchCV(mlp, param_distributions=mlp_param_dist, n_iter=5, # Reduced n_iter to save time
                                scoring='f1', cv=3, random_state=42, n_jobs=-1, verbose=1)
mlp_search.fit(X_train_scaled, y_train)

best_mlp = mlp_search.best_estimator_
print(f"Best MLP parameters: {mlp_search.best_params_}")
mlp_time = time.time() - start_mlp_time
print(f"MLP Training & Tuning Time: {mlp_time:.2f} seconds")

# Visualize Training loss curve
plt.figure(figsize=(10, 6))
plt.plot(best_mlp.loss_curve_)
plt.title('MLP Training Loss Curve')
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.savefig('plots/mlp_loss_curve.png')
plt.close()

# Visualize effect of network depth/width
print("Evaluating effect of MLP architecture...")
architectures = [(64,), (128, 64), (256, 128, 64)]
arch_names = ['Shallow (64,)', 'Medium (128, 64)', 'Deep (256, 128, 64)']
arch_scores = []
for arch in architectures:
    model = MLPClassifier(hidden_layer_sizes=arch, max_iter=300, random_state=42)
    model.fit(X_train_scaled, y_train)
    # Validate on val set
    val_preds = model.predict(X_val_scaled)
    arch_scores.append(f1_score(y_val, val_preds))

plt.figure(figsize=(8, 6))
plt.bar(arch_names, arch_scores, color=['lightblue', 'skyblue', 'steelblue'])
plt.title('Effect of Network Architecture on Validation F1-Score')
plt.ylabel('F1 Score')
plt.savefig('plots/mlp_arch_effect.png')
plt.close()

# ==========================================
# 4. GBDT vs MLP Comparison
# ==========================================
print("\n--- Evaluation on Test Set ---")

# Predictions
gbdt_preds = final_gbdt.predict(X_test)
gbdt_probs = final_gbdt.predict_proba(X_test)[:, 1]

mlp_preds = best_mlp.predict(X_test_scaled)
mlp_probs = best_mlp.predict_proba(X_test_scaled)[:, 1]

# Metrics Function
def evaluate_model(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_pr = average_precision_score(y_true, y_prob)
    return [acc, prec, rec, f1, auc_pr]

gbdt_metrics = evaluate_model(y_test, gbdt_preds, gbdt_probs)
mlp_metrics = evaluate_model(y_test, mlp_preds, mlp_probs)

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-PR'],
    'GBDT': gbdt_metrics,
    'MLP': mlp_metrics
})

print("\nPerformance Comparison Table:")
print(metrics_df.to_markdown(index=False))

# Bar chart comparison
metrics_df_melted = metrics_df.melt(id_vars='Metric', var_name='Model', value_name='Score')
plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_df_melted, x='Metric', y='Score', hue='Model')
plt.title('GBDT vs MLP Test Set Performance Comparison')
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.savefig('plots/comparison_metrics.png')
plt.close()

# Record results to a text file for reporting
results_text = f"GBDT metrics: {gbdt_metrics}\nMLP metrics: {mlp_metrics}\nGBDT Time: {gbdt_time}\nMLP Time: {mlp_time}\n"
with open('results.txt', 'w') as f:
    f.write(results_text)
    f.write("\n\nPerformance Comparison Table:\n")
    f.write(metrics_df.to_markdown(index=False))

print("\nPipeline finished successfully. All plots saved to 'plots/' directory.")
