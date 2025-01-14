import numpy as np
import pandas as pd
import os
import joblib
import json
import shap
import time
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, log_loss)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import optuna

# Set global random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Reduce Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)
# Reduce warnings
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    filename='model_results_confounders/model_training_conf.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


# Get the best model based on mean accuracy
best_fs = 'SFS-CV'
best_model_type = 'RandomForest'
# Model type from best model
model_type = best_model_type

# Load the data with confounders only
df = pd.read_csv("../../ML_Data/FINAL_GENUS_TAXA_CLR.csv")
y = df["Group"]
X = df.iloc[:, -13:]  # Select last 13 columns
X = X.drop(columns=["Epigenetic_deviation", "Group"])

# Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Outer cross-validation loop
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Define base output directory
base_output_dir = 'model_results_confounders'
os.makedirs(base_output_dir, exist_ok=True)

# Initialize results list
results = []

# Nested cross-validation
outer_fold_counter = 1  # To track outer folds

for train_index, test_index in outer_cv.split(X, y_encoded):
    fold_start_time = time.time()

    # Create directory for this outer fold
    outer_fold_dir = os.path.join(base_output_dir, f'outer_fold_{outer_fold_counter}')
    os.makedirs(outer_fold_dir, exist_ok=True)

    # Save outer train/test indices
    np.save(os.path.join(outer_fold_dir, 'train_indices.npy'), train_index)
    np.save(os.path.join(outer_fold_dir, 'test_indices.npy'), test_index)

    X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
    y_train_outer, y_test_outer = y_encoded[train_index], y_encoded[test_index]

    # Define the objective function for Optuna
    def objective(trial):
        # Hyperparameter tuning for RandomForest or XGBoost
        if model_type == 'RandomForest':
            max_depth = trial.suggest_categorical('max_depth', [None, 5, 10, 15, 25])
            n_estimators = trial.suggest_int('n_estimators', 50, 700, step=65)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=SEED,
                n_jobs=-1
            )
        else:
            raise ValueError('Unknown model type')

        # Inner cross-validation loop
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        inner_scores = []

        for inner_train_index, inner_val_index in inner_cv.split(X_train_outer, y_train_outer):
            X_train_inner = X_train_outer.iloc[inner_train_index]
            X_val_inner = X_train_outer.iloc[inner_val_index]
            y_train_inner = y_train_outer[inner_train_index]
            y_val_inner = y_train_outer[inner_val_index]

            # Train the model
            model.fit(X_train_inner, y_train_inner)
            y_pred_val = model.predict(X_val_inner)
            val_accuracy = accuracy_score(y_val_inner, y_pred_val)
            inner_scores.append(val_accuracy)

        return np.mean(inner_scores)

    # Perform hyperparameter optimization with Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25, n_jobs=-1)

    # Get the best hyperparameters
    best_params = study.best_params

    # Recreate the best model with the tuned hyperparameters
    if model_type == 'RandomForest':
        model = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=-1)
    elif model_type == 'XGBoost':
        model = XGBClassifier(**best_params, use_label_encoder=False, verbosity=0, random_state=SEED, n_jobs=-1)
    else:
        raise ValueError('Unknown model type')

    # Train the model on the entire outer training set
    model.fit(X_train_outer, y_train_outer)

    # Save the model
    joblib.dump(model, os.path.join(outer_fold_dir, 'model.pkl'))

    # Evaluate on the outer test data
    y_pred_outer = model.predict(X_test_outer)
    y_proba_outer = model.predict_proba(X_test_outer)

    # Decode labels back to original
    y_test_outer_decoded = label_encoder.inverse_transform(y_test_outer)
    y_pred_outer_decoded = label_encoder.inverse_transform(y_pred_outer)

    # Compute performance metrics
    accuracy = accuracy_score(y_test_outer, y_pred_outer)
    precision = precision_score(y_test_outer, y_pred_outer, average='weighted', zero_division=0)
    recall = recall_score(y_test_outer, y_pred_outer, average='weighted', zero_division=0)
    f1 = f1_score(y_test_outer, y_pred_outer, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test_outer, y_pred_outer)
    roc_auc = roc_auc_score(y_test_outer, y_proba_outer[:, 1]) if len(label_encoder.classes_) == 2 else roc_auc_score(
        y_test_outer, y_proba_outer, multi_class='ovo', average='weighted')

    logloss = log_loss(y_test_outer, y_proba_outer, labels=label_encoder.transform(label_encoder.classes_))

    # Save metrics
    metrics = {
        'model_type': model_type,
        'outer_fold': outer_fold_counter,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'log_loss': logloss
    }

    # After computing metrics
    results.append(metrics)


    with open(os.path.join(outer_fold_dir, 'performance_metrics.json'), 'w') as f:
        json.dump(metrics, f)

    # Save confusion matrix
    np.save(os.path.join(outer_fold_dir, 'confusion_matrix.npy'), conf_matrix)

    # Save predictions
    predictions_df = pd.DataFrame({
        'y_true': y_test_outer_decoded,
        'y_pred': y_pred_outer_decoded
    })

    # Adding probability columns for each class
    for i, class_label in enumerate(label_encoder.classes_):
        predictions_df[class_label] = y_proba_outer[:, i]

    predictions_df.to_csv(os.path.join(outer_fold_dir, 'predictions.csv'), index=False)

    # Compute SHAP values if applicable
    if model_type in ['RandomForest', 'XGBoost']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_outer)
        joblib.dump(shap_values, os.path.join(outer_fold_dir, 'shap_values.pkl'))

    # Logging
    fold_elapsed_time = time.time() - fold_start_time
    logging.info(f"Completed outer fold {outer_fold_counter}")
    logging.info(f"Fold Time: {fold_elapsed_time:.2f} seconds")

    outer_fold_counter += 1

# Save the results summary
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(base_output_dir, 'overall_results.csv'), index=False)

