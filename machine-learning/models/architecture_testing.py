import numpy as np
import pandas as pd
import os
import joblib
import json
import shap
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, log_loss)
from sklearn.preprocessing import LabelEncoder
import optuna
import logging

# Reduce Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)
# Reduce XGBoost verbosity
import warnings
warnings.filterwarnings('ignore')

# Record the start time
start_time = time.time()

# Data
df = pd.read_csv("../ML_Data/FINAL_GENUS_TAXA_CLR.csv")

y = df["Group"]
X = df.iloc[:, 1:-13]

# Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Only for testing: use a very small subset of data (~1/300th)
y_encoded = y_encoded[0:15]
X = X.iloc[0:15, 0:15]

# Define outer cross-validation with StratifiedKFold
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define feature selection methods and model types
fs_methods = ['None', 'LASSO', 'SFS-CV']
model_types = ['RandomForest', 'XGBoost', 'StochasticXGBoost']

# Initialize results list
results = []

# Define base output directory
base_output_dir = 'model_results'
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

# Nested cross-validation
for fs_method in fs_methods:
    for model_type in model_types:
        # Initialize list to store scores from outer folds
        outer_scores = []
        # Create directory for this fs_method and model_type
        model_fs_dir = os.path.join(base_output_dir, f'{fs_method}_{model_type}')
        if not os.path.exists(model_fs_dir):
            os.makedirs(model_fs_dir)

        outer_fold_counter = 1  # Reset counter for each combination
        # Outer CV loop
        for train_index, test_index in outer_cv.split(X, y_encoded):
            # Create directory for this outer fold
            outer_fold_dir = os.path.join(model_fs_dir, f'outer_fold_{outer_fold_counter}')
            if not os.path.exists(outer_fold_dir):
                os.makedirs(outer_fold_dir)

            # Save outer train/test indices
            np.save(os.path.join(outer_fold_dir, 'train_indices.npy'), train_index)
            np.save(os.path.join(outer_fold_dir, 'test_indices.npy'), test_index)

            X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
            y_train_outer, y_test_outer = y_encoded[train_index], y_encoded[test_index]

            # Define the objective function for Optuna
            def objective(trial):
                # Model hyperparameters
                if model_type == 'RandomForest':
                    n_estimators = trial.suggest_int('n_estimators', 50, 200)
                    max_depth = trial.suggest_int('max_depth', 3, 10)
                    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                    model = RandomForestClassifier(
                        n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        min_samples_split=min_samples_split
                    )
                elif model_type == 'XGBoost':
                    # Normal XGBoost without stochasticity
                    max_depth = trial.suggest_int('max_depth', 3, 10)
                    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                    model = XGBClassifier(
                        max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        learning_rate=learning_rate, 
                        use_label_encoder=False,
                        eval_metric='logloss',
                        verbosity=0  # Reduce XGBoost verbosity
                    )
                elif model_type == 'StochasticXGBoost':
                    # Stochastic XGBoost (subsample rows and columns)
                    max_depth = trial.suggest_int('max_depth', 3, 10)
                    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                    subsample = trial.suggest_float('subsample', 0.5, 1.0)  # Sample rows
                    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)  # Sample features
                    model = XGBClassifier(
                        max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        learning_rate=learning_rate, 
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        use_label_encoder=False,
                        eval_metric='logloss',
                        verbosity=0  # Reduce XGBoost verbosity
                    )
                else:
                    raise ValueError('Unknown model type')

                # Inner cross-validation with StratifiedKFold
                inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                inner_scores = []

                for inner_train_index, inner_val_index in inner_cv.split(X_train_outer, y_train_outer):
                    X_train_inner = X_train_outer.iloc[inner_train_index]
                    X_val_inner = X_train_outer.iloc[inner_val_index]
                    y_train_inner = y_train_outer[inner_train_index]
                    y_val_inner = y_train_outer[inner_val_index]

                    # Perform feature selection for this fold
                    if fs_method == 'None':
                        X_train_selected = X_train_inner  # No selection, use full data
                        X_val_selected = X_val_inner
                    elif fs_method == 'LASSO':
                        lasso = Lasso(alpha=0.01, max_iter=10000)
                        selector = SelectFromModel(estimator=lasso)
                        selector.fit(X_train_inner, y_train_inner)
                        X_train_selected = selector.transform(X_train_inner)
                        X_val_selected = selector.transform(X_val_inner)
                    elif fs_method == 'SFS-CV':
                        sfs_estimator = RandomForestClassifier()
                        selector = SequentialFeatureSelector(
                            estimator=sfs_estimator, direction='forward',
                            scoring='accuracy', cv=3, n_jobs=-1
                        )
                        selector.fit(X_train_inner, y_train_inner)
                        X_train_selected = selector.transform(X_train_inner)
                        X_val_selected = selector.transform(X_val_inner)
                    else:
                        raise ValueError('Unknown feature selection method')

                    # Fit the model
                    model.fit(X_train_selected, y_train_inner)
                    y_pred_val = model.predict(X_val_selected)

                    # Evaluate and store the accuracy for this fold
                    val_accuracy = accuracy_score(y_val_inner, y_pred_val)
                    inner_scores.append(val_accuracy)

                # Return the mean score across the inner folds
                return np.mean(inner_scores)

            # Perform hyperparameter optimization with Optuna
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20, show_progress_bar=False)

            # Get the best hyperparameters
            best_params = study.best_params

            # Save best hyperparameters
            with open(os.path.join(outer_fold_dir, 'best_hyperparameters.json'), 'w') as f:
                json.dump(best_params, f)

            # Recreate the best model
            if model_type == 'RandomForest':
                model = RandomForestClassifier(**best_params)
            elif model_type == 'XGBoost':
                model = XGBClassifier(
                    **best_params, use_label_encoder=False, eval_metric='logloss', verbosity=0
                )
            elif model_type == 'StochasticXGBoost':
                model = XGBClassifier(
                    **best_params, use_label_encoder=False, eval_metric='logloss', verbosity=0
                )
            else:
                raise ValueError('Unknown model type')

            # Re-apply the feature selection using all outer training data
            if fs_method == 'None':
                X_train_selected = X_train_outer
                X_test_selected = X_test_outer
                selected_features = X_train_outer.columns
            elif fs_method == 'LASSO':
                lasso = Lasso(alpha=0.01, max_iter=10000)
                selector = SelectFromModel(estimator=lasso)
                selector.fit(X_train_outer, y_train_outer)
                X_train_selected = selector.transform(X_train_outer)
                X_test_selected = selector.transform(X_test_outer)
                feature_mask = selector.get_support()
                selected_features = X_train_outer.columns[feature_mask]
            elif fs_method == 'SFS-CV':
                sfs_estimator = RandomForestClassifier()
                selector = SequentialFeatureSelector(
                    estimator=sfs_estimator, direction='forward',
                    scoring='accuracy', cv=3, n_jobs=-1
                )
                selector.fit(X_train_outer, y_train_outer)
                X_train_selected = selector.transform(X_train_outer)
                X_test_selected = selector.transform(X_test_outer)
                feature_mask = selector.get_support()
                selected_features = X_train_outer.columns[feature_mask]
            else:
                raise ValueError('Unknown feature selection method')

            # Save selected features
            selected_features_df = pd.DataFrame(selected_features, columns=['feature'])
            selected_features_df.to_csv(os.path.join(outer_fold_dir, 'selected_features.csv'), index=False)

            # Fit the model on the selected features from the outer training data
            if model_type.startswith('XGBoost'):
                eval_set = [(X_train_selected, y_train_outer), (X_test_selected, y_test_outer)]
                model.fit(X_train_selected, y_train_outer, eval_set=eval_set, verbose=False)
                # Save training history
                training_history = model.evals_result()
                with open(os.path.join(outer_fold_dir, 'training_history.json'), 'w') as f:
                    json.dump(training_history, f)
            else:
                model.fit(X_train_selected, y_train_outer)

            # Save the model
            joblib.dump(model, os.path.join(outer_fold_dir, 'model.pkl'))

            # Evaluate on the outer test data
            y_pred_outer = model.predict(X_test_selected)
            y_proba_outer = model.predict_proba(X_test_selected)

            # Decode labels back to original
            y_test_outer_decoded = label_encoder.inverse_transform(y_test_outer)
            y_pred_outer_decoded = label_encoder.inverse_transform(y_pred_outer)

            # Compute performance metrics
            accuracy = accuracy_score(y_test_outer, y_pred_outer)
            precision = precision_score(y_test_outer, y_pred_outer, average='weighted', zero_division=0)
            recall = recall_score(y_test_outer, y_pred_outer, average='weighted', zero_division=0)
            f1 = f1_score(y_test_outer, y_pred_outer, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(y_test_outer, y_pred_outer)
            roc_auc = roc_auc_score(y_test_outer, y_proba_outer[:, 1])
            logloss = log_loss(y_test_outer, y_proba_outer, labels=label_encoder.transform(label_encoder.classes_))

            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'log_loss': logloss
            }

            # Save performance metrics
            with open(os.path.join(outer_fold_dir, 'performance_metrics.json'), 'w') as f:
                json.dump(metrics, f)

            # Save confusion matrix
            np.save(os.path.join(outer_fold_dir, 'confusion_matrix.npy'), conf_matrix)

            # Save predictions
            proba_df = pd.DataFrame(y_proba_outer, columns=label_encoder.classes_)
            predictions_df = pd.DataFrame({
                'y_true': y_test_outer_decoded,
                'y_pred': y_pred_outer_decoded
            })
            predictions_df = pd.concat([predictions_df, proba_df.reset_index(drop=True)], axis=1)
            predictions_df.to_csv(os.path.join(outer_fold_dir, 'predictions.csv'), index=False)

            # Save feature importances if available
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
                feat_imp_df = pd.DataFrame({
                    'feature': selected_features,
                    'importance': feature_importances
                })
                feat_imp_df.to_csv(os.path.join(outer_fold_dir, 'feature_importances.csv'), index=False)

            # Compute SHAP values
            if model_type in ['RandomForest', 'XGBoost', 'StochasticXGBoost']:
                explainer = shap.TreeExplainer(model)
                shap_values_train = explainer.shap_values(X_train_selected)
                shap_values_test = explainer.shap_values(X_test_selected)
                # Save SHAP values
                joblib.dump(shap_values_train, os.path.join(outer_fold_dir, 'shap_values_train.pkl'))
                joblib.dump(shap_values_test, os.path.join(outer_fold_dir, 'shap_values_test.pkl'))

            # Append outer score
            outer_scores.append(accuracy)
            outer_fold_counter += 1  # Increment outer fold counter

        # Store the results
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        results.append({
            'fs_method': fs_method,
            'model_type': model_type,
            'mean_accuracy': mean_score,
            'std_accuracy': std_score
        })

        # Status update with elapsed time and average accuracy
        elapsed_time = time.time() - start_time
        print(f"Completed fs_method: {fs_method}, model_type: {model_type}")
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Average Accuracy on Outer Folds: {mean_score:.4f}\n")

# Record the end time
end_time = time.time()
total_time = end_time - start_time

print(f"Total runtime: {total_time:.2f} seconds")

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print(results_df)
