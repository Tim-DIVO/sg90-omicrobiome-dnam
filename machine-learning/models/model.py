import numpy as np
import pandas as pd
import os
import joblib
import json
import shap
import time
import logging
import psutil  # For memory usage logging
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, log_loss)
from sklearn.preprocessing import LabelEncoder
import optuna
import warnings
import traceback

if __name__ == "__main__":
    # Set global random seed for reproducibility
    SEED = 42
    np.random.seed(SEED)

    # Reduce Optuna verbosity
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Reduce XGBoost verbosity
    warnings.filterwarnings('ignore')

    # Set up logging
    logger = logging.getLogger()
    if not logger.handlers:
        logging.basicConfig(
            filename='model_training.log',
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )

    # Function to log memory usage
    def log_memory_usage(stage):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logging.info(f"{stage}: Memory usage: {mem_info.rss / (1024 ** 3):.2f} GB | Virtual memory: {mem_info.vms / (1024 ** 3):.2f} GB")

    # Record the start time
    start_time = time.time()

    # Data
    try:
        df = pd.read_csv("ML_Data/FINAL_GENUS_TAXA_CLR.csv")
    except Exception as e:
        logging.error(f"Error loading data:, Error: {str(e)}", exc_info=True)
        raise e

    y = df["Group"]
    X = df.iloc[:, 1:-13]  # Adjust as per your dataset structure

    # for testing purposes
    #X = X.iloc[:15,:15]
    #y = y[:15]

    # Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Log memory usage after loading data
    log_memory_usage("After loading data")

    # Outer folds
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Define feature selection methods and model types
    fs_methods = ['None', 'L1-LogReg', 'SFS-CV']
    model_types = ['RandomForest', 'XGBoost', 'StochasticXGBoost']

    # Initialize results list
    results = []

    # Define base output directory
    base_output_dir = 'taxa_results'
    os.makedirs(base_output_dir, exist_ok=True)

    # Nested cross-validation
    for fs_method in fs_methods:
        for model_type in model_types:
            # Initialize list to store scores from outer folds
            outer_scores = []
            # List to store performance metrics across folds
            performance_metrics_list = []

            # Create directory for this fs_method and model_type
            model_fs_dir = os.path.join(base_output_dir, f'{fs_method}_{model_type}')
            os.makedirs(model_fs_dir, exist_ok=True)

            outer_fold_counter = 1  # Reset counter for each combination
            # Outer CV loop
            for train_index, test_index in outer_cv.split(X, y_encoded):
                fold_start_time = time.time()

                # Log memory usage at the start of each outer fold
                log_memory_usage(f"Start of outer fold {outer_fold_counter}")

                # Create directory for this outer fold
                outer_fold_dir = os.path.join(model_fs_dir, f'outer_fold_{outer_fold_counter}')
                os.makedirs(outer_fold_dir, exist_ok=True)

                # Save outer train/test indices
                np.save(os.path.join(outer_fold_dir, 'train_indices.npy'), train_index)
                np.save(os.path.join(outer_fold_dir, 'test_indices.npy'), test_index)

                X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
                y_train_outer, y_test_outer = y_encoded[train_index], y_encoded[test_index]

                # Perform feature selection once per outer fold
                try:
                    if fs_method == 'None':
                        X_train_selected = X_train_outer.values  # Convert to NumPy array
                        X_test_selected = X_test_outer.values
                        selected_features = X_train_outer.columns
                    elif fs_method == 'L1-LogReg':
                        # LogisticRegressionCV with L1 penalty
                        logistic_cv = LogisticRegressionCV(
                            Cs=10,
                            cv=3,
                            penalty='l1',
                            solver='saga',
                            random_state=SEED,
                            n_jobs=-1,
                            max_iter=1000
                        )
                        selector = SelectFromModel(estimator=logistic_cv)
                        selector.fit(X_train_outer, y_train_outer)
                        X_train_selected = selector.transform(X_train_outer)
                        X_test_selected = selector.transform(X_test_outer)
                        feature_mask = selector.get_support()
                        selected_features = X_train_outer.columns[feature_mask]
                        del logistic_cv, selector
                    elif fs_method == 'SFS-CV':
                        sfs_estimator = RandomForestClassifier(random_state=SEED, n_jobs=-1)
                        selector = SequentialFeatureSelector(
                            estimator=sfs_estimator,
                            direction='backward',
                            scoring='accuracy',
                            cv=3,
                            n_jobs=-1
                        )
                        selector.fit(X_train_outer, y_train_outer)
                        X_train_selected = selector.transform(X_train_outer)
                        X_test_selected = selector.transform(X_test_outer)
                        feature_mask = selector.get_support()
                        selected_features = X_train_outer.columns[feature_mask]
                        del selector, sfs_estimator
                    else:
                        raise ValueError('Unknown feature selection method')
                except Exception as e:
                    logging.error(f"Feature selection failed for fs_method: {fs_method}, model_type: {model_type}, outer_fold: {outer_fold_counter}, Error: {str(e)}", exc_info=True)
                    continue

                # Check if any features were selected
                if X_train_selected.shape[1] == 0:
                    logging.warning(f"No features selected for fs_method: {fs_method}, model_type: {model_type}, outer_fold: {outer_fold_counter}")
                    continue

                # Save selected features
                selected_features_df = pd.DataFrame(selected_features, columns=['feature'])
                selected_features_df.to_csv(os.path.join(outer_fold_dir, 'selected_features.csv'), index=False)

                # Define the objective function for Optuna
                def objective(trial):
                    # Model hyperparameters
                    try:
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
                        elif model_type == 'XGBoost' or model_type == 'StochasticXGBoost':
                            max_depth = trial.suggest_int('max_depth', 1, 15)
                            min_child_weight = trial.suggest_float('min_child_weight', 0.5, 10.0)
                            learning_rate = trial.suggest_float('learning_rate', 0.001, 0.25, log=True)
                            n_estimators = trial.suggest_int('n_estimators', 50, 1000, step=95)
                            params = {
                                'max_depth': max_depth,
                                'min_child_weight': min_child_weight,
                                'learning_rate': learning_rate,
                                'n_estimators': n_estimators,
                                'use_label_encoder': False,
                                'verbosity': 0,
                                'random_state': SEED,
                                'n_jobs': -1
                            }
                            if model_type == 'StochasticXGBoost':
                                subsample = trial.suggest_float('subsample', 0.5, 1.0)
                                colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
                                params['subsample'] = subsample
                                params['colsample_bytree'] = colsample_bytree
                            if fs_method == 'None':
                                reg_alpha = trial.suggest_float('reg_alpha', 0.0, 0.1)
                                params['reg_alpha'] = reg_alpha
                            model = XGBClassifier(**params)
                        else:
                            raise ValueError('Unknown model type')
                    except Exception as e:
                        logging.error(f"Error setting up the model in objective function:, Error: {str(e)}", exc_info=True)
                        raise e

                    # Inner cross-validation with StratifiedKFold
                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
                    inner_scores = []

                    for inner_train_index, inner_val_index in inner_cv.split(X_train_selected, y_train_outer):
                        try:
                            X_train_inner = X_train_selected[inner_train_index]
                            X_val_inner = X_train_selected[inner_val_index]
                            y_train_inner = y_train_outer[inner_train_index]
                            y_val_inner = y_train_outer[inner_val_index]

                            # Fit the model
                            model.fit(X_train_inner, y_train_inner)
                            y_pred_val = model.predict(X_val_inner)
                            val_accuracy = accuracy_score(y_val_inner, y_pred_val)
                            inner_scores.append(val_accuracy)

                            # Clean up variables to free up memory
                            del X_train_inner, X_val_inner, y_train_inner, y_val_inner, y_pred_val
                        except Exception as e:
                            logging.error(f"Error during inner cross-validation:, Error: {str(e)}", exc_info=True)
                            return 0.0

                    # Clean up model
                    del model

                    return np.mean(inner_scores)

                # Perform hyperparameter optimization with Optuna
                try:
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=25, n_jobs=1)
                except Exception as e:
                    logging.error(f"Error during hyperparameter optimization with Optuna:, Error: {str(e)}", exc_info=True)
                    continue

                # Check if any trials completed successfully
                if len(study.trials) == 0 or study.best_trial is None or study.best_value == 0.0:
                    logging.warning(f"No successful trials for fs_method: {fs_method}, model_type: {model_type}, outer_fold: {outer_fold_counter}")
                    continue

                # Get the best hyperparameters
                try:
                    best_params = study.best_params
                    # Save best hyperparameters for this fold
                    with open(os.path.join(outer_fold_dir, 'best_hyperparameters.json'), 'w') as f:
                        json.dump(best_params, f)
                except ValueError as e:
                    logging.error(f"Error retrieving best hyperparameters:, Error: {str(e)}", exc_info=True)
                    continue

                # Log memory usage after hyperparameter optimization
                log_memory_usage(f"After hyperparameter optimization for outer fold {outer_fold_counter}")

                # Recreate the best model
                try:
                    if model_type == 'RandomForest':
                        model = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=-1)
                    elif model_type == 'XGBoost' or model_type == 'StochasticXGBoost':
                        model = XGBClassifier(**best_params, use_label_encoder=False, verbosity=0, random_state=SEED, n_jobs=-1)
                    else:
                        raise ValueError('Unknown model type')
                except Exception as e:
                    logging.error(f"Error recreating the model with best hyperparameters:, Error: {str(e)}", exc_info=True)
                    continue

                # Fit the model on the selected features from the outer training data
                try:
                    model.fit(X_train_selected, y_train_outer)
                except Exception as e:
                    logging.error(f"Model training failed for fs_method: {fs_method}, model_type: {model_type}, outer_fold: {outer_fold_counter}", exc_info=True)
                    continue

                # Save the model
                try:
                    joblib.dump(model, os.path.join(outer_fold_dir, 'model.pkl'))
                except Exception as e:
                    logging.error(f"Error saving the trained model:, Error: {str(e)}", exc_info=True)

                # Evaluate on the outer test data
                try:
                    y_pred_outer = model.predict(X_test_selected)
                    y_proba_outer = model.predict_proba(X_test_selected)
                except Exception as e:
                    logging.error(f"Error during model evaluation on outer test data:, Error: {str(e)}", exc_info=True)
                    continue

                # Decode labels back to original
                y_test_outer_decoded = label_encoder.inverse_transform(y_test_outer)
                y_pred_outer_decoded = label_encoder.inverse_transform(y_pred_outer)

                # Compute performance metrics
                try:
                    accuracy = accuracy_score(y_test_outer, y_pred_outer)
                    precision = precision_score(y_test_outer, y_pred_outer, average='weighted', zero_division=0)
                    recall = recall_score(y_test_outer, y_pred_outer, average='weighted', zero_division=0)
                    f1 = f1_score(y_test_outer, y_pred_outer, average='weighted', zero_division=0)
                    conf_matrix = confusion_matrix(y_test_outer, y_pred_outer)
                    if len(label_encoder.classes_) == 2:
                        roc_auc = roc_auc_score(y_test_outer, y_proba_outer[:, 1])
                    else:
                        roc_auc = roc_auc_score(y_test_outer, y_proba_outer, multi_class='ovo', average='weighted')
                    logloss = log_loss(y_test_outer, y_proba_outer, labels=label_encoder.transform(label_encoder.classes_))
                except Exception as e:
                    logging.error(f"Error computing performance metrics:, Error: {str(e)}", exc_info=True)
                    roc_auc = np.nan
                    logloss = np.nan

                metrics = {
                    'fs_method': fs_method,
                    'model_type': model_type,
                    'outer_fold': outer_fold_counter,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'log_loss': logloss
                }

                # Save performance metrics
                try:
                    with open(os.path.join(outer_fold_dir, 'performance_metrics.json'), 'w') as f:
                        json.dump(metrics, f)
                except Exception as e:
                    logging.error(f"Error saving performance metrics:, Error: {str(e)}", exc_info=True)

                # Append metrics to the list for aggregation
                performance_metrics_list.append(metrics)

                # Save confusion matrix
                try:
                    np.save(os.path.join(outer_fold_dir, 'confusion_matrix.npy'), conf_matrix)
                except Exception as e:
                    logging.error(f"Error saving confusion matrix:, Error: {str(e)}", exc_info=True)

                # Save predictions
                try:
                    proba_df = pd.DataFrame(y_proba_outer, columns=label_encoder.classes_)
                    predictions_df = pd.DataFrame({
                        'y_true': y_test_outer_decoded,
                        'y_pred': y_pred_outer_decoded
                    })
                    predictions_df = pd.concat([predictions_df, proba_df.reset_index(drop=True)], axis=1)
                    predictions_df.to_csv(os.path.join(outer_fold_dir, 'predictions.csv'), index=False)
                except Exception as e:
                    logging.error(f"Error saving predictions:, Error: {str(e)}", exc_info=True)

                # Save feature importances if available
                try:
                    if hasattr(model, 'feature_importances_'):
                        feature_importances = model.feature_importances_
                        feat_imp_df = pd.DataFrame({
                            'feature': selected_features,
                            'importance': feature_importances
                        })
                        feat_imp_df.to_csv(os.path.join(outer_fold_dir, 'feature_importances.csv'), index=False)
                except Exception as e:
                    logging.error(f"Error saving feature importances:, Error: {str(e)}", exc_info=True)

                # Compute SHAP values
                try:
                    if model_type in ['RandomForest', 'XGBoost', 'StochasticXGBoost']:
                        explainer = shap.TreeExplainer(model)
                        shap_values_train = explainer.shap_values(X_train_selected)
                        shap_values_test = explainer.shap_values(X_test_selected)
                        joblib.dump(shap_values_train, os.path.join(outer_fold_dir, 'shap_values_train.pkl'))
                        joblib.dump(shap_values_test, os.path.join(outer_fold_dir, 'shap_values_test.pkl'))
                        del explainer, shap_values_train, shap_values_test
                except Exception as e:
                    logging.error(f"Error computing SHAP values:, Error: {str(e)}", exc_info=True)

                # Log memory usage after SHAP values computation
                log_memory_usage(f"After SHAP computation for outer fold {outer_fold_counter}")

                # Append outer score
                outer_scores.append(accuracy)
                # Logging the completion of the fold
                fold_elapsed_time = time.time() - fold_start_time
                logging.info(f"Completed fs_method: {fs_method}, model_type: {model_type}, outer_fold: {outer_fold_counter}")
                logging.info(f"Fold Time: {fold_elapsed_time:.2f} seconds")

                # Clean up variables to free up memory
                del X_train_selected, X_test_selected, y_train_outer, y_test_outer, model, y_pred_outer, y_proba_outer, selected_features
                del y_test_outer_decoded, y_pred_outer_decoded, metrics, conf_matrix, proba_df, predictions_df, feat_imp_df

                outer_fold_counter += 1

            if len(outer_scores) == 0:
                logging.warning(f"No successful outer folds for fs_method: {fs_method}, model_type: {model_type}")
                continue

            # Store the results
            mean_score = np.mean(outer_scores)
            std_score = np.std(outer_scores)
            results.append({
                'fs_method': fs_method,
                'model_type': model_type,
                'mean_accuracy': mean_score,
                'std_accuracy': std_score
            })

            # Aggregate performance metrics across folds
            metrics_df = pd.DataFrame(performance_metrics_list)
            metrics_df.to_csv(os.path.join(model_fs_dir, 'performance_metrics_aggregate.csv'), index=False)

            # Logging the completion of the model type and feature selection method
            elapsed_time = time.time() - start_time
            logging.info(f"Completed fs_method: {fs_method}, model_type: {model_type}")
            logging.info(f"Elapsed Time: {elapsed_time:.2f} seconds")
            logging.info(f"Average Accuracy on Outer Folds: {mean_score:.4f}")

            # Clean up variables to free up memory
            del outer_scores, performance_metrics_list, metrics_df

    # Record the end time
    end_time = time.time()
    total_time = end_time - start_time

    logging.info(f"Total runtime: {total_time:.2f} seconds")

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(base_output_dir, 'overall_results.csv'), index=False)

    # Log final memory usage
    log_memory_usage("End of script")

    # Clean up any remaining variables
    del X, y, y_encoded, df, results, results_df
