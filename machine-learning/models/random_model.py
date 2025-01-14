import numpy as np
import pandas as pd
import os
import joblib
import json
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score)
from sklearn.preprocessing import LabelEncoder

# Random perturbation testing function
def random_perturbation_test(y_test, outer_fold_dir, outer_fold, label_encoder, num_trials=1):
    # Ensure the directory exists
    os.makedirs(outer_fold_dir, exist_ok=True)

    # Initialize lists to store results from multiple trials
    accuracy_random_list = []
    precision_random_list = []
    recall_random_list = []
    f1_random_list = []
    roc_auc_random_list = []

    # Create a DataFrame to store predictions for each trial
    predictions_df = pd.DataFrame()

    for trial in range(num_trials):
        # Generate random predictions by shuffling the true labels
        y_random_pred = np.random.permutation(y_test)

        # Compute performance metrics for the random predictions
        accuracy_random = accuracy_score(y_test, y_random_pred)
        precision_random = precision_score(y_test, y_random_pred, average='weighted', zero_division=0)
        recall_random = recall_score(y_test, y_random_pred, average='weighted', zero_division=0)
        f1_random = f1_score(y_test, y_random_pred, average='weighted', zero_division=0)
        
        # For ROC-AUC, use only the probability of the positive class (0.5 for each class in binary setup)
        y_random_proba = np.full(len(y_test), 0.5)  # Probability for the positive class
        roc_auc_random = roc_auc_score(y_test, y_random_proba)

        # Append results to lists
        accuracy_random_list.append(accuracy_random)
        precision_random_list.append(precision_random)
        recall_random_list.append(recall_random)
        f1_random_list.append(f1_random)
        roc_auc_random_list.append(roc_auc_random)

        # Save predictions and probabilities for this trial
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_random_pred_decoded = label_encoder.inverse_transform(y_random_pred)

        # Store true, predicted, and dummy probability values
        trial_predictions = pd.DataFrame({
            'y_true': y_test_decoded,
            'y_pred': y_random_pred_decoded,
            label_encoder.classes_[0]: 0.5,  # Probability for the first class
            label_encoder.classes_[1]: 0.5   # Probability for the second class
        })

        predictions_df = pd.concat([predictions_df, trial_predictions])

    # Save the predictions to a CSV file
    predictions_df.to_csv(os.path.join(outer_fold_dir, f'predictions_random_trial_{outer_fold}.csv'), index=False)

    # Compute the mean of the random metrics
    random_metrics = {
        'model_type': 'Random Model',
        'outer_fold': outer_fold,
        'accuracy': np.mean(accuracy_random_list),
        'precision': np.mean(precision_random_list),
        'recall': np.mean(recall_random_list),
        'f1_score': np.mean(f1_random_list),
        'roc_auc': np.mean(roc_auc_random_list),
        'log_loss': None  # Not applicable for random baseline without probability scores
    }

    # Save the random performance metrics to JSON
    with open(os.path.join(outer_fold_dir, 'avg_random_performance_metrics.json'), 'w') as f:
        json.dump(random_metrics, f)

    # Save raw results from each trial
    random_metrics_df = pd.DataFrame({
        'accuracy_random': accuracy_random_list,
        'precision_random': precision_random_list,
        'recall_random': recall_random_list,
        'f1_score_random': f1_random_list,
        'roc_auc_random': roc_auc_random_list
    })
    random_metrics_df.to_csv(os.path.join(outer_fold_dir, 'random_metrics_trials.csv'), index=False)

    return random_metrics  # Return to collect in main loop

# Main script
if __name__ == "__main__":
    base_output_dir = 'taxa_results'
    os.makedirs(base_output_dir, exist_ok=True)
    
    df = pd.read_csv("../../ML_Data/FINAL_GENUS_TAXA_CLR.csv")
    y = df["Group"]
    X = df.iloc[:, 1:-13]

    best_fs_method = 'SFS-CV'
    best_model_type = 'RandomForest'
    
    # Load label encoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Best model's directory
    model_fs_dir = os.path.join(base_output_dir, f'{best_fs_method}_{best_model_type}')
    os.makedirs(model_fs_dir, exist_ok=True)
    
    all_metrics = []

    # Iterate through each fold
    for outer_fold_counter in range(1, 6):
        outer_fold_dir = os.path.join(model_fs_dir, f'outer_fold_{outer_fold_counter}')
        os.makedirs(outer_fold_dir, exist_ok=True)

        test_indices = np.load(os.path.join(outer_fold_dir, 'test_indices.npy'))
        y_test_outer = y_encoded[test_indices]

        results_dir = os.path.join('model_results_random_baseline', f'outer_fold_{outer_fold_counter}')
        os.makedirs(results_dir, exist_ok=True)

        # Perform random perturbation test and collect metrics
        metrics = random_perturbation_test(y_test_outer, results_dir, outer_fold_counter, label_encoder)
        all_metrics.append(metrics)

    # Save the final metrics
    final_df = pd.DataFrame(all_metrics, columns=['model_type', 'outer_fold', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'log_loss'])
    output_path = os.path.join('model_results_random_baseline', 'avg_random_performance_metrics.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print("Average random performance metrics saved to", output_path)
