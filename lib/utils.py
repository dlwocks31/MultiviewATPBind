from sklearn.metrics import confusion_matrix
from torchdrug import metrics
import torch
from statistics import mean, stdev
import pandas as pd
import numpy as np

def dict_tensor_to_num(d):
    return {k: v.item() if isinstance(v, torch.Tensor) else v
             for k, v in d.items()}
    
def round_dict(d, n):
    return {k: round(v, n) if isinstance(v, float) else v
                for k, v in d.items()}
    
def statistics_per_key(list_of_dict):
    keys = list_of_dict[0].keys()
    result = {}
    for key in keys:
        result[key] = [mean([i[key] for i in list_of_dict]), stdev([i[key] for i in list_of_dict]) if len(list_of_dict) >= 2 else -1, len(list_of_dict)]
    return result

def read_initial_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, IndexError):
        # File does not exist, or it is empty
        return pd.DataFrame()


def compute_mcc_from_cm(tp, tn, fp, fn):
    # Calculate the denominator
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    
    # Check for zero denominator
    if denominator == 0:
        return 0  # or return float('nan')
    
    # Calculate MCC
    mcc = ((tp * tn) - (fp * fn)) / denominator
    return mcc

def generate_mean_ensemble_metrics(df, threshold=0):
    # Get the mean prediction
    sum_preds = df[list(filter(lambda a: a.startswith('pred_'), df.columns.tolist()))].mean(axis=1)

    # Convert the mean predictions to binary based on the threshold
    final_prediction = (sum_preds > threshold).astype(int)

    # Compute the confusion matrix once
    tn, fp, fn, tp = confusion_matrix(df['target'], final_prediction).ravel()

    # Calculate metrics
    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp)
    mcc = compute_mcc_from_cm(tp, tn, fp, fn)
    
    sum_preds_tensor = torch.tensor(sum_preds.values).float()
    target_tensor = torch.tensor(df['target'].values).float()
    auprc = metrics.area_under_prc(sum_preds_tensor, target_tensor).item()
    auroc = metrics.area_under_roc(sum_preds_tensor, target_tensor).item()

    result = {
        "mcc": mcc,
        "micro_auprc": auprc,
        "sensitivity": sensitivity,
        "precision": precision,
        "micro_auroc": auroc,
    }
    return round_dict(result, 4)
    
def generate_mean_ensemble_metrics_auto(df_valid, df_test, start, end, step):
    thresholds = np.arange(start, end, step)
    valid_mccs = []
    for threshold in thresholds:
        valid_mccs.append(generate_mean_ensemble_metrics(df_valid, threshold=threshold)['mcc'])
    
    best_threshold_arg = np.argmax(valid_mccs)
    best_threshold = thresholds[best_threshold_arg]
    
    best_test_metric = generate_mean_ensemble_metrics(df_test, threshold=best_threshold)
    return {**best_test_metric, 'valid_mcc': valid_mccs[best_threshold_arg], 'best_threshold': best_threshold}
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def aggregate_pred_dataframe(files=None, dfs=None, alpha=None, apply_sig=False):
    '''
    alpha: "Amount of say" for each models. If None, equal weight is used. 
    '''
    if dfs is None:
        dfs = [pd.read_csv(f) for f in files]
    final_df = dfs[0].rename(columns={'pred': 'pred_0'})
    for i in range(1, len(dfs)):
        final_df[f'pred_{i}'] = dfs[i]['pred']
    if apply_sig:
        for i in range(len(dfs)):
            final_df[f'pred_{i}'] = final_df[f'pred_{i}'].apply(sigmoid)
    final_df['pred'] = final_df[list(filter(lambda a: a.startswith('pred_'), final_df.columns.tolist()))].mean(axis=1)
    return final_df.reset_index()
