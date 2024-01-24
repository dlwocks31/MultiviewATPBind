
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from lib.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchdrug import utils, data
from lib.disable_logger import DisableLogger
from lib.utils import read_initial_csv
import pandas as pd
from lib.pipeline import create_single_pred_dataframe
from lib.utils import aggregate_pred_dataframe, generate_mean_ensemble_metrics_auto


def make_resiboost_preprocess_fn(negative_use_ratio, mask_positive=False):
    def resiboost_preprocess(pipeline, df_trains):
        # build mask
        if not df_trains:
            print('No previous result, mask nothing')
            return
        masks = pipeline.dataset.masks
        
        final_df = aggregate_pred_dataframe(dfs=df_trains, apply_sig=True)
        
        mask_target_df = final_df if mask_positive else final_df[final_df['target'] == 0] 
        
        # larger negative_use_ratio means more negative samples are used in training
        
        # Create a new column for sorting
        mask_target_df['sort_key'] = mask_target_df.apply(lambda row: 1-row['pred'] if row['target'] == 1 else row['pred'], axis=1)

        # Sort the DataFrame using the new column
        confident_target_df = mask_target_df.sort_values(by='sort_key')[:int(len(mask_target_df) * (1 - negative_use_ratio))]

        # Drop the 'sort_key' column from the sorted DataFrame
        confident_target_df = confident_target_df.drop(columns=['sort_key'])
        
        print(f'Masking out {len(confident_target_df)} samples out of {len(mask_target_df)}. (Originally {len(final_df)}) Most confident samples:')
        print(confident_target_df.head(10))
        for _, row in confident_target_df.iterrows():
            protein_index_in_dataset = int(row['protein_index'])
            # assume valid fold is consecutive: so that if protein index is larger than first protein index in valid fold, 
            # we need to add the length of valid fold as an offset
            if row['protein_index'] >= pipeline.dataset.valid_fold()[0]:
                protein_index_in_dataset += len(pipeline.dataset.valid_fold())
            masks[protein_index_in_dataset][int(row['residue_index'])] = False
        
        pipeline.apply_mask_and_weights(masks=masks)
    return resiboost_preprocess

ALL_PARAMS = {
    # GearNet
    'gearnet': {
        'model': 'gearnet',
        'model_kwargs': {
            'input_dim': 21,
            'hidden_dims': [512, 512, 512, 512],
        },
        'load_path': None,
    },
    # ESM
    'esm-t33': {
        'model': 'esm-t33',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 30,  
        },
        'load_path': None,
    },
    'bert': {
        'model': 'bert',
        'model_kwargs': {
            'freeze_bert': False,
            'freeze_layer_count': 29,  
        },
        'load_path': None,
    },
    # ESM + Ensemble
    'esm-t33-ensemble': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33',
    },
    # ESM + Resiboost
    'esm-t33-rboost50': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.5),
    },
    'esm-t33-rboost25': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.25),
    },
    'esm-t33-rboost10': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.1),
    },
    'esm-t33-aboost50': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.5, mask_positive=True),
    },
    'esm-t33-aboost25': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.25, mask_positive=True),
    },
    'esm-t33-aboost10': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.1, mask_positive=True),
    },
    'esm-t33-aboost05': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.1, mask_positive=True),
    },
    # ESM + GearNet
    'esm-33-gearnet': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 30,
        },
        'load_path': None,
    },
    # ESM + Pretrained
    'esm-t33-pretrained': {
        'model': 'esm-t33',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 30,  
        },
        'load_path': 'esm-t33-pretrained.pth',
    },
    'esm-t33-pretrained-freezelm': {
        'model': 'esm-t33',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 33,  
        },
        'load_path': 'esm-t33-pretrained.pth',
    },
    # ESM + Pretrained + Ensemble
    'esm-t33-pretrained-ensemble': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33-pretrained',
    },
    # ESM + Pretrained + Resiboost
    'esm-t33-pretrained-rboost50': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33-pretrained',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.5),
    },
    'esm-t33-pretrained-rboost10': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33-pretrained',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.1),
    },
    
    # ESM + GearNet + Pretrained
    'esm-33-gearnet-pretrained': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 30,
        },
        'load_path': 'lm-gearnet-pretrained.pth',
    },
    'esm-33-gearnet-pretrained-freezelm': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 33,
        },
        'load_path': 'lm-gearnet-pretrained.pth',
    },
    'esm-33-gearnet-pretrained-freezeall': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 33,
        },
        'load_path': 'lm-gearnet-pretrained.pth',
        'before_train_lambda': lambda pipeline: pipeline.model.freeze_gearnet(freeze_all=True)
    },
    # ESM + GearNet + Ensemble
    'esm-33-gearnet-ensemble': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet',
    },
    # ESM + GearNet + ResiBoost
    'esm-33-gearnet-rboost50': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.5),
    },
    'esm-33-gearnet-rboost25': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.25),
    },
    'esm-33-gearnet-rboost10': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.1),
    },
    'esm-33-gearnet-rboost05': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.05),
    },
    # ESM + GearNet + All Boost
    'esm-33-gearnet-aboost50': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.5, mask_positive=True),
    },
    'esm-33-gearnet-aboost25': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.25, mask_positive=True),
    },
    'esm-33-gearnet-aboost10': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.1, mask_positive=True),
    },
    'esm-33-gearnet-aboost05': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.05, mask_positive=True),
    },
    # ESM + GearNet + Pretrained + Ensemble
    'esm-33-gearnet-pretrained-ensemble': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet-pretrained',
    },
    # ESM + GearNet + Pretrained + ResiBoost
    'esm-33-gearnet-pretrained-rboost50': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet-pretrained',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.5),
    },
    'esm-33-gearnet-pretrained-rboost10': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet-pretrained',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.1),
    },
    'esm-33-gearnet-pretrained-freezelm-ensemble': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet-pretrained-freezelm',
    },
    'esm-33-gearnet-pretrained-freezelm-rboost50': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet-pretrained-freezelm',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.5),
    },
    'esm-33-gearnet-pretrained-freezelm-rboost25': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet-pretrained-freezelm',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.25),
    },
    'esm-33-gearnet-pretrained-freezelm-rboost10': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet-pretrained-freezelm',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.1),
    },
    'esm-33-gearnet-pretrained-freezelm-rboost05': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet-pretrained-freezelm',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.05),
    },
    'esm-33-gearnet-pretrained-freezelm-aboost50': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet-pretrained-freezelm',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.5, mask_positive=True),
    },
    'esm-33-gearnet-pretrained-freezelm-aboost25': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet-pretrained-freezelm',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.25, mask_positive=True),
    },
    'esm-33-gearnet-pretrained-freezelm-aboost10': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet-pretrained-freezelm',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.1, mask_positive=True),
    },
    'esm-33-gearnet-pretrained-freezelm-aboost05': {
        'ensemble_count': 10,
        'model_ref': 'esm-33-gearnet-pretrained-freezelm',
        'before_train_lambda_ensemble': make_resiboost_preprocess_fn(negative_use_ratio=0.05, mask_positive=True),
    },
}

BATCH_SIZE = 1

def run_test(
    dataset_type,
    model,
    model_kwargs,
    load_path,
    fold,
    gpu,
    batch_size=1,
    before_train_lambda=None,
    before_train_lambda_ensemble=None,
    get_inference_df=False,
    df_trains=None,
):
    device = f"cuda:{gpu}"
    batch_size = 8 if dataset_type == 'atpbind3d' else 1
    pipeline = Pipeline(
        model=model,
        dataset=dataset_type,
        gpus=[gpu],
        model_kwargs={
            'gpu': gpu,
            **model_kwargs,
        },
        batch_size=batch_size,
        scheduler='cyclic',
        scheduler_kwargs={
            'base_lr': 3e-4,
            'max_lr': 3e-3,
            'step_size_up': 5,
            'step_size_down': 5,
            'cycle_momentum': False,
        },
        valid_fold_num=fold,
    )
    
    if load_path is not None:
        pipeline.task.load_state_dict(torch.load(load_path, map_location=device))
    
    if before_train_lambda is not None:
        before_train_lambda(pipeline)
    elif before_train_lambda_ensemble is not None:
        before_train_lambda_ensemble(pipeline, df_trains)

    res = pipeline.train_until_fit(max_epoch=10, patience=10)
    if get_inference_df:
        df_train = create_single_pred_dataframe(pipeline=pipeline, dataset=pipeline.train_set)
        df_valid = create_single_pred_dataframe(pipeline=pipeline, dataset=pipeline.valid_set)
        df_test = create_single_pred_dataframe(pipeline=pipeline, dataset=pipeline.test_set)
        return res[-1], df_train, df_valid, df_test
    else:
        return res[-1]

WRITE_DF = False

def run_ensemble_test(dataset_type, ensemble_count, model_ref, fold, gpu, before_train_lambda_ensemble=None):
    df_trains = []
    df_valids = []
    df_tests = []
    for i in range(ensemble_count):
        print(f'ensemble: {i}')
        res, df_train, df_valid, df_test = run_test(
            dataset_type = dataset_type,
            gpu = gpu,
            fold = fold,
            **ALL_PARAMS[model_ref],
            before_train_lambda_ensemble=before_train_lambda_ensemble,
            get_inference_df=True,
            df_trains=df_trains,
        )
        df_trains.append(df_train)
        df_valids.append(df_valid)
        df_tests.append(df_test)
    
        df_valid = aggregate_pred_dataframe(dfs=df_valids, apply_sig=False)
        df_test = aggregate_pred_dataframe(dfs=df_tests, apply_sig=False)
        
        start, end, step = (0.1, 0.9, 0.01)

        me_metric = generate_mean_ensemble_metrics_auto(df_valid=df_valid, df_test=df_test, start=start, end=end, step=step)
        print(f'me_metric: {me_metric}')

    
    if WRITE_DF:
        sum_preds = df_test[list(filter(lambda a: a.startswith('pred_'), df_test.columns.tolist()))].mean(axis=1)
        final_prediction = (sum_preds > me_metric['best_threshold']).astype(int)
        df_test['pred'] = final_prediction
        df_test.to_csv(f'{dataset_type}_{model_ref}_{fold}.csv', index=False)
            
    del me_metric['best_threshold']
    return me_metric

    
def main(dataset_type, param_key, cnt, gpu):
    for i in range(cnt):
        for fold in range(5):
            print(f'param_key: {param_key}, cnt: {i}, fold: {fold}')
            if ALL_PARAMS[param_key].get('ensemble_count', None) is not None:
                res = run_ensemble_test(
                    dataset_type = dataset_type,
                    ensemble_count = ALL_PARAMS[param_key]['ensemble_count'],
                    model_ref = ALL_PARAMS[param_key]['model_ref'],
                    before_train_lambda_ensemble=ALL_PARAMS[param_key].get('before_train_lambda_ensemble', None),
                    fold = fold,
                    gpu = gpu,
                )
            else:
                res = run_test(
                    dataset_type=dataset_type,
                    gpu = gpu,
                    fold = fold, 
                    **ALL_PARAMS[param_key],
                )
                
            record_df = read_initial_csv(f'{dataset_type}.csv')
            record_df = pd.concat([record_df, pd.DataFrame([
                {
                    'model_key': param_key,
                    **res,
                    'fold': fold,
                    'finished_at': pd.Timestamp.now().strftime('%Y-%m-%d %X'),
                }
            ])])
            
            record_df.to_csv(f'{dataset_type}.csv', index=False)


if __name__ == '__main__':
    import argparse
    import re
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, nargs='+', default=['imatinib'])
    parser.add_argument('--param_keys', type=str, nargs='+', default=list(ALL_PARAMS.keys()))
    parser.add_argument('--param_key_regex', type=str, default=None)
    parser.add_argument('--cnt', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--write_df', action='store_true')
    args = parser.parse_args()
    
    WRITE_DF = args.write_df
    
    for dataset_type in args.dataset_type:
        for param_key in args.param_keys:
            if args.param_key_regex is not None and re.search(args.param_key_regex, param_key) is None:
                continue
            main(dataset_type, param_key, args.cnt, args.gpu)
    
    
    
    
    
    
