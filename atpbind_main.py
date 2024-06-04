from lib.pipeline import Pipeline
import pandas as pd
from torchdrug import utils, data
import os
import torch
import inspect
import numpy as np

from lib.utils import generate_mean_ensemble_metrics_auto, read_initial_csv, aggregate_pred_dataframe
GPU = 0


def rus_preprocess(pipeline, prev_results, negative_use_ratio, _):
    # build random mask
    masks = pipeline.dataset.masks
    for i in range(len(masks)):
        positive_mask = torch.tensor(pipeline.dataset.targets['binding'][i]) == 1
        negative_mask = torch.rand(len(masks[i])) < negative_use_ratio
        assert(len(positive_mask) == len(negative_mask))
        mask = positive_mask | negative_mask
        masks[i] = mask
    pipeline.apply_mask_and_weights(masks=masks)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def resiboost_preprocess(pipeline, prev_results, negative_use_ratio, _):
    if not negative_use_ratio:
        raise ValueError('negative_use_ratio must be specified for resiboost_preprocess')
    # build mask
    if not prev_results:
        print('No previous result, mask nothing')
        return
    masks = pipeline.dataset.masks
    # assert all of mask is True
    assert(all([all(m) for m in masks]))
    df_trains = [r['df_train'] for r in prev_results]
    final_df = df_trains[0].rename(columns={'pred': 'pred_0'})
    for i in range(1, len(df_trains)):
        final_df[f'pred_{i}'] = df_trains[i]['pred']
    
    # # apply sigmoid
    # for i in range(len(df_trains)):
    #     col_name = f'pred_{i}'
    #     final_df[col_name] = final_df[col_name].apply(sigmoid)
    
    final_df['pred'] = final_df[[f'pred_{i}' for i in range(len(df_trains))]].mean(axis=1)
    final_df.reset_index(inplace=True)
    negative_df = final_df[final_df['target'] == 0]
    
    # larger negative_use_ratio means more negative samples are used in training
    confident_negative_df = negative_df.sort_values(
        by=['pred'])[:int(len(negative_df) * (1-negative_use_ratio))]
    
    print(f'Masking out {len(confident_negative_df)} negative samples out of {len(negative_df)}. Most confident negative samples:')
    print(confident_negative_df.head(10))
    for _, row in confident_negative_df.iterrows():
        protein_index_in_dataset = int(row['protein_index'])
        # assume valid fold is consecutive: so that if protein index is larger than first protein index in valid fold, 
        # we need to add the length of valid fold as an offset
        if row['protein_index'] >= pipeline.dataset.valid_fold()[0]:
            protein_index_in_dataset += len(pipeline.dataset.valid_fold())
        masks[protein_index_in_dataset][int(row['residue_index'])] = False
    
    pipeline.apply_mask_and_weights(masks=masks)


DEBUG = False

CYCLE_SIZE = 10

ALL_PARAMS = {
    'esm-t33': {
        'model': 'esm-t33',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 30,  
        },
    },
    'bert': {
        'model': 'bert',
        'model_kwargs': {
            'freeze_bert': False,
            'freeze_layer_count': 29,
        },
    },
    'gearnet': {
        'model': 'gearnet',
        'model_kwargs': {
            'input_dim': 21,
            'hidden_dims': [512] * 4,
        },
    },
    'bert-gearnet': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'bert',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 29,
        },
    },
    'esm-33-gearnet': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 30,
        },
    },
    'esm-t33-ensemble': {
        'ensemble_count': 10,
        'model': 'esm-t33',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 30,  
        },
    },
    'esm-t33-resiboost': {
        'ensemble_count': 10,
        'model': 'esm-t33',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 30,  
        },
        'negative_use_ratio': 0.5,
        'pipeline_before_train_fn': resiboost_preprocess,
    },
    'bert-gearnet-ensemble': {
        'ensemble_count': 10,
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'bert',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 29,
        },
    },
    'esm-33-gearnet-ensemble': {
        'ensemble_count': 50,
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 30,
        },
    },
    'esm-33-gearnet-ensemble-rus': {
        'ensemble_count': 10,
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 30,
        },
        'negative_use_ratio': 0.5,
        'pipeline_before_train_fn': rus_preprocess,
    },
    'esm-33-gearnet-resiboost': {
        'ensemble_count': 50,
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 30,
        },
        'batch_size': 8,
        'negative_use_ratio': 0.5,
        'pipeline_before_train_fn': resiboost_preprocess,
    },
}

def create_single_pred_dataframe(pipeline, dataset, gpu=None):
    gpu = gpu or GPU
    df = pd.DataFrame()
    pipeline.task.eval()
    for protein_index, batch in enumerate(data.DataLoader(dataset, batch_size=1, shuffle=False)):
        batch = utils.cuda(batch, device=f'cuda:{gpu}')
        label = pipeline.task.target(batch)['label'].flatten()

        new_data = {
            'protein_index': protein_index,
            'residue_index': list(range(len(label))),
            'target': label.tolist(),
        }
        pred = pipeline.task.predict(batch).flatten()
        assert (len(label) == len(pred))
        new_data[f'pred'] = [round(t, 5) for t in pred.tolist()]
        new_data = pd.DataFrame(new_data)
        df = pd.concat([df, new_data])

    return df

def single_run(
    valid_fold_num,
    model,
    model_kwargs={},
    pipeline_before_train_fn=None,
    prev_result=None,
    batch_size=8,
    negative_use_ratio=None,
    gpu=None,
    prev_weights=None,
    max_slice_length=550,
    padding=100,
):
    print(f'batch_size: {batch_size}')
    gpu = gpu or GPU
    pipeline = Pipeline(
        dataset='atpbind3d-minimal' if DEBUG else 'atpbind3d',
        model=model,
        gpus=[gpu],
        model_kwargs={
            'gpu': gpu,
            **model_kwargs,
        },
        valid_fold_num=valid_fold_num,
        batch_size=batch_size,
        scheduler='cyclic',
        scheduler_kwargs={
            'base_lr': 3e-4,
            'max_lr': 3e-3,
            'step_size_up': CYCLE_SIZE / 2,
            'step_size_down': CYCLE_SIZE / 2,
            'cycle_momentum': False
        },
        dataset_kwargs={
            'to_slice': True,
            'max_slice_length': max_slice_length,
            'padding': padding,
        }
    )
    
    if pipeline_before_train_fn:
        params = inspect.signature(pipeline_before_train_fn).parameters
        if len(params) == 1:
            print('Not using previous result')
            pipeline_before_train_fn(pipeline)
        else: # used by resiboost, which need previous state
            print('Using previous result')
            pipeline_before_train_fn(pipeline, prev_result, negative_use_ratio, prev_weights)
    
    train_record = pipeline.train_until_fit(
        patience=CYCLE_SIZE,
        max_epoch=CYCLE_SIZE,
    )
    
    df_train = create_single_pred_dataframe(pipeline, pipeline.train_set, gpu=gpu)
    df_valid = create_single_pred_dataframe(pipeline, pipeline.valid_set, gpu=gpu)
    df_test = create_single_pred_dataframe(pipeline, pipeline.test_set, gpu=gpu)
    
    last_record = train_record[-1]
    print(f'single_run done. Last MCC: {last_record["mcc"]}')
    return {
        'df_train': df_train,
        'df_valid': df_valid,
        'df_test': df_test,
        'weights': pipeline.dataset.weights,
        'record': last_record,
        'full_record': train_record,
    }

def write_result(model_key, 
                 valid_fold, 
                 result,
                 write_inference=True,
                 result_file='result_cv/result_cv.csv',
                 additional_record={},
                 ):
    # write dataframes to result_cv/{model_key}/fold_{valid_fold}/{train | valid | test}.csv
    # aggregate record to result_cv/result_cv.csv
    if write_inference:
        folder = f'result_cv/{model_key}/fold_{valid_fold}'
        os.makedirs(folder, exist_ok=True)
        result['df_train'].to_csv(f'{folder}/train.csv', index=False)
        result['df_valid'].to_csv(f'{folder}/valid.csv', index=False)
        result['df_test'].to_csv(f'{folder}/test.csv', index=False)
    
    record_df = read_initial_csv(result_file)
    record_df = pd.concat([record_df, pd.DataFrame([
        {
            'model_key': model_key,
            'valid_fold': valid_fold,
            **additional_record,
            **result['record'],
            'finished_at': pd.Timestamp.now().strftime('%Y-%m-%d %X'),
        }
    ])])
    record_df.to_csv(result_file, index=False)

def write_result_intermediate(model_key, valid_fold, result, iter):
    folder = f'result_cv/{model_key}/fold_{valid_fold}/intermediate'
    os.makedirs(folder, exist_ok=True)
    result['df_train'].to_csv(f'{folder}/iter_{iter}_train.csv', index=False)
    result['df_valid'].to_csv(f'{folder}/iter_{iter}_valid.csv', index=False)
    result['df_test'].to_csv(f'{folder}/iter_{iter}_test.csv', index=False)
    
    record_df = read_initial_csv(f'{folder}/agg_record.csv')
    record_df = pd.concat([record_df, pd.DataFrame([
        {
            'model_key': model_key,
            'valid_fold': valid_fold,
            'iter': iter,
            **result['record'],
            'finished_at': pd.Timestamp.now().strftime('%Y-%m-%d %X'),
        }
    ])])
    record_df.to_csv(f'{folder}/agg_record.csv', index=False)

def write_result_ensemble(model_key, valid_fold, metric):
    record_df = read_initial_csv('result_cv/result_cv.csv')
    record_df = pd.concat([record_df, pd.DataFrame([
        {
            'model_key': model_key,
            'valid_fold': valid_fold,
            **metric,
            'finished_at': pd.Timestamp.now().strftime('%Y-%m-%d %X'),
        }
    ])])
    record_df.to_csv('result_cv/result_cv.csv', index=False)

def main(model_key, valid_fold, extra_kwargs={}):
    model = ALL_PARAMS[model_key].copy()
    if 'ensemble_count' not in model: # single run model
        result = single_run(
            valid_fold_num=valid_fold,
            **model,
            **extra_kwargs,
        )
        write_result(
            model_key=model_key,
            valid_fold=valid_fold,
            result=result,
            additional_record=extra_kwargs,
        )
    else:
        results = []
        weights = None
        ensemble_count = model.pop('ensemble_count')
        
        for iter in range(ensemble_count):
            result = single_run(
                valid_fold_num=valid_fold,
                **model,
                **extra_kwargs,
                prev_result=results,
                prev_weights=weights,
            )
            results.append(result)
            write_result_intermediate(
                model_key=model_key,
                valid_fold=valid_fold,
                result=result,
                iter=iter,
            )
            weights = result['weights']
        # do mean ensemble
        print('Doing mean ensemble')
        df_valid = aggregate_pred_dataframe(dfs=[r['df_valid'] for r in results], apply_sig=True)
        df_test = aggregate_pred_dataframe(dfs=[r['df_test'] for r in results], apply_sig=True)
        print('df_valid:')
        print(df_valid.head(10))
        print('df_test:')
        print(df_test.head(10))
        me_metric = generate_mean_ensemble_metrics_auto(df_valid=df_valid, df_test=df_test, start=0.1, end=0.6, step=0.01)
        write_result_ensemble(
            model_key=model_key,
            valid_fold=valid_fold,
            metric=me_metric
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_keys', type=str, nargs='+', default=['esm-33'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--valid_folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])

    args = parser.parse_args()
    GPU = args.gpu
    print(f'Using default GPU {GPU}')
    print(f'Running model keys {args.model_keys}')
    print(f'Running valid folds {args.valid_folds}')
    
    # set this on need
    extra_kwargs = [{'max_slice_length': slice, 'padding': padding} for slice in [650, 550, 450, 350] for padding in [50, 75, 100, 125]]
    
    try:
        for model_key in args.model_keys:
            for valid_fold in args.valid_folds:
                if not extra_kwargs:
                    extra_kwargs = [{}]
                for kwargs in extra_kwargs:
                    print(f'Running {model_key} fold {valid_fold}, extra_kwargs {kwargs}')
                    main(model_key=model_key, valid_fold=valid_fold, extra_kwargs=kwargs)
    except KeyboardInterrupt:
        print('Received KeyboardInterrupt. Exit.')
        exit(0)
