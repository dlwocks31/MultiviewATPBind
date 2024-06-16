from lib.pipeline import Pipeline
import torch
import pandas as pd
import os
import numpy as np

from lib.utils import generate_mean_ensemble_metrics_auto, read_initial_csv, aggregate_pred_dataframe, round_dict
from lib.pipeline import create_single_pred_dataframe

GPU = 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_resiboost_preprocess_fn(negative_use_ratio, mask_positive=False):
    '''
    This function is intended to be called when the file is load.
    This is to easily configure what negative_use_ratio to be used in a specific model.
    '''
    def make_resiboost_preprocess_traintime_fn(df_trains):
        '''
        This function is intended to be called in training time in `ensemble_run`
        when we actaully have access to `df_trains`.
        The final `resiboost_preprocess` is passed to `single_run` as `pipeline_before_train_fn`,
        with the information about previous training (`df_trains`) encapsulated.
        '''
        def resiboost_preprocess(pipeline):
            # build mask
            if not df_trains:
                print('No previous result, mask nothing')
                return
            masks = pipeline.dataset.masks
            
            final_df = aggregate_pred_dataframe(dfs=df_trains, apply_sig=True)
            
            mask_target_df = final_df if mask_positive else final_df[final_df['target'] == 0].copy()
            
            # larger negative_use_ratio means more negative samples are used in training
            
            # Create a new column for sorting
            mask_target_df.loc[:, 'sort_key'] = mask_target_df.apply(
                lambda row: 1-row['pred'] if row['target'] == 1 else row['pred'], axis=1)

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
    return make_resiboost_preprocess_traintime_fn

def load_pretrained_fn(path):
    def load_pretrained(pipeline):
        load_path = get_data_path(path)
        print(f'Loading weight from {load_path}')
        pipeline.task.load_state_dict(torch.load(load_path), strict=True)
        print('Done loading weight')
    return load_pretrained

DEBUG = False
WRITE_DF = False

CYCLE_SIZE = 10

ALL_PARAMS = {
    'esm-t33': {
        'model': 'esm-t33',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 30,  
        },
    },
    'esm-t33-pretrained': {
        'model': 'esm-t33',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 30,
        },
        'pipeline_before_train_fn': load_pretrained_fn('weight/atpbind3d_esm-t33_1.pt'),
    },
    'esm-t36': {
        'model': 'esm-t36',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 35,
        },
    },
    'esm-t30': {
        'model': 'esm-t30',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 27,
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
    'esm-t33-gearnet': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 30,
        },
    },
    'esm-t33-gearnet-pretrained': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 30,
        },
        'pipeline_before_train_fn': load_pretrained_fn('weight/atpbind3d_lm-gearnet_1.pt'),
    },
    'esm-t36-gearnet': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t36',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 35,
        },
    },
    'esm-t30-gearnet': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t30',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 27,
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
    'esm-t33-gearnet-ensemble': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33-gearnet',
    },
    'esm-t33-gearnet-pretrained-ensemble': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33-gearnet-pretrained',
    },
    'esm-t33-gearnet-adaboost-r10': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33-gearnet',
        'pipeline_before_train_fn': make_resiboost_preprocess_fn(negative_use_ratio=0.1, mask_positive=True),
    },
    'esm-t33-gearnet-adaboost-r20': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33-gearnet',
        'pipeline_before_train_fn': make_resiboost_preprocess_fn(negative_use_ratio=0.2, mask_positive=True),
    },
    'esm-t33-gearnet-pretrained-adaboost-r10': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33-gearnet-pretrained',
        'pipeline_before_train_fn': make_resiboost_preprocess_fn(negative_use_ratio=0.1, mask_positive=True),
    },
    'esm-t33-gearnet-pretrained-adaboost-r10': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33-gearnet-pretrained',
        'pipeline_before_train_fn': make_resiboost_preprocess_fn(negative_use_ratio=0.2, mask_positive=True),
    },
    'esm-t33-gearnet-resiboost-r10': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33-gearnet',
        'pipeline_before_train_fn': make_resiboost_preprocess_fn(negative_use_ratio=0.1, mask_positive=False),
    },
    'esm-t33-gearnet-resiboost-r10': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33-gearnet',
        'pipeline_before_train_fn': make_resiboost_preprocess_fn(negative_use_ratio=0.2, mask_positive=False),
    },
    'esm-t33-gearnet-pretrained-resiboost-r10': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33-gearnet-pretrained',
        'pipeline_before_train_fn': make_resiboost_preprocess_fn(negative_use_ratio=0.1, mask_positive=False),
    },
    'esm-t33-gearnet-pretrained-resiboost-r10': {
        'ensemble_count': 10,
        'model_ref': 'esm-t33-gearnet-pretrained',
        'pipeline_before_train_fn': make_resiboost_preprocess_fn(negative_use_ratio=0.2, mask_positive=False),
    },
}


def get_data_path(path):
    data_folder = os.environ.get('DATA_FOLDER', None)
    if data_folder is None:
        return path
    return os.path.join(data_folder, path)


def clear_cache():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def single_run(
    dataset,
    valid_fold_num,
    model,
    model_kwargs={},
    pipeline_before_train_fn=None,
    gpu=None,
    max_slice_length=500,
    padding=50,
    return_df=False,
    save_weight=False,
):
    clear_cache()
    gpu = gpu or GPU
    pipeline = Pipeline(
        dataset=dataset,
        model=model,
        gpus=[gpu],
        model_kwargs={
            'gpu': gpu,
            **model_kwargs,
        },
        valid_fold_num=valid_fold_num,
        batch_size=8 if dataset == 'atpbind3d' else 2,
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
        pipeline_before_train_fn(pipeline)

    train_record = pipeline.train_until_fit(
        patience=CYCLE_SIZE,
        max_epoch=CYCLE_SIZE,
    )
    
    last_record = train_record[-1]
    print(f'single_run done. Last MCC: {last_record["mcc"]}')

    if return_df:
        df_train = create_single_pred_dataframe(pipeline, pipeline.train_set)
        df_valid = create_single_pred_dataframe(pipeline, pipeline.valid_set)
        df_test = create_single_pred_dataframe(
            pipeline, pipeline.test_set, slice=True, max_slice_length=max_slice_length, padding=padding
        )
    else:
        df_train = df_valid = df_test = None
    
    if save_weight:
        print(f'Saving weight to weight/{dataset}_{model}_{valid_fold_num}.pt')
        torch.save(pipeline.task.state_dict(), f'weight/{dataset}_{model}_{valid_fold_num}.pt')
        print('Done saving weight')

    return {
        'df_train': df_train,
        'df_valid': df_valid,
        'df_test': df_test,
        'weights': pipeline.dataset.weights,
        'record': last_record,
        'full_record': train_record,
        'pipeline': pipeline,
    }


def ensemble_run(
    dataset,
    valid_fold_num,
    model_ref,
    ensemble_count,
    gpu=None,
    pipeline_before_train_fn=None,
    save_weight=False,
    original_model_key=None,
    extra_kwargs={},
):
    df_trains = []
    df_valids = []
    df_tests = []
    for i in range(ensemble_count):
        pipeline_before_train_fn_ls = [
            ALL_PARAMS[model_ref].get('pipeline_before_train_fn', None), # fn for single run (ex. load pretrained weight)
            pipeline_before_train_fn(df_trains) if pipeline_before_train_fn else None, # fn for ensemble run (ex. set resiboost mask)
        ]
        pipeline_before_train_fn_ls = [i for i in pipeline_before_train_fn_ls if i]
        
        # remove pipeline_before_train_fn from single_run kwargs to remove possible duplicate
        all_params = ALL_PARAMS[model_ref].copy()
        if 'pipeline_before_train_fn' in all_params:
            del all_params['pipeline_before_train_fn']
        res = single_run(
            dataset=dataset,
            gpu=gpu,
            valid_fold_num=valid_fold_num,
            **all_params,
            pipeline_before_train_fn=lambda pipeline: [fn(pipeline) for fn in pipeline_before_train_fn_ls],
            return_df=True,
            **extra_kwargs,
        )
        if save_weight:
            print(f'Saving weight to weight/{dataset}_{model_ref}_{valid_fold_num}_{i}.pt')
            torch.save(res['pipeline'].task.state_dict(), f'weight/{dataset}_{original_model_key}_{valid_fold_num}_{i}.pt')
            print('Done saving weight')
        df_trains.append(res['df_train'])
        df_valids.append(res['df_valid'])
        df_tests.append(res['df_test'])
    
        apply_sig = False
        df_valid = aggregate_pred_dataframe(dfs=df_valids, apply_sig=apply_sig)
        df_test = aggregate_pred_dataframe(dfs=df_tests, apply_sig=apply_sig)
        
        start, end, step = (0.1, 0.9, 0.01) if apply_sig else (-3, 1, 0.1)

        me_metric = generate_mean_ensemble_metrics_auto(
            df_valid=df_valid, df_test=df_test, start=start, end=end, step=step
        )
        print(f'me_metric: {me_metric}')

    
    if WRITE_DF:
        # TODO check when do I need to write this. Maybe for case study
        sum_preds = df_test[list(filter(lambda a: a.startswith('pred_'), df_test.columns.tolist()))].mean(axis=1)
        final_prediction = (sum_preds > me_metric['best_threshold']).astype(int)
        df_test['pred'] = final_prediction
        # df_test.to_csv(f'{dataset_type}_{model_ref}_{fold}.csv', index=False)
            
    return {
        "record": me_metric,
    }

def write_result(model_key, 
                 valid_fold, 
                 result_dict,
                 write_inference=False,
                 result_file='result/result_cv.csv',
                 additional_record={},
                 ):
    # write dataframes to result/{model_key}/fold_{valid_fold}/{train | valid | test}.csv
    # aggregate record to result/result_cv.csv
    if write_inference:
        folder = f'result/{model_key}_detail/fold_{valid_fold}'
        os.makedirs(folder, exist_ok=True)
        result_dict['df_train'].to_csv(f'{folder}/train.csv', index=False)
        result_dict['df_valid'].to_csv(f'{folder}/valid.csv', index=False)
        result_dict['df_test'].to_csv(f'{folder}/test.csv', index=False)
    
    record_df = read_initial_csv(result_file)
    record_dict = round_dict(result_dict['record'], 4)
    # if there is train_bce and valid_bce, delete record
    remove_keys = ['train_bce', 'valid_bce']
    for key in remove_keys:
        if key in record_dict:
            del record_dict[key]
    record_df = pd.concat([record_df, pd.DataFrame([
        {
            'model_key': model_key,
            'valid_fold': valid_fold,
            **additional_record,
            **record_dict,
            'finished_at': pd.Timestamp.now().strftime('%Y-%m-%d %X'),
        }
    ])])
    record_df.to_csv(result_file, index=False)


def main(dataset, model_key, valid_fold, extra_kwargs={}, save_weight=False):
    model = ALL_PARAMS[model_key]
    if 'ensemble_count' not in model: # single run model
        result_dict = single_run(
            dataset=dataset,
            valid_fold_num=valid_fold,
            save_weight=save_weight,
            **model,
            **extra_kwargs,
        )
    else:
        ensemble_count = model['ensemble_count']
        model_ref = model['model_ref']
        pipeline_before_train_fn = model.get('pipeline_before_train_fn', None)
        result_dict = ensemble_run(
            dataset=dataset,
            original_model_key=model_key,
            ensemble_count=ensemble_count,
            valid_fold_num=valid_fold,
            model_ref=model_ref,
            pipeline_before_train_fn=pipeline_before_train_fn,
            save_weight=save_weight,
            **extra_kwargs,
        )
    
    result_file = get_data_path(f'result/{dataset}_stats.csv')
    
    write_result(
        model_key=model_key,
        valid_fold=valid_fold,
        result_dict=result_dict,
        additional_record=extra_kwargs,
        result_file=result_file
    )
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', default=['atpbind3d'])
    parser.add_argument('--model_keys', type=str, nargs='+', default=['esm-t33'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--valid_folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--save_weight', action='store_true')

    args = parser.parse_args()
    GPU = args.gpu
    print(f'Using default GPU {args.gpu}')
    print(f'Running model keys {args.model_keys}')
    print(f'Running valid folds {args.valid_folds}')
    
    # set this on need
    extra_kwargs = []
    
    try:
        for dataset in args.dataset:
            for model_key in args.model_keys:
                for valid_fold in args.valid_folds:
                    if not extra_kwargs:
                        extra_kwargs = [{}]
                    for kwargs in extra_kwargs:
                        print(f'Running {model_key}, dataset {dataset}, fold {valid_fold}, extra_kwargs {kwargs}')
                        main(
                            dataset=dataset, model_key=model_key, valid_fold=valid_fold, extra_kwargs=kwargs,
                            save_weight=args.save_weight,
                            )
    except KeyboardInterrupt:
        print('Received KeyboardInterrupt. Exit.')
        exit(0)
