from lib.pipeline import Pipeline
import torch
import pandas as pd
import os
import numpy as np
import logging
import datetime
from lib.utils import generate_mean_ensemble_metrics_auto, read_initial_csv, aggregate_pred_dataframe, round_dict, send_to_discord_webhook
from lib.pipeline import create_single_pred_dataframe
from itertools import product
import argparse
import ast

logger = logging.getLogger(__name__)
GPU = 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_resiboost_preprocess_fn(df_trains, negative_use_ratio, mask_positive):
    '''
    This function is intended to be called in training time in `ensemble_run`
    when we actaully have access to `df_trains`, `negative_use_ratio`, and `mask_positive`.
    The generated function will be passed to `single_run` as `pipeline_before_train_fn`.
    '''    
    def resiboost_preprocess(pipeline):
        '''
        This function is intended to be called in training time in `ensemble_run`
        when we actaully have access to `df_trains`.
        '''
        # build mask
        if not df_trains:
            logger.info('No previous result, mask nothing')
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
        
        logger.info(f'Masking out {len(confident_target_df)} samples out of {len(mask_target_df)}. (Originally {len(final_df)}) Most confident samples:')
        logger.info(confident_target_df.head(10))
        for _, row in confident_target_df.iterrows():
            protein_index_in_dataset = int(row['protein_index'])
            # assume valid fold is consecutive: so that if protein index is larger than first protein index in valid fold, 
            # we need to add the length of valid fold as an offset
            if row['protein_index'] >= pipeline.dataset.valid_fold()[0]:
                protein_index_in_dataset += len(pipeline.dataset.valid_fold())
            masks[protein_index_in_dataset][int(row['residue_index'])] = False
        
        pipeline.apply_mask_and_weights(masks=masks)
    return resiboost_preprocess

def make_rus_preprocess_fn(use_ratio, mask_positive=True):
    def make_rus_preprocess_traintime_fn(df_trains):
        '''
        RUS does not use df_trains. is added for consistency with resiboost
        '''
        def rus_preprocess(pipeline):
            masks = pipeline.dataset.masks
            targets = pipeline.dataset.targets['binding']
            valid_fold = pipeline.dataset.valid_fold()
            for i, mask in enumerate(masks):
                if i in valid_fold:
                    continue
                for j in range(len(mask)):
                    if not mask_positive and targets[i][j] == 1:
                        # if mask_positive is false (only mask negative), pass positive residues
                        continue
                    if np.random.rand() > use_ratio:
                        mask[j] = False
            pipeline.apply_mask_and_weights(masks=masks)
        return rus_preprocess
    return make_rus_preprocess_traintime_fn

def load_pretrained_fn(path):
    def load_pretrained(pipeline):
        load_path = get_data_path(path)
        logger.info(f'Loading weight from {load_path}')
        pipeline.task.load_state_dict(torch.load(load_path), strict=False)
        logger.info('Done loading weight')
    return load_pretrained

DEBUG = False
WRITE_DF = False

CYCLE_SIZE = 10


ALL_PARAMS = {
    'gvp': {
        'model': 'gvp-encoder',
        'model_kwargs': {
            'node_in_dim': (6, 3),
            'node_h_dim': (100, 16),
            'edge_in_dim': (32, 1),
            'edge_h_dim': (32, 1),
            'num_layers': 3,
            'drop_rate': 0.1,
            'output_dim': 20,
        },
        'task_kwargs': {
            'node_feature_type': 'gvp_data',
        },
        'cycle_size': 50,
    },
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
    'esm-t33-gvp': {
        'model': 'esm-t33-gvp',
        'model_kwargs': {
            'lm_freeze_layer_count': 30,
        },
        'task_kwargs': {'node_feature_type': 'gvp_data'},
        'hyperparameters': {
            'max_lr': [1e-3, 2e-3],
            'model_kwargs.lm_freeze_layer_count': [30, 31],
            'model_kwargs.node_h_dim': [(256, 16), (512, 16)],
            'model_kwargs.num_layers': [3, 4],
            'model_kwargs.residual': [True, False],
            'cycle_size': [20, 10],
        }
    },
    'esm-t33-ensemble': {
        'ensemble_count': 10,
        'model': 'esm-t33',
        'model_kwargs': {
            'freeze_esm': False,
            'freeze_layer_count': 30,
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
        'max_slice_length': 500,
        'padding': 50,
        'hyperparameters': {
            'model_kwargs.lm_freeze_layer_count': [27, 28, 29, 30, 31, 32, 33],
            'max_slice_length': [300, 400, 500, 600, 700],
            'padding': [25, 50, 75, 100],
        }
    },
    'esm-t33-gearnet-pretrained': {
        'model': 'lm-gearnet',
        'model_kwargs': {
            'lm_type': 'esm-t33',
            'gearnet_hidden_dim_size': 512,
            'gearnet_hidden_dim_count': 4,
            'lm_freeze_layer_count': 30,
        },
        'pretrained_weight_path': 'weight/atpbind3d_esm-t33-gearnet_1.pt',
    },
    'esm-t33-gearnet-resiboost': {
        'model': 'lm-gearnet',
        'ensemble_count': 10,
        'model_ref': 'esm-t33-gearnet',
        'hyperparameters': {
            'boost_negative_use_ratio': [0.1, 0.2, 0.5, 0.9],
            'boost_mask_positive': [False, True],
        }
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

def save_pipeline_weight(pipeline, path):
    path = get_data_path(path)
    logger.info(f'Saving weight to {path}')
    original_state_dict = pipeline.task.state_dict()
    filtered_state_dict = {k: v for k, v in original_state_dict.items() if not (
        k.startswith('model.lm.encoder.layer.') and int(k.split('.')[4]) < 30)}
    torch.save(filtered_state_dict, path)
    logger.info('Done saving weight')
    
def get_batch_size_and_gradient_interval(dataset, batch_size, gradient_interval, max_slice_length):
    if batch_size is None: # use default
        THRESHOLD = 400
        if 'atpbind3d' in dataset and max_slice_length <= THRESHOLD:
            batch_size = 8
            gradient_interval = 1
        elif 'atpbind3d' in dataset and max_slice_length > THRESHOLD:
            batch_size = 4
            gradient_interval = 2
        else:
            batch_size = 2
            gradient_interval = 1
    return batch_size, gradient_interval

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
    batch_size=None,
    gradient_interval=1,
    original_model_key=None,
    cycle_size=CYCLE_SIZE,
    base_lr=3e-4,
    max_lr=3e-3,
    task_kwargs={},
    pretrained_weight_path=None,
):
    logger.info(f'single_run: dataset={dataset}, model={model}, model_kwargs={model_kwargs}, max_slice_length={max_slice_length}, padding={padding}, batch_size={batch_size}, gradient_interval={gradient_interval}')
    batch_size, gradient_interval = get_batch_size_and_gradient_interval(dataset, batch_size, gradient_interval, max_slice_length)
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
        batch_size=batch_size,
        gradient_interval=gradient_interval,
        scheduler='cyclic',
        scheduler_kwargs={
            'base_lr': base_lr,
            'max_lr': max_lr,
            'step_size_up': cycle_size // 2,
            'step_size_down': cycle_size // 2,
            'cycle_momentum': False
        },
        dataset_kwargs={
            'to_slice': True,
            'max_slice_length': max_slice_length,
            'padding': padding,
        },
        task_kwargs=task_kwargs,
    )
    
    if pretrained_weight_path is not None:
        load_path = get_data_path(pretrained_weight_path)
        logger.info(f'Loading weight from {load_path}')
        pipeline.task.load_state_dict(torch.load(load_path), strict=False)
        logger.info('Done loading weight')
    
    if pipeline_before_train_fn is not None: # mainly, for resiboost_preprocess passed from ensemble_run
        pipeline_before_train_fn(pipeline)

    train_record = pipeline.train_until_fit(
        max_epoch=cycle_size, 
        eval_testset_intermediate=False, # Speed up training
    )
    
    last_record = train_record[-1]
    logger.info(f'single_run done. Last MCC: {last_record["mcc"]}')

    if return_df:
        df_train = create_single_pred_dataframe(pipeline, pipeline.train_set)
        df_valid = create_single_pred_dataframe(pipeline, pipeline.valid_set)
        df_test = create_single_pred_dataframe(
            pipeline, pipeline.test_set, slice=True, max_slice_length=max_slice_length, padding=padding
        )
    else:
        df_train = df_valid = df_test = None
    
    if save_weight:
        file = f'weight/{dataset}_{original_model_key}_{valid_fold_num}.pt'
        save_pipeline_weight(pipeline, file)

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
    save_weight=False,
    original_model_key=None,
    boost_negative_use_ratio=None,
    boost_mask_positive=False,
):
    df_trains = []
    df_valids = []
    df_tests = []
    logger.info(f'ensemble_run: dataset={dataset}, model_ref={model_ref}, ensemble_count={ensemble_count}, boost_negative_use_ratio={boost_negative_use_ratio}, boost_mask_positive={boost_mask_positive}')
    for i in range(ensemble_count):
        # remove pipeline_before_train_fn from single_run kwargs to remove possible duplicate
        all_params = ALL_PARAMS[model_ref].copy()
        # pop hyperparameters. On ensemble run, the value specified in the root dictionary
        # (rather than those inside hyperparameters dict) will be used
        all_params.pop('hyperparameters', None)
        
        if boost_negative_use_ratio is not None:
            pipeline_before_train_fn = make_resiboost_preprocess_fn(
                df_trains=df_trains,
                negative_use_ratio=boost_negative_use_ratio,
                mask_positive=boost_mask_positive,
            )
        else:
            pipeline_before_train_fn = None
        
        
        res = single_run(
            dataset=dataset,
            gpu=gpu,
            valid_fold_num=valid_fold_num,
            **all_params,
            pipeline_before_train_fn=pipeline_before_train_fn,
            return_df=True,
        )
        if save_weight:
            file = f'weight/{dataset}_{original_model_key}_{valid_fold_num}_{i}.pt'
            save_pipeline_weight(res['pipeline'], file)
            
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
        logger.info(f'me_metric: {me_metric}')

    
    if WRITE_DF:
        # TODO check when do I need to write this. Maybe for case study
        sum_preds = df_test[list(filter(lambda a: a.startswith('pred_'), df_test.columns.tolist()))].mean(axis=1)
        final_prediction = (sum_preds > me_metric['best_threshold']).astype(int)
        df_test['pred'] = final_prediction
        # df_test.to_csv(f'{dataset_type}_{model_ref}_{fold}.csv', index=False)
            
    return {
        "record": me_metric,
    }

def get_hyperparameter_combinations(hyperparameters):
    keys, values = zip(*hyperparameters.items())
    return [dict(zip(keys, v)) for v in product(*values)]

def parse_hyperparameters(hp_string):
    if not hp_string:
        return {}
    
    result = {}
    for item in hp_string.split(','):
        key, values = item.split('=')
        model, param = key.split(':')
        values = [v.strip() for v in values.split('|')]
        
        # Convert to appropriate types
        try:
            values = [ast.literal_eval(v) for v in values]
        except:
            pass  # Keep as strings if conversion fails
        
        if model not in result:
            result[model] = {}
        result[model][param] = values
    
    return result

def update_nested_dict(d, key, value):
    keys = key.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

def check_if_run_exists(result_file, model_key, valid_fold, hp_combination):
    if not os.path.exists(result_file):
        return False
    
    df = pd.read_csv(result_file)
    
    # Filter for the specific model_key and valid_fold
    df = df[(df['model_key'] == model_key) & (df['valid_fold'] == valid_fold)]
    
    # Check if all hyperparameters match
    for key, value in hp_combination.items():
        if key in df.columns:
            df = df[df[key] == value]
    
    return len(df) > 0

def main_single_run(dataset, model_key, valid_folds, save_weight=False):
    model = ALL_PARAMS[model_key]
    hyperparameters = model.get('hyperparameters', {})
    if hyperparameters:
        combinations = get_hyperparameter_combinations(hyperparameters)
    else:
        combinations = [{}]

    logger.info(f"Hyperparameters for {model_key}: {hyperparameters}")

    result_file = get_data_path(f'result/{dataset}_{model_key}_stats.csv')

    for i, hp_combination in enumerate(combinations):
        for valid_fold in valid_folds:
            if check_if_run_exists(result_file, model_key, valid_fold, hp_combination):
                logger.info(f'Skipping model_key={model_key}, fold={valid_fold}, hp_combination={hp_combination} as it has already been run.')
                continue

            logger.info(f'main: Running single model "{model_key}", valid_fold={valid_fold} with hyperparameters: {hp_combination}')
            updated_model = model.copy()
            updated_model.pop('hyperparameters', None)

            for key, value in hp_combination.items():
                update_nested_dict(updated_model, key, value)
            logger.info(f'main: Updated model: {updated_model}')
            result_dict = single_run(
                original_model_key=model_key,
                dataset=dataset,
                valid_fold_num=valid_fold,
                save_weight=save_weight,
                **updated_model,
            )

            write_result(
                model_key=model_key,
                valid_fold=valid_fold,
                result_dict=result_dict,
                additional_record={**hp_combination, 'hp_combination': i},
                result_file=result_file
            )

def main_ensemble_run(dataset, model_key, valid_folds, save_weight=False):
    model = ALL_PARAMS[model_key]
    ensemble_count = model['ensemble_count']
    model_ref = model['model_ref']
    hyperparameters = model.get('hyperparameters', {})
    if hyperparameters:
        combinations = get_hyperparameter_combinations(hyperparameters)
    else:
        combinations = [{}]
        
    logger.info(f'main_ensemble_run: hyperparameters={hyperparameters}')

    result_file = get_data_path(f'result/{dataset}_{model_key}_stats.csv')

    for i, hp_combination in enumerate(combinations):
        for valid_fold in valid_folds:
            if check_if_run_exists(result_file, model_key, valid_fold, hp_combination):
                logger.info(f'Skipping ensemble of model_key={model_key}, valid_fold={valid_fold}, hp_combination={hp_combination} as it has already been run.')
                continue

            logger.info(f'main: Running ensemble model "{model_key}", valid_fold={valid_fold} with hyperparameters: {hp_combination}')
            result_dict = ensemble_run(
                dataset=dataset,
                original_model_key=model_key,
                ensemble_count=ensemble_count,
                valid_fold_num=valid_fold,
                model_ref=model_ref,
                save_weight=save_weight,
                **hp_combination,
            )

            write_result(
                model_key=model_key,
                valid_fold=valid_fold,
                result_dict=result_dict,
                additional_record={**hp_combination, 'hp_combination': i},
                result_file=result_file
            )

def main(dataset, model_key, valid_folds, save_weight=False):
    if 'ensemble_count' not in ALL_PARAMS[model_key]: # single run model
        main_single_run(dataset, model_key, valid_folds, save_weight)
    else:
        main_ensemble_run(dataset, model_key, valid_folds, save_weight)


def write_result(model_key, 
                 valid_fold, 
                 result_dict,
                 additional_record={},
                 write_inference=False,
                 result_file='result/result_cv.csv',
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
            
    logger.info(f'write_result: record_dict: {record_dict}')
            
    added_record = {
        'model_key': model_key,
        'valid_fold': valid_fold,
        **record_dict,
        **additional_record,
        'finished_at': pd.Timestamp.now().strftime('%Y-%m-%d %X'),
    }
    new_record_df = pd.DataFrame([added_record])
    record_df = pd.concat([record_df, new_record_df], ignore_index=True)
    record_df.to_csv(result_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', default=['atpbind3d'])
    parser.add_argument('--model_keys', type=str, nargs='+', default=['esm-t33'])
    parser.add_argument('--model_key_regex', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--valid_folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--save_weight', action='store_true')
    parser.add_argument('--hyperparameters', type=str, default=None, 
                        help='Hyperparameters to override or add. '
                             'Format: model:param=value1|value2,model:param2=value3|value4')

    args = parser.parse_args()
    GPU = args.gpu
    
    if args.model_key_regex:
        import re
        logger.info(f'Using model key regex {args.model_key_regex}')
        model_keys = list(ALL_PARAMS.keys())
        model_keys = [key for key in model_keys if re.match(args.model_key_regex, key)]
    else:
        model_keys = args.model_keys
        
    logger.info(f'Using default GPU {args.gpu}')
    logger.info(f'Running model keys {model_keys}')
    logger.info(f'Running valid folds {args.valid_folds}')
    
    start_time = datetime.datetime.now()

    
    
    
    if args.hyperparameters:
        if args.hyperparameters == 'none':
            for model_key in ALL_PARAMS:
                if 'hyperparameters' not in ALL_PARAMS[model_key]:
                    ALL_PARAMS[model_key]['hyperparameters'] = {}
        else:
            custom_hyperparameters = parse_hyperparameters(args.hyperparameters)
            for model_key, params in custom_hyperparameters.items():
                if model_key in ALL_PARAMS:
                    ALL_PARAMS[model_key].setdefault('hyperparameters', {})
                    ALL_PARAMS[model_key]['hyperparameters'].update(params)
                    logger.info(f"Updated hyperparameters for {model_key}")
                else:
                    logger.warning(f"Model key '{model_key}' not found in ALL_PARAMS. Skipping.")
    # Read the command line used to start the current process
    with open("/proc/self/cmdline", "r") as f:
        cmdline = f.read().replace('\0', ' ').strip()
    send_to_discord_webhook(f'Started job at {start_time}. Command: `{cmdline}`')
    try:
        for dataset in args.dataset:
            for model_key in model_keys:
                logger.info(f"Running model: {model_key}")
                if model_key in ALL_PARAMS:
                    logger.info(f'Running {model_key}, dataset {dataset}')
                    for valid_fold in args.valid_folds:
                        main(
                            dataset=dataset, model_key=model_key, valid_folds=[valid_fold],
                            save_weight=args.save_weight,
                        )
        send_to_discord_webhook(f'Finished job `{cmdline}` started at {start_time}')
    except KeyboardInterrupt:
        send_to_discord_webhook(f'You requested to stop the job `{cmdline}` started at {start_time}')
        logger.info('Received KeyboardInterrupt. Exit.')
        exit(0)
    except Exception as e:
        send_to_discord_webhook(f'Job `{cmdline}` Error: {e}')
        logger.exception(e)
        exit(1)

