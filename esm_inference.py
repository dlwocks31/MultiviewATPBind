import os
import re

import torch
import pandas as pd

from lib.pipeline import Pipeline
from lib.utils import dict_tensor_to_num
from atpbind_main import ALL_PARAMS


params = [
    {
        'model': 'esm-t33-gearnet',
        'weights': [[f'atpbind3d_lm-gearnet_{i}.pt'] for i in range(5)],
    },
    {
        'model': 'esm-t33-gearnet-adaboost-r10',
        'weights': [
            [f'atpbind3d_esm-t33-gearnet-adaboost-r10_{fold}_{i}.pt' for i in range(10)]
            for fold in range(5)
        ],
    },
    {
        'model': 'esm-t33-gearnet-resiboost-r10',
        'weights': [
            [f'atpbind3d_esm-t33-gearnet-resiboost-r10_{fold}_{i}.pt' for i in range(10)]
            for fold in range(5)
        ],
    }
]

def esm_inference(model_key, weights, weight_base, threshold, gpu):
    model_config = ALL_PARAMS[model_key]
    pipeline = Pipeline(
        gpus=[gpu],
        dataset='atpbind3d-esm',
        model=model_config['model'],
        model_kwargs={
            'gpu': gpu,
            **model_config['model_kwargs']
        },
        dataset_kwargs={
            'to_slice': True,
            'max_slice_length': 500,
            'padding': 50,
        }
    )
    preds = []
    for weight in weights:
        model_weight = torch.load(os.path.join(weight_base, weight))
        pipeline.task.load_state_dict(model_weight)
        pred, target = pipeline.predict_and_target_dataset(pipeline.test_set, 500, 50)
        preds.append(pred)
    pred_sum = torch.zeros_like(preds[0])
    for pred in preds:
        pred_sum += pred
    pred_sum /= len(preds)
    result = pipeline.task.evaluate(pred_sum, target, threshold=threshold)
    result = dict_tensor_to_num(result)
    
    return result


def main(gpu):
    results = []
    for param in params:
        print(f'Running {param["model"]}')
        for fold, weights in enumerate(param['weights']):
            # Searching for the best threshold computed according to the validation set
            # which is stored in the below file in training process
            stat_df = pd.read_csv(f'result/atpbind3d_stats.csv')
            try:
                threshold = stat_df[(stat_df['model_key'] == param['model']) & (
                    stat_df['valid_fold'] == fold)].iloc[0]['best_threshold']
                print(f'Fold {fold}: found best threshold {threshold}')
            except IndexError:
                threshold = -1.0
                print(f'Fold {fold}: best threshold not found, using default {threshold}')
                
            # do inference
            result = esm_inference(
                model_key='esm-t33-gearnet', 
                weights=weights, 
                weight_base='weight', 
                threshold=threshold, 
                gpu=gpu,
            )
            print(f'Fold {fold}: result {result}')
            results.append({'model': param['model'], 'fold': fold, 'threshold': threshold, **result})
            
            # save results
            df = pd.DataFrame(results)
            df.to_csv('result/atpbind3d_esm_stats.csv', index=False)
    
    
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    main(gpu=args.gpu)
    
    