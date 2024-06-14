import os
import re

import torch
import pandas as pd

from lib.pipeline import Pipeline
from lib.utils import dict_tensor_to_num
from atpbind_main import ALL_PARAMS


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

params = [
    {
        'label': 'esm-t33-gearnet',
        'weights': [[f'atpbind3d_lm-gearnet_{i}.pt'] for i in range(5)],
        'thresholds': [-3.0, -2.6, -3.0, -2.2, -0.3]
    },
    {
        'label': 'esm-t33-gearnet-adaboost-r10',
        'weights': [
            [f'atpbind3d_esm-t33-gearnet-adaboost-r10_{fold}_{i}.pt' for i in range(10)] 
            for fold in range(5)
        ],
        'thresholds':  [-1.1, -1.4, -1.7, -0.9, -1.2],
    },
    {
        'label': 'esm-t33-gearnet-resiboost-r10',
        'weights': [
            [f'atpbind3d_esm-t33-gearnet-resiboost-r10_{fold}_{i}.pt' for i in range(10)] 
            for fold in range(5)
        ],
        'thresholds': [-0.1, -0.4, -1.4, -1.3, -0.4]
    }
]

def main(gpu):
    results = []
    for param in params:
        print(f'Running {param["label"]}')
        for weights, threshold in zip(param['weights'], param['thresholds']):
            result = esm_inference(
                model_key='esm-t33-gearnet', 
                weights=weights, 
                weight_base='weight', 
                threshold=threshold, 
                gpu=0
            )
            results.append({'model': param['label'], 'threshold': threshold, **result})
        df = pd.DataFrame(results)
        df.to_csv('result/atpbind3d_esm_stats.csv', index=False)
    
    
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    main(gpu=args.gpu)
    
    