import contextlib
import logging
from copy import deepcopy
from datetime import datetime, timedelta
from functools import cache
from statistics import mean

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils import data as torch_data
from torchdrug import (core, data, layers, metrics, models, tasks, transforms,
                       utils)
from torchdrug.core import Registry as R
from torchdrug.layers import functional, geometry

from .bert import BertWrapModel, EsmWrapModel
from .custom_models import GearNetWrapModel, LMGearNetModel
from .datasets import (CUSTOM_DATASET_TYPES, ATPBind3D, CustomBindDataset, ATPBindTestEsm,
                       get_slices, protein_to_slices)
from .lr_scheduler import CyclicLR, ExponentialLR
from .tasks import NodePropertyPrediction
from .utils import dict_tensor_to_num, round_dict


class DisableLogger():
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


def format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)

    return f"{minutes}m{seconds}s"


@cache
def get_dataset(dataset, to_slice=True, max_slice_length=550, padding=100):
    print(f'get dataset {dataset}')
    if dataset in ['atpbind3d', 'atpbind3d-minimal']:
        protein_view_transform = transforms.ProteinView(view='residue')
        transform = transforms.Compose([protein_view_transform])

        limit = 5 if dataset == 'atpbind3d-minimal' else -1
        return ATPBind3D(transform=transform, limit=limit, to_slice=to_slice, max_slice_length=max_slice_length, padding=padding)
    elif dataset in ['atpbind3d-esm']:
        return ATPBindTestEsm()
    elif dataset in CUSTOM_DATASET_TYPES:
        protein_view_transform = transforms.ProteinView(view='residue')
        transform = transforms.Compose([protein_view_transform])

        return CustomBindDataset(transform=transform, dataset_type=dataset)
    else:
        raise ValueError('Dataset is not supported')


def create_single_pred_dataframe(pipeline, dataset, slice=False, max_slice_length=None, padding=None):
    if slice and not (max_slice_length and padding):
        raise ValueError('max_slice_length and padding must be provided when slice is True')

    df = pd.DataFrame()
    pipeline.task.eval()
    for protein_index, batch in enumerate(data.DataLoader(dataset, batch_size=1, shuffle=False)):
        # unpack a non-cuda protein beforehand - need one for infer_sliced
        protein = batch['graph'].unpack()[0]
        batch = utils.cuda(batch, device=torch.device(
            f'cuda:{pipeline.gpus[0]}'))
        label = pipeline.task.target(batch)['label'].flatten()

        new_data = {
            'protein_index': protein_index,
            'residue_index': list(range(len(label))),
            'target': label.tolist(),
        }
        if slice:
            pred = pipeline.infer_sliced(protein, max_slice_length, padding).flatten()
        else:
            pred = pipeline.task.predict(batch).flatten()
        assert (len(label) == len(pred))
        new_data[f'pred'] = [round(t, 5) for t in pred.tolist()]
        new_data = pd.DataFrame(new_data)
        df = pd.concat([df, new_data])

    return df



            

METRICS_USING = ("mcc", "micro_auprc", "sensitivity", "precision", "micro_auroc")


class Pipeline:
    possible_models = ['bert', 'gearnet', 'lm-gearnet',
                       'cnn', 'esm-t6', 'esm-t12', 'esm-t30', 'esm-t33', 'esm-t36', 'esm-t48']
    threshold = 0

    def __init__(self,
                 model,
                 dataset,
                 gpus,
                 model_kwargs={},
                 optimizer_kwargs={},
                 scheduler=None,
                 scheduler_kwargs={},
                 task_kwargs={},
                 batch_size=1,
                 verbose=False,
                 valid_fold_num=0,
                 dataset_kwargs={},
                 num_mlp_layer=2,
                 ):
        print(f'init pipeline, model: {model}, dataset: {dataset}, gpus: {gpus}')
        self.gpus = gpus

        if model not in self.possible_models and not isinstance(model, torch.nn.Module):
            raise ValueError(
                'Model must be one of {}, or is torch.nn.Module'.format(self.possible_models))

        self.load_model(model, **model_kwargs)

        if dataset_kwargs['max_slice_length'] and dataset_kwargs['padding'] and dataset_kwargs['to_slice']:
            self.max_slice_length = dataset_kwargs['max_slice_length']
            self.padding = dataset_kwargs['padding']
            print(f'get dataset with kwargs: {dataset_kwargs}')                  
        else:
            raise ValueError('Are you sure?')
        self.dataset = get_dataset(dataset, **dataset_kwargs)
        self.valid_fold_num = valid_fold_num
        self.train_set, self.valid_set, self.test_set = self.dataset.initialize_mask_and_weights().split(valid_fold_num=valid_fold_num)
        print("train samples: %d, valid samples: %d, test samples: %d" %
              (len(self.train_set), len(self.valid_set), len(self.test_set)))

        edge_layers = [
            geometry.SpatialEdge(radius=10.0, min_distance=5),
            geometry.KNNEdge(k=10, min_distance=5),
            geometry.SequentialEdge(max_distance=2),
        ]

        graph_construction_model = layers.GraphConstruction(
            node_layers=[geometry.AlphaCarbonNode()],
            edge_layers=edge_layers,
            edge_feature="gearnet"
        )
        task_kwargs = {
            'graph_construction_model': graph_construction_model,
            'normalization': False,
            'num_mlp_layer': num_mlp_layer,
            'metric': METRICS_USING,
            **task_kwargs,
        }

        self.task = NodePropertyPrediction(
            self.model,
            **task_kwargs,
        )

        # it does't matter whether we use self.task or self.model.parameters(), since mlp is added at preprocess time
        # and mlp parameters is then added to optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_kwargs)

        if scheduler == 'cyclic':
            print('use cyclic lr scheduler')
            self.scheduler = CyclicLR(
                self.optimizer, 
                **scheduler_kwargs,
            )
        elif scheduler == 'exponential':
            print('use exponential lr scheduler')
            self.scheduler = ExponentialLR(
                self.optimizer, 
                **scheduler_kwargs,
            )
        else:
            print('no scheduler')
            self.scheduler = None

        self.verbose = verbose
        print(f'pipeline batch_size: {batch_size}')
        self.batch_size = batch_size
        self._init_solver()

    def apply_mask_and_weights(self, masks, weights=None):
        self.train_set, self.valid_set, self.test_set = self.dataset.initialize_mask_and_weights(
            masks=masks, weights=weights).split(valid_fold_num=self.valid_fold_num)
        
        print("train samples: %d, valid samples: %d, test samples: %d" %
              (len(self.train_set), len(self.valid_set), len(self.test_set)))
        self._init_solver()
            
    def _init_solver(self):
        with DisableLogger():
            self.solver = core.Engine(self.task,
                                      self.train_set,
                                      self.valid_set,
                                      self.test_set,
                                      self.optimizer,
                                      scheduler=self.scheduler,
                                      batch_size=self.batch_size,
                                      log_interval=1000000000,
                                      gpus=self.gpus,
                                      )

    def load_model(self, model, **model_kwargs):
        print(f'load model {model}, kwargs: {model_kwargs}')
        with DisableLogger():
            if model == 'bert':
                self.model = BertWrapModel(**model_kwargs)
            elif model == 'gearnet':
                self.model = GearNetWrapModel(**model_kwargs)
            elif model == 'lm-gearnet':
                self.model = LMGearNetModel(**model_kwargs)
            elif model == 'cnn':
                self.model = models.ProteinCNN(**model_kwargs)
            elif model.startswith('esm'):
                self.model = EsmWrapModel(model_type=model, **model_kwargs)
            # pre built model, eg LoraModel. I wonder wheter there is better way to check this
            elif isinstance(model, torch.nn.Module):
                self.model = model

    def train(self, num_epoch):
        return self.solver.train(num_epoch=num_epoch)

    def train_until_fit(self, max_epoch=None, patience=1, use_dynamic_threshold=True):
        from itertools import count
        train_record = []
        last_time = datetime.now()
        for epoch in count(start=1):
            if (max_epoch is not None) and (epoch > max_epoch):
                break
            cm = contextlib.nullcontext() if self.verbose else DisableLogger()
            with cm:
                self.train(num_epoch=1)

                # record
                if use_dynamic_threshold:
                    results = self.valid_dataset_stats()
                    valid_mcc = results['best_mcc']
                    threshold = results['best_threshold']
                    valid_bce = results['loss']
                else:
                    valid_mcc = self.evaluate(split='valid', threshold=0)['mcc']
                    threshold = 0
                
                cur_result = self.evaluate_test_sliced(threshold=threshold)
                cur_result['valid_mcc'] = valid_mcc
                cur_result['train_bce'] = self.get_last_bce()
                cur_result['valid_bce'] = valid_bce
                cur_result['best_threshold'] = threshold
                cur_result = round_dict(cur_result, 4)
                train_record.append(cur_result)
                # logging
                cur_time = datetime.now()
                print(f'{format_timedelta(cur_time - last_time)} {cur_result}')
                last_time = cur_time
                
                # early stop                
                best_index = np.argmax([record['valid_mcc'] for record in train_record])
                if best_index < len(train_record) - patience:
                    break
        return train_record

    def get_last_bce(self):
        from statistics import mean
        meter = self.solver.meter
        index = slice(meter.epoch2batch[-2], meter.epoch2batch[-1])
        bce_records = meter.records['binary cross entropy'][index]
        return mean(bce_records)


    def valid_dataset_stats(self):
        dataloader = data.DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False
        )

        preds = []
        targets = []
        thresholds = np.linspace(-3, 1, num=41)
        mcc_values = [0 for i in range(len(thresholds))]
        self.task.eval()
        metrics = []
        with torch.no_grad():
            for batch in dataloader:
                batch = utils.cuda(
                    batch, device=torch.device(f'cuda:{self.gpus[0]}'))
                pred, target, loss, metric = self.task.predict_and_target_with_metric(batch)
                preds.append(pred)
                targets.append(target)
                metrics.append(metric['binary cross entropy'].item())

        pred = utils.cat(preds)
        target = utils.cat(targets)

        for i, threshold in enumerate(thresholds):
            mcc = self.task.evaluate(
                pred, target, threshold
            )['mcc']
            mcc_values[i] = mcc_values[i] + mcc

        max_mcc_idx = np.argmax(mcc_values)

        return {
            'best_mcc': mcc_values[max_mcc_idx],
            'best_threshold': thresholds[max_mcc_idx],
            'loss': mean(metrics),
        }

    def evaluate(self, split="test", verbose=False, threshold=0):
        self.task.threshold = threshold
        return dict_tensor_to_num(self.solver.evaluate(split=split))
    
    def evaluate_test_sliced(self, threshold=0):
        self.task.threshold = threshold
        pred, target = self.predict_and_target_dataset(
            self.test_set, self.max_slice_length, self.padding)
        metric = self.task.evaluate(pred, target)
        return dict_tensor_to_num(metric)
    
    def predict_and_target_dataset(self, data_set, max_slice_length, padding):
        '''
        Get pred, target after slicing the protein in datasets
        Test set is by default not sliced. 
        (because it has to be evaluated as a whole, unlike train / valid)
        This function and `_infer_sliced` are used to slice test set and aggregate sliced inference
        and make it into complete inference regarding the whole, unsliced protein.
        '''
        preds = []
        targets = []
        for item in data_set:
            pred = self.infer_sliced(item['graph'], max_slice_length=max_slice_length, padding=padding)
            preds.append(pred)

            # self.task.target only receives a batch, so we have to wrap it with a dataloader
            dataloader = data.DataLoader([item], batch_size=1, shuffle=False)    
            target = self.task.target(next(iter(dataloader)))
            targets.append(target)
            
        pred = utils.cat(preds)
        target = utils.cat(targets)
        return pred, target

    def infer_sliced(self, protein, max_slice_length, padding):
        '''
        Given a protein, infer it sliced and combine it together
        to create a inference regarding the whole protein.
        '''
        target = protein.target
        intermediate_preds = []
        sliced_proteins, _ = protein_to_slices(protein, target, max_slice_length=max_slice_length, padding=padding)
        dataloader = data.DataLoader(sliced_proteins, batch_size=1, shuffle=False)
        with torch.no_grad():
            self.task.eval()
            for batch in dataloader:
                batch = utils.cuda(batch, device=torch.device(f'cuda:{self.gpus[0]}'))
                pred = self.task.predict({"graph": batch})
                intermediate_preds.append(pred)
        final_preds = torch.zeros(target.shape)
        for i, (start, end) in enumerate(get_slices(target.shape[0], max_slice_length=max_slice_length, padding=padding)):
            final_preds[start:end] += intermediate_preds[i].cpu()
            if i > 0:
                # TODO this is a hacky way to deal with overlapping slices
                # which assumes that for any point at most two slices overlap
                final_preds[start:start+padding] /= 2
        return final_preds
        
    
            
            
