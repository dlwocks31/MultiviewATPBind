from copy import deepcopy
from torchdrug import transforms, data, core, layers, tasks, metrics, utils, models
from torchdrug.layers import functional, geometry
from torchdrug.core import Registry as R
import torch
from torch.utils import data as torch_data
from torch.nn import functional as F
import contextlib
import logging
import numpy as np
from functools import cache
import pandas as pd
from datetime import datetime, timedelta
from statistics import mean

from .tasks import NodePropertyPrediction
from .datasets import CUSTOM_DATASET_TYPES, ATPBind3D, CustomBindDataset, protein_to_slices
from .bert import BertWrapModel, EsmWrapModel
from .custom_models import GearNetWrapModel, LMGearNetModel
from .utils import dict_tensor_to_num, round_dict
from .lr_scheduler import CyclicLR, ExponentialLR

from timer_cm import Timer

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
def get_dataset(dataset, max_length=350, to_slice=True, max_slice_length=550, padding=100):
    print(f'get dataset {dataset}')
    if dataset in ['atpbind3d', 'atpbind3d-minimal']:
        truncuate_transform = transforms.TruncateProtein(
            max_length=max_length, random=False)
        protein_view_transform = transforms.ProteinView(view='residue')
        transform = transforms.Compose(
            [truncuate_transform, protein_view_transform])

        limit = 5 if dataset == 'atpbind3d-minimal' else -1
        return ATPBind3D(transform=transform, limit=limit, to_slice=to_slice, max_slice_length=max_slice_length, padding=padding)
    elif dataset in CUSTOM_DATASET_TYPES:
        truncuate_transform = transforms.TruncateProtein(
            max_length=max_length, random=False)
        protein_view_transform = transforms.ProteinView(view='residue')
        transform = transforms.Compose(
            [truncuate_transform, protein_view_transform])

        return CustomBindDataset(transform=transform, dataset_type=dataset)


def create_single_pred_dataframe(pipeline, dataset):
    df = pd.DataFrame()
    pipeline.task.eval()
    for protein_index, batch in enumerate(data.DataLoader(dataset, batch_size=1, shuffle=False)):
        batch = utils.cuda(batch, device=torch.device(
            f'cuda:{pipeline.gpus[0]}'))
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


METRICS_USING = ("sensitivity", "precision", "mcc", "micro_auprc",)


class Pipeline:
    possible_models = ['bert', 'gearnet', 'lm-gearnet',
                       'cnn', 'esm-t6', 'esm-t12', 'esm-t30', 'esm-t33', 'esm-t36', 'esm-t48']
    possible_datasets = ['atpbind', 'atpbind3d', 'atpbind3d-minimal'] + CUSTOM_DATASET_TYPES
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
                 dataset_args={},
                 max_length=350,
                 num_mlp_layer=2,
                 ):
        print(f'init pipeline, model: {model}, dataset: {dataset}, gpus: {gpus}')
        self.gpus = gpus

        if model not in self.possible_models and not isinstance(model, torch.nn.Module):
            raise ValueError(
                'Model must be one of {}, or is torch.nn.Module'.format(self.possible_models))

        if dataset not in self.possible_datasets:
            raise ValueError('Dataset must be one of {}'.format(
                self.possible_datasets))

        self.load_model(model, **model_kwargs)

        if dataset_args['max_slice_length']:
            max_length = max(dataset_args['max_slice_length'], max_length)
        self.dataset = get_dataset(dataset, max_length=max_length, **dataset_args)
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
                cur_result = self.evaluate(split='test', threshold=threshold)
                cur_result['valid_mcc'] = valid_mcc
                cur_result['train_bce'] = self.get_last_bce()
                cur_result['valid_bce'] = valid_bce
                cur_result = round_dict(cur_result, 4)
                cur_result['lr'] = round(self.optimizer.param_groups[0]['lr'], 9)
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
    
    def evaluate_with_slicing(self):
        self.task.eval()
        test_set = self.test_set
        PADDING = 50
        for test_item in test_set:
            protein = test_item['graph']
            target = protein.target
            sliced_proteins, sliced_targets = protein_to_slices(protein, target)
            intermediate_preds = []
            for protein in sliced_proteins:
                pred = self.task.predict(protein).flatten()
                intermediate_preds.append(pred)
            final_preds = np.zeros(len(target))
            
            
            
