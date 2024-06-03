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
def get_dataset(dataset, max_length=350):
    print(f'get dataset {dataset}')
    if dataset == 'atpbind3d' or dataset == 'atpbind3d-minimal':
        truncuate_transform = transforms.TruncateProtein(
            max_length=max_length, random=False)
        protein_view_transform = transforms.ProteinView(view='residue')
        transform = transforms.Compose(
            [truncuate_transform, protein_view_transform])

        limit = -1 if dataset == 'atpbind3d' else 5
        return ATPBind3D(transform=transform, limit=limit, to_slice=True)
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


METRICS_USING = ("sensitivity", "specificity", "accuracy",
                 "precision", "mcc", "micro_auroc",)


class Pipeline:
    possible_models = ['bert', 'gearnet', 'lm-gearnet',
                       'cnn', 'esm-t6', 'esm-t12', 'esm-t30', 'esm-t33', 'esm-t36', 'esm-t48']
    possible_datasets = ['atpbind', 'atpbind3d', 'atpbind3d-minimal'] + CUSTOM_DATASET_TYPES
    threshold = 0

    def __init__(self,
                 model,
                 dataset,
                 gpus,
                 task='npp',
                 model_kwargs={},
                 optimizer_kwargs={},
                 task_kwargs={},
                 undersample_kwargs={},
                 batch_size=1,
                 bce_weight=1,
                 verbose=False,
                 optimizer='adam',
                 scheduler=None,
                 scheduler_kwargs={},
                 valid_fold_num=0,
                 max_length=350,
                 num_mlp_layer=2,
                 discriminative_decay_factor=None,
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

        self.dataset = get_dataset(dataset, max_length=max_length)
        self.valid_fold_num = valid_fold_num
        self.train_set, self.valid_set, self.test_set = self.dataset.initialize_mask_and_weights(
            **undersample_kwargs).split(valid_fold_num=valid_fold_num)
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
            'bce_weight': torch.tensor([bce_weight], device=torch.device(f'cuda:{self.gpus[0]}')),
            **task_kwargs,
        }

        if task == 'npp':
            self.task = NodePropertyPrediction(
                self.model,
                **task_kwargs,
            )
        else:
            raise ValueError(
                'Task must be one of {}'.format(['npp', 'me_npp']))
            

        if not optimizer in ['adam', 'adamw']:
            raise ValueError(
                'Optimizer must be one of {}'.format(['adam', 'adamw']))
        # it does't matter whether we use self.task or self.model.parameters(), since mlp is added at preprocess time
        # and mlp parameters is then added to optimizer
        
        if model == 'lm-gearnet' and discriminative_decay_factor is not None:
            print('Adam parameter: discriminative')
            base_lr = optimizer_kwargs.get('lr', 1e-3)
            parameters = self.model.get_parameters_with_discriminative_lr(
                lr=base_lr, lr_decay_factor=discriminative_decay_factor
            )
        else:
            print('Adam parameter: all')
            parameters = self.model.parameters()

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(parameters, **optimizer_kwargs)
        elif optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(parameters, **optimizer_kwargs)

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

    def train_until_fit(self, max_epoch=None, patience=1, early_stop_metric='valid_mcc', return_preds=False, return_state_dict=False, use_dynamic_threshold=True):
        from itertools import count
        train_record = []
        train_preds = None
        valid_preds = None
        test_preds = None
        state_dict = None
        best_metric = -1

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
                should_replace_best_metric = cur_result['valid_mcc'] > best_metric
                if should_replace_best_metric:
                    if return_preds:
                        best_metric = cur_result['valid_mcc']
                        train_preds = create_single_pred_dataframe(
                            self, self.train_set)
                        valid_preds = create_single_pred_dataframe(
                            self, self.valid_set)
                        test_preds = create_single_pred_dataframe(
                            self, self.test_set)
                    elif return_state_dict:
                        state_dict = deepcopy(self.task.state_dict())
                        
                best_index = np.argmax([record['valid_mcc'] for record in train_record])
                if best_index < len(train_record) - patience:
                    break


        if return_preds:
            return (train_record, train_preds, valid_preds, test_preds)
        elif return_state_dict:
            return (train_record, state_dict)
        else:
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
            
            
            
