from torch_geometric import data as gep_data
import torch
from torchdrug import data
import os
from collections import defaultdict
import numpy as np

from gvp import data as gvp_data
from torch import utils
from Bio.PDB import PDBParser
from .gvp_util import parse_protein_to_json_record
from rdkit import Chem
import logging
import tqdm
from lib.utils import protein_to_sequence

logger = logging.getLogger(__name__)




class ATPBindTestEsm(data.ProteinDataset):
    def __init__(self, **kwargs):
        pdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../data/atp-esm'))
        target_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../data/atp'))
        
        num_samples, sequences, targets, pdb_ids = read_file(os.path.join(target_path, 'test.txt'))
        self.test_sample_count = num_samples
        
        pdb_files = [os.path.join(pdb_path, f'{pdb_id}.pdb')
                        for pdb_id in pdb_ids]
        
        self.load_pdbs(pdb_files, **kwargs)
        self.targets = defaultdict(list)
        self.targets["binding"] = targets
        
        print(f'ATPBindTestEsm: loaded {len(self.data)} proteins')
        print(f'ATPBindTestEsm: loaded {len(self.targets["binding"])} targets')
        assert(len(self.data) == len(self.targets["binding"]))
        
        # do not do slicing, because it is test set and should be evaluated as a whole
        for protein, binding in zip(self.data, self.targets["binding"]):
            if protein.num_residue != len(binding):
                print(f'ERROR: protein: {protein.num_residue}, binding: {len(binding)}')

    def get_item(self, index):
        if self.lazy:
            graph = data.Protein.from_sequence(
                self.sequences[index], **self.kwargs)
        else:
            graph = self.data[index]

        with graph.residue():
            target = torch.as_tensor(
                self.targets["binding"][index], dtype=torch.long).view(-1, 1)
            graph.target = target
            graph.mask = torch.ones(len(target)).bool().view(-1, 1)
            graph.weight = torch.ones(len(target)).float().view(-1, 1)
        graph.view = 'residue'
        item = {"graph": graph}
        if self.transform:
            item = self.transform(item)
        return item
    
    def initialize_mask_and_weights(self, _=None, __=None):
        return self
    
    
    def split(self, valid_fold_num=0):
        splits = [
            utils.data.Subset(self, [0]),  # train
            utils.data.Subset(self, [0]),  # valid
            utils.data.Subset(self, list(range(self.test_sample_count))),  # test
        ]
        return splits
    

class ATPBind3D(data.ProteinDataset):
    splits = ["train", "valid", "test"]

    fold_count = 5
    
    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0, **kwargs):
        """
        Adapted from torchdrug.data.ProteinDataset.load_pdbs
        
        Load the dataset from pdb files.

        Parameters:
            pdb_files (list of str): pdb file names
            transform (Callable, optional): protein sequence transformation function
            lazy (bool, optional): if lazy mode is used, the proteins are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """
        num_sample = len(pdb_files)

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            if not lazy or i == 0:
                mol = Chem.MolFromPDBFile(pdb_file, sanitize=False) # sanitize=False is added from original code
                if not mol:
                    logger.debug("Can't construct molecule from pdb file `%s`. Ignore this sample." % pdb_file)
                    continue
                protein = data.Protein.from_molecule(mol, **kwargs)
                if not protein:
                    logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % pdb_file)
                    continue
            else:
                protein = None
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(protein.to_sequence() if protein else None)
    
    def load_proteins(self, proteins, **kwargs):
        self.data = proteins
        self.pdb_files = [None] * len(proteins)
        self.sequences = [protein_to_sequence(protein) if protein else None for protein in proteins]
    
    def __init__(self, proteins=None, path=None, limit=-1, to_slice=False, max_slice_length=500, padding=50, transform=None):
        if path is None:
            joined_path = os.path.join(os.path.dirname(__file__), '../data/atp')
            data_path = os.path.normpath(joined_path)
        _, targets, pdb_ids = self.get_seq_target(data_path, limit)
        pdb_files = [os.path.join(data_path, f'{pdb_id}.pdb')
                     for pdb_id in pdb_ids]

        self.load_pdbs(pdb_files, transform=transform, atom_feature=None)
        self.targets = defaultdict(list)
        self.targets["binding"] = targets["binding"]

        assert (len(self.data) == len(self.targets["binding"]))
        for protein, binding in zip(self.data, self.targets["binding"]):
            if protein.num_residue != len(binding):
                print(f'ERROR: protein: {protein.num_residue}, binding: {len(binding)}')

        logger.info(f'ATPBind3D: loaded {len(self.data)} proteins and {len(self.targets["binding"])} targets')
        if to_slice:
            new_data = []
            new_targets = []
            # only slice the training set / validation set. because test set has to be evaluated as a whole
            for protein, target in zip(self.data[:self.train_sample_count], self.targets["binding"][:self.train_sample_count]):
                sliced_proteins, sliced_targets = protein_to_slices(
                    protein, 
                    target, 
                    max_slice_length=max_slice_length, 
                    padding=padding
                )
                new_data += sliced_proteins
                new_targets += sliced_targets
            self.data = new_data + self.data[self.train_sample_count:]
            self.targets["binding"] = new_targets + self.targets["binding"][self.train_sample_count:]
            self.train_sample_count = len(new_data)
            
            logger.info(f'ATPBind3D: sliced original data into {len(self.data)} proteins')
            
        self.fold_ranges = np.array_split(np.arange(self.train_sample_count), self.fold_count)
        
        # load gvp data
        gvp_json_records = [parse_protein_to_json_record(protein)
                            for protein in self.data]
        self.gvp_dataset = gvp_data.ProteinGraphDataset(gvp_json_records)
        assert(len(self.data) == len(self.gvp_dataset))
        for protein, json_record in zip(self.data, gvp_json_records):
            if protein.num_residue != json_record['coords'].shape[0]:
                print(f'ERROR: protein: {protein.num_residue}, gvp_graph: {json_record["coords"].shape[0]}')
        logger.info(f'ATPBind3D: loaded {len(self.gvp_dataset)} gvp graphs. length of data: {len(self.data)}')


    def initialize_mask_and_weights(self, masks=None, weights=None):
        if masks is not None:
            print('Initialize Undersampling: fixed mask')
            self.masks = masks
        else:
            print('Initialize Undersampling: all ones')
            self.masks = [
                torch.ones(len(target)).bool()
                for target in self.targets["binding"]
            ]
        
        if weights is not None:
            print('Initialize Weighting: fixed weight')
            self.weights = weights
        else:
            print('Initialize Weighting: all ones')
            self.weights = [
                torch.ones(len(target)).float()
                for target in self.targets["binding"]
            ]
        return self

    def get_seq_target(self, path, limit):
        sequences, targets, pdb_ids = [], [], []

        for file in ['train.txt', 'test.txt']:
            num_samples, seq, tgt, ids = read_file(os.path.join(path, file))

            if limit > 0:
                seq = seq[:limit]
                tgt = tgt[:limit]
                ids = ids[:limit]

            sequences += seq
            targets += tgt
            pdb_ids += ids

            if file == 'train.txt':
                self.train_sample_count = len(seq)
            elif file == 'test.txt':
                self.test_sample_count = len(seq)
            else:
                raise NotImplementedError

        return sequences, {"binding": targets}, pdb_ids


    def _is_train_set(self, index):
        return (index < self.train_sample_count) and (index not in self.fold_ranges[self.valid_fold_num])

    def _get_mask(self, index):
        if not self._is_train_set(index) or self.masks is None:
            # if not train set, do not mask!
            return torch.ones(len(self.targets["binding"][index])).bool()
        return self.masks[index]

    def _get_weight(self, index):
        if not self._is_train_set(index) or self.weights is None:
            # if not train set, do not weight!
            return torch.ones(len(self.targets["binding"][index])).float()
        return self.weights[index]
    
    def valid_fold(self):
        return self.fold_ranges[self.valid_fold_num]

    def get_item(self, index):
        graph = self.data[index]
        gvp_data = self.gvp_dataset[index]
        
        with graph.residue():
            target = torch.as_tensor(
                self.targets["binding"][index], dtype=torch.long).view(-1, 1)
            graph.target = target
            
            graph.mask = self._get_mask(index).view(-1, 1)
            graph.weight = self._get_weight(index).view(-1, 1)
        graph.view = 'residue'
        item = {"graph": graph, "gvp_data": gvp_data}
        if self.transform:
            item = self.transform(item)
        # print(f'get_item {index}, mask {item["graph"].mask.sum()} / {len(item["graph"].mask)}')
        return item

    def split(self, valid_fold_num=0):
        assert(valid_fold_num < self.fold_count and valid_fold_num >= 0)

        self.valid_fold_num = valid_fold_num

        splits = [
            utils.data.Subset(self, to_int_list(
                np.concatenate(self.fold_ranges[:valid_fold_num] + self.fold_ranges[valid_fold_num+1:])
            )), # train
            utils.data.Subset(self, to_int_list(self.fold_ranges[valid_fold_num])), # valid
            utils.data.Subset(self, list(range(self.train_sample_count, self.train_sample_count + self.test_sample_count))), # test
        ]
        return splits

def to_int_list(np_arr):
    return [int(i) for i in np_arr]

CUSTOM_DATASET_TYPES = ['imatinib', 'dasatinib', 'bosutinib']

class CustomBindDataset(data.ProteinDataset):
    splits = ['train', 'valid', 'test']
    fold_count = 5
    
    def __init__(self, dataset_type='imatinib', to_slice=False, max_slice_length=500, padding=50, **kwargs):
        # if data_path is none, set to ../data/imatinib
        self.dataset_type = dataset_type
        joined_path = os.path.join(os.path.dirname(__file__), f'../data/{dataset_type}')
        data_path = os.path.normpath(joined_path)
        print(f'CustomBindDataset: data_path is set to {data_path}')
        
        _, targets, pdb_ids = self._get_seq_target(data_path)
        pdb_files = [os.path.join(data_path, '%s.pdb' % pdb_id)
                     for pdb_id in pdb_ids]
        
        self.load_pdbs(pdb_files, **kwargs)
        self.targets = defaultdict(list)
        self.targets["binding"] = targets["binding"]
        
        if to_slice:
            new_data = []
            new_targets = []
            # only slice the training set / validation set. because test set has to be evaluated as a whole
            for protein, target in zip(self.data[:self.train_sample_count], self.targets["binding"][:self.train_sample_count]):
                sliced_proteins, sliced_targets = protein_to_slices(
                    protein,
                    target,
                    max_slice_length=max_slice_length,
                    padding=padding
                )
                new_data += sliced_proteins
                new_targets += sliced_targets
            
            self.data = new_data + self.data[self.train_sample_count:]
            self.targets["binding"] = new_targets + \
                self.targets["binding"][self.train_sample_count:]
            self.train_sample_count = len(new_data)            
        
        self.fold_ranges = np.array_split(np.arange(self.train_sample_count), self.fold_count)
        assert (len(self.data) == len(self.targets["binding"]))
        for protein, binding in zip(self.data, self.targets["binding"]):
            if protein.num_residue != len(binding):
                print(f'ERROR: protein: {protein.num_residue}, binding: {len(binding)}')
        
    def initialize_mask_and_weights(self, masks=None, weights=None):
        if masks is not None:
            print('Initialize Undersampling: fixed mask')
            self.masks = masks
        else:
            print('Initialize Undersampling: all ones')
            self.masks = [
                torch.ones(len(target)).bool()
                for target in self.targets["binding"]
            ]

        if weights is not None:
            print('Initialize Weighting: fixed weight')
            self.weights = weights
        else:
            print('Initialize Weighting: all ones')
            self.weights = [
                torch.ones(len(target)).float()
                for target in self.targets["binding"]
            ]
        return self
        

    def _get_seq_target(self, path):
        sequences, targets, pdb_ids = [], [], []
        num_samples, sequences, targets, pdb_ids = read_file(os.path.join(path, f'{self.dataset_type}_binding.txt'))
        self.train_sample_count = int(num_samples * 0.8)
        self.test_sample_count = num_samples - self.train_sample_count
        
        return sequences, {"binding": targets}, pdb_ids

    def _is_train_set(self, index):
        return (index < self.train_sample_count) and (index not in self.fold_ranges[self.valid_fold_num])

    def _get_mask(self, index):
        if not self._is_train_set(index) or self.masks is None:
            # if not train set, do not mask!
            return torch.ones(len(self.targets["binding"][index])).bool()
        return self.masks[index]

    def _get_weight(self, index):
        if not self._is_train_set(index) or self.weights is None:
            # if not train set, do not weight!
            return torch.ones(len(self.targets["binding"][index])).float()
        return self.weights[index]

    def valid_fold(self):
        return self.fold_ranges[self.valid_fold_num]

    def get_item(self, index):
        graph = self.data[index]
        with graph.residue():
            graph.target = torch.as_tensor(
                self.targets["binding"][index], dtype=torch.long).view(-1, 1)
            graph.mask = self._get_mask(index).view(-1, 1)
            graph.weight = self._get_weight(index).view(-1, 1)
        graph.view = 'residue'
        item = {"graph": graph}
        if self.transform:
            item = self.transform(item)
        return item
    
    def split(self, valid_fold_num=0):
        print(f'CustomBindDataset: split with valid_fold_num {valid_fold_num}')
        assert(valid_fold_num < self.fold_count and valid_fold_num >= 0)
        self.valid_fold_num = valid_fold_num

        # Create train split excluding the validation fold
        train_indices = [i for fold in (self.fold_ranges[:valid_fold_num] + self.fold_ranges[valid_fold_num + 1:]) for i in to_int_list(fold)]
        train_split = utils.data.Subset(self, train_indices)

        # Create validation split using the validation fold
        validation_split = utils.data.Subset(self, to_int_list(self.fold_ranges[valid_fold_num]))

        # Create test split based on the predefined range
        test_split = utils.data.Subset(self, list(range(self.train_sample_count, self.train_sample_count + self.test_sample_count)))
        return [train_split, validation_split, test_split]

    
    
    def initialize_mask_and_weights(self, masks=None, weights=None):
        if masks is not None:
            print('Initialize Undersampling: fixed mask')
            self.masks = masks
        else:
            print('Initialize Undersampling: all ones')
            self.masks = [
                torch.ones(len(target)).bool()
                for target in self.targets["binding"]
            ]

    def initialize_mask_and_weights(self, masks=None, weights=None):
        if masks is not None:
            print('Initialize Undersampling: fixed mask')
            self.masks = masks
        else:
            print('Initialize Undersampling: all ones')
            self.masks = [
                torch.ones(len(target)).bool()
                for target in self.targets["binding"]
            ]
        
        
        if weights is not None:
            print('Initialize Weighting: fixed weight')
            self.weights = weights
        else:
            print('Initialize Weighting: all ones')
            self.weights = [
                torch.ones(len(target)).float()
                for target in self.targets["binding"]
            ]
        return self

def read_file(path):
    '''
    Read from ATPBind dataset.
    
    return:
        num_samples: number of samples in the file
        sequences: list of sequences
        targets: list of targets
        pdb_ids: list of pdb_ids
    '''
    sequences, targets, pdb_ids = [], [], []
    with open(path) as f:
        lines = f.readlines()
        num_samples = len(lines)
        for line in lines:
            sequence = line.split(' : ')[-1].strip()
            sequences.append(sequence)

            target = line.split(' : ')[-2].split(' ')
            target_indices = []
            for index in target:
                target_indices.append(int(index[1:]))
            target = []
            for index in range(len(sequence)):
                if index+1 in target_indices:
                    target.append(1)
                else:
                    target.append(0)
            targets.append(target)

            pdb_id = line.split(' : ')[0]
            pdb_ids.append(pdb_id)
    return num_samples, sequences, targets, pdb_ids



def get_slices(total_length, max_slice_length, padding):
    '''
    Find how to slice a protein of length `total_length` into
    consecutive slices, each of maximum length `max_slice_length`
    Two consecutive slices will overlap by `padding` residues.
    
    Ex.
    - get_slices(350, 350, 100) = [(0, 350)] 
    - get_slices(566, 350, 100) = [(0, 308), (258, 566)]
    '''
    cnt = 1
    while True:
        if cnt * max_slice_length - (cnt - 1) * padding >= total_length:
            break
        cnt += 1
    
    slices = []
    avg_length = (total_length + (cnt - 1) * padding) / cnt
    for i in range(cnt):
        start = int(i * (avg_length - padding))
        end = int(i * (avg_length - padding) + avg_length)
        slices.append((start, end))
    return slices


def protein_to_slices(protein, targets, max_slice_length, padding):
    '''
    Slice a protein-target pair into consecutive slices.
    '''
    slices = get_slices(protein.num_residue, max_slice_length, padding)
    sliced_proteins = []
    sliced_targets = []
    for start, end in slices:
        masks = [True if start <= i < end else False for i in range(protein.num_residue)]
        sliced_proteins.append(protein.subresidue(masks))
        sliced_targets.append(list(np.array(targets)[masks]))
    return sliced_proteins, sliced_targets