from Bio.PDB import Select, PDBIO
from Bio.PDB.PDBParser import PDBParser
from torchdrug import utils, layers, data, utils
from torchdrug.layers import geometry
import os
import pylcs
from torchdrug import data
import warnings
from itertools import combinations
import random

# Reference: https://stackoverflow.com/a/47584587/12134820


class ChainSelect(Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        if chain.get_id() == self.chain:
            return 1
        else:
            return 0

    def accept_residue(self, residue):
        hetflag, _, _ = residue.get_id()
        return hetflag == " "  # Should only accept normal residue (not HETATM)

    def accept_atom(self, atom):
        # Should only accept backbone atoms
        # to get backbone atom: ["N", "CA", "C", "O"]
        return atom.get_name() in ["CA"]
        
        # if (not atom.is_disordered()) or atom.get_altloc() == "A":
        #     atom.set_altloc(" ")  # Eliminate alt location ID before output.
        #     return True
        # else:  # Alt location was not one to be output.
        #     return False


class LCSSelect(Select):
    def __init__(self, residue_seq):
        self.residue_seq = set(residue_seq)

    def accept_residue(self, residue):
        _, resseq, _ = residue.get_id()
        # Should only accept residue in longest substring
        return resseq in self.residue_seq


three_to_one_letter = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


def read_file(path):
    '''
    Read from ATPBind dataset.
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


def write_parsed_pdb_from_pdb_id(atpbind_sequence, pdb_id, save_pdb_folder):
    if os.path.exists("./pdb_tmp/%s.pdb" % pdb_id[:4]):
        file_path = "./pdb_tmp/%s.pdb" % pdb_id[:4]
    else:
        file_path = utils.download(
            "https://files.rcsb.org/download/%s.pdb" % pdb_id[:4], "./pdb_tmp")

    # First stage: filter for desired chain, normal residue, backbone atom
    chain_id = pdb_id[4]
    p = PDBParser(PERMISSIVE=1)
    structure = p.get_structure(pdb_id[:4], file_path)
    select = ChainSelect(chain_id)
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save('pdb_tmp/%s.pdb' % pdb_id, select)

    # Second Stage: filter for longest substring
    file_path = 'pdb_tmp/%s.pdb' % pdb_id
    p = PDBParser(PERMISSIVE=1)
    structure = p.get_structure(pdb_id, file_path)
    raw_sequence = ''.join(three_to_one_letter[residue.get_resname()]
                           for residue in structure.get_residues())
    idx_to_use = pylcs.lcs_string_idx(atpbind_sequence, raw_sequence)
    if -1 in idx_to_use:
        print('warn: -1 in lss. Using lcs instead. %s' % pdb_id)
        idx_to_use = pylcs.lcs_sequence_idx(atpbind_sequence, raw_sequence)
        assert(-1 not in idx_to_use)
    resseq_ids = [residue.get_id()[1] for i, residue in enumerate(
        structure.get_residues()) if i in idx_to_use]
    select = LCSSelect(resseq_ids)
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(f'../{save_pdb_folder}/{pdb_id}.pdb', select)


def generate_all_in_file(filename, save_pdb_folder):
    _, sequences, _, pdb_ids = read_file(filename)
    
    deny_list = ['3CRCA', '2C7EG', '3J2TB', '3VNUA',
                 '4QREA', '5J1SB', '1MABB', '3LEVH', '3BG5A',
                 '4TU0A',
                 ]
    for sequence, pdb_id in zip(sequences, pdb_ids):
        if pdb_id not in deny_list:
            continue
        print('Generating %s..' % pdb_id)
        write_parsed_pdb_from_pdb_id(sequence, pdb_id, save_pdb_folder)


def try_loading_pdb(file_path):
    try:
        protein = data.Protein.from_pdb(file_path)
        return protein
    except Exception as e:
        print("Error loading %s" % file_path)
        return None


def validate(base_path, filename):
    _, sequences, _, pdb_ids = read_file(os.path.join(base_path, filename))
    for sequence, pdb_id in zip(sequences, pdb_ids):
        # print('Validating %s..' % pdb_id)
        file_path = os.path.join(base_path, '%s.pdb' % pdb_id)
        protein = try_loading_pdb(file_path)
        if not protein:
            raise Exception('Error loading %s' % file_path)
        
        # Get sequence from protein after graph construction model leaving only alpha carbon nodes
        graph_construction_model = layers.GraphConstruction(
            node_layers=[geometry.AlphaCarbonNode()],
            edge_layers = [
                geometry.SpatialEdge(radius=10.0, min_distance=5),
                geometry.KNNEdge(k=10, min_distance=5),
                geometry.SequentialEdge(max_distance=2),
            ],
            edge_feature="gearnet",
        )
        dataloader = data.DataLoader([protein], batch_size=1)
        batch = utils.cuda(next(iter(dataloader)))
        batch = graph_construction_model(batch)
        
        protein_sequence = ''.join(
            i for i in batch.to_sequence()[0] if i != '.'
        )
        if protein_sequence != sequence:
            print('validation failed for %s: sequence unmatch. length of alphacarbons: %d, length of given sequence: %d' %
                  (pdb_id, len(protein_sequence), len(sequence)))
            print('sequence from protein')
            print(protein_sequence)
            print('seqeuence from txt')
            print(sequence)
        elif protein.num_residue != len(sequence):
            print('validation failed for %s: length unmatch. len: %d %d' %
                  (pdb_id, protein.num_residue, len(sequence)))
            continue


def find_close_edit_distance(base_path, filename, ratio_threshold=0.6):
    _, sequences, _, pdb_ids = read_file(os.path.join(base_path, filename))
    iter = zip(sequences, pdb_ids)
    unions = [[id] for id in pdb_ids]
    ratios = []
    for (sequence1, pdb_id1), (sequence2, pdb_id2) in combinations(iter, 2):
        edit_distance = pylcs.edit_distance(sequence1, sequence2)
        ratio = edit_distance * 2 / (len(sequence1) + len(sequence2))
        ratios.append((ratio, edit_distance, pdb_id1, pdb_id2))
        if ratio < ratio_threshold:
            # merge two list
            def find_union_idx(id):
                for i, union in enumerate(unions):
                    if id in union:
                        return i
                return None
        
            idx1 = find_union_idx(pdb_id1)
            idx2 = find_union_idx(pdb_id2)
            if idx1 is None or idx2 is None:
                print('error: union not found')
                continue
            if idx1 == idx2:
                continue
            unions[idx1] += unions[idx2]
            unions.pop(idx2)
    
    unions.sort(key=lambda x: len(x), reverse=True)
    print(f'{len(unions)} groups:')
    print(unions)
    ratios.sort(key=lambda x: x[0])
    for ratio, edit_distance, pdb_id1, pdb_id2 in ratios[:10]:
        print(f'{pdb_id1} {pdb_id2} {ratio} {edit_distance}')
    return unions

def shuffle_lines(filename):
    '''
    Read from ATPBind dataset.
    '''
    with open(filename) as f:
        lines = f.readlines()
        lines = list(lines)
        print(lines)
        
    
    with open(filename, 'w') as f:
        random.shuffle(lines)
        f.writelines(lines)
        
def save_lines(filename, pdb_id_order):
    with open(filename) as f:
        lines = f.readlines()
        lines = list(lines)

    lines.sort(key=lambda x: pdb_id_order.index(x.split(' : ')[0]))    
    with open(filename, 'w') as f:
        f.writelines(lines)

'''
Some generated PDB files can't be loaded using data.Protein.from_pdb because of errors
from RDKit ("Explicit valence for atom # 320 O, 3, is greater than permitted" error).
This error not only occurs when loading generated PDB files, but also when loading
the original PDB file from the PDB database.
Filtering the PDB files with `ChainSelect` resolves a few cases, but some cases 
(3CRCA, 2C7EG, 3J2TB, 3VNUA, 4QREA) still have same issues.

Also, a few generated PDB files show different residue counts when using 
data.Protein.from_pdb compared to their original sequences in ATPBind dataset. 
The generated PDB file is filtered using BioPython so that it matches the
original sequence in ATPBind dataset, so this is probably due to different implementation 
of loading PDB files in bioPython and torchprotein. (Need to investigate further)
The affected PDB files are 5J1SB, 1MABB, 3LEVH, and 3BG5A.
'''
if __name__ == '__main__':
    warnings.filterwarnings("ignore", message=".*discontinuous at line.*")
    warnings.filterwarnings("ignore", message=".*Unknown.*")
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--task', type=str)
    # parser.add_argument('--dataset', type=str)
    # args = parser.parse_args()

    # task = args.task
    # dataset_type = args.dataset
    
    # print('Generate train set..')
    # generate_all_in_file('../../lib/train.txt')
    # print('Generate test set..')
    # generate_all_in_file('../../lib/test.txt')
    
    # print('Generating..')
    # generate_all_in_file(f'../{dataset_type}/{dataset_type}_binding.txt', dataset_type)
    
    # if args.task == 'edit':
    #     print(f'Finding close edit distance..')
    #     base_path = os.path.join(os.path.dirname(__file__), f'../../data/{dataset_type}')
    #     find_close_edit_distance(base_path, f'{dataset_type}_binding.txt', ratio_threshold=0.6)
    
    # if args.task == 'edit_save':
    #     print(f'Finding close edit distance..')
    #     base_path = os.path.join(os.path.dirname(__file__), f'../../data/{dataset_type}')
    #     unions = find_close_edit_distance(base_path, f'{dataset_type}_binding.txt', ratio_threshold=0.6)
    #     pdb_id_order = []
    #     for union in unions:
    #         pdb_id_order += union
    #     random.shuffle(pdb_id_order[:int(len(pdb_id_order)*0.8)])
    #     save_lines(os.path.join(base_path, f'{dataset_type}_binding.txt'), pdb_id_order)
    # base_path = os.path.join(os.path.dirname(__file__), f'../../lib')
    # find_close_edit_distance(base_path, f'test.txt')

    # print('Validating..')
    # base_path = os.path.join(os.path.dirname(__file__), f'../../data/{dataset_type}')
    # validate(base_path, f'{dataset_type}_binding.txt')

    # print('Shuffling..')
    # base_path = os.path.join(os.path.dirname(__file__), f'../{dataset_type}/{dataset_type}_binding.txt')
    # shuffle_lines(base_path)
    
    # genreate
    # generate_all_in_file('../atp/train.txt', 'atp')
    
    # validate:
    base_path = os.path.join(os.path.dirname(__file__), f'../atp')
    validate(base_path, 'train.txt')
