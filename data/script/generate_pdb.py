from Bio.PDB import Select, PDBIO
from Bio.PDB.PDBParser import PDBParser
from torchdrug import utils, layers, data, utils
from torchdrug.layers import geometry
import os
import pylcs
from torchdrug import data
import warnings
import logging

# Reference: https://stackoverflow.com/a/47584587/12134820

logger = logging.getLogger(__name__)

class ChainSelect(Select):
    def __init__(self, chain):
        self.chain = chain
        self.residue_to_altloc = {}

    def accept_chain(self, chain):
        if chain.get_id() == self.chain:
            return 1
        else:
            return 0

    def accept_residue(self, residue):
        hetflag, resseq, icode = residue.get_id()
        return hetflag == " " and icode == " "  # Accept normal residues without insertion code

    def accept_atom(self, atom):
        # Should only accept backbone atoms
        if atom.get_name() not in ["N", "CA", "C", "O"]:
            return False
        # Ref: https://biopython.org/wiki/Remove_PDB_disordered_atoms
        if not atom.is_disordered():
            return True
        else:
            # Find all disordered atoms in the residue
            residue_id = atom.get_parent().get_id()[1]
            if residue_id not in self.residue_to_altloc:
                self.residue_to_altloc[residue_id] = atom.get_altloc()
                # print(f"Debug: Residue ID: {residue_id}, Altloc: {atom.get_altloc()}, in dict: {self.residue_to_altloc[residue_id]}. First seen")
                res = True
            else:
                # print(f"Debug: Residue ID: {residue_id}, Altloc: {atom.get_altloc()}, in dict: {self.residue_to_altloc[residue_id]}, return: {self.residue_to_altloc[residue_id] == atom.get_altloc()}")
                res = self.residue_to_altloc[residue_id] == atom.get_altloc()
            if res and atom.get_altloc() != 'A':
                # Set to standard altloc. If altloc is not A, it will be ignored by torchprotein
                atom.set_altloc('A')
            return res



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
    # 0th stage: download raw pdb file
    pdb_tmp_folder = '../pdb_tmp'
    if os.path.exists(f"{pdb_tmp_folder}/{pdb_id[:4]}.pdb"):
        file_path = f"{pdb_tmp_folder}/{pdb_id[:4]}.pdb"
    else:
        file_path = utils.download(
            f"https://files.rcsb.org/download/{pdb_id[:4]}.pdb", pdb_tmp_folder)

    # First stage: filter for desired chain, normal residue, backbone atom, save it back to pdb_tmp_folder
    chain_id = pdb_id[4]
    p = PDBParser(PERMISSIVE=1)
    structure = p.get_structure(pdb_id[:4], file_path)
    select = ChainSelect(chain_id)
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(f'{pdb_tmp_folder}/{pdb_id}.pdb', select)

    # Second Stage: filter for longest substring
    file_path = f'{pdb_tmp_folder}/{pdb_id}.pdb'
    p = PDBParser(PERMISSIVE=1)
    structure = p.get_structure(pdb_id, file_path)
    raw_sequence = ''.join(three_to_one_letter[residue.get_resname()]
                           for residue in structure.get_residues())
    if atpbind_sequence == raw_sequence:
        logger.info(f'{pdb_id} is already aligned')
        idx_to_use = list(range(len(atpbind_sequence)))
    else:
        logger.info(f'{pdb_id} is not aligned')
        idx_to_use = pylcs.lcs_string_idx(atpbind_sequence, raw_sequence)
        if -1 in idx_to_use:
            logger.warning(f'{pdb_id}: -1 in lss. Using lcs instead.')
            idx_to_use = pylcs.lcs_sequence_idx(atpbind_sequence, raw_sequence)
            if -1 in idx_to_use:
                logger.error(f'{pdb_id}: -1 in lcs. using full sequence instead')
                idx_to_use = list(range(len(atpbind_sequence)))
    resseq_ids = [residue.get_id()[1] for i, residue in enumerate(
        structure.get_residues()) if i in idx_to_use]
    select = LCSSelect(resseq_ids)
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(f'../{save_pdb_folder}/{pdb_id}.pdb', select)


def generate_all_in_file(filename, save_pdb_folder):
    _, sequences, _, pdb_ids = read_file(filename)
    
    for sequence, pdb_id in zip(sequences, pdb_ids):
        logger.info('Generating %s..' % pdb_id)
        write_parsed_pdb_from_pdb_id(sequence, pdb_id, save_pdb_folder)


def try_loading_pdb(file_path):
    try:
        protein = data.Protein.from_pdb(file_path)
        return protein
    except Exception as e:
        print("Error loading %s" % file_path)
        return None


def validate(base_path, filename):
    logger = logging.getLogger('validate')
    _, sequences, _, pdb_ids = read_file(os.path.join(base_path, filename))
    validation_results = []
    for sequence, pdb_id in zip(sequences, pdb_ids):
        logger.debug('Validating %s..', pdb_id)
        file_path = os.path.join(base_path, '%s.pdb' % pdb_id)
        protein = try_loading_pdb(file_path)
        if not protein:
            validation_results.append({
                'pdb_id': pdb_id,
                'status': 'error',
                'message': 'protein not loaded'
            })
            logger.error('%s: error: protein not loaded', pdb_id)
            continue

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
            result = {
                'pdb_id': pdb_id,
                'status': 'failed',
                'reason': 'sequence unmatch',
                'alpha_carbon_length': len(protein_sequence),
                'given_sequence_length': len(sequence),
                'protein_sequence': protein_sequence,
                'given_sequence': sequence
            }
            validation_results.append(result)
            logger.warning('validation failed for %s: sequence unmatch. length of alphacarbons: %d, length of given sequence: %d',
                           pdb_id, len(protein_sequence), len(sequence))
            logger.debug('sequence from protein: %s', protein_sequence)
            logger.debug('sequence from txt: %s', sequence)
        elif protein.num_residue != len(sequence):
            result = {
                'pdb_id': pdb_id,
                'status': 'failed',
                'reason': 'length unmatch',
                'protein_residue_count': protein.num_residue,
                'sequence_length': len(sequence)
            }
            validation_results.append(result)
            logger.warning('validation failed for %s: length unmatch. len: %d %d',
                           pdb_id, protein.num_residue, len(sequence))
        else:
            validation_results.append({
                'pdb_id': pdb_id,
                'status': 'success'
            })
            logger.info('validation successful for %s', pdb_id)
    
    return validation_results


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
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to process (atp, imatinib, dasatinib, or bosutinib)')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message=".*discontinuous at line.*")
    warnings.filterwarnings("ignore", message=".*Unknown.*")
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if args.dataset == 'atp':
        print('--- Generate train set.. ---')
        generate_all_in_file('../atp/train.txt', 'atp')
        print('--- Generate test set.. ---')
        generate_all_in_file('../atp/test.txt', 'atp')
        print('--- Validate.. ---')
        base_path = os.path.join(os.path.dirname(__file__), '..', 'atp')
        result = []
        result.extend(validate(base_path, 'train.txt'))
        result.extend(validate(base_path, 'test.txt'))
    elif args.dataset in ['imatinib', 'dasatinib', 'bosutinib']:
        print('--- Generate train set.. ---')
        generate_all_in_file(f'../{args.dataset}/{args.dataset}_binding.txt', args.dataset)
        print('--- Validate.. ---')
        base_path = os.path.join(os.path.dirname(__file__), '..', args.dataset)
        result = validate(base_path, f'{args.dataset}_binding.txt')
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    print('--- Validation Results ---')
    for item in result:
        if item['status'] != 'success':
            print(f"PDB ID: {item['pdb_id']}")
            print(f"Status: {item['status']}")
            if 'reason' in item:
                print(f"Reason: {item['reason']}")
            print(f"Message: {item['message']}")
            print()

