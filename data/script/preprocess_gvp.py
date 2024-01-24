import pylcs
import os
import json
from torchdrug import utils


def parse_pdb(file_path, chain_id):
    '''
    Parses a PDB file and returns the atom positions and the corresponding amino acid sequence.

    Args:
    file_path (str): Path to the PDB file.
    chain_id (str): Chain identifier for the target protein chain. (ex. "A" if PDB ID is "3EPSA")
                    This is required because some PDB files contain multiple chains.

    Returns:
    positions (list): A list of atom positions for each residue in the format required by `gvp-pytorch`.
                     Each entry contains the positions of N, CA, C, and O atoms as [x, y, z] coordinates.
                     The list structure is: [residue_number][atom_type][coordinate]
                     Refer to https://github.com/drorlab/gvp-pytorch
    sequence (str): A string containing the one-letter amino acid codes for the parsed protein sequence.
    '''
    three_to_one_letter = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

    atom_order = {'N': 0, 'CA': 1, 'C': 2, 'O': 3}

    positions = []
    sequence = ''
    prev_residue_number = None
    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM'):
                current_chain_id = line[21].strip()
                if current_chain_id == chain_id:
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    residue_number = int(line[22:26].strip())
                    if atom_name in atom_order:
                        if residue_number != prev_residue_number:
                            one_letter_code = three_to_one_letter.get(
                                residue_name, 'X')
                            sequence += one_letter_code
                            positions.append([None, None, None, None])
                            prev_residue_number = residue_number
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        positions[-1][atom_order[atom_name]] = [x, y, z]
    return positions, sequence


def get_parsed_data_from_pdb_id(orig_sequence, pdb_id):
    '''
    Fetches a PDB file from the RCSB PDB database, parses it to obtain atom positions, and aligns the positions
    with the original sequence.

    This function utilizes the `pylcs` package to find the longest common subsequence (LCS) between the fetched
    sequence and the original sequence from the ATPBind Dataset. It then filters the fetched positions to only
    include those corresponding to the LCS. This step is necessary because some fetched sequences contain extra
    residues compared to the original sequence(11/388 in the training set). This discrepancy may be due to the
    ATPBind Dataset having truncated some of the sequences (uncertain).

    Args:
    orig_sequence (str): The original protein sequence from the ATPBind Dataset.
    pdb_id (str): The PDB ID corresponding to the target protein structure.

    Returns:
    positions (list): A list of atom positions aligned with the longest common subsequence between the original
                      and fetched sequences.
    '''
    if os.path.exists("./pdb/%s.pdb" % pdb_id[:4]):
        file_path = "./pdb/%s.pdb" % pdb_id[:4]
    else:
        file_path = utils.download(
            "https://files.rcsb.org/download/%s.pdb" % pdb_id[:4], "./pdb")
    chain_id = pdb_id[4]
    positions, sequence = parse_pdb(file_path, chain_id)
    lcs_indices = pylcs.lcs_sequence_idx(orig_sequence, sequence)
    assert(-1 not in lcs_indices)
    filtered_positions = [positions[i] for i in lcs_indices]
    filtered_sequence = ''.join(sequence[i] for i in lcs_indices)
    assert(filtered_sequence == orig_sequence)
    return filtered_positions


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


def load_all_in_file(filename):
    '''
    Download and parse all PDB files in the given ATPBind dataset file.
    The output json matches the format required by `gvp-pytorch`.
    See https://github.com/drorlab/gvp-pytorch#loading-data
    '''
    _, sequences, _, pdb_ids = read_file(filename)
    result = []
    for sequence, pdb_id in zip(sequences, pdb_ids):
        coords = get_parsed_data_from_pdb_id(sequence, pdb_id)
        result.append({
            'name': pdb_id,
            'seq': sequence,
            'coords': coords,
        })
        print('Loaded %s' % pdb_id)

    with open(f"../{filename.split('/')[-1].split('.')[0]}-gvp.json", 'w') as o:
        json.dump(result, o)

    return result


if __name__ == "__main__":
    print('Loading train data...')
    load_all_in_file('../../lib/train.txt')
    print('Loading test data...')
    load_all_in_file('../../lib/test.txt')
