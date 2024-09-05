'''
Adapted from https://github.com/aws-samples/lm-gvp/blob/main/data/prepare_data.py
Does parsing of pdb files for gvp dataset
'''

import numpy as np
from torchdrug import data
from lib.utils import protein_to_sequence


def get_residue_atom_data(protein, residue_index):
    """Extract atom data for a single residue."""
    atom_indices = protein.residue2atom(residue_index)
    atom_positions = protein.node_position[atom_indices]
    atom_name_ids = protein.atom_name[atom_indices]
    return atom_positions, atom_name_ids


def get_residue_coords(protein, residue_index, target_atoms):
    """Extract coordinates for target atoms of a single residue."""
    atom_positions, atom_name_ids = get_residue_atom_data(
        protein, residue_index)

    residue_coords = []
    existing_coords = []
    for target_atom in target_atoms:
        target_atom_id = data.Protein.atom_name2id[target_atom]
        matching_indices = (atom_name_ids == target_atom_id).nonzero().flatten()
        
        if matching_indices.numel() > 0:
            matching_position = atom_positions[matching_indices[0]]
            coord = matching_position.cpu().numpy()
            residue_coords.append(coord)
            existing_coords.append(coord)
        else:
            residue_coords.append(None)

    # Fallback to mean coordinate if some atoms are missing
    if len(existing_coords) > 0:
        mean_coord = np.mean(existing_coords, axis=0)
        residue_coords = [
            coord if coord is not None else mean_coord for coord in residue_coords]
    else:
        raise ValueError(f"No coordinates found for residue {residue_index+1}")

    return residue_coords

def parse_protein_to_json_record(protein, name=""):
    """Convert a torchprotein Protein structure to coordinates of target atoms from all AAs

    Args:
        protein: a torchprotein.Protein object representing the protein structure
        name: String. Name of the protein

    Return:
        Dictionary with the protein sequence, atom 3D coordinates and name.
    """
    output = {}

    # Get AA sequence
    sequence = protein_to_sequence(protein)
    if "." in sequence:
        raise ValueError("Sequence contains invalid character '.'")
    output["seq"] = sequence

    # Get atom coordinates
    coords = []
    target_atoms = ["N", "CA", "C", "O"]
    for residue_index in range(protein.num_residue):
        residue_coords = get_residue_coords(
            protein, residue_index, target_atoms)
        coords.append(residue_coords)

    coords = np.asarray(coords)

    output["coords"] = coords
    output["name"] = name

    return output
