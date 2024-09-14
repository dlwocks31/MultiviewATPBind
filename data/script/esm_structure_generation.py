'''
A script to generate PDB files for the PATP-1930 dataset using ESMFold.

Adapted from https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb#scrollTo=90ee986d
'''

from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import torch
import os
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def read_fasta(fasta_file):
    with open(fasta_file, 'r') as file:
        fasta_content = file.read().strip().split('\n')

    # Extract the sequences from the FASTA file
    proteins = []
    for i in range(0, len(fasta_content), 2):
        if i + 1 < len(fasta_content):
            header = fasta_content[i]
            sequence = fasta_content[i + 1]
            proteins.append({'header': header, 'sequence': sequence})
        else:
            logger.info("The FASTA file does not contain a valid sequence.")
    
    return proteins

def make_model(use_cuda=True):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
    )

    if use_cuda:
        model = model.cuda()
        model.esm = model.esm.half()
        torch.backends.cuda.matmul.allow_tf32 = True
    
    return tokenizer, model

def main(use_cuda=True):
    proteins = read_fasta("../atp-1930/PATP-1930_seq.fa")
    logger.info(f"Proteins read")
    
    proteins = [protein for protein in proteins if not os.path.exists(f"../atp-1930/{protein['header'][1:]}.pdb")]
    logger.info(f"Proteins filtered, {len(proteins)} proteins left.")

    logger.info(f"use_cuda: {use_cuda}")
    tokenizer, model = make_model(use_cuda=use_cuda)
    logger.info(f"Model and tokenizer loaded")
    failed_proteins = []
    for protein in tqdm(proteins, desc="Generating PDB files"):
        header = protein['header'][1:]
        sequence = protein['sequence']
        pdb_filename = f"../atp-1930/{header}.pdb"
        if os.path.exists(pdb_filename):
            continue

        tokenized_input = tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
        if use_cuda:
            tokenized_input = tokenized_input.cuda()
        else:
            tokenized_input = tokenized_input.cpu()

        try:
            with torch.no_grad():
                output = model(tokenized_input)
            pdb = convert_outputs_to_pdb(output)

            with open(pdb_filename, "w") as f:
                f.write("".join(pdb))
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.info(f"CUDA out of memory for {header}, len {len(sequence)}. Adding to failed list.")
                failed_proteins.append((header, len(sequence)))
                torch.cuda.empty_cache()
            else:
                raise e

    if failed_proteins:
        logger.info("The following proteins failed due to CUDA out of memory:")
        for header, length in failed_proteins:
            logger.info(f"Header: {header}, Length: {length}")
    
    

if __name__ == '__main__':
    main(use_cuda=True)