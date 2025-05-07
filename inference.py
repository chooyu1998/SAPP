import os
import sys
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("../")

from src.datasets.dataprocess import get_Data
from src.utils import get_RSA
from src.models.Model import SAPP_Model
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser

warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB.DSSP")

def load_and_group_data(input_file, ptm_to_residue):
    grouped_info = defaultdict(list)
    rsa_dict = defaultdict(dict)
    seq_dict = defaultdict(dict)

    with open(input_file, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) < 6:
                raise ValueError("Each line must have at least 6 tab-separated fields")

            protein_id, seq, site, label, ptm_type, rsa_or_af_path = tokens[:6]
            site = int(site)
            label = int(label)

            assert seq[site] in ptm_to_residue[ptm_type], f"Residue mismatch at site {site} in {protein_id}"

            try:
                if rsa_or_af_path.endswith('.npy'):
                    rsa_values = np.load(rsa_or_af_path)
                elif rsa_or_af_path.endswith('.cif'):
                    structure = MMCIFParser(QUIET=True).get_structure('model', rsa_or_af_path)
                    rsa_values = get_RSA(structure[0], rsa_or_af_path)[1]
                elif rsa_or_af_path.endswith('.pdb'):
                    structure = PDBParser(QUIET=True).get_structure('model', rsa_or_af_path)
                    rsa_values = get_RSA(structure[0], rsa_or_af_path)[1]
                else:
                    raise ValueError(f"Unsupported RSA source format: {rsa_or_af_path}")
            except Exception as e:
                print(f"Failed to get RSA for {protein_id} from {rsa_or_af_path}: {e}")
                continue

            grouped_info[ptm_type].append((protein_id, site, label, ptm_type))
            rsa_dict[ptm_type][protein_id] = rsa_values
            seq_dict[ptm_type][protein_id] = seq

    return grouped_info, rsa_dict, seq_dict

def run_inference(input_path, output_path, config):
    ptm_to_residue = {
        'SAPPPhos': ['S', 'T'], 'SAPPmethylR': ['R'], 'SAPPphosY': ['Y'],
        'SAPPsumoK': ['K'], 'SAPPmethylK': ['K'],
        'SAPPacetylK': ['K'], 'SAPPubiquitinK': ['K']
    }

    device = config.get("device","cpu")
    batch_size = config.get("batch_size", 128)

    grouped_info, rsa_dict, seq_dict = load_and_group_data(input_path, ptm_to_residue)
    all_results = []

    for ptm_type, info_list in grouped_info.items():
        print(f"Running inference for {ptm_type}...")
        window = config.get("window", 25)
        seq_tensor, rsa_tensor, mask_tensor, rsa_mask_tensor, label_tensor = get_Data(
            info_list, seq_dict[ptm_type], rsa_dict[ptm_type], window
        )

        dataset = TensorDataset(seq_tensor, rsa_tensor, mask_tensor, rsa_mask_tensor, label_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = SAPP_Model(
            vocab_size=config.get("embedding_dim", 22),
            window = window, 
            hidden=config.get("hidden", 256),
            n_layers=config.get("n_layers", 2),
            attn_heads=config.get("attn_heads", 4),
            feed_forward_dim=config.get("feed_forward_dim", 758),
            device=device
        ).to(device)

        pred_list = []
        model_dir = f"data/models/{ptm_type}"
        weight_files = os.listdir(model_dir)

        for weight_file in weight_files:
            model.load_state_dict(torch.load(os.path.join(model_dir, weight_file)))
            model.eval()

            preds = []
            for batch in tqdm(loader, desc=f"{ptm_type} - {weight_file}"):
                batch = [b.to(device) for b in batch]
                pred, _ = model(batch[0], batch[1], batch[2], batch[3])
                preds.append(pred.view(-1).detach().cpu().numpy())

            pred_list.append(np.concatenate(preds))

        averaged_preds = np.mean(np.stack(pred_list), axis=0)

        result_df = pd.DataFrame({
            'ProteinID': [info[0] for info in info_list],
            'Site': [info[1] for info in info_list],
            'Pred': averaged_preds,
            'Label': label_tensor.cpu().numpy(),
            'PTMType': ptm_type
        })

        all_results.append(result_df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(output_path, index=False)
    print(f"Inference completed. Results saved to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on PTM data using SAPP")
    parser.add_argument('--input', type=str, required=True, help='Input .txt file with inference data')
    parser.add_argument('--output', type=str, default='inference_result.csv', help='Output CSV path')
    parser.add_argument('--config', type=str,required=True, help='Path to config JSON file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    with open(args.config) as f:
        config = json.load(f)

    run_inference(args.input, args.output, config)
