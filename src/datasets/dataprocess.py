import torch
from tqdm import tqdm 
import numpy as np

aa_dict = {'A': 1,'G': 2,'V': 3,'S': 4,'E': 5,'R': 6,'T': 7,
'I': 8,'D': 9,'P': 10,'K': 11,'Q': 12,'N': 13,'F': 14,'Y': 15,
'M': 16,'H': 17,'W': 18,'C': 19,'-': 0,'L': 20}

def torch_binning(data,max_val,steps):
    bin_edges = torch.linspace(0,max_val,steps=steps)
    digitized = torch.bucketize(data.unsqueeze(-1)[:,:,-1],bin_edges)
    one_hot = torch.nn.functional.one_hot(digitized, num_classes=len(bin_edges)+1)
    
    return digitized,one_hot

def get_Data(data_info,protein_dic,rsa_dic,window):
    data_list = []
    rsa_list = []
    mask_list =[]
    rsamask_list = []
    y_list = []

    for li in tqdm(data_info):
        protein = li[0]
        site = li[1]
        label = li[2]
        seq = protein_dic[protein]
        RSA = rsa_dic[protein]
        onehot = np.zeros(window*2+1)
        rsa_feat = np.zeros(window*2+1)
        
        mask = np.zeros(window*2+1)
        rsa_mask = np.zeros(window*2+1)
        start_idx = max(0,site-window)
        end_idx = min(site+window,len(seq)-1)
        j = window - (site-start_idx)
        for i in range(start_idx, end_idx+1):
            onehot[j] = aa_dict[seq[i]]
            rsa_feat[j] = RSA[i]
            if RSA[i] != -1:
                rsa_mask[j] = 1

            if aa_dict[seq[i]] != 0:
                mask[j] = 1
            j += 1
        
        data_list.append(onehot)
        rsa_list.append(rsa_feat)
        mask_list.append(np.expand_dims(mask,-1)*np.expand_dims(mask,-1).T)
        rsamask_list.append(np.expand_dims(rsa_mask,-1)*np.expand_dims(rsa_mask,-1).T)
        y_list.append(label)
    
    data_list = torch.tensor(np.array(data_list), dtype=torch.long)
    y_list = torch.tensor(np.array(y_list), dtype=torch.float32)
    rsa_list = torch.tensor(np.array(rsa_list), dtype=torch.float32)
    rsamask_list = torch.BoolTensor(rsamask_list)
    mask_list = torch.BoolTensor(mask_list)
    
    _,rsa_list = torch_binning(rsa_list,1, 20)
    rsa_list = torch.tensor(np.array(rsa_list), dtype=torch.float32)

    return data_list,rsa_list,mask_list,rsamask_list,y_list

