import os
from random import shuffle

import anndata
import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import issparse
from sklearn import preprocessing
import torch
from torch.utils.data import TensorDataset, DataLoader

import couplevae



def train_test_split(adata, train_frac=0.8, test_frac=0.1):
    train_size = int(adata.shape[0] * train_frac)
    valid_size = int(adata.shape[0] * (1-test_frac))
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    valid_idx = indices[train_size:valid_size]
    test_idx = indices[valid_size:]

    train_data = adata[train_idx, :]
    valid_data = adata[valid_idx, :]
    test_data = adata[test_idx, :]

    return train_data, valid_data, test_data





def load_h5ad_to_dataloader(data, condition_key, cell_type_key, 
                            cell_type, ctrl_key, pert_key, device, batch_size=32, shuffle=False):
    

    data_c = data[(data.obs[condition_key]==ctrl_key)&(data.obs[cell_type_key]==cell_type)]  
    data_p = data[(data.obs[condition_key]==pert_key)&(data.obs[cell_type_key]==cell_type)]
    

    if sparse.issparse(data_c.X):
        data_c = data_c.X.A
    else:
        data_c = data_c.X
    
    if sparse.issparse(data_p.X):
        data_p = data_p.X.A
    else:
        data_p = data_p.X
        
    adata_c = torch.tensor(data_c).float().to(device)
    adata_p = torch.tensor(data_p).float().to(device)
    
    dataset = TensorDataset(adata_c, adata_p)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader






    
