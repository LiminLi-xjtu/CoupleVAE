
# CoupleVAE

This is a PyTorch implementation for our paper:

CoupleVAE: coupled variational autoencoders for predicting perturbational single-cell RNA sequencing data

Yahao Wu, Jing Liu, Songyan Liu, Yanni Xiao, Shuqin Zhang and Limin Li

![image](https://github.com/LiminLi-xjtu/CoupleVAE/blob/master/img/couplevae_arch.png)

    
## Getting Started

To run the CoupleVAE you need following packages :
### `Requirements`

    python                                               3.11.0 
    anndata                                              0.10.7
    adjustText                                           0.8
    scanpy                                               1.9.2
    torch                                                2.3.1
    torchaudio                                           2.3.1
    torchcision                                          0.18.1
    numpy                                                1.26.0
    scipy                                                1.13.1
    sklearn                                              1.5.0
    pandas                                               1.5.2
    matplotlib                                           3.6.0
    seaborn                                              0.13.2
    
## Installation

install the development version via pip:
```bash
pip install git+https://github.com/LiminLi-XJTU/CoupleVAE.git
```

## Example
The data utilize in CoupleVAE is in the h5ad format. If your data is in other format, please refer to Scanpy' or anndata's tutorial for how to create one.

### Input
The data input to CoupleVAE are better the normalized and scaled data, you can use follow codes for this purpose.
```Python
import scanpy as sc

sc.pp.filter_genes(adata, min_counts=10)
sc.pp.filter_cells(adata, min_counts=3)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)
```

### How to use it
The following is an example of the dataset COVID-19


```Python
from couplevae import *
import scanpy as sc

# Load Data
train = sc.read(train_path)
valid = sc.read(valid_path)
test = sc.read(test_path) 

data_name = "covid"
cell_type = "Macrophages"
condition_key = "condition"
cell_type_key = "celltype"
pert_key = "severe COVID-19"
ctrl_key = "control"
device = "cuda"

trainloader = load_h5ad_to_dataloader(train, condition_key, cell_type_key, 
                                    cell_type, ctrl_key, pert_key, device)
validloader = load_h5ad_to_dataloader(valid, condition_key, cell_type_key, 
                                    cell_type, ctrl_key, pert_key, device)

test_adata_c = test[(test.obs[condition_key]==ctrl_key)&(test.obs[cell_type_key]==cell_type)]  
test_adata_p = test[(test.obs[condition_key]==pert_key)&(test.obs[cell_type_key]==cell_type)]

# Create Model
network = VAE(x_dim=train.X.shape[1],
              z_dim=200,
              alpha=0.00005,
              beta=0.05,
              dropout_rate=0.1,
              learning_rate=0.0001)
trainer = Trainer(model=network, learning_rate=0.0001, n_epochs=200, patience=20, batch_size=32)
                     
# Train
trainer.train(train_loader=trainloader, valid_loader=validloader)

# Test
pred = network.predict(test_adata_c, test_adata_p)
```
Then you can complete the training process and get the predicted data.

For more examples, refer to the folder ```/example/example.py ```

## Datasets
You can download the preprocessed data [here](https://drive.google.com/drive/folders/1VkKqwFd9AfVRG9E2ue8XZLBW1QUPq5Qb)

