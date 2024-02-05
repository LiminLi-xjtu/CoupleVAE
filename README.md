
# CoupleVAE

This is the official Implementation for our paper:

CoupleVAE: coupled variational autoencoders for predicting perturbational single-cell RNA sequencing data

Yahao Wu, Jing Liu, Songyan Liu, Yanni Xiao, Shuqin Zhang and Limin Li

![image](https://github.com/LiminLi-xjtu/CoupleVAE/blob/master/img/couplevae_arch.png)

    
## Getting Started

To run the CoupleVAE you need following packages :
### `Requirements`

    python                                               3.6 
    anndata                                              0.7.4
    scanpy                                               1.6.0
    tensorflow                                           2.0.0
    numpy                                                1.20.3
    scipy                                                1.5.3
    pandas                                               1.1.3
    matplotlib                                           3.4.3
    seaborn                                              0.11.2
    
## Installation

install the development version via pip:
```bash
pip install git+https://github.com/LiminLi-XJTU/CoupleVAE.git
```

## How to use it



```Python
import couplevae
import scanpy as sc

# Create Model
train_ctrl = sc.read(data)
train_pert = sc.read(data)
network = couple.VAE(x_dimension=train_ctrl.X.shape[1],
                     z_dimension=100,
                     alpha=0.00005,
                     dropout_rate=0.2,
                     learning_rate=0.001)
                     
# Training
network.train(train_ctrl, train_pert, n_epochs=n_epochs, batch_size=batch_size)

# Testing
pred, _ = network.predict(train_ctrl,
                          test,
                          conditions = {"ctrl": ctrl_key, "pert": pert_key},
                          cell_type_key=cell_type_key,
                          condition_key=condition_key,
                          celltype_to_predict=cell_type,
                          biased=True)
```
Then you can complete the training process and get the predicted data.

In order to reproduce paper results visit [here](https://drive.google.com/drive/folders/1VkKqwFd9AfVRG9E2ue8XZLBW1QUPq5Qb?usp=sharing)
