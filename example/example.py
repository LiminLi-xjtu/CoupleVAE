from couplevae import *
import anndata
import numpy as np
import scanpy as sc
import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot
import pandas as pd
from scipy import stats, sparse
from adjustText import adjust_text



def autolabel(rects):

    for rect in rects:
        height = rect.get_height()
        pyplot.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                '%.2f' % float(height),
                ha='center', va='bottom', fontsize=18)
        
        
def reg_mean_plot(adata, condition_key, axis_keys, labels, path_to_save="./reg_mean.pdf", gene_list=None, top_100_genes=None,
                  show=False,
                  legend=True, title=None,
                  x_coeff=0.30, y_coeff=0.8, fontsize=14, **kwargs):

    import seaborn as sns
    sns.set()
    sns.set(color_codes=True)
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    diff_genes = top_100_genes
    stim = adata[adata.obs[condition_key] == axis_keys["y"]]
    ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
    if diff_genes is not None:
        if hasattr(diff_genes, "tolist"):
            diff_genes = diff_genes.tolist()
        adata_diff = adata[:, diff_genes]
        stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
        ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
        x_diff = np.average(ctrl_diff.X, axis=0)
        y_diff = np.average(stim_diff.X, axis=0)
        m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(x_diff, y_diff)
        print(r_value_diff ** 2)
    if "y1" in axis_keys.keys():
        real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
    x = np.average(ctrl.X, axis=0)
    y = np.average(stim.X, axis=0)
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    print(r_value ** 2)
    df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
    ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df, scatter_kws={'rasterized': True})
    ax.tick_params(labelsize=fontsize)
    if "range" in kwargs:
        start, stop, step = kwargs.get("range")
        ax.set_xticks(np.arange(start, stop, step))
        ax.set_yticks(np.arange(start, stop, step))
    # _p1 = pyplot.scatter(x, y, marker=".", label=f"{axis_keys['x']}-{axis_keys['y']}")
    # pyplot.plot(x, m * x + b, "-", color="green")
    ax.set_xlabel(labels["x"], fontsize=fontsize)
    ax.set_ylabel(labels["y"], fontsize=fontsize)
    # if "y1" in axis_keys.keys():
        # y1 = np.average(real_stim.X, axis=0)
        # _p2 = pyplot.scatter(x, y1, marker="*", c="red", alpha=.5, label=f"{axis_keys['x']}-{axis_keys['y1']}")
    if gene_list is not None:
        texts = []
        for i in gene_list:
            j = adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            texts.append(pyplot.text(x_bar, y_bar , i, fontsize=11, color ="black"))
            pyplot.plot(x_bar, y_bar, 'o', color="red", markersize=5)
            # if "y1" in axis_keys.keys():
                # y1_bar = y1[j]
                # pyplot.text(x_bar, y1_bar, i, fontsize=11, color="black")
    if gene_list is not None:
        adjust_text(texts, x=x, y=y, arrowprops=dict(arrowstyle="->", color='grey', lw=0.5), force_points=(0.0, 0.0))
    if legend:
        pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if title is None:
        pyplot.title(f"", fontsize=fontsize)
    else:
        pyplot.title(title, fontsize=fontsize)
    ax.text(max(x) - max(x) * x_coeff, max(y) - y_coeff * max(y), r'$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= ' + f"{r_value ** 2:.2f}", fontsize=kwargs.get("textsize", fontsize))
    if diff_genes is not None:
        ax.text(max(x) - max(x) * x_coeff, max(y) - (y_coeff+0.15) * max(y), r'$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= ' + f"{r_value_diff ** 2:.2f}", fontsize=kwargs.get("textsize", fontsize))
    pyplot.savefig(f"{path_to_save}", bbox_inches='tight', dpi=300)
    if show:
        pyplot.show()
    pyplot.close()
    
    

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

# Save
pred_adata = anndata.AnnData(pred, 
                             obs={condition_key: [f"{cell_type}_pred_pert"] * len(pred),
                                  cell_type_key: [cell_type] * len(pred)},
                             var={"var_names": test_adata_c.var_names})
if sparse.issparse(test_adata_c.X):
    test_adata_c.X = test_adata_c.X.A
else:
    test_adata_c.X = test_adata_c.X
ctrl_adata = anndata.AnnData(test_adata_c.X,
                             obs={condition_key: [f"{cell_type}_ctrl"] * len(test_adata_c),
                                  cell_type_key: [cell_type] * len(test_adata_c)},
                             var={"var_names": test_adata_c.var_names})
if sparse.issparse(test_adata_p.X):
    test_adata_p.X = test_adata_p.X.A
else:
    test_adata_p.X = test_adata_p.X
real_stim_adata = anndata.AnnData(test_adata_p.X,
                                  obs={condition_key: [f"{cell_type}_real_pert"] * len(test_adata_p),
                                       cell_type_key: [cell_type] * len(test_adata_p)},
                                  var={"var_names": test_adata_p.var_names})

all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
all_data.write_h5ad(f"./coupleVAE{data_name}_{cell_type}.h5ad")


# Plot
result=sc.read(f"./coupleVAE{data_name}_{cell_type}.h5ad")

sc.tl.rank_genes_groups(result,groupby="condition",n_genes=100,method="wilcoxon")

diff_genes_covid=result.uns["rank_genes_groups"]["names"][f"{cell_type}_real_pert"]
conditions={"ctrl":f"{cell_type}_ctrl","pred_stim":f"{cell_type}_pred_pert","real_stim":f"{cell_type}_real_pert"}
reg_mean_plot(result,
              condition_key="condition",
              axis_keys={"x":conditions["pred_stim"],"y":conditions["real_stim"]},
              gene_list=diff_genes_covid[:5],
              top_100_genes=diff_genes_covid,
              legend=False,
              labels={"x":"pred","y":"real"},
              title=f"CoupleVAE_{cell_type}",
              path_to_save=f"./CoupleVAE_{cell_type}.pdf",
              show=False,
             )
