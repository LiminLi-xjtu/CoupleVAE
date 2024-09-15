from couplevae import *
import anndata
import numpy as np
import scanpy as sc
import seaborn as sns
mpl.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import issparse
from adjustText import adjust_text

        
        
def plot_gene_correlation(data, cond_col, axes_map, label_map, file_path='./output_plot.pdf', 
                          genes_of_interest=None, top_genes=None, show_plot=False, include_legend=True, 
                          plot_title='', x_offset=0.3, y_offset=0.8, text_size=14):
    # Check if optional parameters were passed and set default values
    if genes_of_interest is None:
        genes_of_interest = []
    if top_genes is None:
        top_genes = []

    # Convert sparse matrix to dense matrix
    if issparse(data.X):
        data.X = data.X.toarray()

    # Retrieve groups of cells based on the condition
    group_y = data[data.obs[cond_col] == axes_map['y'], :]
    group_x = data[data.obs[cond_col] == axes_map['x'], :]

    # If top_genes are provided, handle the differential gene part
    if len(top_genes) > 0:
        subset_data_x = group_x[:, top_genes].X
        subset_data_y = group_y[:, top_genes].X

        avg_x_genes = np.mean(subset_data_x, axis=0)
        avg_y_genes = np.mean(subset_data_y, axis=0)

        # Calculate R² value
        r_squared_top = np.corrcoef(avg_x_genes, avg_y_genes)[0, 1] ** 2
        print(f'R-squared for top genes: {r_squared_top:.2f}')

    # Compute the average expression for all genes
    avg_x = np.mean(group_x.X, axis=0)
    avg_y = np.mean(group_y.X, axis=0)

    # Calculate R² value
    r_squared_all = np.corrcoef(avg_x, avg_y)[0, 1] ** 2
    print(f'R-squared for all genes: {r_squared_all:.2f}')

    # Create a DataFrame with x and y data
    df = pd.DataFrame({label_map['x']: avg_x, label_map['y']: avg_y})

    # Create a scatter plot and use sns.regplot to draw the regression line
    plt.figure()
    ax = sns.regplot(x=label_map['x'], y=label_map['y'], data=df, scatter_kws={'s': 10}, line_kws={'color': 'green'})
    
    plt.xlabel(label_map['x'], fontsize=text_size)
    plt.ylabel(label_map['y'], fontsize=text_size)

    # If a title is specified
    if plot_title:
        plt.title(plot_title, fontsize=text_size)

    # If specific genes are provided, label them
    if len(genes_of_interest) > 0:
        texts = []
        for gene in genes_of_interest:
            gene_idx = np.where(data.var_names == gene)[0][0]  # Find the gene index in var_names
            x_pos = avg_x[gene_idx]
            y_pos = avg_y[gene_idx]
            texts.append(plt.text(x_pos, y_pos, gene, fontsize=10, color='black'))
            plt.scatter(x_pos, y_pos, color='red', s=40)
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='grey', lw=0.5))

    # Display R² value
    plt.text(max(avg_x) - max(avg_x) * x_offset, max(avg_y) - y_offset * max(avg_y),
             r'$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$='+ f"{r_squared_all:.2f}", fontsize=text_size)

    if len(top_genes) > 0:
        plt.text(max(avg_x) - max(avg_x) * x_offset, max(avg_y) - (y_offset + 0.15) * max(avg_y),
                r'$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$='+ f"{r_squared_top:.2f}", fontsize=text_size)

    # Display legend if needed
    if include_legend:
        plt.legend(loc='best')

    # Save the plot
    plt.savefig(file_path, bbox_inches='tight', dpi=300)

    # Show the plot if required
    if show_plot:
        plt.show()

    plt.close()
    
    

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
plot_gene_correlation(result, 
                      cond_col="condition",
                      axes_map={"x":conditions["pred_stim"],"y":conditions["real_stim"]},
                      genes_of_interest=diff_genes_covid[:5],
                      top_genes=diff_genes_covid,
                      include_legend=False,
                      label_map={"x":"pred","y":"real"},
                      plot_title=f"CoupleVAE_{cell_type}",
                      file_path=f"./CoupleVAE_{cell_type}.pdf",
                      show_plot=True,
                      ) 
