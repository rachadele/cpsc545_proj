#import PyWGCNA
import anndata as ad
import scanpy as sc
import numpy as np
import scvi
import warnings
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
warnings.simplefilter(action='ignore', category=FutureWarning)
scvi.settings.seed = 0
#from adata_functions import *

projPath="/space/grp/rschwartz/rschwartz/cpsc545_proj/"

outdir=os.path.join(projPath,"results_subsample_1000")
model_file_path = "/space/grp/rschwartz/rschwartz/cpsc545_proj/bin/scvi-human-2024-07-01"

adata_mapped= ad.read_h5ad(os.path.join(projPath,"mapped_queries/velmeshev/whole_cortex/subsample_1000/query_mapped.h5ad"))
adata=ad.read_h5ad(os.path.join(projPath,"data/velmeshev_subsampled_1000.h5ad"))
#scvi.model.SCVI.prepare_query_anndata(adata, model_file_path)
#vae_q = scvi.model.SCVI.load_query_data(adata, model_file_path)
# Set the model to trained and get latent representation
#vae_q.is_trained = True
#latent = vae_q.get_latent_representation()
#adata.obsm["scvi"] = latent
adata.obs=adata_mapped.obs
scvi.model.SCVI.setup_anndata(adata, batch_key='batch')
# Create the SCVI model
model_scvi = scvi.model.SCVI(adata)
# Train the mmodel_scvi
model_scvi.train(max_epochs=100, early_stopping=True)

adata.layers["X_normalized_scVI"] = model_scvi.get_normalized_expression(library_size="latent")
# Log normalize the counts
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
# Save a copy of the log_normalized counts in the layer log_normalized
adata.layers['log_normalized'] = adata.X.copy()
# List of genes to plot
genes_to_plot = ["OLIG1","CUX2","SST","VIP","AQP4","SLC17A7","FEZF2","GAD1","RORB","C3","PVALB","LAMP5","SNCG"]
#indexes = adata.var[adata.var["feature_name"].isin(genes_to_plot)].index
# Plot expression for these genes


rcParams.update({
    "font.size": 20,  # Increase this value to make fonts larger globally
    "axes.titlesize": 25,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
})
# Set up a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # 1 row, 2 columns

# Plot the first matrixplot for 'log_normalized'
sc.pl.matrixplot(
    adata,
    var_names=genes_to_plot,
    use_raw=False,
    cmap="viridis",
    groupby="predicted_rachel_subclass",
    gene_symbols="feature_name",
    layer="log_normalized",
    ax=axes[0],  # Specify the axis for this plot
    show=False,
    dendrogram=True # Suppress immediate display
)
axes[0].set_title("Log-Normalized")

# Plot the second matrixplot for 'X_normalized_scVI'
sc.pl.matrixplot(
    adata,
    var_names=genes_to_plot,
    use_raw=False,
    cmap="viridis",
    groupby="predicted_rachel_subclass",
    gene_symbols="feature_name",
    layer="X_normalized_scVI",
    ax=axes[1],  # Specify the axis for this plot
    show=False,
    dendrogram=True  # Suppress immediate display
)
axes[1].set_title("SCVI-Normalized")

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(outdir,"matrixplot_comparison.png"), dpi=300)
plt.show()

