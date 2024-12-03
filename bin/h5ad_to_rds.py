
from pathlib import Path
import os
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.io import mmwrite
import os

filepath = "/space/grp/rschwartz/rschwartz/cpsc545_proj/data/velmeshev_subsampled_10000.h5ad"
adata =ad.read_h5ad(filepath)
map_path ="/space/grp/rschwartz/rschwartz/cpsc545_proj/mapped_queries/velmeshev/whole_cortex/subsample_10000/query_mapped.h5ad"
adata_mapped = ad.read_h5ad(map_path)
base_dir = os.path.dirname(map_path)
file_name = os.path.basename(filepath).replace(".h5ad","")

# Define output file paths
mtx_file = os.path.join(base_dir, f"{file_name}_mapped.mtx")
obs_file = os.path.join(base_dir, f"{file_name}_obs_mapped.tsv")
var_file = os.path.join(base_dir, f"{file_name}_var_mapped.tsv")

# Export .X to .mtx (sparse matrix format)
# use original
mmwrite(mtx_file, adata.X.T)
# Export .obs to .tsv
# only use updated cell ids
adata_mapped.obs.to_csv(obs_file, sep='\t')

# Export .var to .tsv
# use original genes as sciv
adata.var.to_csv(var_file, sep='\t')

    
