#!/user/bin/python3



import subprocess
import importlib
from pathlib import Path
import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import cellxgene_census
import scvi
from scipy.sparse import csr_matrix
import warnings
import cellxgene_census
import cellxgene_census.experimental
import scvi
import torch
from sklearn.ensemble import RandomForestClassifier
import importlib
import adata_functions
from adata_functions import *
from pathlib import Path
#current_directory = Path.cwd()
projPath = "/space/grp/rschwartz/rschwartz/cpsc545_proj/"
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import json
scvi.settings.seed = 0
torch.set_float32_matmul_precision("high")
from scipy.stats import pearsonr
sc.set_figure_params(figsize=(10, 10), frameon=False)

importlib.reload(adata_functions)
from adata_functions import *

# Read the JSON file
with open(os.path.join(projPath,"meta",'master_hierarchy.json'), 'r') as file:
    
    tree = json.load(file)



n_cells=10000

outpath=f"/space/grp/rschwartz/rschwartz/cpsc545_proj/results_subsample_{n_cells}"
os.makedirs(outpath, exist_ok=True)

# Make these command line args in workflow
# Keys for harmonized labels at 3 levels of granularity
ref_keys = ["rachel_subclass","rachel_class","rachel_family"]
organism="homo_sapiens"
#random.seed(1)
census_version="2024-07-01"
subsample=50
split_column="tissue"
dims=20
directory="/space/grp/rschwartz/rschwartz/cpsc545_proj/data"
test_files=os.listdir(directory)
full_paths = [os.path.join(directory, file) for file in test_files if 'subsample' not in file]

# only need to do once
# Get model file link and download
model_file_path=setup(organism="homo_sapiens", version="2024-07-01")

# when i make workflow, make this part of setup

#for file in full_paths:
 #   subsample_and_save(file, n_cells=n_cells)
    
#in workflow, use get_refs to separate this step
# save as pickle instead of h5ad?
refs=adata_functions.get_census(organism="homo_sapiens", 
                                subsample=100, split_column="dataset_id", dims=20, 
                                ref_collections=["Transcriptomic cytoarchitecture reveals principles of human neocortex organization",
                                                 "SEA-AD: Seattle Alzheimer’s Disease Brain Cell Atlas"],
                                relabel_path=f"{projPath}meta/census_map_human.tsv")

mapping_df = pd.read_csv(f"{projPath}meta/census_map_human.tsv", sep="\t")

test_files=os.listdir(directory)
subsample_paths = [os.path.join(directory, file) for file in test_files if f"subsampled_{n_cells}.h5ad" in file]

queries = {}
for file in subsample_paths:
    query_name = os.path.basename(file).replace("_subsampled_" + str(n_cells) + ".h5ad","")
    relabel_path = os.path.join(projPath, "meta", query_name  + "_relabel.tsv")
    adata = ad.read_h5ad(file)
    sc.pp.calculate_qc_metrics(adata, inplace=True)
   # sc.pp.filter_cells(adata, min_genes=100)
  #  sc.pp.filter_genes(adata, min_cells=3)  
  #  sc.pp.normalize_total(adata)
  #  sc.pp.scale(adata)
   # sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata.obs["batch"] =  adata.obs["Capbatch"].astype(str) + "_" + adata.obs["Seqbatch"].astype(str)
    adata = process_query(adata, model_file_path, batch_key="batch")
    sc.pp.neighbors(adata, use_rep="scvi",n_neighbors=30)
    sc.tl.leiden(adata)
    sc.tl.umap(adata)
    join_key = pd.read_csv(relabel_path, sep="\t").columns[0]
    adata = relabel(adata, relabel_path=relabel_path, join_key=join_key, sep='\t')
    queries[query_name] = adata


# %# Initialize defaultdict for thresholds and confusion
confusion = defaultdict(lambda: defaultdict(dict))
rocs = defaultdict(lambda: defaultdict(dict))
probs = defaultdict(lambda: defaultdict(dict))
class_metrics = defaultdict(lambda: defaultdict(dict))


for query_name, query in queries.items():

    for ref_name,ref in refs.items():
        all_probs = rfc_pred(ref=ref, query=query, ref_keys=ref_keys)
        probs[query_name][ref_name] = all_probs
        rocs[query_name][ref_name] = roc_analysis(probabilities=all_probs, 
                                                    query=query, key=ref_keys[0])
        new_query_name = query_name.replace(" ", "_").replace("/", "_")
        new_ref_name = ref_name.replace(" ", "_").replace("/", "_")
        outdir=os.path.join(outpath,"roc",new_query_name, new_ref_name)
        os.makedirs(outdir, exist_ok=True)  # Create the directory if it doesn't exist
        plot_roc_curves(metrics=rocs[query_name][ref_name],
                       title=f"{query_name} vs {ref_name}",
                        save_path=os.path.join(outdir,"roc_results.png"))




roc_df= process_all_rocs(rocs, queries)

plot_distribution(roc_df, "optimal_threshold", outdir=outpath, split="ref")
plot_distribution(roc_df, "auc", outdir=outpath, split="ref")
plot_distribution(roc_df, "optimal_threshold", outdir=outpath,split="query")
plot_distribution(roc_df, "auc", outdir=outpath, split="query")


queries_classified = defaultdict(lambda: defaultdict(dict))

for query_name, query in queries.items():
    for ref_name, ref in refs.items():
        probabilities = probs[query_name][ref_name][ref_keys[0]]["probabilities"]
        class_labels =  probs[query_name][ref_name][ref_keys[0]]["class_labels"]
        probabilities_df = pd.DataFrame(probabilities, columns=class_labels)
        query = classify_cells(query, ref_keys, 0, probabilities_df, tree)
        queries_classified[query_name][ref_name] = query 
        
        new_query_name = query_name.replace(" ", "_").replace("/", "_")
        new_ref_name = ref_name.replace(" ", "_").replace("/", "_") 
        os.makedirs(os.path.join(outpath,"meta",new_query_name, new_ref_name), exist_ok=True)  # Create the directory if it doesn't exist
        query.obs.to_csv(os.path.join(outpath,"meta",new_query_name, new_ref_name,"meta_transfer.tsv"),sep="\t")
       
        plt.figure(figsize=(8, 10))  # Adjust the width and height as needed
 
        sc.pl.umap(
            query, 
            color=["predicted_" + key for key in ref_keys] + ["cluster"], 
            ncols=1, na_in_legend=True, legend_fontsize=20, 
            show=False  # Prevents immediate display, so we can save it with plt
        )
        outdir =os.path.join(outpath, "umaps",new_query_name,new_ref_name)
        os.makedirs(outdir, exist_ok=True)  # Create the directory if it doesn't exist

        # Save the figure using plt.savefig()
        plt.savefig(
            os.path.join(outdir, "umap.png"), 
            dpi=300, 
            bbox_inches='tight'
        )
      
        plt.close()
        
        class_metrics[query_name][ref_name] = eval(query, 
                                                    ref_keys, mapping_df)
        
        


class_metrics = update_classification_report(class_metrics, ref_keys)

all_f1_scores=combine_f1_scores(class_metrics, ref_keys) # Combine f1 scores into data frame
outdir =os.path.join(outpath, "heatmaps")
os.makedirs(outdir, exist_ok=True)  # Create the directory if it doesn't exist
plot_label_f1_heatmaps(all_f1_scores, threshold=0, outpath=outdir)
plot_aggregated_f1_heatmaps(all_f1_scores, threshold=0, outpath=outdir, ref_keys=ref_keys)

for query_name in queries:
    for ref_name in refs:
            for key in ref_keys:
                new_query_name = query_name.replace(" ", "_").replace("/", "_")
                new_ref_name = ref_name.replace(" ", "_").replace("/", "_") 
                plot_confusion_matrix(query_name, ref_name, key,
                                      class_metrics[query_name][ref_name][key]["confusion"],
                                      output_dir=os.path.join(outpath,'confusion',new_query_name,new_ref_name))
 
 
## Save plots for each query and reference
#corr_dict = defaultdict(lambda: defaultdict(dict))
#for query_name, query in queries.items():
    #for key in ref_keys:
        #true_labels = query.obs[key]
        #class_proportions = pd.Series(true_labels).value_counts(normalize=True)    
        #corr_dict[key] = all_f1_scores[key].merge(
            #class_proportions.rename("proportion"), 
            #left_on="label", 
            #right_index=True
        #)

# Initialize plot
#sns.set(style="whitegrid")

## Loop through each DataFrame in corr_dict
#for key, df in corr_dict.items():
    ## Ensure that the necessary columns are available
    #if 'f1_score' in df.columns and 'proportion' in df.columns:
        
        ## Get unique combinations of query and reference
        #unique_combos = df[['query', 'reference']].drop_duplicates()

        #for query, reference in unique_combos.values:
            ## Filter data for the current query-reference combo
            #query_ref_df = df[(df['query'] == query) & (df['reference'] == reference)]
           ## Calculate the correlation between F1-score and class proportion
            #correlation = query_ref_df[['f1_score', 'proportion']].corr().iloc[0, 1]
            
            ## Create a scatter plot with a regression line
            #plt.figure(figsize=(8, 6))
            
            #sns.regplot(
                #data=query_ref_df, 
                #x='proportion', 
                #y='f1_score', 
                #scatter_kws={'s': 50, 'color': 'blue'},  # Scatter plot style
                #line_kws={'color': 'red', 'lw': 2},  # Regression line style
                #ci=None  # Disable confidence intervals for the regression line
            #)
            
            ## Title with correlation coefficient
            #plt.title(f'Correlation between F1-score and Proportion\n{key} - Query: {query} - Reference: {reference}\nCorrelation: {correlation:.2f}', fontsize=14)
            #plt.xlabel('Class Proportion', fontsize=12)
            #plt.ylabel('F1 Score', fontsize=12)
            
            ## Display or save the plot
            #plt.tight_layout()
            #outdir=os.path.join(outpath,"class_imbalance")
            #os.makedirs(outdir,exist_ok=True)
            #plt.savefig(os.path.join(outdir,f"{key}_{query}_{reference}_f1_vs_proportion_with_line.png"))
            #plt.show()

#for query_name, query in queries.items():
    #for key, df in all_f1_scores.items():
        ## Filter the DataFrame for the specific query
        #query_f1_scores = df[df["query"] == query_name]
        
        ## Find the row with the maximum weighted F1 score
        #max_f1_row = query_f1_scores.loc[query_f1_scores['weighted_f1'].idxmax()]
        
        ## Extract the reference (ref) at which the max F1 score occurs
        #max_f1_ref = max_f1_row['reference']
        
        ## Print the result
        #print(f"Reference with maximum weighted F1 score for {query_name} in {key}: {max_f1_ref} with F1 score {max_f1_row['weighted_f1']:.3f}")

query_name="velmeshev"
ref_name="whole cortex"

adata_ref = refs[ref_name]
adata_query = queries[query_name] 
adata_query.obs["dataset_title"] = query_name
adata_query.obs["tissue"] = adata_query.obs["region"]
adata_concat = ad.concat([adata_query, adata_ref])

new_query_name = query_name.replace(" ", "_").replace("/", "_")
new_ref_name = ref_name.replace(" ", "_").replace("/", "_")   

outdir=os.path.join(projPath,"mapped_queries",new_query_name,new_ref_name,f"subsample_{n_cells}")
os.makedirs(outdir, exist_ok=True)  # Create the directory if it doesn't exist

# Make sure 'scvi' embeddings exist
sc.pp.neighbors(adata_concat, use_rep="scvi")
sc.tl.umap(adata_concat)
# Plot UMAP with additional metadata (e.g., 'rachel_class', 'tissue', 'dataset_title')

sc.pl.umap(adata_concat, color=['rachel_class', 'tissue', 'dataset_title'],
           show=False, title="UMAP: Whole Cortex vs Velmeshev", ncols=2)

# Save the figure using plt.savefig()
plt.savefig(
    os.path.join(outdir, "umap.png"), 
    dpi=300, 
    bbox_inches='tight'
)

# Close the plot to free up memory
plt.close()


queries_classified["velmeshev"]["whole cortex"].write_h5ad(os.path.join(outdir, "query_mapped.h5ad"))

