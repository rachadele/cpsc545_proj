import pandas as pd
import numpy as np
import scanpy as sc
import random
import cellxgene_census
import cellxgene_census.experimental
import os
import anndata as ad
import scvi
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
scvi.settings.seed = 0
from pathlib import Path
#current_directory = Path.cwd()
projPath = "/space/grp/rschwartz/rschwartz/cpsc545_proj/"

import subprocess


def setup(organism="homo_sapiens", version="2024-07-01"):
    organism=organism.replace(" ", "_") 
    #census = cellxgene_census.open_soma(census_version=version)
    outdir = f"scvi-human-{version}"  # Concatenate strings using f-string
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Check if the model file exists
    model_file_path = os.path.join(outdir, "model.pt")
    #if not os.path.exists(model_file_path):
        # Get scVI model info
    scvi_info = cellxgene_census.experimental.get_embedding_metadata_by_name(
            embedding_name="scvi",
            organism=organism,
            census_version=version,
        )

        # Extract the model link
    model_link = scvi_info["model_link"]
    date = model_link.split("/")[5]
    url = os.path.join("https://cellxgene-contrib-public.s3.us-west-2.amazonaws.com/models/scvi/", date, organism, "model.pt")

    # Download the model using wget if it doesn't already exist
    subprocess.run(["wget", "--no-check-certificate", "-q", "-O", model_file_path, url])
# else:
     #   print(f"File already exists at {model_file_path}, skipping download.")

    return(outdir)

def subsample_and_save(dataset_path, n_cells=1000):
    dataset = ad.read_h5ad(dataset_path)
    subsampled_dataset = dataset[np.random.choice(dataset.n_obs, size=n_cells, replace=False), :] if dataset.n_obs > n_cells else dataset
    dir_name, base_name = os.path.split(dataset_path)
    file_name, ext = os.path.splitext(base_name)
    output_path = os.path.join(dir_name, f"{file_name}_subsampled_{n_cells}{ext}")
    subsampled_dataset.write_h5ad(output_path)
    print(f"Saved to {output_path}")
    
# Subsample x cells from each cell type if there are n>x cells present
#ensures equal representation of cell types in reference
def subsample_cells(data, filtered_ids, subsample=500):
    # Filter data based on filtered_ids
    obs = data[data['soma_joinid'].isin(filtered_ids)]
    celltypes = obs['cell_type'].unique()
    final_idx = []
    for celltype in celltypes:
        celltype_ids = obs[obs['cell_type'] == celltype]['soma_joinid'].values
        # Sample if there are enough observations, otherwise take all
        if len(celltype_ids) > subsample:
            subsampled_cell_idx = random.sample(list(celltype_ids), subsample)
        else:
            subsampled_cell_idx = celltype_ids.tolist()
        # Append subsampled indices to final list
        final_idx.extend(subsampled_cell_idx)

    # Return final indices
    return final_idx

def relabel(adata, relabel_path, join_key, sep="\t"):
    # Read the relabel table from the file
    relabel_df = pd.read_csv(relabel_path, sep=sep)  # Adjust the separator as needed
    # Take the first column as the join key
   # join_key = relabel_df.columns[0]
    # Ensure the join_key is in both the AnnData object and the relabel DataFrame
    if join_key not in adata.obs.columns:
        raise ValueError(f"{join_key} not found in AnnData object observations.")
    if join_key not in relabel_df.columns:
        raise ValueError(f"{join_key} not found in relabel DataFrame.")
    # Perform the left join to update the metadata
    adata.obs = adata.obs.merge(relabel_df, on=join_key, how='left', suffixes=(None, "_y"))
    columns_to_drop = [col for col in adata.obs.columns if col.endswith('_y')]
    adata.obs.drop(columns=columns_to_drop, inplace=True)
    return adata


def extract_data(data, filtered_ids, subsample=10, organism=None, census=None, 
    obs_filter=None, cell_columns=None, dataset_info=None, dims=20, relabel_path="biof501_proj/meta/relabel/census_map_human.tsv'"):
    
    brain_cell_subsampled_ids = subsample_cells(data, filtered_ids, subsample)
    # Assuming get_seurat is defined to return an AnnData object
    adata = cellxgene_census.get_anndata(
        census=census,
        organism=organism,
        obs_value_filter=obs_filter,  # Ensure this is constructed correctly
        obs_column_names=cell_columns,
        obs_coords=brain_cell_subsampled_ids,
        var_value_filter = "nnz > 10",
        obs_embeddings=["scvi"])
    sc.pp.filter_genes(adata, min_cells=3)
    print("Subsampling successful.")
    newmeta = adata.obs.merge(dataset_info, on="dataset_id", suffixes=(None,"y"))
    adata.obs = newmeta
    # Assuming relabel_wrapper is defined
    adata = relabel(adata, relabel_path=relabel_path, join_key="cell_type", sep='\t')
    # Convert all columns in adata.obs to factors (categorical type in pandas)
    return adata

def split_and_extract_data(data, split_column, subsample=500, organism=None, census=None, 
                           cell_columns=None, dataset_info=None, dims=20, relabel_path="/biof501_proj/meta/relabel/census_map_human.tsv"):
    # Get unique split values from the specified column
    unique_values = data[split_column].unique()
    refs = {}

    for split_value in unique_values:
        # Filter the brain observations based on the split value
        filtered_ids = data[data[split_column] == split_value]['soma_joinid'].values
        obs_filter = f"{split_column} == '{split_value}'"
        
        adata = extract_data(data, filtered_ids, subsample, organism, census, obs_filter, 
                             cell_columns, dataset_info, dims=dims, relabel_path=relabel_path)
        dataset_titles = adata.obs['dataset_title'].unique()

        if len(dataset_titles) > 1:
            name_to_use = split_value
        else:
            name_to_use = dataset_titles[0]

        refs[name_to_use] = adata

    return refs

def get_census(census_version="2024-07-01", organism="homo_sapiens", subsample=10, split_column="tissue", dims=20, 
               ref_collections=["Transcriptomic cytoarchitecture reveals principles of human neocortex organization"],
               relabel_path=f"{projPath}meta/census_map_human.tsv"):

    census = cellxgene_census.open_soma(census_version=census_version)
    dataset_info = census.get("census_info").get("datasets").read().concat().to_pandas()
    brain_obs = cellxgene_census.get_obs(census, organism,
        value_filter=(
            "tissue_general == 'brain' and "
            "is_primary_data == True and "
            "disease == 'normal' "
        ))
    
    brain_obs = brain_obs.merge(dataset_info, on="dataset_id", suffixes=(None,"_y"))
    brain_obs.drop(columns=['soma_joinid_y'], inplace=True)
    brain_obs_filtered = brain_obs
    # Filter based on organism
    if organism == "homo_sapiens":
        brain_obs_filtered = brain_obs[
            brain_obs['collection_name'].isin(ref_collections)]
        brain_obs_filtered = brain_obs_filtered[~brain_obs_filtered['cell_type'].isin(["unknown", "glutamatergic neuron"])]
    elif organism == "mus_musculus":
        brain_obs_filtered = brain_obs[
            brain_obs['collection_name'].isin(ref_collections) 
        ]
    else:
       raise ValueError("Unsupported organism")

    # Adjust organism naming for compatibility
    organism_name_mapping = {
        "homo_sapiens": "Homo sapiens",
        "mus_musculus": "Mus musculus"
    }
    organism = organism_name_mapping.get(organism, organism)

    cell_columns = [
        "assay", "cell_type", "tissue",
        "tissue_general", "suspension_type",
        "disease", "dataset_id", "development_stage",
        "soma_joinid"
    ]
    # Get individual datasets and embeddings
    refs = split_and_extract_data(
        brain_obs_filtered, split_column=split_column,
        subsample=subsample, organism=organism,
        census=census, cell_columns=cell_columns,
        dataset_info=dataset_info, dims=dims,
        relabel_path=relabel_path
    )
    # Get embeddings for all data together
    filtered_ids = brain_obs_filtered['soma_joinid'].values
    adata = extract_data(
        brain_obs_filtered, filtered_ids,
        subsample=subsample, organism=organism,
        census=census, obs_filter=None,
        cell_columns=cell_columns, dataset_info=dataset_info, dims=dims,
        relabel_path=relabel_path
    )
    refs["whole cortex"] = adata

    for name, ref in refs.items():
        dataset_title = name.replace(" ", "_")
        for col in ref.obs.columns:
            if ref.obs[col].dtype.name =='category':
    # Convert to Categorical and remove unused categories
                ref.obs[col] = pd.Categorical(ref.obs[col].cat.remove_unused_categories())
    
    return refs



def process_query(query, model_file_path, batch_key="sample"):
    # Ensure the input AnnData object is valid
    if not isinstance(query, ad.AnnData):
        raise ValueError("Input must be an AnnData object.")

    # Assign ensembl_id to var
    #query.var["ensembl_id"] = query.var["feature_id"]
    if "feature_id" in query.var.columns:
        query.var.set_index("feature_id", inplace=True)

    query.obs["n_counts"] = query.X.sum(axis=1)
    query.obs["joinid"] = list(range(query.n_obs))
    query.obs["batch"] = query.obs[batch_key]

    # Filter out missing HGNC features
    #query = query[:, query.var["feature_name"].notnull().values].copy()

    # Prepare the query AnnData for scVI
    scvi.model.SCVI.prepare_query_anndata(query, model_file_path)
    vae_q = scvi.model.SCVI.load_query_data(query, model_file_path)

    # Set the model to trained and get latent representation
    vae_q.is_trained = True
    latent = vae_q.get_latent_representation()
    query.obsm["scvi"] = latent

    return query


# Function to find a node's parent in the tree
def find_parent_label(tree, target_label, current_path=None):
    if current_path is None:
        current_path = []
    for key, value in tree.items():
        # Add the current node to the path
        current_path.append(key)
        # If we found the target, return the parent label if it exists
        if key == target_label:
            if len(current_path) > 1:
                return current_path[-2]  # Return the parent label
            else:
                return None  # No parent if we're at the root
        # Recurse into nested dictionaries if present
        if isinstance(value, dict):
       #     print(value)
            result = find_parent_label(value, target_label, current_path)
           # print(result)
            if result:
                return result
        # Remove the current node from the path after processing
        current_path.pop()
    return None

# Recursive function to get the closest valid label
def get_valid_label(original_label, query_labels, tree):
    # Base case: if the label exists in query, return it
    if original_label in query_labels:
        return original_label
    # Find the parent label in the tree
    parent_label = find_parent_label(tree, original_label)
    # Recursively check the parent label if it exists
    if parent_label:
        return get_valid_label(parent_label, query_labels, tree)
    else:
        return None  # Return None if no valid parent found

# Example usage

def map_valid_labels(query, ref_keys, mapping_df):
    # deal with differing levels of granularity
    for key in ref_keys:
        original=query.obs[key].unique()
        for og in original:
            if og not in mapping_df[key].unique():
                level = mapping_df.columns[mapping_df.apply(lambda col: og in col.values, axis=0)]
                og_index = query.obs.index[query.obs[key] == og]
                # Replace the value in "predicted_" column with corresponding predicted value at `level`
                for idx in og_index:
                    # Find the replacement value from `mapping_df` for this level
                    replacement = query.obs.loc[idx, "predicted_" + level]
                    # replace predicted id with appropriate level
                    query.obs["predicted_" + key] = query.obs["predicted_" + key].astype("object")
                    query.obs.loc[idx, "predicted_" + key] = replacement.iloc[0]
                    query.obs["predicted_" + key] = query.obs["predicted_" + key].astype("category")

    return query            




def find_node(tree, target_key):
    """
    Recursively search the tree for the target_key and return the corresponding node. 
    """
    for key, value in tree.items():
        if isinstance(value, dict):
            if key == target_key:  # If we've found the class at this level
                return value  # Return the current node
            else:
                # Recurse deeper into the tree
                result = find_node(value, target_key)
                if result:
                    return result
    return None  # Return None if the target key is not found


# Helper function to recursively gather all subclasses under a given level
def get_subclasses(node, colname):
    subclasses = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict) and value.get("colname") == colname:
                subclasses.append(key)
            else:
                subclasses.extend(get_subclasses(value, colname))
    return subclasses


def rfc_pred(ref, query, ref_keys):
    """
    Fit a RandomForestClassifier at the most granular level and aggregate probabilities for higher levels.
    
    Parameters:
    - ref: Reference data with labels.
    - query: Query data for prediction.
    - ref_keys: List of ordered keys from most granular to highest level (e.g., ["rachel_subclass", "rachel_class", "rachel_family"]).
    - tree: Dictionary representing the hierarchy of classes.
    
    Returns:
    - probabilities: Dictionary with probabilities for each level of the hierarchy.
    """
    probabilities = {}
    
    # The most granular key is the first in the ordered list
    granular_key = ref_keys[0]
    
    # Initialize and fit the RandomForestClassifier at the most granular level
    rfc = RandomForestClassifier(class_weight='balanced', random_state=42, max_depth=20, )
    rfc.fit(ref.obsm["scvi"], ref.obs[granular_key].values)
    # Predict probabilities at e most granular level
    probs_granular = rfc.predict_proba(query.obsm["scvi"])
    class_labels_granular = rfc.classes_
    base_score = rfc.score(query.obsm["scvi"], query.obs[granular_key].values)

    # Store granular level probabilities
    probabilities[granular_key] = {
        "probabilities": probs_granular,
        "class_labels": class_labels_granular,
        "accuracy": base_score
    }
    
    return probabilities 



def roc_analysis(probabilities, query, key):
    optimal_thresholds = {}
    metrics={}
  #  for key in ref_keys:
       # print(key) 
    probs = probabilities[key]["probabilities"]
    class_labels = probabilities[key]["class_labels"]
    optimal_thresholds[key] = {}
        
    # Binarize the class labels for multiclass ROC computation
    true_labels = label_binarize(query.obs[key].values, classes=class_labels)
        
    # Find the optimal threshold for each class
    metrics[key] = {}
    for i, class_label in enumerate(class_labels):
        optimal_thresholds[key][class_label] = {}
        # check for positive samples
        # usually positive samples are 0 when a ref label is
        # replaced with a parent label
        # since it is not in the original query labels
        # but it is being annotated during the label transfer
        # these should not be evaluated ?
        # incorrect classifications will be reflected in the AUC and F1 of the og label
        # eg. ET is not in query so it is changed to "deep layer non-IT"
        # but these cells are CT or NP in the ref, so they're still incorrect
        # not entirely sure how to deal with this
        positive_samples = np.sum(true_labels[:, i] == 1)
        if positive_samples == 0:
            print(f"Warning: No positive samples for class {class_label}, skipping eval and setting threshold to 0.5")
            optimal_thresholds[key][class_label] = 0.5
        elif positive_samples > 0:
            metrics[key][class_label]={}
            # True label one hot encoding at class label index = 
            # vector of all cells which are either 1 = label or 0 = not label
            # probs = probability vector for all cells given class label
            fpr, tpr, thresholds = roc_curve(true_labels[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr) # change to roc_auc_score, ovo, average= macro, labels               
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            if optimal_threshold == float('inf'):
                optimal_threshold = 0 
            optimal_thresholds[key][class_label]=optimal_threshold
            metrics[key][class_label]["tpr"] = tpr
            metrics[key][class_label]["fpr"] = fpr
            metrics[key][class_label]["auc"] = roc_auc
            metrics[key][class_label]["optimal_threshold"] = optimal_threshold

    return metrics


def process_all_rocs(rocs, queries): 
    # Populate the list with threshold data
    data = []

    for query_name, query_dict in rocs.items():
        for ref_name, ref_data in query_dict.items():
            for key, roc in ref_data.items():
                if roc:
                    for class_label, class_data in roc.items():
                        if class_data:
                            data.append({
                                "ref": ref_name,
                                "query": query_name,
                                "key": key, 
                                "label": class_label, 
                                "auc": class_data["auc"],
                                "optimal_threshold": class_data["optimal_threshold"]
                              #   f'{var}': class_data[var]
                            })

    # Create DataFrame from the collected data
    df = pd.DataFrame(data)
    return df


def process_roc(rocs, ref_name, query_name):
    data=[]
    for key, roc in rocs.items():
        if roc:
            for class_label, class_data in roc.items():
                if class_data:
                        data.append({
                                "ref": ref_name,
                                "query": query_name,
                                "key": key, 
                                "label": class_label, 
                                "auc": class_data["auc"],
                                "optimal threshold": class_data["optimal_threshold"]
                              #   f'{var}': class_data[var]
                            })

    # Create DataFrame from the collected data
    roc_df = pd.DataFrame(data)
    return roc_df 

def plot_distribution(df, var, outdir, split="query"):
    # Set up the figure size
    plt.figure(figsize=(17, 6))
    # Create the violin plot with faceting by the 'query' column
    sns.violinplot(data=df, y=var, x='label', palette="Set2", hue=split, split=False)
    # Set the labels and title
    plt.xlabel('Key', fontsize=14)
    plt.ylabel(f"{var}", fontsize=14)
    plt.title(f'Distribution of {var} across {split}', fontsize=20)
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    # Move the legend outside the plot
    plt.legend(title=split, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    # Adjust layout to ensure everything fits
    plt.tight_layout()
    # Save the plot as a PNG file
    os.makedirs(outdir, exist_ok=True)
    var = var.replace(" ", "_")
    save_path = os.path.join(outdir, f"{var}_{split}_distribution.png")
    plt.savefig(save_path, bbox_inches="tight")  # Use bbox_inches="tight" to ensure the legend is included
    plt.close()


def check_column_ties(probabilities, class_labels):
    """
    Checks for column ties (multiple columns with the same maximum value) in each row.

    Parameters:
    - probabilities (numpy.ndarray): 2D array where rows represent samples and columns represent classes.
    - class_labels (list): List of class labels corresponding to the columns.

    Returns:
    - tie_rows (list): List of row indices where ties occur.
    - tie_columns (dict): Dictionary where the key is the row index and the value is a list of tied column indices and their class labels.
    """
    # Find the maximum probability for each row
    max_probs = np.max(probabilities, axis=1)

    # Check if there are ties (multiple columns with the maximum value in the row)
    ties = np.sum(probabilities == max_probs[:, None], axis=1) > 1

    # Get the indices of rows with ties
    tie_rows = np.where(ties)[0]

    # Find the columns where the tie occurs and associated class labels
    tie_columns = {}
    for row in tie_rows:
        tied_columns = np.where(probabilities[row] == max_probs[row])[0]
        tie_columns[row] = [(col, class_labels[col]) for col in tied_columns]
    
    return tie_rows, tie_columns

def classify_cells(query, ref_keys, cutoff, probabilities, tree):
    class_metrics = {}
    
    # Only use the first ref_key
    key = ref_keys[0]
    class_metrics[key] = {}

    # Extract the class labels and probabilities (DataFrame structure)
    class_labels = probabilities.columns.values  # Class labels are the column names
    class_probs = probabilities.values  # Probabilities as a numpy array
    
    predicted_classes = []
    
    if cutoff > 0:
        # Find the class with the maximum probability for each cell
        max_class_indices = np.argmax(class_probs, axis=1)  # Get the index of the max probability
        max_class_probs = np.max(class_probs, axis=1)  # Get the max probability
        
        # Set predicted classes to "unknown" if the max probability does not meet the threshold
        predicted_classes = [
            class_labels[i] if prob > cutoff else "unknown"
            for i, prob in zip(max_class_indices, max_class_probs)
        ]
    else:
        # Direct prediction without threshold filtering
        predicted_classes = class_labels[np.argmax(class_probs, axis=1)]
    
    # Store predictions and confidence in `query`
    query.obs["predicted_" + key] = predicted_classes
    query.obs["confidence"] = np.max(class_probs, axis=1)  # Store max probability as confidence
    
    # Aggregate predictions (you can keep this logic as needed)
    query = aggregate_preds(query, ref_keys, tree)
    
    return query


def aggregate_preds(query, ref_keys, tree):
    
    preds = np.array(query.obs["predicted_" + ref_keys[0]])
    query.obs.index = query.obs.index.astype(int)

    for higher_level_key in ref_keys[1:]: 
        query.obs["predicted_" + higher_level_key] = "unknown"  # Initialize to account for unknowns preds
        # Skip the first (granular) level
        ## Get all possible classes for this level (e.g. "GABAergic", "Glutamatergic", "Non-neuron")
        subclasses = get_subclasses(tree, higher_level_key) 
        
        for higher_class in subclasses: # eg "GABAergic"
            node = find_node(tree, higher_class) # find position in tree dict
            valid = get_subclasses(node, ref_keys[0]) # get all granular labels falling under this class
            ## eg all GABAergic subclasses
            if not valid:
                print("no valid subclasses")
                continue  # Skip if no subclasses found   

            # Get the indices of cells in `preds` that match any of the valid subclasses
            cells_to_agg = np.where(np.isin(preds, valid))[0]
            cells_to_agg = [int(cell) for cell in cells_to_agg] # Ensure cells_to_agg is in integers (if not already)

            # Assign the higher-level class label to the identified cells
            query.obs.loc[cells_to_agg, "predicted_" + higher_level_key] = higher_class

    return query

def eval(query, ref_keys, mapping_df):
    class_metrics = defaultdict(lambda: defaultdict(dict))
    for key in ref_keys:
        
       #threshold = kwargs.get('threshold', True)  # Or some other default value    
        query = map_valid_labels(query, ref_keys, mapping_df)  
        class_labels = query.obs[key].unique()
        pred_classes = query.obs[f"predicted_{key}"].unique()
        true_labels= query.obs[key]
        predicted_labels = query.obs["predicted_" + key]
        labels = list(set(class_labels).union(set(pred_classes)))

    # Calculate accuracy and confusion matrix after removing "unknown" labels
        accuracy = accuracy_score(true_labels, predicted_labels)
        conf_matrix = confusion_matrix(
            true_labels, predicted_labels, 
            labels=labels
        )
        class_metrics[key]["confusion"] = {
            "matrix": conf_matrix,
            "labels": labels
            #"accuracy": accuracy
        }
        # Classification report for predictions
        class_metrics[key]["classification_report"] = classification_report(true_labels, predicted_labels, 
                                        labels=labels,output_dict=True, zero_division=np.nan)

    return class_metrics

def update_classification_report(class_metrics, ref_keys):
    for query_name, query in class_metrics.items():
        for ref_name, ref in query.items():
            for key in ref_keys:
                for key, val in ref[key]["classification_report"].items():
                    if isinstance(val, dict):
                        if val['support'] == 0.0:
                            val['f1-score'] = np.nan 
    return class_metrics

def plot_confusion_matrix(query_name, ref_name, key, confusion_data, output_dir):

    new_query_name = query_name.replace(" ", "_").replace("/", "_").replace("(","").replace(")","")
    new_ref_name = ref_name.replace(" ", "_").replace("/", "_").replace("(","").replace(")","")
    output_dir = os.path.join(new_query_name, new_ref_name, output_dir)
    os.makedirs(output_dir, exist_ok=True) 
    # Extract confusion matrix and labels from the confusion data
    conf_matrix = confusion_data["matrix"]
    labels = confusion_data["labels"]

    # Plot the confusion matrix
    plt.figure(figsize=(28, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Reds', xticklabels=labels, yticklabels=labels, annot_kws={"size": 20})
    plt.xlabel('Predicted', fontsize =20)
    plt.ylabel('True', fontsize= 20)
    plt.title(f'Confusion Matrix: {query_name} vs {ref_name} - {key}', fontsize=17)
    # Adjust tick parameters for both axes
    #plt.tick_params(axis='both', which='major', labelsize=15, width=1)  # Increase tick label size and make ticks thicker

    # Rotate both x and y tick labels by 90 degrees
    plt.xticks(rotation=45, fontsize=15)  # Rotate x-axis labels by 90 degrees
    plt.yticks(rotation=45, fontsize=15)  # Rotate y-axis labels by 90 degrees

    # Save the plot
   # output_dir = os.path.join(projPath, 'results', 'confusion')
    
    #os.makedirs(os.path.join(output_dir, new_query_name, new_ref_name), exist_ok=True)  # Create the directory if it doesn't exist
    plt.savefig(os.path.join(output_dir,f"{key}_confusion.png"))
    plt.close() 

def plot_roc_curves(metrics, title="ROC Curves for All Keys and Classes", save_path=None):
    """
    Plots ROC curves for each class at each key level from the metrics dictionary on the same figure.
    
    Parameters:
    metrics (dict): A dictionary with structure metrics[key][class_label] = {tpr, fpr, auc, optimal_threshold}.
    title (str): The title of the plot.
    save_path (str, optional): The file path to save the plot. If None, the plot is not saved.
    """
    fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure and axis

    # Create a subplot for each key
    for key in metrics:
        for class_label in metrics[key]:

            if isinstance(metrics[key][class_label], dict):
                if all(k in metrics[key][class_label] for k in ["tpr", "fpr", "auc"]):
                    tpr = metrics[key][class_label]["tpr"]
                    fpr = metrics[key][class_label]["fpr"]
                    roc_auc = metrics[key][class_label]["auc"]

                    # Find the index of the optimal threshold
                    optimal_idx = np.argmax(tpr - fpr)
                    optimal_fpr = fpr[optimal_idx]
                    optimal_tpr = tpr[optimal_idx]

                    # Plot the ROC curve for the current class
                    #plt.plot(fpr, tpr, lw=2, label=f"Class {class_label} (AUC = {roc_auc:.3f})")
                 #   curve = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=class_label) #drop_intermediate=False)
                   # curve.plot(ax=ax)  # Add to the shared axis
                    
                    ax.step(fpr, tpr, where='post', lw=2, label=f"{key}: {class_label} (AUC = {roc_auc:.3f})")

                    # Plot the optimal threshold as a point
                    ax.scatter(optimal_fpr, optimal_tpr, color='red', marker='o') 
                          #  label=f"Optimal Threshold (Class {class_label})")
                    
    # Plot the reference line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label="Random Classifier")

    # Add title, labels, legend, and grid
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('False Positive Rate', fontsize = 15)
    ax.set_ylabel('True Positive Rate', fontsize = 15)
    ax.legend(loc='lower right', bbox_to_anchor=(1.05, 0), fontsize='medium', borderaxespad=0)
    ax.grid(True)

    # Adjust layout and save the plot if a path is provided
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    
def combine_f1_scores(class_metrics, ref_keys):
   # metrics = class_metrics
    # Dictionary to store DataFrames for each key
    all_f1_scores = {}
    #cutoff = class_metrics["cutoff"]
    # Iterate over each key in ref_keys
    for key in ref_keys:
        # Create a list to store F1 scores for each query-ref combo
        f1_data = [] 
        # Iterate over all query-ref combinations
        for query_name in class_metrics:
            for ref_name in class_metrics[query_name]:
                # Extract the classification report for the current query-ref-key combination
                classification_report = class_metrics[query_name][ref_name][key]["classification_report"]
                # Extract F1 scores for each label
                if classification_report:
                    for label, metrics in classification_report.items():
                        if label not in ["macro avg","micro avg","weighted avg","accuracy"]:
                         #   if isinstance(metrics, dict) and 'f1-score' in metrics:
                                f1_data.append({
                                    'query': query_name,
                                    'reference': ref_name,
                                    'label': label,
                                    'f1_score': metrics['f1-score'],                         
                                    'macro_f1': classification_report.get('macro avg', {}).get('f1-score', None),
                                    'micro_f1': classification_report.get('micro avg', {}).get('f1-score', None),
                                    'weighted_f1': classification_report.get('weighted avg', {}).get('f1-score', None), #,
                                    'precision': metrics['precision'],
                                    'recall': metrics['recall']        
                                })

        # Create DataFrame for the current key
        df = pd.DataFrame(f1_data)

        # Store the DataFrame in the dictionary for the current key
        all_f1_scores[key] = df

    return all_f1_scores

def plot_label_f1_heatmaps(all_f1_scores, threshold, outpath):
    """
    Plot heatmaps for label-level F1 scores for each query and save one figure per key.
    
    Parameters:
        all_f1_scores (dict): Dictionary with keys as reference names and values as DataFrames containing F1 scores.
        threshold (float): Threshold value to display in plot titles.
        outpath (str): Directory to save the generated heatmaps.
    """
    sns.set(style="whitegrid")
    os.makedirs(outpath, exist_ok=True)

    for key, df in all_f1_scores.items():
        # Get the unique queries for this key
        unique_queries = df['query'].unique()
        cols = 3  # Max 3 columns per figure
        rows = int(np.ceil(len(unique_queries) / cols))  # Determine rows based on queries

        # Create a new figure for this key
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 8, rows * 6))
        axes = axes.flatten()  # Flatten for easy indexing

        for idx, query in enumerate(unique_queries):
            query_df = df[df['query'] == query]

            # Pivot DataFrame to create the heatmap
            pivot_df = query_df.pivot_table(index='reference', columns='label', values='f1_score')
            mask = pivot_df.isnull()

            sns.heatmap(
                pivot_df,
                annot=True,
                cmap='YlOrRd',
                cbar_kws={'label': 'F1 Score'},
                mask=mask,
                ax=axes[idx],
                linewidths=0,
                annot_kws={"size": 6}
            )

            axes[idx].set_title(f'Query: {query}\nThreshold = {threshold:.2f}', fontsize=12)
            axes[idx].set_ylabel('Reference', fontsize=10)
            axes[idx].set_xlabel('Label', fontsize=10)
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=90, fontsize=8)
            axes[idx].set_yticklabels(axes[idx].get_yticklabels(), fontsize=8)

        # Remove any unused subplots
        for ax in axes[len(unique_queries):]:
            ax.remove()

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, f'f1_heatmaps_{key}_threshold_{threshold:.2f}.png'))
        plt.close()


def plot_aggregated_f1_heatmaps(all_f1_scores, threshold, outpath, ref_keys):
    """
    Plot heatmaps for weighted F1 scores, with one heatmap per level (key in ref_keys).
    
    Parameters:
        all_f1_scores (dict): Dictionary with keys as reference names and values as DataFrames containing F1 scores.
        threshold (float): Threshold value to display in plot titles.
        outpath (str): Directory to save the generated heatmaps.
        ref_keys (list): List of reference keys to ensure consistent ordering across levels.
    """
    sns.set(style="whitegrid")
    os.makedirs(outpath, exist_ok=True)

    # Combine all F1 scores into a single DataFrame
    final_f1_data = pd.DataFrame()
    for key, df in all_f1_scores.items():
        macro = df.drop(columns=['label', 'f1_score'])
        macro["key"] = key
        final_f1_data = pd.concat([final_f1_data, macro], ignore_index=True)

    # Filter for weighted F1 data
    weighted_f1_data = final_f1_data[['reference', 'key', 'query', 'weighted_f1']]

    # Iterate through each level (key in ref_keys)
    for level in ref_keys:
        # Filter data for the current level
        level_data = weighted_f1_data[weighted_f1_data['key'] == level]

        # Pivot the data for the current level
        pivot_f1 = level_data.pivot_table(
            index='reference',
            columns='query',
            values='weighted_f1'
        )

        # Plot the heatmap for the current level
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(
            pivot_f1,
            annot=True,
            cmap='YlOrRd',
            cbar_kws={'label': 'Weighted F1 Score'},
            fmt='.3f',
            annot_kws={"size": 10},
            ax=ax
        )

        # Customize plot appearance
        ax.set_xticklabels(pivot_f1.columns, rotation=45, ha="right", fontsize=15)
        ax.set_yticklabels(pivot_f1.index, fontsize=15)
        ax.set_title(f'Weighted F1 Score for Level: {level}\nThreshold = {threshold:.2f}', fontsize=20)
        ax.set_xlabel('Query', fontsize=15)
        ax.set_ylabel('Reference', fontsize=14)

        # Save the heatmap for the current level
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, f'weighted_f1_heatmap_level_{level}_threshold_{threshold:.2f}.png'))
        plt.close()


        
def get_test_data(census_version, test_name, subsample=10, 
                  organism="homo_sapiens", 
                  split_key="dataset_title"):
    census = cellxgene_census.open_soma(census_version=census_version)
    dataset_info = census.get("census_info").get("datasets").read().concat().to_pandas()
    brain_obs = cellxgene_census.get_obs(census, organism,
        value_filter=(
            "tissue_general == 'brain' and "
            "is_primary_data == True and "
            "tissue == 'frontal cortex' " # putting this in to speed things up for docker
        ))
    
    brain_obs = brain_obs.merge(dataset_info, on="dataset_id", suffixes=(None,"_y"))
    brain_obs.drop(columns=['soma_joinid_y'], inplace=True)
    # Filter based on organism
    test_obs = brain_obs[brain_obs[split_key].isin([test_name])]
    filtered_ids = list(test_obs["soma_joinid"])
    subsample_ids = random.sample(filtered_ids, subsample)#subsample_cells(brain_obs, filtered_ids, subsample=subsample)
    # Adjust organism naming for compatibility
    organism_name_mapping = {
        "homo_sapiens": "Homo sapiens",
        "mus_musculus": "Mus musculus"
    }
    organism = organism_name_mapping.get(organism, organism)
    cell_columns = [
        "assay", "cell_type", "tissue",
        "tissue_general", "suspension_type",
        "disease", "dataset_id", "development_stage",
        "soma_joinid"
    ]
    

    random.seed(1)
    test = cellxgene_census.get_anndata(
            census=census,
            organism=organism,
           # obs_value_filter= "development_stage"  
           # need to filter out fetal potentially?
            var_value_filter = "nnz > 50",
          #  obs_column_names=cell_columns,
            obs_coords=subsample_ids)
    test.obs= test.obs.merge(dataset_info,  on="dataset_id", suffixes=(None,"_y"))
    columns_to_drop = [col for col in test.obs.columns if col.endswith('_y')]
    test.obs.drop(columns=columns_to_drop, inplace=True)
    return test

def split_anndata_by_obs(adata, obs_key="dataset_title"):
    """
    Split an AnnData object into multiple AnnData objects based on unique values in an obs key.

    Parameters:
    - adata: AnnData object to split.
    - obs_key: Key in `adata.obs` on which to split the data.

    Returns:
    - A dictionary where keys are unique values in `obs_key` and values are corresponding AnnData subsets.
    """
    # Dictionary comprehension to create a separate AnnData for each unique value in obs_key
    split_data = {
        value: adata[adata.obs[obs_key] == value].copy() 
        for value in adata.obs[obs_key].unique()
    }
    
    return split_data


