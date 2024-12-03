projPath="/space/grp/rschwartz/rschwartz/single_cell_stuff"
source("/space/grp/rschwartz/rschwartz/single_cell_stuff/src/malat1_function.R")
source("/space/grp/rschwartz/rschwartz/single_cell_stuff/src/seurat_functions.r")
source("/space/grp/rschwartz/rschwartz/goldowitz/single_cell/scripts/sc_functions.R") 
library(reticulate)
use_condaenv("~/anaconda3/envs/r4.3/")
load_required_packages()
library(argparse)
library(gplots)
set.seed(123)
library(dplyr)
library(ggplot2)

files_to_convert <- list.files(file.path(projPath,"rds/queries"),full.names=TRUE)

for (file in files_to_convert) {
    prefix <- strsplit(file,"/")[[1]][10] %>% gsub("_raw.rds","",.)
    obj <- readRDS(file)
    obj[["RNA"]] <-  as(object = obj[["RNA"]], Class = "Assay")
    browser()
    SaveH5Seurat(obj, filename=file.path("/space/grp/rschwartz/rschwartz/census-stuff/h5ad",prefix),overwrite=TRUE)
    Convert(file.path("/space/grp/rschwartz/rschwartz/census-stuff/h5ad",paste0(prefix,".h5seurat")), dest = "h5ad")
}

