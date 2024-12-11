library(CSCORE)
library(Seurat)
library(WGCNA)
library(flashClust)
library(dplyr)
library(ggplot2)
library(enrichR)
library(VennDiagram)

regions = c("PFC","ACC") 
n_cells = 10000 
cutHeight = 0.75

# Define a custom theme with global font size settings and white background
custom_theme <- theme_minimal() +
  theme(
    text = element_text(size = 18),  # Global text size for all text
    axis.text = element_text(size = 18),  # Axis text size
    axis.title = element_text(size = 18),  # Axis title size
    plot.title = element_text(size = 18, hjust = 0.5),  # Plot title size and alignment
    legend.text = element_text(size = 18),  # Legend text size
    legend.title = element_text(size = 18),  # Legend title size
    strip.text = element_text(size = 18),  # Facet labels size
    panel.background = element_rect(fill = "white"),  # Set panel background to white
    plot.background = element_rect(fill = "white"),  # Set plot background to white
    panel.grid = element_blank() # Remove grid lines (optional)
   # plot.margin = margin(10, 10, 10, 10)  # Optional: adjust margins around the plot
  )

# Set the custom theme globally
theme_set(custom_theme)

par(
  cex = 2,          # Global scaling of text (2x larger than default)
  cex.axis = 2,     # Axis labels font size (2x larger)
  cex.lab = 2,      # Axis titles font size (2x larger)
  cex.main = 3      # Main title font size (3x larger)
)

projPath="/space/grp/rschwartz/rschwartz/cpsc545_proj"

if (length(regions) == 1) {
outdir=file.path(projPath, paste0("CSCORE_results/subsample_",n_cells),regions)
} else {
  # Combine the regions into a single string separated by underscores
regions_combined <- paste(regions, collapse = "_")

# Create the directory path using the combined region name
outdir <- file.path(projPath, paste0("CSCORE_results/subsample_", n_cells), regions_combined)
}
# Create the directory if it doesn't already exist
if (!dir.exists(outdir)) {
  dir.create(outdir, recursive = TRUE)
}



basedir=paste0("/space/grp/rschwartz/rschwartz/cpsc545_proj/mapped_queries/velmeshev/whole_cortex/subsample_",n_cells)
# File paths
mtx_file <- file.path(basedir, paste("velmeshev_subsampled_", n_cells, "_mapped.mtx", sep = ""))
obs_file <- file.path(basedir, paste("velmeshev_subsampled_", n_cells, "_obs_mapped.tsv", sep = ""))
var_file <- file.path(basedir, paste("velmeshev_subsampled_", n_cells, "_var_mapped.tsv", sep = ""))
  
  #

# Read the count matrix
counts <- ReadMtx(mtx_file,cells=obs_file,features=var_file, skip.cell=1, skip.feature=1, feature.column=2)
velmeshev <- CreateSeuratObject(counts)
# Read the metadata
cell_metadata <- read.table(obs_file, header = TRUE, sep = "\t", row.names = 5)
gene_metadata <- read.table(var_file, header = TRUE, sep = "\t", row.names = 1)

# Ensure row and column names are set correctly
colnames(velmeshev) <- rownames(cell_metadata)

velmeshev@meta.data <- cell_metadata
velmeshev$RNA[[]]["feature_id"] <- rownames(gene_metadata)
DefaultAssay(velmeshev) <- 'RNA'

#velmeshev <- velmeshev %>% NormalizeData() %>% FindVariableFeatures() %>% ScaleData()
# Remove ribosomal genes and mitochondrial genes
ribosomal_genes <- grep("^RP|^RP", rownames(velmeshev), value = TRUE)
mitochondrial_genes <- grep("^MT-", rownames(velmeshev), value = TRUE)
genes_to_remove <- union(ribosomal_genes, mitochondrial_genes)
velmeshev <- velmeshev[!rownames(velmeshev) %in% genes_to_remove, ]

velmeshev <- velmeshev[,velmeshev$region %in% regions] 

# Normalize
velmeshev_IT = velmeshev[,velmeshev$predicted_rachel_class %in% 'L2/3-6 IT']
mean_exp = rowMeans(velmeshev_IT@assays$RNA$counts/velmeshev_IT$nCount_RNA)
genes_selected = names(sort.int(mean_exp, decreasing = T))[1:20000]

CSCORE_result <- CSCORE(velmeshev_IT, genes = genes_selected)
CSCORE_coexp <- CSCORE_result$est
CSCORE_p <- CSCORE_result$p_value
p_matrix_BH = matrix(0, length(genes_selected), length(genes_selected))
p_matrix_BH[upper.tri(p_matrix_BH)] = p.adjust(CSCORE_p[upper.tri(CSCORE_p)], method = "BH")
p_matrix_BH <- p_matrix_BH + t(p_matrix_BH)
# Set your desired p-value threshold (e.g., 0.05)
#p_threshold <- 0.05
# Create a mask for significant gene pairs (where p-value is less than the threshold)
#significant_mask <- p_matrix_BH < p_threshold
# Set non-sgnificant gene correlations to 0 in CSCORE_coexp
##CSCORE_coexp[!significant_mask] <- 0
# You can now proceed with the rest of your analysis, using this filtered TOM

sft <- pickSoftThreshold.fromSimilarity(CSCORE_coexp)
power <- sft$power
adj = WGCNA::adjacency.fromSimilarity(CSCORE_coexp, power = power, type="signed")
TOM = WGCNA::TOMsimilarity(adj)
dissTOM = 1-TOM
rownames(dissTOM) <- colnames(dissTOM) <- genes_selected

geneTree = hclust(as.dist(dissTOM), method = "average") 
Modules = dynamicTreeCut::cutreeDynamic(dendro = geneTree, distM = dissTOM, deepSplit = 2,
pamRespectsDendro = FALSE, minClusterSize = 10)
ModuleColors <- labels2colors(Modules)

# Save the dendrogram plot
dendro_plot_path <- file.path(outdir, "gene_dendrogram_and_module_colors.png")
png(dendro_plot_path, width = 1000, height = 600)
plotDendroAndColors(geneTree, ModuleColors, "Module", dendroLabels = FALSE, 
hang = 0.03, addGuide = TRUE, guideHang = 0.05, main = "Gene dendrogram and module colors")
dev.off()

module_genes <- split(genes_selected, ModuleColors)  # Split genes by module
# Create the data frame with module counts
module_counts <- data.frame(
  Module = names(module_genes),
  Count = unlist(lapply(module_genes, length))
)
filename <- file.path(outdir, "genes_across_modules.tsv")
# Write the table to a TSV file
write.table(module_counts, file = filename, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)



# Aggregate expression by sample
seurat_subset <- subset(velmeshev_IT, features = genes_selected)
#pseudobulk <- AggregateExpression(object = seurat_subset, group.by = "sample", assays = "RNA", slot = "counts", return.seurat = TRUE)

#expression.data = t(as.matrix(pseudobulk$RNA$data))
#rownames(expression.data) <- sub("^g", "", rownames(expression.data))
seurat_subset <- seurat_subset %>% NormalizeData()
expression.data = t(as.matrix(seurat_subset@assays$RNA$data))

MElist <- moduleEigengenes(expression.data, colors = ModuleColors, impute=FALSE)
MEs <- MElist$eigengenes 
ME.dissimilarity = 1-cor(MElist$eigengenes, use="complete")
METree = hclust(as.dist(ME.dissimilarity), method = "average")
# Save the eigengene tree plot
eigengene_tree_plot_path <- file.path(outdir, "eigengene_tree.png")
png(eigengene_tree_plot_path, width = 1000, height = 600)
par(mar = c(0,4,2,0))
par(cex = 0.6)
plot(METree)

abline(h=cutHeight, col = "red") 
dev.off()


merge <- mergeCloseModules(exprData=expression.data, colors=ModuleColors, MEs=MEs, cutHeight = cutHeight, impute=FALSE, equalizeQuantiles	
=FALSE)
mergedColors = merge$colors
mergedMEs = merge$newMEs

merged_module_genes <- split(genes_selected, mergedColors)  # Split genes by module
# Create the data frame with module counts
merged_module_counts <- data.frame(
  Module = names(merged_module_genes),
  Count = unlist(lapply(merged_module_genes, length))
)
filename <- file.path(outdir, paste0("merged_genes_across_modules_",cutHeight,".tsv"))
write.table(merged_module_counts, file = filename, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)

# Save the merged modules dendrogram plot
merged_modules_plot_path <- file.path(outdir, "gene_dendrogram_and_merged_modules.png")
png(merged_modules_plot_path, width = 1000, height = 600)
plotDendroAndColors(geneTree, cbind(ModuleColors, mergedColors), c("Original Module", "Merged Module"), dendroLabels = FALSE, hang = 0.03, 
addGuide = TRUE, guideHang = 0.05, main = "Gene dendrogram and module colors for original and merged modules")
dev.off()

datTraits <- velmeshev_IT@meta.data %>% 
      dplyr::select(c("diagnosis","post.mortem.interval..hours.","age"))

datTraits$diagnosis <- ifelse(datTraits$diagnosis == "Control", 0, 1)
# Define numbers of genes and samples
nGenes = ncol(expression.data)
nSamples = nrow(expression.data)
module.trait.correlation = cor(mergedMEs, datTraits, use = "p") #p for pearson correlation coefficient 
module.trait.Pvalue = corPvalueStudent(module.trait.correlation, nSamples) #calculate the p-value associated with the correlation
textMatrix = paste(signif(module.trait.correlation, 2), "\n(", signif(module.trait.Pvalue, 1), ")", sep = "")
dim(textMatrix) = dim(module.trait.correlation)

# Save the module-trait relationships heatmap
heatmap_plot_path <- file.path(outdir, "module_trait_relationships_heatmap.png")
png(heatmap_plot_path, width = 1000, height = 600)
labeledHeatmap(Matrix = module.trait.correlation, xLabels = names(datTraits), yLabels = names(mergedMEs), ySymbols = names(mergedMEs), colorLabels = FALSE, colors = blueWhiteRed(50), textMatrix = textMatrix, setStdMargins = FALSE, cex.text = 0.4, zlim = c(-1,1), main = paste("Module-trait relationships"))
dev.off()


if (length(regions) > 1) {
traitData <- velmeshev_IT@meta.data %>% 
      dplyr::select(c("diagnosis","sex","individual","region","post.mortem.interval..hours.","age"))
} else {
  traitData <- velmeshev_IT@meta.data %>% 
      dplyr::select(c("diagnosis","sex","individual","post.mortem.interval..hours.","age"))
}

# Iterate over each module eigengene and perform ANOVA
anova_results <- lapply(colnames(mergedMEs), function(module) {
  module_eigengene <- mergedMEs[, module]
  aov_result <- aov(module_eigengene ~ ., data=traitData)
  summary(aov_result)
})

# Name the results by module
names(anova_results) <- colnames(mergedMEs)

# Extract F-values and p-values from ANOVA results
anova_summary <- do.call(rbind, lapply(names(anova_results), function(module) {
  summary_df <- as.data.frame(anova_results[[module]][[1]])
  summary_df$Predictor <- rownames(summary_df)
  summary_df$Module <- module
  summary_df
}))
colnames(anova_summary) <- c("SumSq", "Df", "MeanSq", "FValue", "PValue", "Predictor", "Module")
# Apply trimws() to all columns in the data frame
anova_summary[] <- lapply(anova_summary, function(x) if (is.character(x)) trimws(x) else x)
# Remove rows where Predictor is exactly "Residuals"
anova_summary <- anova_summary[!grepl("^Residuals$", anova_summary$Predictor), ]
anova_summary$FDR <- p.adjust(anova_summary$PValue, method = "fdr") 
anova_summary$SignificanceLabel <- ifelse(anova_summary$FDR < 0.05, "FDR < 0.05", "FDR > 0.05")
anova_summary$NegLogPValue <- -log10(anova_summary$FDR)
anova_plot_path <- file.path(outdir, "anova_significance_plot.png")


# Save the ANOVA F-values plot
anova_plot <- ggplot(anova_summary, aes(x = Predictor, y = NegLogPValue, fill = Module)) +
  geom_bar(
    stat = "identity",
    position = position_dodge(width = 0.8),
    aes(color = SignificanceLabel),
    size = 1.2
  ) +
  coord_flip() +
  labs(
    title = "ANOVA for WGCNA Module Eigengenes by Predictor",
    x = "Predictor",
    y = "-log10(p-value)",
    fill = "Module",
    color = "Significance"
  ) +
  scale_color_manual(values = c("FDR < 0.05" = "red", "FDR > 0.05" = "transparent")) +
  custom_theme +
  theme(legend.position = "right")


# Save the plot using ggsave
ggsave(anova_plot_path, plot = anova_plot, width = 20, height = 8)

library(clusterProfiler)
library(org.Hs.eg.db) 
library(biomaRt)# Replace with the appropriate organism db

# Connect to the Ensembl database for human gene information
ensembl <- useMart("ensembl", 
                   dataset = "hsapiens_gene_ensembl", 
                   host = "https://useast.ensembl.org")

# Map HGNC symbols to Entrez IDs
gene_entrez_mapping <- getBM(
  attributes = c("hgnc_symbol", "entrezgene_id"),
  filters = "hgnc_symbol",
  values = genes_selected,
  mart = ensembl
)

# Merge Entrez IDs with your selected genes
gene_entrez_ids <- merge(data.frame(SYMBOL = genes_selected), gene_entrez_mapping, by.x = "SYMBOL", by.y = "hgnc_symbol")

merged_module_genes <- split(genes_selected, mergedColors)  # Split genes by module

sig_modules= anova_summary %>% filter(Predictor == "diagnosis" & FDR < 0.05)
sig_modules$Module <- gsub("ME","",sig_modules$Module)
go_results <- list()
# Iterate over each module in your list of modules
for (module in sig_modules$Module) {
# Select the genes for the current module
    module_genes <- merged_module_genes[[module]]
    
    # Merge the module genes with the Entrez IDs
    module_entrez_ids <- merge(data.frame(SYMBOL = module_genes), gene_entrez_ids, by = "SYMBOL")
    
    # Perform GO enrichment analysis
    go_enrichment <- tryCatch({
        enrichGO(
            module_entrez_ids$entrezgene_id,
            OrgDb = org.Hs.eg.db,
            ont = "BP",  # Biological Process (can change to "MF" or "CC" for Molecular Function or Cellular Component)
            pAdjustMethod = "BH",
            pvalueCutoff  = 0.05,
            qvalueCutoff  = 0.05
        )
    }, error = function(e) {
        message(paste("GO enrichment failed for module", module, ": ", e$message))
        return(NULL)
    })
    
    # If enrichment is not null, store it in the list
    if (!is.null(go_enrichment)) {
        go_results[[module]] <- go_enrichment
        
        # Plot the GO enrichment results for the current module
        tryCatch({
            dotplot(go_enrichment) + ggtitle(paste("GO Enrichment for Module", module))
            # Optionally save the plot
            ggsave(file.path(outdir,paste0("GO_enrichment_module_", module, ".png")), width=15, height=8)
        }, error = function(e) {
            message(paste("Plotting failed for module", module, ": ", e$message))
        })
    }
}
