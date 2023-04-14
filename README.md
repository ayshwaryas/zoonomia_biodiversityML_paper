# zoonomia_biodiversityML_paper

## Ayshwarya Subramanian and Anish Mudide

# Non-linear prediction models (RF) for assessment of environmental conservation status and genome elements

## Notebooks

### Preprocess-Window170.ipynb

Input: Raw data files for het, roh, snpphylop, miscons and miscount (179 species total before filtering). 

Description: \
-- Filters out species with more than 30,000 NaNs (out of ~57,000 total features). We decided on this threshold after creating a histogram that showed the number of NaNs for each of the feature matrices. \
-- Normalizes the miscons and miscount matrixes w.r.t. the total number of coding variants per genome. \
-- Adds IUCN labels to matrices. \
-- Splits each matrix into two output matrices: one containing all DD species, and the other with all LC/NT/VU/EN/CR species.

### Preprocess-Window200.ipynb

Input: Raw data files for het, roh, snpphylop, miscons and miscount (207 or 240 species total before filtering). 

Description: See 'Preprocess-Window170'.

### Preprocess-GenomeSummary.ipynb

Input: Raw data file for genome summary statistics (241 species total before filtering). 

Description: \
-- Subsets to 13 features of interest. \
-- Removes Homo sapiens species. \
-- Adds IUCN labels. \
-- Splits the subsetted matrix into two output matrices: one containing all DD species, and the other with all LC/NT/VU/EN/CR species.

### Preprocess-Eco.ipynb

Input: Raw data file for ecological features (241 species total before filtering). 

Description: \
-- Subsets to 42 features of interest. \
-- Removes species with more than 31 NaNs (out of a total of 42 features). We decided on this threshold after creating a histogram that showed the number of NaNs for each of the feature matrix. \
-- Adds IUCN labels.\
-- Splits each matrix into two output matrices: one containing all DD species, and the other with all LC/NT/VU/EN/CR species.


## Scripts

### train-median.py

Task: train a binary classification algorithm using random forests (LC = 0, NT/VU/EN/CR = 1).

Input: Seed (e.g. 1), performance metric (e.g. roc), dataset (e.g. het), cv folds (e.g. 5).

Description: \
-- Splits into train (75%) and test (25%) sets according to the seed value. \
-- Removes features with too many NaNs within the train set. \
-- Performs k-fold cross validation on the training set to optimize hyperparameters (e.g. number of features selected, number of trees in forest etc.). \
-- Imputes missing values using the median of existing values. Reports test set performance of the model architecture chosen by CV. Saves model.

V2: Added compatibility for smaller datasets (e.g. eco or summary-eco)

V3: Saves txt file describing results.

V4: Excludes feature selection (used for datasets which combine selected window metrics with genomic summary and/or ecological variables).

### train-phylo.py

Analogous to 'train-median'; performs phylogenetic imputation instead of median-based imputation.

