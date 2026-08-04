import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.stats.mstats import gmean
import anndata as ad
import numpy as np
import scanpy as sc
import episcanpy.api as epi

def TFIDF(count_mat):
    row_sums = np.array(count_mat.sum(axis=1)).flatten()
    tf = count_mat.multiply(1.0 / row_sums[:, None])
    peak_counts = (count_mat > 0).sum(axis=0).A1
    idf = np.log(1 + count_mat.shape[0] / peak_counts)
    multiply_title = sp.diags(idf)
    tfidf_mat = tf.dot(multiply_title)
    return tfidf_mat, row_sums, multiply_title


def RNA_data_preprocessing(
        RNA_data,
        normalize_total=True,
        log1p=True,
        use_hvg=True,
        n_top_genes=3000
):
    RNA_data_processed = RNA_data.copy()
    RNA_data_processed.var_names_make_unique()
    if normalize_total:
        sc.pp.normalize_total(RNA_data_processed)
    if log1p:
        sc.pp.log1p(RNA_data_processed)
    if use_hvg:
        sc.pp.highly_variable_genes(RNA_data_processed, n_top_genes=n_top_genes,flavor="seurat_v3",)
        RNA_data_processed = RNA_data_processed[:, RNA_data_processed.var["highly_variable"]]
    return RNA_data_processed


def ATAC_data_preprocessing(
        ATAC_data,
        binary_data=True,
        filter_features=True,
        fpeaks=0.005,
        tfidf=True,
        normalize=True
):
    ATAC_data_processed = ATAC_data.copy()
    divide_title, multiply_title, max_temp = None, None, None
    if binary_data:
        epi.pp.binarize(ATAC_data_processed)
    if filter_features:
        epi.pp.filter_features(ATAC_data_processed, min_cells=np.ceil(fpeaks * ATAC_data.shape[0]))
    if tfidf:
        count_mat = ATAC_data_processed.X.copy()
        ATAC_data_processed.X, divide_title, multiply_title = TFIDF(count_mat)
    if normalize:
        max_temp = np.max(ATAC_data_processed.X)
        ATAC_data_processed.X = ATAC_data_processed.X / max_temp
    return ATAC_data_processed, divide_title, multiply_title, max_temp


def ADT_data_preprocessing(ADT_data):
    ADT_matrix = ADT_data.X.todense()
    gmean_list = []
    for i in range(ADT_matrix.shape[0]):
        temp = []
        for j in range(ADT_matrix.shape[1]):
            if not ADT_matrix[i, j] == 0:
                temp.append(ADT_matrix[i, j])
        gmean_temp = gmean(temp)
        gmean_list.append(gmean_temp)
        for j in range(ADT_matrix.shape[1]):
            if not ADT_matrix[i, j] == 0:
                ADT_matrix[i, j] = np.log(ADT_matrix[i, j] / gmean_temp)
    ADT_data_processed = ad.AnnData(csr_matrix(ADT_matrix), obs=ADT_data.obs, var=ADT_data.var)
    return ADT_data_processed, gmean_list
