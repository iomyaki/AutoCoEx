import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from statsmodels.stats.multitest import fdrcorrection


def extension_loader(file):
    extension = file.split('.')[-1].lower()

    if extension == 'csv':
        df = pd.read_csv(file, index_col=0).T
    elif extension == 'tsv':
        df = pd.read_csv(file, index_col=0, sep='\t').T
    elif extension == 'xlsx':
        df = pd.read_excel(file, index_col=0).T
    else:
        print('Incorrect input format')
        return None

    return df


def add_noise(batch, noise_factor=0.25):
    noisy_data = batch + torch.randn_like(batch) * noise_factor
    return noisy_data


def get_percentile(array, threshold=5):
    """
    must receive Pandas DataFrame converted to a NumPy array, and scaled;
    default percentile is 5th
    """

    df_scaled = pd.DataFrame(array)
    series = df_scaled.to_numpy().reshape(1, len(df_scaled.columns) * len(df_scaled.index))

    wo_zeros = np.delete(series, np.where(series == 0))
    wo_borders = np.delete(wo_zeros, np.where(wo_zeros == 1))

    percentile = np.percentile(wo_borders, threshold)

    return percentile


def get_p_values(train_in, train_out, test_in, test_out, arr_length):
    mw_p = []

    for i in range(arr_length):
        # perform the Mann–Whitney test for each difference vector
        mw_p.append(
            stats.mannwhitneyu(
                np.subtract(train_out[:, i], train_in[:, i]),
                np.subtract(test_out[:, i], test_in[:, i])
            ).pvalue
        )

    # multiple comparisons
    fdr_p = fdrcorrection(mw_p)[1]

    return fdr_p


def ensembl2symbol(file, mode='forward'):
    id2symbol = {}
    with open(file) as fin:
        for line in fin:
            if '#' not in line and 'gene_id "' in line and 'gene_name "' in line:
                ens_id = line.split('gene_id "')[1].split('"')[0]
                symbol = line.split('gene_name "')[1].split('"')[0]

                if mode == 'forward':
                    id2symbol[ens_id] = symbol
                else:
                    id2symbol[symbol] = ens_id

    exceptions = [['ENSG00000011638', 'TMEM159'], []]

    return id2symbol
