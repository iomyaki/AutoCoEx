import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from archs import AE
from funcs import extension_loader, get_p_values


def find_anomalies(train_input: str, test_input: str):
    # load data

    df_train = extension_loader(train_input)
    df_test = extension_loader(test_input)

    genes = df_train.columns.values.tolist()

    scaler = MinMaxScaler()

    df_train_scaled = scaler.fit_transform(df_train.to_numpy())
    df_test_scaled = scaler.fit_transform(df_test.to_numpy())

    # set parameters

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    bottleneck_size = 100
    external_layer_size = len(df_train.columns)

    model = AE(external_layer_size, bottleneck_size).to(device)

    # prepare data for the NN

    train_in = torch.tensor(df_train_scaled).to(torch.float32).to(device)
    test_in = torch.tensor(df_test_scaled).to(torch.float32).to(device)

    # load the model and get reconstructions

    with torch.no_grad():
        model.load_state_dict(torch.load('model.pt'))
        model.eval()

        _, train_out = model(train_in)
        _, test_out = model(test_in)

    torch.cuda.empty_cache()

    # get p-values

    fdr_p_val = get_p_values(
        df_train_scaled,
        np.array(train_out.detach().cpu()),
        df_test_scaled,
        np.array(test_out.detach().cpu()),
        external_layer_size
    )

    # present results

    df_fdr_p_val = pd.DataFrame(fdr_p_val, index=genes, columns=['FDR-corrected p-values'])
    df_fdr_p_val.to_excel('fdr_p_val.xlsx')

    print('List of genes with differential co-expression has been saved')
