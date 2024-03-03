import logging
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.stats.multitest import fdrcorrection
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def data_loader(file: str) -> pd.DataFrame or None:
    extension = file.split('.')[-1].lower()

    if extension == 'csv' or extension == 'tsv':
        df = pd.read_csv(file, index_col=0).T
    elif extension == 'xlsx':
        df = pd.read_excel(file, index_col=0).T
    else:
        logging.error('Incorrect input format')
        return None

    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)

    return df


def get_std_vector(df: pd.DataFrame) -> list:
    std_vector = []

    for i in df:
        std_vector.append(df[i].std())

    return std_vector


def repeat_dataset(df: pd.DataFrame, multiplier: int) -> pd.DataFrame:
    return df.loc[df.index.repeat(multiplier)].reset_index(drop=True)


def add_noise(df: pd.DataFrame, std_vector: list, noise_factor: float) -> pd.DataFrame:
    shape = df.shape
    noise = np.random.normal([0] * shape[1], np.array(std_vector) * noise_factor, shape)
    noisy_df = df + noise

    return noisy_df


def concat_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df1, df2], ignore_index=True)


def scale_dataset(df: pd.DataFrame, scaling: str) -> np.ndarray:
    if scaling.lower() == 'log':
        return np.log((df + 1).to_numpy())
    elif scaling.lower() == 'standard':
        scaler = StandardScaler()
        return scaler.fit_transform(df.to_numpy())
    else:
        scaler = MinMaxScaler()
        return scaler.fit_transform(df.to_numpy())


def gpu_train_test_split(df: np.ndarray,
                         test_size: float,
                         batch_size: int,
                         device: torch.device) -> tuple:
    """
    Perform a train/test split and send data to the GPU
    """

    df_tensor = torch.tensor(df).to(torch.float32).to(device)
    len_df_tensor = len(df_tensor)
    split_num = int(len_df_tensor * test_size)
    train_data, test_data = torch.utils.data.random_split(df_tensor, [len_df_tensor - split_num, split_num])

    del df_tensor
    del len_df_tensor
    del split_num

    return (DataLoader(train_data, batch_size=batch_size, shuffle=True),
            DataLoader(test_data, batch_size=batch_size, shuffle=True))


def get_bottlenecks(external_layer_size: int) -> list:
    bottleneck_sizes = []

    size = external_layer_size
    while size > int(external_layer_size / 1000):
        size //= 2
        bottleneck_sizes.append(size)

    bottleneck_sizes.reverse()

    del size

    return bottleneck_sizes


def training_loop(device: torch.device,
                  model,
                  criterion,
                  optimizer,
                  epochs: int,
                  train_data,
                  test_data) -> tuple:
    train_loss, eval_loss = [], []

    for epoch in range(epochs):
        start_time = datetime.now()

        # train the model
        model.train()
        train_loss_epoch = []

        for batch in train_data:
            batch = batch.to(device)

            _, output = model(batch)

            # comparison — forward
            loss_train_value = criterion(output, batch)

            # change weights — backward
            optimizer.zero_grad()
            loss_train_value.backward()
            optimizer.step()

            train_loss_epoch.append(loss_train_value.detach().cpu().numpy())

        train_loss.append(np.mean(train_loss_epoch))

        # evaluate the model
        model.eval()
        eval_loss_epoch = []

        with torch.no_grad():
            for batch in test_data:
                batch = batch.to(device)

                _, output = model(batch)

                loss_eval_value = criterion(output, batch)

                eval_loss_epoch.append(loss_eval_value.detach().cpu().numpy())

        eval_loss.append(np.mean(eval_loss_epoch))

        end_time = datetime.now()
        time_spent = end_time - start_time

        logging.info(f'Epoch: {epoch + 1}/{epochs}, '
                     f'train_loss: {np.mean(train_loss_epoch):4f}, '
                     f'eval_loss: {np.mean(eval_loss_epoch):4f}, '
                     f't_spent: {time_spent}'
                     )

    return train_loss, eval_loss


def draw_loss_plots(model_name: str, train_loss: list, eval_loss: list) -> None:
    train_loss[0] = None  # because usually is too big and breaks the scale

    train_loss_chart = plt.plot(train_loss, color='red')
    eval_loss_chart = plt.plot(eval_loss, color='blue')

    plt.title(f'Training dynamics of {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.legend((train_loss_chart[0], eval_loss_chart[0]), ('training', 'evaluation'))
    plt.grid(True)

    plt.savefig(f'loss_of_{model_name}.png')
    plt.clf()


def find_saturation_point(x: list, y: list, external_layer_size: int) -> int:
    diff_vector = np.diff(y)
    for i in range(len(diff_vector)):
        if abs(diff_vector[i]) < (max(y) - min(y)) * 0.001:
            return x[i]
    return external_layer_size // 2


def get_p_values(train_in, train_out, test_in, test_out, arr_length):
    mw_p = []

    for i in range(arr_length):
        # perform the Mann–Whitney test for each difference vector
        mw_p.append(
            stats.mannwhitneyu(
                np.abs(np.subtract(train_out[:, i], train_in[:, i])),
                np.abs(np.subtract(test_out[:, i], test_in[:, i]))
            ).pvalue
        )

    # multiple comparisons
    fdr_p = fdrcorrection(mw_p)[1]

    return fdr_p


def calculate_correlation_matrix(weight_matrix, external_layer_size):
    corr_matrix = np.empty((external_layer_size, external_layer_size))

    for i in range(external_layer_size):
        corr_matrix[i][i] = 1.0
        for j in range(i + 1, external_layer_size):
            corr = np.corrcoef(weight_matrix[i], weight_matrix[j])[0][1]
            corr_matrix[i][j], corr_matrix[j][i] = corr, corr

    return corr_matrix
