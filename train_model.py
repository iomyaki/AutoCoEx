import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader
from archs import AE
from funcs import extension_loader, get_percentile
from training import training_loop


def train_model(train_input: str):
    # load data

    df = extension_loader(train_input)

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df.to_numpy())

    # set parameters

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    criterion = nn.MSELoss()
    torch.set_printoptions(precision=6)
    torch.manual_seed(3047)
    epochs = 20
    batch_size = 2 ** 3
    learning_rate = 1e-4
    bottleneck_size = 100
    multiplier = 500
    external_layer_size = len(df.columns)

    model = AE(external_layer_size, bottleneck_size).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-6)

    # prepare data

    data_tensor = torch.tensor(df_scaled).to(torch.float32).to(device)
    multiplied_data = torch.repeat_interleave(data_tensor, multiplier, 0)
    split_num = len(multiplied_data) // 5
    train_d, test_d = torch.utils.data.random_split(multiplied_data,
                                                    [len(multiplied_data) - split_num, split_num]
                                                    )
    train_data = DataLoader(train_d, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_d, batch_size=batch_size, shuffle=True)

    # training loop

    train_loss, eval_loss = training_loop(
        device,
        model,
        criterion,
        optimizer,
        epochs,
        train_data,
        test_data
    )

    train_loss[0] = None

    # plot loss function graphs

    train_loss_chart = plt.plot(train_loss, color='red')
    eval_loss_chart = plt.plot(eval_loss, color='blue')

    plt.title('Training dynamics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.legend((train_loss_chart[0], eval_loss_chart[0]), ('training', 'evaluation'))
    plt.grid(True)

    plt.savefig('loss_plot.png')

    # check the accuracy of the model

    torch.save(model.state_dict(), 'model.pt')

    with torch.no_grad():
        model.load_state_dict(torch.load('model.pt'))
        model.eval()

        _, output = model(data_tensor)

    torch.cuda.empty_cache()

    print(f'The model has been trained; '
          f'r2_score: {r2_score(df_scaled, np.array(output.detach().cpu()))}, '
          f'mse: {mean_squared_error(df_scaled, np.array(output.detach().cpu()))}, '
          f'5th percentile: {get_percentile(df_scaled)}'
          )
