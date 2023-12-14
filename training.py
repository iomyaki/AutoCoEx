import numpy as np
import torch
from datetime import datetime
from funcs import add_noise


def training_loop(device, model, criterion, optimizer, epochs, train_data, test_data):
    train_loss, eval_loss = [], []

    for epoch in range(epochs):
        start_time = datetime.now()

        # train the model
        model.train()
        train_loss_epoch = []

        for batch in train_data:
            initial_data = batch.to(device)
            noisy_data = add_noise(batch).to(device)

            _, output = model(noisy_data)

            # comparison — forward
            loss_train_value = criterion(output, initial_data)

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
                initial_data = batch.to(device)
                noisy_data = add_noise(batch).to(device)

                _, output = model(noisy_data)

                loss_eval_value = criterion(output, initial_data)

                eval_loss_epoch.append(loss_eval_value.detach().cpu().numpy())

        eval_loss.append(np.mean(eval_loss_epoch))

        end_time = datetime.now()
        # sec_spent = int((end_time - start_time).total_seconds())
        time_spent = end_time - start_time

        print(f'Epoch: {epoch + 1}/{epochs}, '
              f'train_loss: {np.mean(train_loss_epoch):4f}, '
              f'eval_loss: {np.mean(eval_loss_epoch):4f}, '
              f'τ_spent: {time_spent}'
              )

    return train_loss, eval_loss
