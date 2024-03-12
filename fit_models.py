import gc
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error

from archs import AE
from funcs import (
    data_loader,
    get_std_vector,
    repeat_dataset,
    add_noise,
    get_general_min_max,
    custom_min_max_scaling,
    gpu_train_test_split,
    get_bottlenecks,
    training_loop,
    draw_loss_plots,
    find_saturation_point
)


def fit_models(device,
               control_file,
               not_control_file,
               name,
               precision=6,
               multiplier=500,
               noise_factor=0.15,
               scaling='minmax',
               test_size=0.2,
               batch_size=8,
               lr=1e-4,
               weight_decay=1e-6,
               epochs=10,
               n_models=2
               ):
    def iterate_bottleneck_sizes(device,
                                 train_data,
                                 test_data,
                                 control_scaled,
                                 control_scaled_tensor,
                                 external_layer_size,
                                 name,
                                 bottleneck_sizes,
                                 scaling,
                                 lr,
                                 weight_decay,
                                 epochs,
                                 criterion,
                                 characteristics_type='prelim'
                                 ):
        characteristics = pd.DataFrame(columns=[
            'model_name',
            'bottleneck_size',
            'MSE',
            'R^2'
        ]
        )

        logging.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logging.info(f'Creating {characteristics_type} model set: {name}')
        logging.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        for bottleneck_size in bottleneck_sizes:
            logging.info('========================================================================================')
            logging.info(f'Current bottleneck size: {bottleneck_size}')

            # initialize a model
            model = AE(external_layer_size, bottleneck_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=lr,
                                         weight_decay=weight_decay
                                         )
            model_name = f'{name}_scaling_{scaling}_neck_{bottleneck_size}_{characteristics_type}'
            logging.info(f'Model {model_name} initialized')

            # train the model
            train_loss, eval_loss = training_loop(device,
                                                  model,
                                                  criterion,
                                                  optimizer,
                                                  epochs,
                                                  train_data,
                                                  test_data
                                                  )
            logging.info(f'Training of {model_name} complete')

            # plot loss function graphs
            draw_loss_plots(model_name, train_loss, eval_loss)

            # save the model
            torch.save(model.state_dict(), f'model_{model_name}.pt')

            # check and display the accuracy of the trained model
            with torch.no_grad():
                model.load_state_dict(torch.load(f'model_{model_name}.pt'))
                model.eval()

                _, output = model(control_scaled_tensor)

            mse = mean_squared_error(control_scaled, np.array(output.detach().cpu()))
            r_sq = r2_score(control_scaled, np.array(output.detach().cpu()), multioutput='variance_weighted')

            characteristics = pd.concat([characteristics,
                                         pd.DataFrame({'model_name': [f'{model_name}'],
                                                       'bottleneck_size': [bottleneck_size],
                                                       'MSE': [mse],
                                                       'R^2': [r_sq]
                                                       }
                                                      )
                                         ], ignore_index=True
                                        )

            logging.info(f'Model {model_name} trained and saved')
            logging.info(f'MSE of {model_name}: {mse}')
            logging.info(f'R^2 of {model_name}: {r_sq}')

            # clear memory
            del model
            del optimizer
            del model_name
            del train_loss
            del eval_loss
            plt.close()
            torch.cuda.empty_cache()
            gc.collect()

        # save parameters of  models
        characteristics.to_csv(f'characteristics_{characteristics_type}_{name}.csv', index=False)
        logging.info(f'Parameters of {characteristics_type} models saved')

        return characteristics

    # miscellaneous
    torch.set_printoptions(precision=precision)
    # torch.manual_seed(manual_seed)

    # load data
    control = data_loader(control_file)
    not_control = data_loader(not_control_file)
    external_layer_size = len(control.columns)
    logging.info(f'Data loaded')

    # calculate standard deviation for each parameter in control data
    std_vector = get_std_vector(control)
    logging.info(f'Std vector calculated')

    # multiply control data
    control_multiplied = repeat_dataset(control, multiplier)
    logging.info(f'Control multiplied')

    # add noise to the multiplied control data
    control_multiplied_noisy = add_noise(control_multiplied, std_vector, noise_factor)
    logging.info(f'Noise added to the multiplied control')

    # calculate vectors of min. and max. for noisy multiplied control data and (ordinary) not control data
    general_min, general_max = get_general_min_max(control_multiplied_noisy, not_control)
    logging.info(f'Vectors of min. and max. calculated')

    # scale noisy multiplied control data
    control_multiplied_noisy_scaled = custom_min_max_scaling(control_multiplied_noisy, general_min, general_max)
    logging.info(f'Noisy multiplied control data scaled')

    # perform a train/test split and send data to the GPU
    train_data, test_data = gpu_train_test_split(control_multiplied_noisy_scaled.to_numpy(),
                                                 test_size,
                                                 batch_size,
                                                 device
                                                 )
    logging.info(f'Train/test split performed')

    # generate the list of bottleneck sizes for preliminary models
    bottleneck_sizes_prelim = get_bottlenecks(external_layer_size)
    logging.info(f'Bottlenecks calculated')

    # generate models with different bottleneck sizes
    criterion = nn.MSELoss()

    control_scaled = custom_min_max_scaling(control, general_min, general_max)
    control_scaled_tensor = torch.tensor(control_scaled).to(torch.float32).to(device)

    preliminary_model_characteristics = iterate_bottleneck_sizes(device,
                                                                 train_data,
                                                                 test_data,
                                                                 control_scaled,
                                                                 control_scaled_tensor,
                                                                 external_layer_size,
                                                                 name,
                                                                 bottleneck_sizes_prelim,
                                                                 scaling,
                                                                 lr,
                                                                 weight_decay,
                                                                 epochs,
                                                                 criterion,
                                                                 'prelim'
                                                                 )

    # find the optimal bottleneck
    optimal_bottleneck_mse = find_saturation_point(bottleneck_sizes_prelim,
                                                   preliminary_model_characteristics['MSE'].tolist(),
                                                   'mse'
                                                   )
    optimal_bottleneck_r_sq = find_saturation_point(bottleneck_sizes_prelim,
                                                    preliminary_model_characteristics['R^2'].tolist(),
                                                    'r_sq'
                                                    )
    optimal_bottleneck = (optimal_bottleneck_mse + optimal_bottleneck_r_sq) // 2

    # generate the list of bottleneck sizes for predictive models
    bottleneck_sizes_predict = list(range(optimal_bottleneck,
                                          external_layer_size,
                                          (external_layer_size - optimal_bottleneck) // n_models)
                                    )
    bottleneck_sizes_predict = bottleneck_sizes_predict[:n_models]

    # generate predictive models
    iterate_bottleneck_sizes(device,
                             train_data,
                             test_data,
                             control_scaled,
                             control_scaled_tensor,
                             external_layer_size,
                             name,
                             bottleneck_sizes_predict,
                             scaling,
                             lr,
                             weight_decay,
                             epochs,
                             criterion,
                             'predict'
                             )

    return external_layer_size, general_min, general_max
