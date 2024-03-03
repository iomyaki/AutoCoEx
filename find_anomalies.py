import gc
import logging

import numpy as np
import pandas as pd
import torch

from archs import AE
from funcs import concat_datasets, data_loader, get_p_values, scale_dataset


def find_anomalies(device,
                   control_file,
                   not_control_file,
                   name,
                   scaling='minmax',
                   n_models=2
                   ):
    # miscellaneous
    # torch.manual_seed(manual_seed)

    # load data
    control = data_loader(control_file)
    not_control = data_loader(not_control_file)
    samples_initial = len(control)
    external_layer_size = len(control.columns)
    parameter_names = control.columns.values.tolist()
    logging.info(f'Data loaded')

    # concatenate control and not control data
    merged_data = concat_datasets(control, not_control)
    logging.info(f'Control concatenated with not control')

    # scale merged data
    merged_data_scaled = scale_dataset(merged_data, scaling)
    logging.info(f'Merged data scaled')

    # separate scaled merged data & send them to the device
    control_scaled = merged_data_scaled[:samples_initial]
    not_control_scaled = merged_data_scaled[samples_initial:]

    control_scaled_tensor = torch.tensor(control_scaled).to(torch.float32).to(device)
    not_control_scaled_tensor = torch.tensor(not_control_scaled).to(torch.float32).to(device)

    logging.info(f'Scaled merged data split')

    # load data of predictive models
    predictive_model_characteristics = pd.read_csv(f'characteristics_predict_{name}.csv')

    # for each model, find p-values
    pvalues = pd.DataFrame(index=parameter_names)

    iterator = zip(predictive_model_characteristics['model_name'].to_list(),
                   predictive_model_characteristics['bottleneck_size'].to_list()
                   )

    logging.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logging.info(f'Using predictive model set: {name}')
    logging.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for model_name, bottleneck_size in iterator:
        logging.info('========================================================================================')
        logging.info(f'Current model: {model_name}')

        # initialize a model
        model = AE(external_layer_size, bottleneck_size).to(device)
        logging.info('Model initialized')

        # send data to the model
        with torch.no_grad():
            model.load_state_dict(torch.load(f'model_{model_name}.pt'))
            model.eval()

            _, control_out = model(control_scaled_tensor)
            _, not_control_out = model(not_control_scaled_tensor)
        logging.info('Model has reconstructed its input')

        # calculate p-values for parameters
        fdr_p_val = get_p_values(control_scaled,
                                 np.array(control_out.detach().cpu()),
                                 not_control_scaled,
                                 np.array(not_control_out.detach().cpu()),
                                 external_layer_size
                                 )

        pvalues[f'neck {bottleneck_size}'] = fdr_p_val

        logging.info('p-values calculated')

        # clear memory
        del model
        del fdr_p_val
        torch.cuda.empty_cache()
        gc.collect()

    # calculate the proportion of models where p-value is significant for parameters
    pvalues['significant, % of models'] = pvalues.apply(lambda row: (row < 0.05).sum() / n_models, axis=1)

    # save the result
    pvalues.to_csv(f'fdr_p_values_{name}.csv')
