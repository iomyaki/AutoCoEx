import gc
import logging

import numpy as np
import pandas as pd
import torch

from archs import AE
from funcs import data_loader, custom_min_max_scaling, get_p_values


def find_anomalies(device,
                   control_file,
                   not_control_file,
                   name,
                   general_min,
                   general_max,
                   scaling='minmax',
                   cutoff=100,
                   n_models=2
                   ):
    # miscellaneous
    # torch.manual_seed(manual_seed)

    # load data
    control = data_loader(control_file)
    not_control = data_loader(not_control_file)
    external_layer_size = len(control.columns)
    parameter_names = control.columns.values.tolist()
    logging.info(f'Data loaded')

    # scale merged data
    control_scaled = custom_min_max_scaling(control, general_min, general_max)
    not_control_scaled = custom_min_max_scaling(not_control, general_min, general_max)
    logging.info(f'Data scaled')

    # send scaled data to the device
    control_scaled_tensor = torch.tensor(control_scaled).to(torch.float32).to(device)
    not_control_scaled_tensor = torch.tensor(not_control_scaled).to(torch.float32).to(device)
    logging.info(f'Scaled data sent to the device')

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
    pvalues['robustness_score'] = pvalues.apply(lambda row: (row < 0.05).sum() / n_models, axis=1)

    # range parameters based on robustness score and select the top
    pvalues_sorted = pvalues.sort_values(by=['robustness_score'], ascending=False)[:cutoff]
    pvalues_sorted_short = pvalues_sorted[['robustness_score']]

    # save the result
    pvalues.to_csv(f'fdr_p_values_{name}.csv')

    return pvalues_sorted_short, parameter_names
