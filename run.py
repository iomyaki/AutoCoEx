import logging
import sys

import pandas as pd

from analyze_models_method_1 import analyze_models_method_1
from analyze_models_method_2 import analyze_models_method_2
from find_anomalies import find_anomalies
from fit_models import fit_models
from funcs import get_device


if __name__ == '__main__':
    control_file = sys.argv[1]
    not_control_file = sys.argv[2]
    name = sys.argv[3]

    # configure logging
    logging.basicConfig(format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                        datefmt='%d/%m/%Y %I:%M:%S %p',
                        level=logging.INFO,
                        filename=f'logs_{name}.log',
                        filemode='w'
                        )

    # activate CUDA
    device = get_device()
    logging.info(f'Device: {device}')

    # fit models and find anomalies
    external_layer_size, general_min, general_max = fit_models(device,
                                                               control_file,
                                                               not_control_file,
                                                               f'{name}_ctrl'
                                                               )
    pvalues_sorted_short, parameter_names = find_anomalies(device,
                                                           control_file,
                                                           not_control_file,
                                                           f'{name}_ctrl',
                                                           general_min,
                                                           general_max
                                                           )
    fit_models(device, not_control_file, control_file, f'{name}_not_ctrl')

    # create model iterators for further analysis
    characteristics_ctrl = pd.read_csv(f'characteristics_predict_{name}_ctrl.csv')
    characteristics_not_ctrl = pd.read_csv(f'characteristics_predict_{name}_not_ctrl.csv')

    iterator_ctrl = zip(characteristics_ctrl['model_name'].to_list(),
                        characteristics_ctrl['bottleneck_size'].to_list()
                        )

    iterator_not_ctrl = zip(characteristics_not_ctrl['model_name'].to_list(),
                            characteristics_not_ctrl['bottleneck_size'].to_list()
                            )

    # perform model analysis
    analyze_models_method_1(device,
                            external_layer_size,
                            iterator_ctrl,
                            iterator_not_ctrl,
                            pvalues_sorted_short,
                            parameter_names
                            )
    analyze_models_method_2(device, external_layer_size, iterator_ctrl, iterator_not_ctrl, parameter_names)
