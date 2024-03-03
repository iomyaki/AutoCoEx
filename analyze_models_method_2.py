import gc
import logging

import pandas as pd
import torch

from archs import AE
from funcs import calculate_correlation_matrix


def analyze_models_method_2(device,
                            external_layer_size: int,
                            iterator_ctrl,
                            iterator_not_ctrl
                            ):
    def iterate_model_set(device,
                          external_layer_size,
                          iterator
                          ):
        for model_name, bottleneck_size in iterator:
            logging.info(f'Now analyzing {model_name}')

            # initiate a model
            model = AE(external_layer_size, bottleneck_size).to(device)

            with torch.no_grad():
                model.load_state_dict(torch.load(f'model_{model_name}.pt'))
                model.eval()

                for param_name, param in model.named_parameters():
                    if param_name == 'decoder.0.weight':
                        weight_matrix = param.detach().cpu().numpy()
                        break

            # calculate correlation matrix
            corr_matrix = pd.DataFrame(calculate_correlation_matrix(weight_matrix, external_layer_size))

            corr_matrix.to_csv(f'analysis_result_method_2_{model_name}.csv', index=False)
            logging.info(f'Model {model_name} analysis (m. 2) result saved')

            # clear memory
            del model
            del weight_matrix
            del corr_matrix
            torch.cuda.empty_cache()
            gc.collect()

    # analyze every model
    logging.info('Analysis by method #2 started')

    iterate_model_set(device,
                      external_layer_size,
                      iterator_ctrl
                      )
    logging.info('Control models analysed (m. 2)')

    iterate_model_set(device,
                      external_layer_size,
                      iterator_not_ctrl
                      )
    logging.info('Not control models analysed (m. 2)')

    logging.info('Analysis by method #2 finished')
