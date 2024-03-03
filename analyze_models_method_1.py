import gc
import logging

import numpy as np
import pandas as pd
import torch

from archs import AE


def analyze_models_method_1(device,
                            external_layer_size: int,
                            iterator_ctrl,
                            iterator_not_ctrl
                            ):
    def iterate_model_set(device,
                          external_layer_size,
                          identity_matrix_tensor,
                          iterator
                          ):
        for model_name, bottleneck_size in iterator:
            logging.info(f'Now analyzing {model_name}')

            # initiate a model
            model = AE(external_layer_size, bottleneck_size).to(device)

            with torch.no_grad():
                model.load_state_dict(torch.load(f'model_{model_name}.pt'))
                model.eval()

                _, output = model(identity_matrix_tensor)

            results = pd.DataFrame(np.array(output.detach().cpu()))

            results.to_csv(f'analysis_result_method_1_{model_name}.csv', index=False)
            logging.info(f'Model {model_name} analysis (m. 1) result saved')

            # clear memory
            del model
            del results
            torch.cuda.empty_cache()
            gc.collect()

    # create testing data
    identity_matrix = np.eye(external_layer_size)
    identity_matrix_tensor = torch.tensor(identity_matrix).to(torch.float32).to(device)

    # analyze every model
    logging.info('Analysis by method #1 started')

    iterate_model_set(device,
                      external_layer_size,
                      identity_matrix_tensor,
                      iterator_ctrl
                      )
    logging.info('Control models analysed (m. 1)')

    iterate_model_set(device,
                      external_layer_size,
                      identity_matrix_tensor,
                      iterator_not_ctrl
                      )
    logging.info('Not control models analysed (m. 1)')

    logging.info('Analysis by method #1 finished')
