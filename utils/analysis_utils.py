import pathlib as pl
import os
import pickle
import gpflow
import numpy as np
import pandas as pd
from utils.plotting_utils import unpack_series
from utils.utils import mo_indi_init, lmc_init, avg_init, lvmogp_init


def load_learning_curve_results():
    """Load the results of the learning curves for all the different models and return it as a dataframe
    """
    path = pl.Path.home() / 'Transfer_Learning_GP_Results'
    n_new_fun = 10
    surface_types = ['unrelated', 'linear_relation', 'non-linear_relation']
    results_list = []
    for n_new_points in range(2, 30):
        for surface_type in surface_types:
            for seed in [1, 2, 3]:
                for data_seed in range(6):
                    if os.path.exists(
                            path / f'results/results_{surface_type}_{n_new_points}_{n_new_fun}_{seed}_dataseed_{data_seed}.pkl'):
                        with open(
                                path / f'results/results_{surface_type}_{n_new_points}_{n_new_fun}_{seed}_dataseed_{data_seed}.pkl',
                                'rb') as f:
                            df = pickle.load(f)
                            df['n_new_points'] = n_new_points
                            df['seed'] = seed
                            df['data_seed'] = data_seed
                            df['surface_type'] = surface_type
                            results_list.append(df)
    results_df = pd.concat(results_list, ignore_index=True)
    return results_df

def load_all_hyperparameters():
    """Load the results of the learning curves for all the different models and return it as a dataframe
    """

    path = pl.Path.home() / 'Transfer_Learning_GP_Results/hyperparameters'
    n_new_fun = 10
    surface_types = ['unrelated', 'linear_relation', 'non-linear_relation']
    results_list = []
    test_type = f'two_observed_{n_new_fun}_new'
    for n_new_points in range(2, 30):
        for surface_type in surface_types:
            for seed in [1, 2, 3]:
                for data_seed in range(6):
                    if os.path.exists(
                            path / f'hyperparameters_{surface_type}_{test_type}_{n_new_points}_points_seed_{seed}_dataseed_{data_seed}.pkl'):
                        with open(
                                path / f'hyperparameters_{surface_type}_{test_type}_{n_new_points}_points_seed_{seed}_dataseed_{data_seed}.pkl',
                                'rb') as f:
                            df = pickle.load(f)
                            df['n_new_points'] = n_new_points
                            df['seed'] = seed
                            df['data_seed'] = data_seed
                            df['surface_type'] = surface_type
                            results_list.append(df)
    hyp_df = pd.concat(results_list, ignore_index=True)
    return hyp_df


def load_hyperparameters(path):
    """Load the hyperparameters from the saved pickle files"""
    with open(path, 'rb') as f:
        hyperparameters = pickle.load(f)
    return hyperparameters


def fit_model_from_hyperparameters(mod_df):
    """Fit a model from the hyperparameters and data in the Series object
    :param mod_df: pandas Series object containing the hyperparameters and data"""

    best_row = mod_df[mod_df['final_lml'] == np.max(mod_df['final_lml'])].iloc[0]
    model_name = best_row['model']
    lmc_rank = 2
    data_X, data_y, fun_nos, observed_dims, x_new, fs_new, ys_new, n_grid_points, hyp, n_fun = unpack_series(best_row)

    if model_name == 'mo_indi':
        gp = mo_indi_init(data_X, data_y, fun_nos, n_fun, observed_dims, lmc_rank)
    elif model_name == 'lmc':
        gp = lmc_init(data_X, data_y, fun_nos, n_fun, observed_dims, lmc_rank)
        if any(hyp['.kernel.kernels[1].kappa'] < 1e-9):
            hyp['.kernel.kernels[1].kappa'] = hyp['.kernel.kernels[1].kappa'] + 1e-6
    elif model_name == 'avg':
        gp = avg_init(data_X, data_y, fun_nos, observed_dims)
    else:
        latent_dims = 2
        gp = lvmogp_init(data_X, data_y, fun_nos, lengthscales=[1] * (observed_dims + latent_dims))
    print('build complete')
    gpflow.utilities.multiple_assign(gp, hyp)
    print('assignment complete')

    return gp