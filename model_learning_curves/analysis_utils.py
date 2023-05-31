import pathlib as pl
import os
import pickle
import pandas as pd

def load_learning_curve_results():
    """Load the results of the learning curves for all the different models and return it as a dataframe
    :return:
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
    :return:
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