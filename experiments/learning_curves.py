## A general script for running learning curves on the hpc. should be able to specify the test type, the seed and the
## number of data points. The script should fit each of the models with multiple inits for multiple random restarts and
# save the lml and rsme and nlpd for every restart. It should also have the option to plot the predictions of each
# model and save outside the repo.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.utils import build_models, train_models, get_final_models_dict, get_metrics, save_models, save_results, \
    get_gridpoints
from utils.plotting_utils import plot_lmls, plot_predictions, plot_lvmogp_latent_variables
import os
import pathlib as pl
import sys
import argparse

# mpl.style.use('models/mystyle.mplstyle')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# make sure tensorflow doesn't use too many cpus
tf.config.threading.set_intra_op_parallelism_threads(5)
tf.config.threading.set_inter_op_parallelism_threads(5)

# define variables

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='random  for creating data')
parser.add_argument('--n_new_points', type=int, default=10, help='number of points on new surfaces')
parser.add_argument('--data_seed', type=int, default=None, help='random function for creating dataset')
parser.add_argument('--relation_type', type=str, default='unrelated',
                    help='the correlation structure between surfaces')
args = parser.parse_args()
seed = args.seed
n_new_points = args.n_new_points
data_seed = args.data_seed
surface_type = args.relation_type

if surface_type == 'unrelated':
    from models.test_functions import TestFunUncorrelated as TestFun
elif surface_type == 'linear_relation':
    from models.test_functions import TestFunLinearCorrelation as TestFun
elif surface_type == 'non-linear_relation':
    from models.test_functions import TestFunNonLinearCorrelation as TestFun
else:
    raise ValueError('relation_type must be in unrelated, linear_relation, non-linear_relation')

set_seed = True

if set_seed:
    np.random.seed(seed)
    tf.random.set_seed(seed)
else:
    seed = None

plot_figs = False
train_inducing = True  # whether we train the inducing points or not

model_names = ['mo_indi', 'lvmogp', 'lmc', 'avg'] #
n_new_funs = 10
data_type_name = f'two_observed_{n_new_funs}_new'
n_fun = 2 + n_new_funs
total_train = 30 * 2 + 3 * n_new_points  # total number of training points
n_points = [30] * 2 + [n_new_points] * n_new_funs  # number of points on each surface

domain = [0, 2]
latent_dims = 2
latent_dims_lvmogp = 2
observed_dims = 1
max_points = 60
same_points = False

noise = 0.05
n_grid_points = 100
continuous_dims = ['x1', 'x2'][:observed_dims]
lengthscales_x = [0.3, 0.3]

plot_path = pl.Path.home() / f'Transfer_Learning_GP_Results/plots/'

# make test function

test_fun = TestFun(domain, seed, n_fun, observed_dims, latent_dims, max_points, noise, n_grid_points, same_points,
                   lengthscales_x, data_seed=data_seed)

if data_seed:
    np.random.seed(data_seed)

data_X, fun_nos, data_y = test_fun.create_data(n_points=n_points,
                                               random_idx=np.random.permutation(range(0, max_points)))

# have option to plot and save test function

if plot_figs:
    test_fun.plot_data_seperate_plots()
    plt.savefig(plot_path / f'data_seperate_{surface_type}_{data_type_name}_{n_new_points}_points_seed_{seed}_dseed_{data_seed}.png')
    plt.close()

# build and train the models, then for each model type select the one with the best lml to be the final model
models_dict = build_models(model_names, data_X, data_y, fun_nos, n_fun, observed_dims, latent_dims, latent_dims_lvmogp,
                           domain, n_restarts=3)

models_dict, lmls = train_models(models_dict)

final_models_dict = get_final_models_dict(models_dict)

if plot_figs:
    plot_predictions(final_models_dict, test_fun, domain, n_fun, observed_dims, data_y, n_grid_points=100)
    plot_lmls(lmls)
    if 'lvmogp' in model_names:
        plot_lvmogp_latent_variables(final_models_dict['lvmogp'], save=False, path=None, file_name=None)
    plt.show()

# get the NLPD and RMSE of the models
results_df = get_metrics(final_models_dict, test_fun, domain, n_fun, observed_dims, n_new_funs, n_grid_points=100)

print('got metrics')
path = pl.Path.home() / f'Transfer_Learning_GP_Results/'

# get test data and the true functions
x_new, _, _, _ = get_gridpoints(domain, n_fun, final_models_dict, observed_dims,
                                                             n_points=100)
ys_new = []
fs_new = []
for fun in test_fun.functions:
    f_new, _ = fun.predict_y(x_new.reshape(100, 1))
    y_new = test_fun.function_with_noise(fun, x_new.reshape(100, 1), test_fun.noise)
    ys_new.append(y_new)
    fs_new.append(f_new)

# save the model hyperparameters, training log marginal likelihoods and data models can be reconstructed. Also save the
# test data and the true functions
save_models(models_dict, lmls, data_X, data_y, fun_nos, x_new, ys_new, fs_new, path,
            file_name=f'hyperparameters/hyperparameters_{surface_type}_{data_type_name}_{n_new_points}_points_seed_{seed}_dataseed_{data_seed}.pkl')
print('saved models')

# save the NLPDs and RMSEs
save_results(results_df, path, seed, n_new_points, surface_type, n_new_funs, data_seed)

print('run complete')