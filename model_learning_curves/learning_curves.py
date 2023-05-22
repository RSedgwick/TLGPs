## A general script for running learning curves on the hpc. should be able to specify the test type, the seed and the
## number of data points. The script should fit each of the models with multiple inits for multiple random restarts and
# save the lml and rsme and nlpd for every restart. It should also have the option to plot the predictions of each
# model and save outside the repo.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.utils import build_models, train_models, get_final_models_dict ,plot_lmls

import os
import pathlib as pl
from gpflow.utilities import read_values
import pickle
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# make sure tensorflow doesn't use too many cpus
tf.config.threading.set_intra_op_parallelism_threads(5)
tf.config.threading.set_inter_op_parallelism_threads(5)

# define variables
sys_args = [0, 0, 1, 2] #sys.argv[:]

n_new_points = int(sys_args[1])  # number of points on new surfaces
seed = int(sys_args[2])  # seed
relation_type = int(sys_args[3])  # type of relationship between surfaces

if relation_type == 1:
    surface_type = 'unrelated'
    from models.test_functions import TestFunUncorrelated as TestFun
elif relation_type == 2:
    surface_type = 'linear_relation'
    from models.test_functions import TestFunLinearCorrelation as TestFun
elif relation_type == 3:
    surface_type = 'non-linear_relation'
    from models.test_functions import TestFunNonLinearCorrelation as TestFun

else:
    raise ValueError('relation_type must be in {1, 2, 3}')

set_seed = True

if set_seed:
    np.random.seed(seed)
    tf.random.set_seed(seed)
else:
    seed = None

plot_figs = True
train_inducing = True  # whether we train the inducing points or not

model_names = ['lmc', 'mo_indi', 'lvmogp', 'avg']
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
                   lengthscales_x)

data_X, fun_nos, data_y = test_fun.create_data(n_points=n_points)

# have option to plot and save test function

if plot_figs:
    test_fun.plot_data_seperate_plots()
    plt.savefig(plot_path / f'data_seperate_{surface_type}_{data_type_name}_{n_new_points}_points_seed_{seed}.png')
    plt.close()

models_dict = build_models(model_names, data_X, data_y, fun_nos, n_fun, observed_dims, latent_dims, latent_dims_lvmogp,
                           domain, n_restarts=2)

models_dict, lmls = train_models(models_dict)

final_models_dict = get_final_models_dict(models_dict)

plot_lmls(lmls)

x = 0

# for each model: fit each initialisation for n restarts and save the lml and rmse and nlpd for each. also save the results
# for the best of each model
# have option to plot and save the predictions of each model, and save the hyperparameters
