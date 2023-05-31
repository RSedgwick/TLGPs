
import numpy as np
import pathlib as pl
import os
import pandas as pd
import warnings
warnings.simplefilter("ignore")
from copy import copy
import tensorflow as tf
from gpflow.utilities import read_values
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from plotting_utils import load_hyperparameters
# mpl.style.use('mystyle.mplstyle')

full_width = 7.5
halfwidth = 3.5


surface_type = 'linear_relation'
test_type = 'two_observed_10_new'
n_new_points = 1
seed = 3
dataseed = 4


path = pl.Path.home()/ \
       f'Transfer_Learning_GP_Results/hyperparameters/hyperparameters_{surface_type}_{test_type}_{n_new_points}_points_seed_{seed}_dataseed_{dataseed}.pkl'

hyperparams = load_hyperparameters(path)

print(hyperparams)

