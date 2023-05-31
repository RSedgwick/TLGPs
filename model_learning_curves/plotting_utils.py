import pickle
import numpy as np
from models.utils import mo_indi_init, lmc_init, avg_init, lvmogp_init, get_gridpoints, plot_model_predictions
import gpflow
import matplotlib.pyplot as plt

full_width = 7.5
halfwidth = 3.5

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
    if model_name == 'lmc':
        gp = lmc_init(data_X, data_y, fun_nos, n_fun, observed_dims, lmc_rank)
        if any(hyp['.kernel.kernels[1].kappa'] < 1e-9):
            hyp['.kernel.kernels[1].kappa'] = hyp['.kernel.kernels[1].kappa'] + 1e-6
    if model_name == 'avg':
        gp = avg_init(data_X, data_y, fun_nos, observed_dims)
    if model_name == 'lvmogp':
        latent_dims = 2
        gp = lvmogp_init(data_X, data_y, fun_nos, lengthscales=[1] * (observed_dims + latent_dims))
    print('build complete')
    gpflow.utilities.multiple_assign(gp, hyp)
    print('assignment complete')

    return gp

def unpack_series(best_row):
    """Unpack the Series object into the relevant variables"""
    data_X = best_row['data_X']
    data_y = best_row['data_y']
    fun_nos = best_row['fun_nos']
    observed_dims = data_X.shape[1]
    x_new = best_row['x_new']
    fs_new = best_row['f_news']
    ys_new = best_row['y_news']
    n_grid_points = len(best_row['x_new'])
    hyp = best_row['hyperparameters']
    n_fun = len(np.unique(fun_nos))
    return data_X, data_y, fun_nos, observed_dims, x_new, fs_new, ys_new, n_grid_points, hyp, n_fun

def plot_model_from_hyperparameters(model, mod_df, subfig, plot_new_ys=True):
    """Plot the model from the hyperparameters and data in the Series object
    :param model_series: pandas Series object containing the hyperparameters and data"""


    best_row = mod_df[mod_df['final_lml'] == np.max(mod_df['final_lml'])].iloc[0]
    model_name = best_row['model']
    data_X, data_y, fun_nos, observed_dims, x_new, fs_new, ys_new, n_grid_points, hyp, n_fun = unpack_series(best_row)

    x_new, fun_nos_new, x_new_lmc, x_new_lvmogp = get_gridpoints([-1, 1], n_fun, {model_name: model}, observed_dims,
                                                                 x_new=x_new, n_points=100)
    if model_name == 'avg':
        pred_mu, pred_var = model.predict_y(x_new)
    elif model_name == 'lvmogp':
        pred_mu, pred_var = model.predict_y(x_new_lvmogp)
    else:
        pred_mu, pred_var = model.predict_y(x_new_lmc)

    if not plot_new_ys:
        ys_new = None

    subfig = plot_model_predictions(model_name, n_fun, data_X, data_y, fun_nos, x_new, pred_mu, pred_var, fs_new, ys_new,
                                 subfig)
    return subfig

def plot_all_models_from_hyperparameters(model_dict, hyperparams, plot_new_ys=True, save_fig=False, save_path=None):
    """Plot all the models from the hyperparameters in the dictionary"""

    n_fun = len(np.unique(hyperparams['fun_nos'][0]))

    fig, axs = plt.subplots(nrows=len(model_dict), ncols=1, figsize=(20, 12))

    for ax in axs:
        ax.remove()

    gridspec = axs[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]

    row=0
    for model_name, model in model_dict.items():
        mod_df = hyperparams[hyperparams['model'] == model_name]

        subfig = subfigs[row]
        subfig.suptitle(f'{model_name}')
        subfig = plot_model_from_hyperparameters(model, mod_df, subfig, plot_new_ys=plot_new_ys)
        row += 1

    plt.tight_layout()

    return fig
