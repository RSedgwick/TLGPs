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
    mod_names = {'mo_indi': 'MOGP', 'lmc': 'LMC', 'avg': 'AvgGP', 'lvmogp': 'LVMOGP'}

    fig, axs = plt.subplots(nrows=len(model_dict), ncols=1, figsize=(25, 12))

    for ax in axs:
        ax.remove()

    gridspec = axs[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]

    row=0
    for model_name, model in model_dict.items():
        mod_df = hyperparams[hyperparams['model'] == model_name]

        subfig = subfigs[row]
        subfig.suptitle(f'{mod_names[model_name]}')
        subfig = plot_model_from_hyperparameters(model, mod_df, subfig, plot_new_ys=plot_new_ys)
        row += 1

    return fig

def plot_learning_curve_results(results_df, seeds, mean=False):
    """Plot the RMSE and the NLPD against iteration from the results dataframe"""
    model_names = ['mo_indi', 'avg', 'lmc', 'lvmogp', ]
    colors = get_colors()

    surface_type_names = {'unrelated': 'Uncorrelated',
                          'linear_relation': 'Linearly\nCorrelated',
                          'non-linear_relation': 'Non-Linearly\nCorrelated'}

    mod_names = {'mo_indi': 'MOGP', 'lmc': 'LMC', 'avg': 'AvgGP', 'lvmogp': 'LVMOGP'}
    metric_names = {'rmse_new_surface': 'RMSE', 'nlpd_new_surface': 'NLPD'}

    cm = 1 / 2.54
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(17 * cm, 4))
    surface_types = ['unrelated', 'linear_relation', 'non-linear_relation']

    for i, surface_type in enumerate(surface_types):
        for j, metric in enumerate(['rmse_new_surface', 'nlpd_new_surface']):
            for k, model_name in enumerate(model_names):
                if mean:
                    temp_df_ = results_df[(results_df['surface_type'] == surface_type)
                                          & (results_df['model'] == model_name) & (results_df['seed'].isin(seeds))]
                    temp_df = temp_df_.groupby('n_new_points').mean().reset_index()
                    label = mod_names[model_name]
                    axs[j, i].plot(temp_df['n_new_points'], temp_df[metric], alpha=0.8, color=colors[model_name],
                                   linewidth=1, label=label)

                else:
                    for data_seed in results_df['data_seed'].unique():
                        temp_df = results_df[(results_df['surface_type'] == surface_type)
                                             & (results_df['model'] == model_name) & (results_df['seed'].isin(seeds))
                                             & (results_df['data_seed'] == data_seed)]
                        if data_seed == 1:
                            label = mod_names[model_name]
                        else:
                            label = None
                        axs[j, i].plot(temp_df['n_new_points'], temp_df[metric], alpha=0.8, color=colors[model_name],
                                       linewidth=1, label=label)

            axs[j, 0].set_ylabel(metric_names[metric])
        axs[0, i].set_title(surface_type_names[surface_type])
        axs[1, i].set_ylim(-2.5, 10)
        axs[0, i].set_ylim(0, 1)
    axs[1, 1].set_xlabel('number of points on new functions')
    axs[1, 1].legend(bbox_to_anchor=(0.5, -0.65), loc='lower center', ncol=4)

def get_colors():
    """Get the colors for the different models so they are the same in all plots"""
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {'lmc': cols[0], 'mo_indi': cols[3], 'lvmogp': cols[1], 'avg': cols[2]}
    return colors

def plot_log_marginal_likelihoods_of_restarts(hyp_df, seed, data_seed):
    """Plot the log marginal likelihoods of the different restarts of the hyperparameters"""

    surface_type_names = {'unrelated': 'Uncorrelated',
                          'linear_relation': 'Linearly\nCorrelated',
                          'non-linear_relation': 'Non-Linearly\nCorrelated'}

    surface_types = ['unrelated', 'linear_relation', 'non-linear_relation']

    mod_names = {'mo_indi': 'MOGP', 'lmc': 'LMC', 'avg': 'AvgGP', 'lvmogp': 'LVMOGP'}

    labels = {'mo_indi': {'random restart 1': 'random', 'random restart 2': 'random', 'random restart 3': 'random'},
              'lmc': {'random_W_and_k': 'both W and $kappa$ random',
                      'random_W_and_k_0': 'W random and $kappa=0$',
                      'random_W_and_k_1': 'W random and $kappa=1$'},
              'lvmogp': {
                  'random': 'random',
                  'gpy': 'gpy',
                  'mo_pca': 'PCA'},
              'avg': {'random restart 1': 'random', 'random restart 2': 'random', 'random restart 3': 'random'}}

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for model_name in ['mo_indi', 'avg', 'lmc', 'lvmogp']:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(17, 6.2))
        ax = axs.flatten()
        for k, surface_type in enumerate(surface_types):
            hyp_df_model = hyp_df[
                (hyp_df['model'] == model_name) & (hyp_df['seed'] == seed) & (hyp_df['data_seed'] == data_seed)
                & (hyp_df['surface_type'] == surface_type)]
            for i, init_name in enumerate(hyp_df_model['init_type'].unique()):
                for restart in hyp_df_model['restart'].unique():
                    df = hyp_df_model[(hyp_df_model['init_type'] == init_name)
                                      & (hyp_df_model['restart'] == restart)]
                    ax[k].plot(df['n_new_points'], df['final_lml'], color=colors[i],
                               label=labels[model_name][init_name])
            ax[k].set_title(surface_type_names[surface_type])
            ax[k].set_ylabel('log marginal likelihood')
            ax[k].set_xlabel('no. new points')
            if model_name == 'lvmogp':
                ax[k].set_ylim(-150, 500)
        plt.suptitle(mod_names[model_name])
        plt.legend()
        plt.show()
