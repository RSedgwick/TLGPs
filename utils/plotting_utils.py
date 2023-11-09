import os

import numpy as np
from utils.utils import get_gridpoints, full_width, column_width
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pathlib as pl

path = pl.Path(os.getcwd())
mpl.style.use(path /'mystyle.mplstyle')

inch_conversion = 1/25.4
full_width = 5.5984252
page_height = 7.85
halfwidth = 2.645669


def unpack_series(best_row):
    """Unpack the Series object into the relevant variables
    :param best_row: pandas Series object containing the hyperparameters and data
    :return: data_X, data_y, fun_nos, observed_dims, x_new, fs_new, ys_new, n_grid_points, hyp, n_fun"""
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
    :param model: GPflow model object
    :param mod_df: pandas Series object containing the hyperparameters and data
    :param subfig: matplotlib subfigure object
    :return: subfig
    """
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

    subfig, axs = plot_model_predictions(model_name, n_fun, data_X, data_y, fun_nos, x_new, pred_mu, pred_var, fs_new, ys_new,
                                 subfig)
    return subfig, axs

def plot_all_models_from_hyperparameters(model_dict, hyperparams, plot_new_ys=True, save_fig=False, save_path=None):
    """Plot all the models from the hyperparameters in the dictionary
    :param model_dict: dictionary of GPflow model objects
    :param hyperparams: pandas dataframe containing the hyperparameters and data
    :return: figure"""

    n_fun = len(np.unique(hyperparams['fun_nos'][0]))
    mod_names = {'mo_indi': 'MOGP', 'lmc': 'LMC', 'avg': 'AvgGP', 'lvmogp': 'LVMOGP'}

    fig, axs = plt.subplots(nrows=len(model_dict), ncols=1, figsize=(266 * inch_conversion, 140 * inch_conversion),
                            sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.05)
    plot_new_ys = False

    for ax in axs:
        ax.remove()

    gridspec = axs[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]

    row = 0
    for model_name, model in model_dict.items():
        mod_df = hyperparams[hyperparams['model'] == model_name]

        subfig = subfigs[row]
        subfig, axs = plot_model_from_hyperparameters(model, mod_df, subfig, plot_new_ys=plot_new_ys)
        for ax in axs:
            #
            # ax.set_xticks([0, 1, 2])
            ax.xaxis.set_ticklabels([])
            ax.xaxis.set_major_locator(MultipleLocator(1))

            # For the minor ticks, use no labels; default NullFormatter.
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.grid(None)

        axs[0].set_ylabel('y', labelpad=0, rotation=0)

        labels = {'mo_indi': 'MOGP', 'avg': 'AvgGP', 'lmc': 'LMC', 'lvmogp': 'LVMOGP'}
        axs[0].text(-1.7, 0, f'{labels[model_name]}', fontsize=8, ha="center", va="center", rotation=90)

        for ax in axs[1:]:
            ax.yaxis.set_ticklabels([])

            if row == 3:
                for ax in axs:
                    ax.set_xticks([0, 1, 2])
                    ax.xaxis.set_major_formatter('{x:.0f}')
        row += 1

        subfig.subplots_adjust(wspace=0.1)

    fig.subplots_adjust(hspace=0, bottom=0)
    subfigs[3].text(0.51, -0.15, f'x', fontsize=8, ha="center", va="center", rotation=0)

    if save_fig:
        plt.savefig(f'{save_path}.svg', bbox_inches='tight')
        plt.savefig(f'{save_path}.png', dpi=500, bbox_inches='tight')
    return fig

def plot_learning_curve_results(results_df, seeds, mean=False, save=False, path=None, file_name=None,
                                potrait_orientation=False):
    """Plot the RMSE and the NLPD against iteration from the results dataframe
    :param results_df: pandas dataframe containing the results
    :param seeds: list of seeds
    :param mean: boolean, if True, plot the mean and standard deviation of the results"""

    model_names = ['mo_indi', 'avg', 'lmc', 'lvmogp', ]
    colors = get_colors()
    linestyles = {'lmc': 'dashdot', 'mo_indi': 'dotted', 'lvmogp': 'solid', 'avg': 'dashed'}

    surface_type_names = {'unrelated': 'uncorrelated',
                          'linear_relation': 'linearly\ncorrelated',
                          'non-linear_relation': 'non-linearly\ncorrelated'}

    mod_names = {'mo_indi': 'MOGP', 'lmc': 'LMC', 'avg': 'AvgGP', 'lvmogp': 'LVMOGP'}
    metric_names = {'rmse_new_surface': 'RMSE', 'nlpd_new_surface': 'NLPD'}

    if potrait_orientation:
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(85 * inch_conversion, 114 * inch_conversion))
    else:
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(170 * inch_conversion, 4))

    surface_types = ['unrelated', 'linear_relation', 'non-linear_relation']

    for i, surface_type in enumerate(surface_types):
        for j, metric in enumerate(['rmse_new_surface', 'nlpd_new_surface']):
            if potrait_orientation:
                ax = axs[:, j]
            else:
                ax = axs[j, :]
            for k, model_name in enumerate(model_names):
                ax[i].set_xlim(0, 30)
                if mean:
                    temp_df_ = results_df[(results_df['surface_type'] == surface_type)
                                          & (results_df['model'] == model_name) & (results_df['seed'].isin(seeds))]
                    temp_df = temp_df_.groupby('n_new_points').mean().reset_index()
                    label = mod_names[model_name]
                    ax[i].plot(temp_df['n_new_points'], temp_df[metric], alpha=0.8, color=colors[model_name],
                               linewidth=2, label=label, linestyle=linestyles[model_name])

                else:
                    for data_seed in results_df['data_seed'].unique():
                        temp_df = results_df[(results_df['surface_type'] == surface_type)
                                             & (results_df['model'] == model_name) & (results_df['seed'].isin(seeds))
                                             & (results_df['data_seed'] == data_seed)]
                        if data_seed == 1:
                            label = mod_names[model_name]
                        else:
                            label = None
                        ax[i].plot(temp_df['n_new_points'], temp_df[metric], alpha=0.8, color=colors[model_name],
                                   linewidth=1, label=label, linestyle=linestyles[model_name])
            if potrait_orientation:
                ax[0].set_title(metric_names[metric])
            else:
                axs[j, 0].set_ylabel(metric_names[metric])
        if potrait_orientation:
            axs[i, 0].set_ylabel(surface_type_names[surface_type])
            axs[i, 1].set_ylim(-2.5, 5)
            axs[i, 1].set_yticks([-2.5, 0, 2.5, 5])
            axs[0, 0].set_ylim(0, 2)
            axs[1, 0].set_ylim(0, 1)
            axs[2, 0].set_ylim(0, 0.4)
        else:
            axs[0, i].set_title(surface_type_names[surface_type])
            axs[1, i].set_ylim(-2.5, 10)
            axs[0, i].set_ylim(0, 1)

    plt.subplots_adjust(wspace=0.3, bottom=0.08)
    fig.supxlabel('number of points on new functions', fontsize=8)
    if potrait_orientation:
        axs[2,1].legend(bbox_to_anchor=(-0.35, -0.58), loc='lower center', ncol=4, fontsize=8)
        file_name = f'{file_name}_potrait'
    else:
        axs[1, 1].legend(bbox_to_anchor=(0.5, -0.5), loc='lower center', ncol=4)

    if save:
        plt.savefig(path / f'{file_name}.svg', bbox_inches='tight')
        plt.savefig(path / f'{file_name}.png', dpi=500, bbox_inches='tight')


def get_colors():
    """Get the colors for the different models so they are the same in all plots"""
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {'lmc': cols[0], 'mo_indi': cols[3], 'lvmogp': cols[1], 'avg': cols[2]}
    return colors

def plot_log_marginal_likelihoods_of_restarts(hyp_df, seed, data_seed, save=False, path=None, file_name=None):
    """Plot the log marginal likelihoods of the different restarts of the hyperparameters
    :param hyp_df: pandas dataframe containing the hyperparameters and data
    :param seed: seed of the experiment
    :param data_seed: seed of the data"""

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
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(full_width, halfwidth))
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
        # plt.subplots(hspace=0.8)
        ax[2].legend(bbox_to_anchor=(0.5, -0.5), loc='lower center', ncol=len(hyp_df_model['init_type'].unique()))
        if save:
            plt.savefig(path / f'{file_name}_seed_{seed}_dataseed_{data_seed}_{model_name}.svg', bbox_inches='tight')
            plt.savefig(path / f'{file_name}_seed_{seed}_dataseed_{data_seed}_{model_name}.png', dpi=500, bbox_inches='tight')


def plot_lmls(LMLs, save=False, path=None, file_name=None):
    """plot the log marginal likelihoods from the training process
    :param LMLs: dictionary containing the log marginal likelihoods for each model"""
    fig, axs = plt.subplots(ncols=len(LMLs), figsize=(20, 5))
    i = 0
    labels = {'mo_indi': 'MOGP', 'avg': 'AvgGP', 'lmc': 'LMC', 'lvm': 'LVMOGP'}
    for model_name, model_dict in LMLs.items():

        for restart, lml_dict in model_dict.items():
            for init_type, lml in lml_dict.items():
                if lml is not None:
                    axs[i].plot(range(len(lml)), lml, label=f'{init_type}, restart {restart}')
                    axs[i].legend()
                    axs[i].set_title(labels[model_name])
                    axs[i].set_xlabel('iteration')
                    axs[i].set_ylabel('log marginal likelihood')
        i += 1
    plt.suptitle('Log Marginal Likelihoods')
    plt.tight_layout()
    if save:
        plt.savefig(path / file_name, bbox_inches='tight')


def plot_predictions(final_models_dict, test_fun, domain, n_fun, observed_dims, n_grid_points=100, save=False,
                     path=None, file_name=None):
    """plot the predictions of each of the models
    :param final_models_dict: dictionary of the models
    :param test_fun: the test functions object
    :param domain: domain of the data
    :param n_fun: the total number of functions
    :param observed_dims: the dimensions of the observed space
    :param n_grid_points: the number of grid points to use for the RMSE and NLPD"""

    x_new, fun_nos, x_new_lmc, x_new_lvmogp = get_gridpoints(domain, n_fun, final_models_dict, observed_dims,
                                                             n_points=n_grid_points)

    model_x_news = {'avg': x_new, 'mo_indi': x_new_lmc, 'lmc': x_new_lmc, 'lvmogp': x_new_lvmogp}

    # get the true data values at the grid points
    ys_new = []
    fs_new = []
    for fun in test_fun.functions:
        f_new, _ = fun.predict_y(x_new.reshape(100, 1))
        y_new = test_fun.function_with_noise(fun, x_new.reshape(100, 1), test_fun.noise)
        ys_new.append(y_new)
        fs_new.append(f_new)

    labels = {'mo_indi': 'MOGP', 'avg': 'AvgGP', 'lmc': 'LMC', 'lvm': 'LVMOGP'}

    for model_name, model in final_models_dict.items():
        x = model_x_news[model_name]
        pred_mu, pred_var = model.predict_y(x)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(full_width, page_height))

        for ax in axs:
            ax.remove()

        gridspec = axs[0].get_subplotspec().get_gridspec()
        subfigs = [fig.add_subfigure(gs) for gs in gridspec]

        plot_model_predictions(model_name, n_fun, test_fun.X, test_fun.y, test_fun.fun_no, x_new, pred_mu, pred_var,
                               fs_new, ys_new, subfigs[0])

        plt.suptitle(f'{labels[model_name]} Predictions')

        fig.set_figwidth(full_width)
        fig.set_figheight(page_height)
        plt.tight_layout()

        if save:
            plt.savefig(path / f'{model_name}_{file_name}.svg', bbox_inches='tight')
            plt.savefig(path / f'{model_name}_{file_name}.png', dpi=500, bbox_inches='tight')



def plot_model_predictions(model_name, n_fun, data_X, data_y, fun_nos, x_new, pred_mu, pred_var, fs_new, ys_new, subfig):
    """plot predictions of a given model for each function
    :param model_name: the name of the model
    :param n_fun: the total number of functions
    :param data_X: the data inputs
    :param data_y: the data values
    :param fun_nos: the function numbers
    :param x_new: the new inputs
    :param pred_mu: the predicted mean
    :param pred_var: the predicted variance
    :param fs_new: the true function values
    :param ys_new: the true data values
    :param axs: the axes to plot on
    """

    axs = subfig.subplots(nrows=1, ncols=n_fun)

    for i, fun_no in enumerate(range(n_fun)):
        if model_name == 'avg':
            idx = range(len(x_new))
        else:
            idx = range(len(x_new) * i, len(x_new) * (i + 1))
        axs[i].plot(x_new, pred_mu.numpy()[idx].flatten(), linewidth=2)
        axs[i].fill_between(x_new.flatten(),
                            pred_mu.numpy()[idx].flatten() + np.sqrt(pred_var.numpy()[idx].flatten()),
                            pred_mu.numpy()[idx].flatten() - np.sqrt(pred_var.numpy()[idx].flatten()), alpha=0.2)

        axs[i].plot(x_new, fs_new[i].numpy(), linestyle=':', color='k', linewidth=1, zorder=4, alpha=0.3)
        if ys_new is not None:
            axs[i].scatter(x_new, ys_new[i].numpy(), s=5, color='k', alpha=0.5)
        idx_train = np.where([fun_nos == fun_no])[1]
        axs[i].scatter(data_X[idx_train], data_y[idx_train], s=20, zorder=5, alpha=0.8)
        axs[i].set_ylim(np.min(data_y) - 0.5, np.max(data_y) + 0.5)

        if model_name == 'mo_indi':
            axs[i].set_title(f'{fun_no}')


    return subfig, axs

def plot_data(hyperparams, surface_type, test_type, dataseed, seed, plot_data_points=True, save=False, save_path=None, file_name=None):
    """Make plot of the synthetic data and the functions it is drawn from
    :param hyperparams: the hyperparameters
    :param surface_type: the type of surface
    :param test_type: the type of test
    :param dataseed: the seed for the data
    :param seed: the seed for the experiment
    :param plot_data_points: whether to plot the data points
    :param save: whether to save the plot
    :param save_path: the path to save the plot to
    :param file_name: the name of the file to save
    """

    fun_nos = hyperparams['fun_nos'][0]
    n_fun = len(np.unique(fun_nos))
    x_new = hyperparams['x_new'][0]
    fs_new = hyperparams['f_news'][0]
    data_X = hyperparams['data_X'][0]
    data_y = hyperparams['data_y'][0]

    inch_conversion = 1 / 25.4

    surface_type_names = {'unrelated': 'Uncorrelated',
                          'linear_relation': 'Linearly Correlated',
                          'non-linear_relation': 'Non-Linearly Correlated'}

    fig, axs = plt.subplots(nrows=2, ncols=int(n_fun / 2), figsize=(80 * inch_conversion, 38 * inch_conversion),
                            sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.2, wspace=0.1, bottom=0.1, right=1, left=0.1)
    axs = axs.flatten()
    for i, fun_no in enumerate(range(n_fun)):
        axs[i].plot(x_new, fs_new[i].numpy(), color='k')
        if plot_data_points:
            idx_train = np.where([fun_nos == fun_no])[1]
            axs[i].scatter(data_X[idx_train], data_y[idx_train])
            data_str = '_datapoints'
        else:
            data_str = ''
        axs[i].set_ylim(np.min(data_y) - 0.5, np.max(data_y) + 0.5)

    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

    for ax in axs:
        #
        ax.set_xticks([0, 1, 2])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter('{x:.0f}')

        # For the minor ticks, use no labels; default NullFormatter.
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(None)

    axs[0].set_ylabel('y', rotation=0)
    axs[6].set_ylabel('y', rotation=0)

    plt.subplots_adjust(bottom=0.15)
    fig.supxlabel('x', fontsize=8)
    plt.suptitle(f'{surface_type_names[surface_type]}', fontsize=8)

    save_path = pl.Path().home() / 'TLGP_plots'
    file_name = f'data_generating_functions_{surface_type}_{test_type}_points_seed_{seed}_dataseed_{dataseed}'

    plt.savefig(save_path / f'{file_name}{data_str}.svg', bbox_inches='tight')
    plt.savefig(save_path / f'{file_name}{data_str}.png', dpi=500, bbox_inches='tight')

    return fig



def plot_lvmogp_latent_variables(lvmogp, save=False, path=None, file_name=None):
    """plot the latent variables of the LVMOGP model
    :param lvmogp: LVMOGP model
    :return: plot of latent variables"""
    from matplotlib.patches import Ellipse
    fig = plt.figure(figsize=(column_width, column_width))
    axsH = plt.gca()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 3
    for i, H_coord in enumerate(lvmogp.H_data_mean.numpy()):
        axsH.scatter(H_coord[0], H_coord[1], label=f'{i + 1}', color=colors[i])
        axsH.annotate(f'{i + 1}', (H_coord[0], H_coord[1]))
        axsH.set_title(f"LVMOGP Latent Space")
        axsH.set_xlabel(f'latent dimension 1')
        axsH.set_ylabel(f'latent dimension 2')
        circle1 = Ellipse((H_coord[0], H_coord[1]),
                          1.95 * np.sqrt(lvmogp.H_data_var.numpy()[i, 0]),
                          1.95 * np.sqrt(lvmogp.H_data_var.numpy()[i, 1])
                          , color=colors[i], alpha=0.3, zorder=0)
        axsH.add_patch(circle1)
    plt.tight_layout()

    if save:
        plt.savefig(path / f'{file_name}.svg', bbox_inches='tight')
        plt.savefig(path / f'{file_name}.png', dpi=500, bbox_inches='tight')
        plt.close()