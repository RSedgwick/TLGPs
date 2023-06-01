import numpy as np
from utils.utils import get_gridpoints, full_width, column_width
import matplotlib.pyplot as plt

full_width = 7.5
halfwidth = 3.5

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
    fig.subplots_adjust(bottom=0.2, top=0.8, hspace=1)
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
        subfig.subplots_adjust(wspace=0.3)
    if save_fig:
        plt.savefig(save_path, dpi=500)
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
    plt.subplots_adjust(wspace=0.3)
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


def plot_lmls(LMLs, save=False, path=None, file_name=None):
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
        plt.close()


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
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(full_width, full_width))

        for ax in axs:
            ax.remove()

        gridspec = axs[0].get_subplotspec().get_gridspec()
        subfigs = [fig.add_subfigure(gs) for gs in gridspec]

        plot_model_predictions(model_name, n_fun, test_fun.X, test_fun.y, test_fun.fun_no, x_new, pred_mu, pred_var,
                               fs_new, ys_new, subfigs[0])

        plt.suptitle(f'{labels[model_name]} Predictions')
        plt.tight_layout()

        if save:
            plt.savefig(path / f'{model_name}_{file_name}', bbox_inches='tight')
            plt.close()


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
        axs[i].plot(x_new, pred_mu.numpy()[idx].flatten())
        axs[i].fill_between(x_new.flatten(),
                            pred_mu.numpy()[idx].flatten() + np.sqrt(pred_var.numpy()[idx].flatten()),
                            pred_mu.numpy()[idx].flatten() - np.sqrt(pred_var.numpy()[idx].flatten()), alpha=0.2)

        axs[i].plot(x_new, fs_new[i].numpy(), linestyle=':', color='k')
        if ys_new is not None:
            axs[i].scatter(x_new, ys_new[i].numpy(), s=5, color='k', alpha=0.5)
        idx_train = np.where([fun_nos == fun_no])[1]
        axs[i].scatter(data_X[idx_train], data_y[idx_train])
        axs[i].set_ylim(np.min(data_y) - 0.5, np.max(data_y) + 0.5)
        axs[i].set_title(f'Function {fun_no}')
    return subfig


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
        plt.savefig(path / file_name, bbox_inches='tight')
        plt.close()