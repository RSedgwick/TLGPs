import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf
from gpflow.base import _cast_to_dtype
from gpflow.config import default_float
from models.lvmogp import LVMOGP
from gpflow.utilities import to_default_float, ops, read_values
from models.initializations import get_initialisations_mo_indi, get_initialisations_avg, get_initialisations_lvmogp, \
    get_initialisations_lmc
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib as mpl


inch_conversion = 3.93701/100
page_width = 142.4 *inch_conversion
column_width = 67.2* inch_conversion


def build_models(models, data_X, data_y, fun_nos, n_fun, observed_dims, lmc_rank, latent_dims_lvmogp, domain,
                 n_restarts=1):
    """a function for building the models, which first gets the initialisations then builds the models
    :param models: list of models to build
    :param data_X: list of data for each function
    :param data_y: list of data for each function
    :param fun_nos: list of function numbers
    :param n_fun: number of functions
    :param observed_dims: list of observed dimensions
    :param lmc_rank: rank of the lmc kernel
    :param latent_dims_lvmogp: number of latent dimensions for the lvmogp model
    :param domain: domain of the functions
    :param n_restarts: number of restarts
    :return: dictionary of dictionaries of models"""

    models_dict = {model_name: {restart: {} for restart in range(n_restarts)} for model_name in models}

    for model_name in models:
        if model_name == 'lmc':
            inits = get_initialisations_lmc(data_X, data_y, fun_nos, n_fun, observed_dims, lmc_rank,
                                            n_restarts=n_restarts)
            for restart, init in inits.items():
                for init_type, init_val in init.items():
                    # try:
                    models_dict[model_name][restart][init_type] = lmc_init(**init_val)
                    print(f'successful build model {model_name}, restart {restart}, init_type {init_type}')
                # except:
                #     print(f'failed to build model {model_name}, restart {restart}, init_type {init_type}')
                #     models_dict[model_name][restart][init_type] = None
        elif model_name == 'mo_indi':
            inits = get_initialisations_mo_indi(data_X, data_y, fun_nos, n_fun, observed_dims, n_restarts=n_restarts)
            for restart, init in inits.items():
                for init_type, init_val in init.items():
                    # try:
                    models_dict[model_name][restart][init_type] = mo_indi_init(**init_val)
                    print(f'successful build model {model_name}, restart {restart}, init_type {init_type}')
                # except:
                #     print(f'failed to build model {model_name}, restart {restart}, init_type {init_type}')
                #     models_dict[model_name][restart][init_type] = None
        elif model_name == 'avg':
            inits = get_initialisations_avg(data_X, data_y, fun_nos, observed_dims, n_restarts=n_restarts)
            for restart, init in inits.items():
                for init_type, init_val in init.items():
                    # try:
                    models_dict[model_name][restart][init_type] = avg_init(**init_val)
                    print(f'successful build model {model_name}, restart {restart}, init_type {init_type}')
                # except:
                #     print(f'failed to build model {model_name}, restart {restart}, init_type {init_type}')
                #     models_dict[model_name][restart][init_type] = None
        elif model_name == 'lvmogp':
            inits = get_initialisations_lvmogp(data_X, data_y, fun_nos, n_fun, observed_dims, latent_dims_lvmogp,
                                               domain, n_restarts=n_restarts)
            for restart, init in inits.items():
                for init_type, init_val in init.items():
                    # try:
                    models_dict[model_name][restart][init_type] = lvmogp_init(**init_val)
                    print(f'successful build model {model_name}, restart {restart}, init_type {init_type}')
                # except:
                #     print(f'failed to build model {model_name}, restart {restart}, init_type {init_type}')
                #     models_dict[model_name][restart][init_type] = None
        else:
            raise ValueError(f'model_name {model_name} not recognised. model_name must be in'
                             f' ["lmc", "mo_indi", "avg", "lvmogp"]')
    return models_dict


def train_models(models_dict):
    """train the models in the models_dict and record the log marginal likelihoods or ELBOs of the models in the training
    process.
    :param models_dict: dictionary of dictionaries of models
    :return: dictionary of dictionaries of models, dictionary of dictionaries of log marginal likelihoods"""

    LMLs = {model_name: {} for model_name in models_dict.keys()}

    for model_name, model_dict in models_dict.items():
        for restart, mod_dict in model_dict.items():
            LMLs[model_name][restart] = {}
            for init_type, model in mod_dict.items():
                try:
                    model, lmls = train_gp(model)
                    models_dict[model_name][restart][init_type] = model
                    LMLs[model_name][restart][init_type] = lmls
                    print(f'successful train {model_name}, restart {restart}, init_type {init_type}')
                except Exception as e:
                    print(f'{model_name} restart {restart} init_type {init_type} failed to train')
                    print(f'Exception: {e}')
                    models_dict[model_name][restart][init_type] = None
                    LMLs[model_name][restart][init_type] = None

    return models_dict, LMLs


def get_final_models_dict(models_dict):
    """select the restart that has the best log marginal likelihood for each model.
    :param models_dict: dictionary of dictionaries for all model restarts for all models
    :return final_models_dict: dictionary of the best model (highest log marginal likelihood or elbo) for each model"""

    final_models_dict = {model_name: None for model_name in models_dict.keys()}

    for model_name, model_dict in models_dict.items():
        final_lmls = []
        gps = []
        for restart, mod_dict in model_dict.items():
            for init_type, model in mod_dict.items():
                final_lmls.append(model.maximum_log_likelihood_objective())
                gps.append(model)
        arg_best_lml = np.argmax(final_lmls)
        final_models_dict[model_name] = gps[arg_best_lml]

    return final_models_dict


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


def lmc_init(data_X, data_y, fun_nos, n_fun, observed_dims, lmc_rank, lengthscales_X, kernel_var_init, lik_var_init,
             W_init, kappa_init):
    """function for initialising the LMC

    :param data_X: the X data
    :param data_y: y data
    :param fun_nos: numpy array containing the function number of each data point
    :param n_fun: the number of output functions
    :param observed_dims: the dimensions of the observed space
    :param lengthscales_X: lengthscale of the observed space
    :param kernel_var_init: initial value for kernel variance
    :param lik_var_init: initial value for the likelihood variance
    :param lmc_rank: the rank of the LMC. If not specified it is taken to be the latent dims
    :param W_init: initialisation of W. If not none should be numpy array of shape [n_fun, latent_dims]
    :param kappa_init: initialisation of kappa. If not none should be numpy array of shape [n_fun]
   """

    X_lmc = np.hstack([data_X, fun_nos])
    y = data_y
    k = gpflow.kernels.RBF(lengthscales=lengthscales_X, variance=kernel_var_init, active_dims=range(observed_dims))
    coreg_k = gpflow.kernels.Coregion(output_dim=n_fun, rank=lmc_rank, active_dims=[observed_dims])
    cov = k * coreg_k
    lmc = gpflow.models.GPR(data=(tf.convert_to_tensor(X_lmc, dtype=default_float()),
                                  tf.convert_to_tensor(y, dtype=default_float())), kernel=cov)
    lmc.likelihood.variance.assign(lik_var_init)
    lmc.kernel.kernels[1].W.assign(W_init)
    lmc.kernel.kernels[1].kappa.assign(kappa_init)

    return lmc


def mo_indi_init(data_X, data_y, fun_nos, n_fun, observed_dims, lmc_rank, lengthscales_X, kernel_var_init,
                 lik_var_init):
    """function for initialising the multioutput independent Gaussian process. This method uses the LMC method but
    setting W=0 and kappa=1 and not trainable to make it behave like an independent GP, with no transfer learning
    between surfaces. This is so we can observed different points ond different functions as the GPflow MOGP model
    does not allow this.

    :param data_X: the X data
    :param data_y: y data
    :param fun_nos: numpy array containing the function number of each data point
    :param n_fun: the number of output functions
    :param observed_dims: the dimensions of the observed space
    :param lengthscales_X: lengthscale of the observed space
    :param kernel_var_init: initial value for kernel variance
    :param lik_var_init: initial value for the likelihood variance
    :param lmc_rank: the rank of the LMC. If not specified it is taken to be the latent dims
   """

    W_init = np.ones((n_fun, lmc_rank)) * 1e-6  # very small rather than zero to avoid issue with transform
    kappa_init = np.ones(n_fun)

    mo_indi = lmc_init(data_X, data_y, fun_nos, n_fun, observed_dims, lmc_rank, lengthscales_X, kernel_var_init,
                       lik_var_init, W_init, kappa_init)

    # make W and kappa not trainable
    gpflow.set_trainable(mo_indi.kernel.kernels[1].W, False)
    gpflow.set_trainable(mo_indi.kernel.kernels[1].kappa, False)

    return mo_indi


def avg_init(data_X, data_y, fun_nos, observed_dims, lengthscales_X, kernel_var_init, lik_var_init):
    """function for initialising the average GP. This is where we just fit one function to all the data, disregarding
    the fact that it came from different functions.

    :param data_X: the X data
    :param data_y: y data
    :param fun_nos: numpy array containing the function number of each data point
    :param observed_dims: the dimensions of the observed space
    :param lengthscales_X: lengthscale of the observed space
    :param kernel_var_init: initial value for kernel variance
    :param lik_var_init: initial value for the likelihood variance
   """

    y = data_y
    k = gpflow.kernels.RBF(lengthscales=lengthscales_X, variance=kernel_var_init, active_dims=range(observed_dims))
    avg_gp = gpflow.models.GPR(data=(tf.convert_to_tensor(data_X, dtype=default_float()),
                                     tf.convert_to_tensor(y, dtype=default_float())), kernel=k)
    avg_gp.likelihood.variance.assign(lik_var_init)

    return avg_gp


def lvmogp_init(data_X, data_y, fun_nos, lengthscales, kernel_var, H_mean, H_var, lik_variance,
                train_inducing, inducing_points=None, n_u=100):
    """function for initialising the LMC

    :param data_X: the X data
    :param data_y: y data
    :param fun_nos: numpy array containing the function number of each data point
    :param lengthscales: lengthscale of the observed space and latent space
    :param kernel_var: kernel variance
    :param H_mean: initial mean of the latent space
    :param H_var: initial variance of the latent space
    :param lik_variance: initial value for the likelihood variance
    :param train_inducing: whether to train the inducing points
    :param inducing_points: initialisation of inducing points
    :param n_u: number of inducing points
    """

    if inducing_points is not None and n_u is not None:
        raise ValueError("Cannot specify both inducing points and number of inducing points n_u")

    kernel_lvmogp = gpflow.kernels.RBF(lengthscales=lengthscales, variance=kernel_var)

    lvmogp = LVMOGP(data=data_y,
                    X_data=data_X,
                    X_data_fn=fun_nos,
                    H_data_mean=H_mean,
                    H_data_var=H_var,
                    kernel=kernel_lvmogp,
                    num_inducing_variables=n_u,
                    inducing_variable=inducing_points,
                    H_prior_mean=None,
                    H_prior_var=None)

    if not train_inducing:
        gpflow.utilities.set_trainable(lvmogp.inducing_variable.Z, False)

    lvmogp.likelihood.variance.assign(to_default_float(lik_variance))

    return lvmogp


def train_gp(gp):
    """function to train any of the GP models using L-BFGS-B.

        :param gp: the GP to train
        :return gp: the trained GP
        :return lmls: the log marginal likelihoods or elbos at each step
        """

    lmls = []

    def step_callback(step, variables, values):
        lmls.append(gp.maximum_log_likelihood_objective().numpy())

    maxiter = 2000
    opt = gpflow.optimizers.Scipy().minimize(
        gp.training_loss, gp.trainable_variables, options=dict(maxiter=maxiter), method="L-BFGS-B",
        step_callback=step_callback)

    return gp, lmls


def get_metrics(final_models_dict, test_fun, domain, n_fun, observed_dims, n_new_funs, n_grid_points=100):
    """calculate the root mean squared error (RMSE) and negative log predictive density (NLPD) of each of the models
    returning a dataframe with the results in.
    :param final_models_dict: dictionary of the models
    :param test_fun: the test functions object
    :param domain: domain of the data
    :param n_fun: the total number of functions
    :param n_new_funs: the number of new functions
    :param observed_dims: the dimensions of the observed space
    :param n_grid_points: the number of grid points to use for the RMSE and NLPD
    :return results_df: dataframe containing the RMSE and NLPD and LML for each of the models"""
    # format Xnew
    x_new, fun_nos, x_new_lmc, x_new_lvmogp = get_gridpoints(domain, n_fun, final_models_dict, observed_dims,
                                                             n_points=100)

    model_x_news = {'avg': x_new, 'mo_indi': x_new_lmc, 'lmc': x_new_lmc, 'lvmogp': x_new_lvmogp}

    # get the true data values at the grid points
    ys_new = []
    for fun in test_fun.functions:
        y_new, _ = fun.predict_y(x_new.reshape(100, 1))
        ys_new.append(y_new)

    res_dfs = []
    for model_name, model in final_models_dict.items():
        NLPDs = []
        RMSEs = []
        NLPDs_new_only = []
        RMSEs_new_only = []
        x = model_x_news[model_name]
        pred_mu, pred_var = model.predict_y(x)
        for i, fun_no in enumerate(range(n_fun)):
            if model_name == 'avg':
                idx = range(len(x_new))
            else:
                idx = range(n_grid_points * i, n_grid_points * (i + 1))
            mu = pred_mu.numpy()[idx].flatten()
            sig2 = pred_var.numpy()[idx].flatten()
            nlpds = get_nlpd(mu, sig2, ys_new[i].numpy().flatten())
            abs_errors = get_abs_error(mu, ys_new[i].numpy().flatten())
            NLPDs.append(nlpds)
            RMSEs.append(abs_errors)
            if fun_no >= (n_fun - n_new_funs):
                # print(fun_no)
                NLPDs_new_only.append(nlpds)
                RMSEs_new_only.append(abs_errors)
        nlpd = np.mean(NLPDs)
        rmse = np.sqrt(np.mean(np.square(RMSEs)))
        nlpd_new_only = np.mean(NLPDs_new_only)
        rmse_new_only = np.sqrt(np.mean(np.square(RMSEs_new_only)))
        row_df = pd.DataFrame(data=[[model_name, nlpd, rmse, nlpd_new_only, rmse_new_only,
                                     model.maximum_log_likelihood_objective().numpy()]],
                              columns=['model', 'nlpd', 'rmse', 'nlpd_new_surface', 'rmse_new_surface', 'lml'])
        res_dfs.append(row_df)

    results_df = pd.concat(res_dfs)

    return results_df


def plot_predictions(final_models_dict, test_fun, domain, n_fun, observed_dims, data_y, n_grid_points=100, save=False,
                     path=None, file_name=None):
    """plot the predictions of each of the models
    :param final_models_dict: dictionary of the models
    :param test_fun: the test functions object
    :param domain: domain of the data
    :param n_fun: the total number of functions
    :param observed_dims: the dimensions of the observed space
    :param data_y: the data values
    :param n_grid_points: the number of grid points to use for the RMSE and NLPD"""

    x_new, fun_nos, x_new_lmc, x_new_lvmogp = get_gridpoints(domain, n_fun, final_models_dict, observed_dims,
                                                             n_points=100)

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
        fig, axs = plt.subplots(nrows=1, ncols=n_fun, figsize=(20, 3))
        for i, fun_no in enumerate(range(n_fun)):
            if model_name == 'avg':
                idx = range(len(x_new))
            else:
                idx = range(n_grid_points * i, n_grid_points * (i + 1))
            axs[i].plot(x_new, pred_mu.numpy()[idx].flatten())
            axs[i].fill_between(x_new.flatten(),
                                pred_mu.numpy()[idx].flatten() + np.sqrt(pred_var.numpy()[idx].flatten()),
                                pred_mu.numpy()[idx].flatten() - np.sqrt(pred_var.numpy()[idx].flatten()), alpha=0.2)

            axs[i].plot(x_new, fs_new[i].numpy(), linestyle=':', color='k')
            axs[i].scatter(x_new, ys_new[i].numpy(), s=5, color='k', alpha=0.5)
            idx_train = np.where([test_fun.fun_no == fun_no])[1]
            axs[i].scatter(test_fun.X[idx_train], test_fun.y[idx_train])
            axs[i].set_ylim(np.min(data_y) - 0.4, np.max(data_y) + 0.4)
            axs[i].set_title(f'Function {fun_no}')
        plt.suptitle(f'{labels[model_name]} Predictions')
        plt.tight_layout()

        if save:
            plt.savefig(path / f'{model_name}_{file_name}', bbox_inches='tight')
            plt.close()


def get_gridpoints(domain, n_fun, final_models_dict, observed_dims, n_points=100):
    """return grid points across domain on each function, formatted in the different ways the models need"""

    n_points = 100
    x_new = np.linspace(domain[0], domain[1], n_points).reshape(n_points, 1)
    fun_nos = np.hstack([[fun_no] * n_points for fun_no in range(n_fun)]).reshape(n_points * n_fun, observed_dims)
    test = np.tile(x_new.flatten(), n_fun)
    x_new_lmc = np.hstack([np.tile(x_new.flatten(), n_fun).reshape(n_points * n_fun, observed_dims), fun_nos])

    if 'lvmogp' in final_models_dict.keys():
        lvmogp = final_models_dict['lvmogp']

        H_mean_vect = tf.reshape(tf.gather(_cast_to_dtype(lvmogp.H_data_mean, dtype=default_float()),
                                           _cast_to_dtype(fun_nos, dtype=tf.int64)),
                                 [len(np.tile(x_new.flatten(), n_fun)), lvmogp.H_data_mean.numpy().shape[1]])
        H_var_vect = tf.reshape(tf.gather(_cast_to_dtype(lvmogp.H_data_var, dtype=default_float()),
                                          _cast_to_dtype(fun_nos, dtype=tf.int64)),
                                [len(np.tile(x_new.flatten(), n_fun)), lvmogp.H_data_mean.numpy().shape[1]])

        Xnew_mean = tf.concat(
            [tf.convert_to_tensor(np.tile(x_new.flatten(), n_fun).reshape(n_points * n_fun, 1), default_float()),
             H_mean_vect], axis=1)
        Xnew_var = tf.concat(
            [tf.zeros(np.tile(x_new.flatten(), n_fun).reshape(n_points * n_fun, 1).shape, dtype=default_float()),
             H_var_vect], axis=1)
        x_new_lvmogp = [Xnew_mean, Xnew_var]
    else:
        x_new_lvmogp = None

    return x_new, fun_nos, x_new_lmc, x_new_lvmogp


def get_abs_error(mu, y_true):
    abs_error = np.sqrt(np.square(y_true.ravel() - mu))
    return abs_error


def get_nlpd(mu, sig2, y_true):
    nlpd = - (-0.5 * np.log(2 * np.pi) - 0.5 * np.log(sig2)
              - 0.5 * (np.square(y_true.ravel().reshape(len(y_true.ravel()), ) - mu)) / sig2)
    return nlpd


def save_models(models_dict, lmls, data_X, data_y, fun_nos, x_new, y_news, f_news, path, file_name):
    """save the hyperparameters and the training log marginal likelihoods for each model. Also save the data so the
    models can be reconstructed.
    :param models_dict: dictionary of models
    :param lmls: dictionary of log marginal likelihoods
    :param data_X: data X
    :param data_y: data y
    :param fun_nos: function numbers
    :return: dataframe of hyperparameters and lmls"""
    hyp_df = pd.DataFrame(
        columns=['model', 'init_type', 'restart', 'lmls', 'hyperparameters', 'data_X', 'data_y', 'fun_nos', 'x_new', 'y_news', 'f_news'])

    for model_name, mod_dict in models_dict.items():
        for restart, model_dict in mod_dict.items():
            for init_type, model in model_dict.items():
                hyp_df.loc[len(hyp_df)] = [model_name, init_type, restart,
                                           lmls[model_name][restart][init_type],
                                           read_values(model), data_X, data_y, fun_nos, x_new, y_news, f_news]
    hyp_df.to_pickle(path / file_name)


def save_results(results_df, path, seed, n_new_points, surface_type, n_new_fun):
    """save the results dataframe to specfied location.
    :param results_df: dataframe of results
    :param path: path to save results
    :param seed: seed
    :param n_new_points: number of new points
    :param surface_type: surface type
    :param n_new_fun: number of new functions"""

    results_df['seed'] = seed
    results_df['n_new_points'] = n_new_points
    results_df['surface_type'] = surface_type
    results_df['n_new_fun'] = n_new_fun
    results_df.to_pickle(path / f'results/results_{surface_type}_{n_new_points}_{n_new_fun}_{seed}.pkl')


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
