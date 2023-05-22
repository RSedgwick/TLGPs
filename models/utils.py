import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf
from gpflow.base import _cast_to_dtype
from gpflow.config import default_float
from models.lvmogp import LVMOGP
from gpflow.utilities import to_default_float, ops, print_summary
from models.initializations import get_initialisations_mo_indi, get_initialisations_avg, get_initialisations_lvmogp, \
    get_initialisations_lmc
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

def build_models(models, data_X, data_y, fun_nos, n_fun, observed_dims, lmc_rank, latent_dims_lvmogp, domain, n_restarts=1):
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
                gps.append(gps)
        arg_best_lml = np.argmax(final_lmls)
        final_models_dict[model_name] = gps[arg_best_lml]

    return final_models_dict


def plot_lmls(LMLs):

    fig, axs = plt.subplots(ncols=len(LMLs), figsize=(4, 16))

    for i, model_name, model_dict in enumerate(LMLs.items()):
        for restart, lml_dict in model_dict.items():
            for init_type, lml in lml_dict.items():
                if lml is not None:
                    axs[i].plot(lml, label=f'{init_type}, restart {restart}')
                    axs[i].legend()
                    axs[i].set_title(model_name)
    plt.show()

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

def mo_indi_init(data_X, data_y, fun_nos, n_fun, observed_dims, lmc_rank, lengthscales_X, kernel_var_init, lik_var_init):
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

    X_lmc = np.hstack([data_X, fun_nos])
    y = data_y
    k = gpflow.kernels.RBF(lengthscales=lengthscales_X, variance=kernel_var_init, active_dims=range(observed_dims))
    avg_gp = gpflow.models.GPR(data=(tf.convert_to_tensor(X_lmc, dtype=default_float()),
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

def get_metrics(final_models_dict, domain):

    x_new = np.linspace(domain[0], domain[1], 100)

    for model_name, model in final_models_dict.items():
        pred_mu, pred_var = model.predict_y(Xnew)

        nlpds = get_nlpd(pred_mu, pred_var,  y_true)
        abs_errors = get_abs_error(pred_mu, y_true)

    pass

def get_abs_error(mu, y_true):
    abs_error = np.sqrt(np.square(y_true.ravel() - mu))
    return abs_error

def get_nlpd(mu, sig2, y_true):
    nlpd = - (-0.5 * np.log(2 * np.pi) - 0.5 * np.log(sig2)
              - 0.5 * (np.square(y_true.ravel().reshape(len(y_true.ravel()), ) - mu)) / sig2)
    return nlpd
