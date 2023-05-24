import gpflow
import numpy as np
import tensorflow as tf
from gpflow.config import default_float
import tensorflow_probability as tfp


def get_initialisations_lmc(data_X, data_y, fun_nos, n_fun, observed_dims, latent_dims, n_restarts=1):
    """Define the initialisations for the LMC. This returns a dictionary of dictionaries, with each of the initialisation
    types for each random restart. The initialisations are:
    - random_W_and_k: random initialisation of W and kappa
    - random_W_and_k_0: random initialisation of W and kappa, but with kappa set to 1e-4
    - random_W_and_k_1: random initialisation of W and kappa, but with kappa set to 1
    :param data_X: the X data
    :param data_y: the y data
    :param fun_nos: the function numbers
    :param n_fun: the number of functions
    :param observed_dims: the observed dimensions
    :param latent_dims: the latent dimensions
    :param n_restarts: the number of restarts
    :return: a dictionary of dictionaries, with each of the initialisation types for each random restart"""

    base_init_dict = {"data_X": data_X,
                      "data_y": data_y,
                      "fun_nos": fun_nos,
                      "n_fun": n_fun,
                      "observed_dims": observed_dims,
                      "lmc_rank": 2,
                      "kernel_var_init": 1,
                      "lik_var_init": np.random.uniform(0, 0.1), #np.var(data_y) * 0.01,
                      "W_init": np.random.uniform(0.1, 1, [n_fun, latent_dims]),
                      "kappa_init": np.random.uniform(0.1, 1, [n_fun, ])}

    initialisations = {restart: {'random_W_and_k': {**base_init_dict, **{
        "lengthscales_X": np.random.uniform(0, 1, observed_dims),
        "W_init": np.random.uniform(0.1, 1, [n_fun, latent_dims]),
        "kappa_init": np.random.uniform(0.1, 1, [n_fun, ])}},
                                 'random_W_and_k_0': {**base_init_dict, **{
                                     "lengthscales_X": np.random.uniform(0, 1, observed_dims),
                                     "W_init": np.random.uniform(0.1, 1, [n_fun, latent_dims]),
                                     "kappa_init": np.ones(n_fun, ) * 1e-4}},
                                 'random_W_and_k_1': {**base_init_dict, **{
                                     "lengthscales_X": np.random.uniform(0, 1, observed_dims),
                                     "W_init": np.random.uniform(0.1, 1, [n_fun, latent_dims]),
                                     "kappa_init": np.ones(n_fun, )}}} for restart in range(n_restarts)}

    return initialisations


def get_initialisations_mo_indi(data_X, data_y, fun_nos, n_fun, observed_dims, n_restarts=1):
    """define the initialisations for the mo indi model. Here there is only one type of initialisation as the mo_indi
    model is much easier to fit.
    :param data_X: the X data
    :param data_y: the y data
    :param fun_nos: the function numbers
    :param n_fun: the number of functions
    :param observed_dims: the observed dimensions
    :param n_restarts: the number of restarts
    :return: dictionary of dictionaries, with each of the initialisation types for each random restart

     """

    base_init_dict = {"data_X": data_X,
                      "data_y": data_y,
                      "fun_nos": fun_nos,
                      "n_fun": n_fun,
                      "observed_dims": observed_dims,
                      "lmc_rank": 2,
                      "kernel_var_init": 1,
                      "lik_var_init": np.random.uniform(0, 0.1)}  # }np.var(data_y) * 0.01}

    initialisations = {restart: {'random restart 1': {**base_init_dict, **{
        "lengthscales_X": np.random.uniform(0, 1, observed_dims)}},
                                 'random restart 2': {**base_init_dict, **{
                                     "lengthscales_X": np.random.uniform(0, 1, observed_dims)}},
                                 'random restart 3': {**base_init_dict, **{
                                     "lengthscales_X": np.random.uniform(0, 1, observed_dims)}},
                                 } for restart in range(n_restarts)}

    return initialisations


def get_initialisations_avg(data_X, data_y, fun_nos, observed_dims, n_restarts=1):
    """define the initialisations for the average GP model. Here there is only one type of initialisation as the average
    GP model is much easier to fit.
    :param data_X: the X data
    :param data_y: the y data
    :param fun_nos: list of the function numbers for each function
    :param observed_dims: the observed dimensions
    :param n_restarts: the number of restarts
    :return: dictionary of dictionaries, with each of the initialisation types for each random restart
     """

    base_init_dict = {"data_X": data_X,
                      "data_y": data_y,
                      "fun_nos": fun_nos,
                      "observed_dims": observed_dims,
                      "kernel_var_init": 1,
                      "lik_var_init": np.random.uniform(0, 0.1)} #np.var(data_y) * 0.01}

    initialisations = {restart: {'random restart 1': {**base_init_dict, **{
                                "lengthscales_X": np.random.uniform(0, 1, observed_dims)}},
                                 'random restart 2': {**base_init_dict, **{
                                     "lengthscales_X": np.random.uniform(0, 1, observed_dims)}},
                                 'random restart 3': {**base_init_dict, **{
                                     "lengthscales_X": np.random.uniform(0, 1, observed_dims)}}
                                 } for restart in range(n_restarts)}

    return initialisations


def get_initialisations_lvmogp(data_X, data_y, fun_nos, n_fun, observed_dims, latent_dims_lvmogp, domain, n_restarts=1):
    """Define the initialisations for the lvmogp. Here there are three types of initialisation:
    - random: random initialisation of the latent variables H_mean and H_var
    - mo_pca: initialisation of the latent variables H_mean and H_var using PCA on the predictions of a mo_indi model
    - gpy: initialisation of the latent variables H_mean and H_var based on the method used in GPy package

    :param data_X: the X data
    :param data_y: the y data
    :param fun_nos: the function numbers
    :param n_fun: the number of functions
    :param observed_dims: the observed dimensions
    :param latent_dims_lvmogp: the latent dimensions of the LVMOGP
    :param domain: the domain of the data
    :param n_restarts: the number of restarts
    :return: dictionary of dictionaries, with each of the initialisation types for each random restart

     """

    initialisations = {restart: {'random': lvmogp_random_initalisations(data_X, data_y, fun_nos, n_fun, observed_dims,
                                                                        latent_dims_lvmogp),
                                 'mo_pca': lvmogp_mo_pca_initalisations(data_X, data_y, fun_nos, observed_dims,
                                                                        latent_dims_lvmogp, n_fun, domain),
                                 'gpy': lvmogp_gpy_initalisations(data_X, data_y, fun_nos, observed_dims,
                                                                  latent_dims_lvmogp, n_fun, domain)} for restart in
                       range(n_restarts)}

    return initialisations


def lvmogp_random_initalisations(data_X, data_y, fun_nos, n_fun, observed_dims, latent_dims_lvmogp):
    """Create an initalisatin dictoinary for the lvmogp model with random initialisation of the latent variables.
    :param data_X: the X data
    :param data_y: the y data
    :param fun_nos: the function numbers
    :param n_fun: the number of functions
    :param observed_dims: the observed dimensions
    :param latent_dims_lvmogp: the latent dimensions of the LVMOGP
    :return: dictionary of the the initialisations"""
    H_mean = tf.convert_to_tensor(np.random.uniform(-1, 1, (n_fun, latent_dims_lvmogp)), dtype=default_float())
    H_var = tf.ones((n_fun, latent_dims_lvmogp), dtype=default_float()) * 1e-6

    kern_var = np.random.uniform(0, 1)
    Ls_observed = np.random.uniform(0, 1, observed_dims)
    Ls_latent = np.random.uniform(0, 1, latent_dims_lvmogp)
    Ls = Ls_observed.tolist() + Ls_latent.tolist()
    lengthscales = tf.convert_to_tensor(Ls, dtype=default_float(), name='lengthscales')
    kern_variance = tf.convert_to_tensor(kern_var, dtype=default_float(), name='variance')
    lik_variance = np.random.uniform(0, 0.5)

    initialisation = {"data_X": data_X,
                      "data_y": data_y,
                      "fun_nos": fun_nos,
                      "lengthscales": [0.1, 0.1][:observed_dims] + np.random.uniform(0, 1,
                                                                                     latent_dims_lvmogp).tolist(),
                      "kernel_var": 1.0,
                      "H_mean": H_mean,
                      "H_var": H_var,
                      "lik_variance": np.random.uniform(0, 0.1), # np.var(data_y) * 0.01,
                      "train_inducing": True,
                      "n_u": 100}

    return initialisation


def lvmogp_mo_pca_initalisations(data_X, data_y, fun_nos, observed_dims, latent_dims_lvmogp, n_fun, domain,
                                 for_gpy_init=False):
    """Create an initialisation dictionary for the LVMOGP model with initialisation of the latent variables using PCA on
    the predictions of a MOGP model. The observed lengthscales of the LVMOGP are initialised as the lengthscales of
    the MOGP.
    :param data_X: the X data
    :param data_y: the y data
    :param fun_nos: the function numbers
    :param observed_dims: the observed dimensions
    :param latent_dims_lvmogp: the latent dimensions of the LVMOGP
    :param n_fun: the number of functions
    :param domain: the domain of the data
    :param for_gpy_init: boolean to indicate if the initialisation is for the GPy package
    :return: dictionary of the the initialisations"""
    lengthscales_x = np.random.uniform(0, 1, observed_dims)

    # make the inducing points
    inducing_point_xs = tf.convert_to_tensor(
        np.vstack([np.linspace(domain[0], domain[1], 20).reshape(20, 1)] * len(np.unique(fun_nos))),
        dtype=default_float())

    fun_nos_new = np.hstack([[fun_no] * 20 for fun_no in np.unique(fun_nos)])
    inducing_point_fns = tf.convert_to_tensor(fun_nos_new.reshape(len(fun_nos_new), 1), dtype=default_float())
    inducing_points = [inducing_point_xs, inducing_point_fns]

    k = gpflow.kernels.RBF(lengthscales=lengthscales_x, active_dims=range(observed_dims))
    coreg_k = gpflow.kernels.Coregion(output_dim=n_fun, rank=latent_dims_lvmogp,
                                      active_dims=[observed_dims])
    cov = k * coreg_k

    inducing_variable = gpflow.inducing_variables.InducingPoints(tf.concat(inducing_points, axis=1))

    mo_indi = gpflow.models.SGPR(data=(tf.convert_to_tensor(np.hstack([data_X, fun_nos]), dtype=default_float()),
                                       tf.convert_to_tensor(data_y, dtype=default_float())), kernel=cov,
                                 inducing_variable=inducing_variable)
    # mo_indi.inducing_variable.Z.assign(inducing_variable)
    gpflow.set_trainable(mo_indi.inducing_variable, False)
    mo_indi.likelihood.variance.assign(np.var(data_y) * 0.01)

    mo_indi.kernel.kernels[1].W.assign(np.hstack([np.array([[1e-6] * n_fun]).T, np.array([[1e-6] * n_fun]).T]))
    gpflow.set_trainable(mo_indi.kernel.kernels[1].W, False)
    mo_indi.kernel.kernels[1].kappa.assign(np.array([1] * n_fun))
    gpflow.set_trainable(mo_indi.kernel.kernels[1].kappa, False)

    opt = gpflow.optimizers.Scipy()
    maxiter = 2000
    res = opt.minimize(
        mo_indi.training_loss,
        method="L-BFGS-B",
        variables=mo_indi.trainable_variables,
        options=dict(maxiter=maxiter),
    )

    mo_indi_mu, mo_indi_sig2 = mo_indi.predict_f(mo_indi.inducing_variable.Z)

    mo_indi_mean = mo_indi_mu.numpy().reshape(int(len(np.unique(fun_nos))),
                                              int(len(mo_indi.inducing_variable.Z.numpy()) / len(np.unique(fun_nos))))

    H_mean_init, fracs = pca_reduce(tf.convert_to_tensor(mo_indi_mean, dtype=default_float()),
                                    latent_dims_lvmogp)

    fracs = 0.1 * fracs
    Ls = tf.reduce_max(fracs) / fracs
    Ls = mo_indi.kernel.kernels[
             0].lengthscales.numpy().ravel().tolist() + Ls.numpy().ravel().tolist()

    lengthscales = tf.convert_to_tensor(Ls, dtype=default_float(), name='lengthscales')

    initialisation = {"data_X": data_X,
                      "data_y": data_y,
                      "fun_nos": fun_nos,
                      "lengthscales": lengthscales,
                      "kernel_var": mo_indi.kernel.kernels[0].variance.numpy(),
                      "H_mean": H_mean_init,
                      "H_var": tf.ones((len(H_mean_init), latent_dims_lvmogp), dtype=default_float()) * 1e-6,
                      "lik_variance": mo_indi.likelihood.variance.numpy(),
                      "train_inducing": True,
                      "n_u": 100}

    if for_gpy_init:
        return initialisation, mo_indi_mean
    else:
        return initialisation


def lvmogp_gpy_initalisations(data_X, data_y, fun_nos, observed_dims, latent_dims_lvmogp, n_fun, domain):
    """Create an initalisation dictoinary for the LVMOGP model based on the initialisation used in the GPy package:
    https://github.com/SheffieldML/GPy/blob/devel/GPy/models/gp_multiout_regression_md.py. This involves:
     - fitting a sparse MOGP model to the data
     - doing PCA on the posterior prediction of that model at it's inducing points
     - using the PCA reduced latent variables as the initialisation for a Bayesian Gaussian process latent variable
     model (BGPLVM)
     - using the BGPLVM latent variables as the initialisation for the LVMOGP latent variables
     - using the BGPLVM lengthscales as the initialisation for the LVMOGP latent lengthscales and the MOGP lengthscales
     to initialise the observed lengthscales
     - using the mean of the kernel variances of the MOGP and the BGPLVM to initialise the LVMOGP kernel variance
     :param data_X: the X data
     :param data_y: the y data
     :param fun_nos: the function numbers
     :param observed_dims: the observed dimensions
     :param latent_dims_lvmogp: the latent dimensions of the LVMOGP
     :param n_fun: the number of functions
     :param domain: the domain of the data
     :return: dictionary of the the initialisations
     """

    # get the MOGP initialisations and predictions
    mogp_pca_init, mo_indi_mean = lvmogp_mo_pca_initalisations(data_X, data_y, fun_nos, observed_dims,
                                                               latent_dims_lvmogp, n_fun, domain, for_gpy_init=True)

    # initialise lengthscales of BGPLVM

    Ls_H = mogp_pca_init["lengthscales"].numpy().ravel().tolist()[observed_dims:]
    H_mean_init = mogp_pca_init["H_mean"]

    kernel_H = gpflow.kernels.RBF(lengthscales=Ls_H,
                                  active_dims=list(range(0, latent_dims_lvmogp)))

    # Initialise the variances randomly between 0 and 0.1
    H_var_init = tf.ones((len(H_mean_init.numpy()), latent_dims_lvmogp), dtype=default_float()) * np.random.uniform(0,
                                                                                                                    0.1)
    # Fit the BGPLVM
    gplvm = gpflow.models.BayesianGPLVM(tf.convert_to_tensor(mo_indi_mean, dtype=default_float()),
                                        X_data_mean=H_mean_init,
                                        X_data_var=H_var_init,
                                        kernel=kernel_H,
                                        num_inducing_variables=n_fun)

    gplvm.likelihood.variance.assign(
        np.var(data_y) * 0.01)  # initalise the variance as 0.01 times the variance of the data

    opt = gpflow.optimizers.Scipy()
    maxiter = 2000
    res = opt.minimize(
        gplvm.training_loss,
        method="L-BFGS-B",
        variables=gplvm.trainable_variables,
        options=dict(maxiter=maxiter),
    )

    # intialise LVMOGP

    Ls = mogp_pca_init["lengthscales"].numpy().ravel().tolist()[:observed_dims] \
         + gplvm.kernel.lengthscales.numpy().ravel().tolist()
    lengthscales = tf.convert_to_tensor(Ls, dtype=default_float(), name='lengthscales')
    kern_variance = np.mean([mogp_pca_init["kernel_var"].ravel(), gplvm.kernel.variance.numpy()])[0]

    initialisation = {"data_X": data_X,
                      "data_y": data_y,
                      "fun_nos": fun_nos,
                      "lengthscales": lengthscales,
                      "kernel_var": kern_variance,
                      "H_mean": gplvm.X_data_mean,
                      "H_var": gplvm.X_data_var,
                      "lik_variance": np.random.uniform(0, 0.1), # np.var(data_y) * 0.01,
                      "train_inducing": True,
                      "n_u": 100}

    return initialisation


def pca_reduce(X: tf.Tensor, latent_dim: tf.Tensor) -> tf.Tensor:
    """
    A helpful function for linearly reducing the dimensionality of the input
    points X to `latent_dim` dimensions. Adapted from the GPflow implementation but also returns the fraction of
    variance explained by each latent dimension.

    :param X: data array of size N (number of points) x D (dimensions)
    :param latent_dim: Number of latent dimensions Q < D
    :return: PCA projection array of size [N, Q] and the fraction of variance explained by each latent dimension
    """
    if latent_dim > X.shape[1]:  # pragma: no cover
        raise ValueError("Cannot have more latent dimensions than observed")
    X_cov = tfp.stats.covariance(X)
    evals, evecs = tf.linalg.eigh(X_cov)
    W = evecs[:, -latent_dim:]
    fracs = evals / tf.reduce_sum(evals)
    return (X - tf.reduce_mean(X, axis=0, keepdims=True)) @ W, fracs[-latent_dim:]
