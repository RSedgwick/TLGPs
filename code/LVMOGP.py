from typing import Optional

import numpy as np
import tensorflow as tf

from gpflow import covariances, kernels, likelihoods
from gpflow.base import Parameter, _cast_to_dtype
from gpflow.config import default_float, default_jitter
from gpflow.expectations import expectation
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction, Zero
from gpflow.probability_distributions import DiagonalGaussian
from gpflow.utilities import positive, to_default_float, triangular
from gpflow.utilities.ops import pca_reduce
from gpflow.models.gpr import GPR
from gpflow.models.model import GPModel, MeanAndVariance
from gpflow.models.training_mixins import InputData, InternalDataTrainingLossMixin, OutputData
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper

from gpflow import kullback_leiblers
from gpflow.covariances.dispatch import Kuf, Kuu

tf.config.run_functions_eagerly(True)


class LVMOGP(GPModel, InternalDataTrainingLossMixin):
    def __init__(
            self,
            data: OutputData,
            X_data: tf.Tensor,
            X_data_fn: tf.Tensor,
            H_data_mean: tf.Tensor,
            H_data_var: tf.Tensor,
            kernel: Kernel,
            num_inducing_variables: Optional[int] = None,
            inducing_variable=None,
            H_prior_mean=None,
            H_prior_var=None,
    ):
        """
        Initialise Latent Variable Multioutput Gaussian process (LVMOGP) object. This method only works with a Gaussian
        likelihood. This method is an implementation of the LVMOGP introduced by Dai, Zhenwen, Mauricio Álvarez,
        and Neil Lawrence. "Efficient modeling of latent information in supervised learning using gaussian processes."
        Advances in Neural Information Processing Systems 30 (2017).

        :param data: matrix of observations, size N (number of points) x output dimensionality
        :param X_data: observed data, size N (number of points) x D (dimensions)
        :param X_data_fn: vector indicating which function each data point belongs to, size N (number of points) x 1
        :param H_data_mean: initial latent positions, size N (number of points) x Q (latent dimensions)
        :param H_data_var: variance of latent positions ([N, Q]), for the initialisation of the latent space
        :param kernel: kernel specification, by default Squared Exponential
        :param num_inducing_variables: number of inducing points, M
        :param inducing_variable: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
            random permutation of X_data_mean
        :param H_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_data_mean
        :param H_prior_var: prior variance used in KL term of bound. By default 1
        """
        num_data, num_latent_gps = X_data.shape
        num_fns, num_latent_dims = H_data_mean.shape
        super().__init__(kernel, likelihoods.Gaussian(), num_latent_gps=num_latent_gps)
        self.data = data_input_to_tensor(data)

        self.X_data = Parameter(X_data, trainable=False)
        self.X_data_fn = Parameter(X_data_fn, trainable=False)
        self.H_data_mean = Parameter(H_data_mean)
        self.H_data_var = Parameter(H_data_var, transform=positive())

        self.num_fns = num_fns
        self.num_latent_dims = num_latent_dims
        self.num_data = num_data
        self.output_dim = self.data.shape[-1]

        assert X_data.shape[0] == self.data.shape[0], "X mean and Y must be same size."
        assert H_data_mean.shape[0] == H_data_var.shape[0], "H mean and var should be the same length"

        if (inducing_variable is None) == (num_inducing_variables is None):
            raise ValueError(
                "LVMOGP needs exactly one of `inducing_variable` and `num_inducing_variables`"
            )

        if inducing_variable is None:
            # By default we initialize by subset of initial data points (randomly permuted)
            # Note that tf.random.shuffle returns a copy, it does not shuffle in place
            X_mean_tilde, X_var_tilde = self.fill_Hs()
            Z = tf.random.shuffle(X_mean_tilde)[:num_inducing_variables]
            inducing_variable = InducingPoints(Z)

        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        assert X_data.shape[1] == self.num_latent_gps

        # deal with parameters for the prior mean variance of H
        if H_prior_mean is None:
            H_prior_mean = tf.zeros((self.num_fns, self.num_latent_dims), dtype=default_float())
        if H_prior_var is None:
            H_prior_var = tf.ones((self.num_fns, self.num_latent_dims))

        self.H_prior_mean = tf.convert_to_tensor(np.atleast_1d(H_prior_mean), dtype=default_float())
        self.H_prior_var = tf.convert_to_tensor(np.atleast_1d(H_prior_var), dtype=default_float())

        assert self.H_prior_mean.shape[0] == self.num_fns, "H prior mean should have length equal to the number of " \
                                                           "functions "
        assert self.H_prior_mean.shape[1] == self.num_latent_dims, "H prior mean should have one column for each " \
                                                                   "latent dimension "
        assert self.H_prior_var.shape[0] == self.num_fns, "H prior mean should have length equal to the number of " \
                                                          "functions"
        assert self.H_prior_var.shape[1] == self.num_latent_dims, "H prior var should have one column for each " \
                                                                  "latent dimension"

    def fill_Hs(self):
        """append Hs to Xs by function number"""

        H_mean_vect = tf.reshape(tf.gather(_cast_to_dtype(self.H_data_mean, dtype=default_float()),
                                           _cast_to_dtype(self.X_data_fn, dtype=tf.int64)),
                                 [self.num_data, self.num_latent_dims])
        H_var_vect = tf.reshape(tf.gather(_cast_to_dtype(self.H_data_var, dtype=default_float()),
                                          _cast_to_dtype(self.X_data_fn, dtype=tf.int64)),
                                [self.num_data, self.num_latent_dims])

        return tf.concat([self.X_data, H_mean_vect], axis=1), \
               tf.concat([tf.zeros(self.X_data.shape, dtype=default_float()), H_var_vect], axis=1)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()



    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        Y_data = self.data
        mu, var = self.fill_Hs()
        pH = DiagonalGaussian(mu, var)
        # pX = DiagonalGaussian(self.X_data, self.X_data_var)

        num_inducing = self.inducing_variable.num_inducing
        psi0 = tf.reduce_sum(expectation(pH, self.kernel))
        psi1 = expectation(pH, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(
            expectation(
                pH, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
            ),
            axis=0,
        )
        cov_uu = covariances.Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        # print(cov_uu)
        # tf.print('cov_uu', np.linalg.cond(cov_uu))
        # tf.print('psi2', np.linalg.cond(psi2))

        L = tf.linalg.cholesky(cov_uu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        # tf.print(B)
        LB = tf.linalg.cholesky(B)
        # tf.print('B', np.linalg.cond(B))
        log_det_B = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma

        # KL[q(x) || p(x)]
        dH_data_var = (
            self.H_data_var
            if self.H_data_var.shape.ndims == 2
            else tf.linalg.diag_part(self.H_data_var)
        )
        NQ = to_default_float(tf.size(self.H_data_mean))
        D = to_default_float(tf.shape(Y_data)[1])
        KL = -0.5 * tf.reduce_sum(tf.math.log(dH_data_var))
        KL += 0.5 * tf.reduce_sum(tf.math.log(self.H_prior_var))
        KL -= 0.5 * NQ
        KL += 0.5 * tf.reduce_sum(
            (tf.square(self.H_data_mean - self.H_prior_mean) + dH_data_var) / self.H_prior_var
        )


        # compute log marginal bound
        ND = to_default_float(tf.size(Y_data))
        bound = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(Y_data)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.linalg.diag_part(AAT)))
        bound2 = bound

        # kl = self.prior_kl()

        bound -= KL

        # tf.print('elbo: ', bound, ' KL: ', KL, ' data fit term: ', bound2)
        self.KLH = KL
        self.datafit = bound2

        # tf.print(bound)
        return bound

    # def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
    #     """
    #     Compute the mean and variance of the latent function at some new points
    #     Xnew. For a derivation of the terms in here, see the associated SGPR
    #     notebook.
    #     """
    #     Y_data = self.data
    #     X_data = self.X_data
    #     X_mean_tilde, X_var_tilde = self.fill_Hs()
    #     num_inducing = self.inducing_variable.num_inducing
    #     err = Y_data - self.mean_function(X_data)
    #     kuf = Kuf(self.inducing_variable, self.kernel, X_mean_tilde)
    #     kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
    #     Kus = Kuf(self.inducing_variable, self.kernel, Xnew)
    #     sigma = tf.sqrt(self.likelihood.variance)
    #     L = tf.linalg.cholesky(kuu)
    #     A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
    #     B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(num_inducing, dtype=default_float())
    #     LB = tf.linalg.cholesky(B)
    #     Aerr = tf.linalg.matmul(A, err)
    #     c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma
    #     tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
    #     tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
    #     mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
    #     if full_cov:
    #         var = (
    #             self.kernel(Xnew)
    #             + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
    #             - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
    #         )
    #         var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
    #     else:
    #         var = (
    #             self.kernel(Xnew, full_cov=False)
    #             + tf.reduce_sum(tf.square(tmp2), 0)
    #             - tf.reduce_sum(tf.square(tmp1), 0)
    #         )
    #         var = tf.tile(var[:, None], [1, 1]) # self.num_latent_gps
    #     # tf.print(self.kernel(Xnew, full_cov=False), summarize=-1)
    #     # tf.print(A, summarize=-1)
    #     # tf.print(B, summarize=-1)
    #     return mean + self.mean_function(Xnew), var

    # def predict_f(self, Xnew: (InputData), full_cov=False, full_output_cov=False) -> MeanAndVariance:
    #     """
    #     Compute the mean and variance of the latent function at some new points
    #     Xnew. For a derivation of the terms in here, see the associated SGPR
    #     notebook.
    #     """
    #
    #     Y_data = self.data
    #     X_data = self.X_data
    #     X_mean_tilde, X_var_tilde = self.fill_Hs()
    #     pH = DiagonalGaussian(X_mean_tilde, X_var_tilde)
    #
    #     psi0 = tf.reduce_sum(expectation(pH, self.kernel))
    #     psi1 = expectation(pH, (self.kernel, self.inducing_variable))
    #     psi2 = tf.reduce_sum(
    #         expectation(
    #             pH, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
    #         ),
    #         axis=0,
    #     )
    #     test = expectation(pH, self.kernel)
    #     Xnew_mean = Xnew[0]
    #     Xnew_var = Xnew[1]
    #     pH_new = DiagonalGaussian(Xnew_mean, Xnew_var)
    #
    #     kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
    #
    #     # do I need to change somthing to ensure its k(x*,x*) not just k(x,x)??
    #     psi0_new = expectation(pH_new, self.kernel)
    #     psi1_new = expectation(pH_new, (self.kernel, self.inducing_variable))
    #     psi2_new = tf.reduce_sum(
    #         expectation(
    #             pH_new, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
    #         ),
    #         axis=0,
    #     )
    #
    #     beta = 1/tf.sqrt(self.likelihood.variance)
    #
    #     # lamda1 = kuu + beta * psi2
    #     # lamda2 = tf.matmul(Y_data, psi1_new)
    #     # lamda3 = tf.matmul(psi1, lamda2, transpose_a=True)
    #     # mean = beta * tf.linalg.triangular_solve(lamda1, lamda3, lower=True)
    #
    #     lamda1 = kuu + beta * psi2
    #     lamda2 = tf.matmul(psi1, Y_data, transpose_a=True)
    #     L_lamda1 = tf.linalg.cholesky(lamda1)
    #     lamda = beta * tf.linalg.triangular_solve(L_lamda1, lamda2, lower=True)
    #
    #     mean = tf.matmul(tf.transpose(lamda), psi1_new, transpose_b=True)
    #
    #     num_inducing = self.inducing_variable.num_inducing
    #     num_new = Xnew_mean.shape[0]
    #     test = tf.matmul(psi1_new, psi1_new, transpose_a=True)
    #     tmp = psi2_new - tf.matmul(psi1_new, psi1_new, transpose_a=True)
    #     tmp2 = tf.matmul(tmp, lamda)
    #     tmp3 = tf.matmul(lamda, tmp2, transpose_a=True)
    #     Kuu_L = tf.linalg.cholesky(kuu)
    #     L2 = tf.linalg.cholesky(kuu+beta*psi2)
    #     tmp4 = tf.linalg.triangular_solve(Kuu_L, psi2_new)
    #     tmp5 = tf.linalg.triangular_solve(L2, psi2_new)
    #     tmp6 = tf.linalg.trace(tmp4 - tmp5) * tf.eye(num_new, dtype=default_float())
    #     var = tmp3 + psi0_new * tf.eye(num_new, dtype=default_float()) - tmp6
    #
    #
    #     # num_inducing = self.inducing_variable.num_inducing
    #     # err = Y_data - self.mean_function(X_data)
    #     # sigma = tf.sqrt(self.likelihood.variance)
    #     # L = tf.linalg.cholesky(kuu)
    #     # A = tf.linalg.triangular_solve(L, psi1, lower=True) / sigma
    #     #
    #     #
    #     # B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(num_inducing, dtype=default_float())
    #     # LB = tf.linalg.cholesky(B)
    #     # Aerr = tf.linalg.matmul(A, err)
    #     # c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma
    #     # tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
    #     # tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
    #     # mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
    #     # if full_cov:
    #     #     var = (
    #     #             self.kernel(Xnew_mean)
    #     #             + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
    #     #             - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
    #     #     )
    #     #     var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
    #     # else:
    #     #     var = (
    #     #             self.kernel(Xnew_mean, full_cov=False)
    #     #             + tf.reduce_sum(tf.square(tmp2), 0)
    #     #             - tf.reduce_sum(tf.square(tmp1), 0)
    #     #     )
    #     #     var = tf.tile(var[:, None], [1, 1])  # self.num_latent_gps
    #     # # tf.print(self.kernel(Xnew, full_cov=False), summarize=-1)
    #     # # tf.print(A, summarize=-1)
    #     # # tf.print(B, summarize=-1)
    #
    #     test = self.mean_function(Xnew_mean)
    #     return tf.transpose(mean) , tf.linalg.diag_part(var)

    def predict_log_density(self, data: OutputData) -> tf.Tensor:
        raise NotImplementedError

    def predict_f(
            self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points.
        Note that this is very similar to the SGPR prediction, for which
        there are notes in the SGPR notebook.
        Note: This model does not allow full output covariances.
        :param Xnew: points at which to predict
        """
        if full_output_cov:
            raise NotImplementedError

        Y_data = self.data
        X_data = self.X_data
        X_mean_tilde, X_var_tilde = self.fill_Hs()
        pH = DiagonalGaussian(X_mean_tilde, X_var_tilde)

        # psi0 = tf.reduce_sum(expectation(pH, self.kernel))
        # psi1 = expectation(pH, (self.kernel, self.inducing_variable))
        # psi2 = tf.reduce_sum(
        #     expectation(
        #         pH, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
        #     ),
        #     axis=0,
        # )
        # test = expectation(pH, self.kernel)
        Xnew_mean = Xnew[0]
        Xnew_var = Xnew[1]
        pH_new = DiagonalGaussian(Xnew_mean, Xnew_var)

        # kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())

        # do I need to change somthing to ensure its k(x*,x*) not just k(x,x)??
        # psi0_new = expectation(pH_new, self.kernel)
        psi1_new = expectation(pH_new, (self.kernel, self.inducing_variable))
        # psi2_new = tf.reduce_sum(
        #     expectation(
        #         pH_new, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
        #     ),
        #     axis=0,
        # )

        pX = pH

        Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(
            expectation(
                pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
            ),
            axis=0,
        )
        jitter = default_jitter()
        # Kus = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)
        sigma2 = self.likelihood.variance
        L = tf.linalg.cholesky(covariances.Kuu(self.inducing_variable, self.kernel, jitter=jitter))

        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True)
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma2
        tmp1 = tf.linalg.triangular_solve(L, tf.transpose(psi1_new), lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                    self.kernel(Xnew_mean)
                    + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                    - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            shape = tf.stack([1, 1, tf.shape(Y_data)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = (
                    self.kernel(Xnew_mean, full_cov=False)
                    + tf.reduce_sum(tf.square(tmp2), axis=0)
                    - tf.reduce_sum(tf.square(tmp1), axis=0)
            )
            shape = tf.stack([1, tf.shape(Y_data)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew_mean), var
