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
        """Create the concatenated data, where the latent variables are appended to the observed data, by function
        number. The variances for observed data points are set to zeros.
        :return: concatenated data (N, D+Q), concatenated variance (N, D+Q)"""

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
        A tensorflow function to compute the evidence lower bound on the marginal likelihood. This lower bound is adapted
        from the lower bound in the GPflow BayesianGPLVM model, which is in turn adapted from Titsias 2009.
        :return: a tensorflow Tensor representing the lower bound on the marginal likelihood
        """

        Y_data = self.data
        mu, var = self.fill_Hs()
        pH = DiagonalGaussian(mu, var)

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

        L = tf.linalg.cholesky(cov_uu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        log_det_B = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma

        # KL[q(x_tilde) || p(x_tilde)]
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

        bound -= KL
        self.KLH = KL
        self.datafit = bound2

        return bound

    def predict_log_density(self, data: OutputData) -> tf.Tensor:
        raise NotImplementedError

    def predict_f(
            self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Predict the mean and variance of observations at new input locations. This method can be used for predicting
        function values at new locations on already observed surfaces, Xnew, or for prediction on previously unseen
        surfaces, i.e. at new values of H. This method integrates over the distribution of the latent variables, H,
        rather than taking H to be a point estimate.

        Note: This model does not allow full output covariances.
        :param Xnew: points at which to predict
        :param full_cov: whether to return the full covariance matrix, or just the diagonal`
        :param full_output_cov: whether to return the full covariance matrix, or just the diagonal`
        :return: a tuple of the mean and variance of the latent function at the points Xnew
        """
        if full_output_cov:
            raise NotImplementedError

        # concatenate the training data with the H means and variances
        X_mean_tilde, X_var_tilde = self.fill_Hs()
        pH = DiagonalGaussian(X_mean_tilde, X_var_tilde)

        Xnew_mean = Xnew[0]
        Xnew_var = Xnew[1]
        pH_new = DiagonalGaussian(Xnew_mean, Xnew_var)
        psi1_new = expectation(pH_new, (self.kernel, self.inducing_variable))

        Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        psi1 = expectation(pH, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(
            expectation(
                pH, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
            ),
            axis=0,
        )
        jitter = default_jitter()
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
