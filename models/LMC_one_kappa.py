from typing import Optional

import numpy as np
import tensorflow as tf

from gpflow.base import Parameter
from gpflow.utilities import positive, to_default_float
from gpflow.kernels.base import ActiveDims, Kernel



class Coregion(Kernel):
    """
    A Coregionalization kernel. The inputs to this kernel are _integers_ (we
    cast them from floats as needed) which usually specify the *outputs* of a
    Coregionalization model.

    The kernel function is an indexing of a positive-definite matrix:

      K(x, y) = B[x, y] .

    To ensure that B is positive-definite, it is specified by the two
    parameters of this kernel, W and kappa:

      B = W Wᵀ + diag(kappa) .

    We refer to the size of B as "output_dim x output_dim", since this is the
    number of outputs in a coregionalization model. We refer to the number of
    columns on W as 'rank': it is the number of degrees of correlation between
    the outputs.

    NB. There is a symmetry between the elements of W, which creates a local
    minimum at W=0. To avoid this, it is recommended to initialize the
    optimization (or MCMC chain) using a random W.
    """

    def __init__(
        self,
        output_dim: int,
        rank: int,
        *,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
    ):
        """
        :param output_dim: number of outputs expected (0 <= X < output_dim)
        :param rank: number of degrees of correlation between outputs
        """

        # assert input_dim == 1, "Coregion kernel in 1D only"
        super().__init__(active_dims=active_dims, name=name)

        self.output_dim = output_dim
        self.rank = rank
        W = 0.1 * np.ones((self.output_dim, self.rank))
        self.kappa = Parameter([1], transform=positive())
        self.W = Parameter(W)
        # self.kappa = Parameter(kappa, transform=positive())

    def output_covariance(self):

        B = tf.linalg.matmul(self.W, self.W, transpose_b=True) + tf.linalg.diag(tf.tile(self.kappa, [self.output_dim]))
        return B

    def output_variance(self):
        B_diag = tf.reduce_sum(tf.square(self.W), 1) + tf.tile(self.kappa, [self.output_dim])
        return B_diag

    def K(self, X, X2=None):
        shape_constraints = [
            (X, [..., "N", 1]),
        ]
        if X2 is not None:
            shape_constraints.append((X2, [..., "M", 1]))
        tf.debugging.assert_shapes(shape_constraints)

        X = tf.cast(X[..., 0], tf.int32)
        if X2 is None:
            X2 = X
        else:
            X2 = tf.cast(X2[..., 0], tf.int32)

        B = self.output_covariance()
        return tf.gather(tf.transpose(tf.gather(B, X2)), X)

    def K_diag(self, X):
        tf.debugging.assert_shapes([(X, [..., "N", 1])])
        X = tf.cast(X[..., 0], tf.int32)
        B_diag = self.output_variance()
        return tf.gather(B_diag, X)