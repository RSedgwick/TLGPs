import matplotlib.pyplot as plt
import numpy as np
import gpflow
import tensorflow as tf
from gpflow import default_float

class _TestFun:

    def __init__(self, domain, seed, n_fun, observed_dims, latent_dims=2, max_points=100, noise=0.1,
                 n_grid_points=100, same_points=False, lengthscales_X=None):
        """ Class for generating test functions for the experiments
        :param domain: domain of the function
        :param int seed: seed for the random number generator
        :param int n_fun: number of functions to generate
        :param int observed_dims: number of observed dimensions
        :param int latent_dims: number of latent dimensions
        :param max_points: maximum number of data points to be generated per surface - i.e. the max number that will
         be needed in the experiments
        :param noise: noise level
        :param int n_grid_points: number of grid points to generate
        :param same_points: whether to use the same points for all functions
        :param lengthscales_X: lengthscales for the latent dimensions
        """

        x_full = np.linspace(domain[0], domain[1], n_grid_points)  # grid points for plotting

        if observed_dims == 1:
            self.x_full = x_full.reshape(len(x_full), 1)
        else:
            x_full_1, x_full_2 = np.meshgrid(x_full, x_full)
            X_full = np.hstack([np.expand_dims(x_full_1.ravel(), axis=1),
                                np.expand_dims(x_full_2.ravel(), axis=1)])
            self.x_full = X_full
            self.xs_full = (x_full_1, x_full_2)

        self.n_fun = n_fun
        self.latent_dims = latent_dims
        self.observed_dims = observed_dims
        self.domain = domain
        self.max_points = max_points
        self.xs = []
        self.ys = []
        self.fun_no = []
        self.noise = noise
        self.h_new = None
        self.y = None
        self.seed = seed

        # lengthscales for observed dimensions if not defined
        if lengthscales_X is None:
            self.lengthscales_X = [0.1, 0.1]
        else:
            self.lengthscales_X = lengthscales_X

        self.functions = self.create_functions()

        if same_points:
            points = np.sort(np.random.uniform(self.domain[0], self.domain[1], (self.max_points, self.observed_dims)),
                             axis=0)
            for i in range(self.n_fun):
                self.xs.append(points)
        else:
            for i in range(self.n_fun):
                self.xs.append(np.random.uniform(self.domain[0], self.domain[1], (self.max_points, self.observed_dims)))

        for i in range(self.n_fun):
            self.ys.append(self.function_with_noise(self.functions[i], self.xs[i], noise=self.noise))

    def create_functions(self):
        """Creates the test functions, this function is defined in the child classes"""
        raise NotImplementedError("create_functions not implemented")

    def create_data(self, n_points, random_idx=None):
        """Create the data. This is done by randomly choosing input values then evaluating the functions at those  points
        with noise.

        Returns:
        _______
        X: numpy array
            inputs
        fun_no: numpy array
            function numbers
        y: numpy array
            output values
            """

        self.X = np.array([[]]).reshape(0, self.observed_dims)  # input values
        self.y = np.array([[]]).T  # output values
        self.fun_no = np.array([[]]).T  # number of the function that the point is observed on

        if type(n_points) is int:
            n_points = [n_points] * self.n_fun
        np.random.seed(self.seed)
        for i, fun in enumerate(self.functions):
            if random_idx is not None:
                idx = random_idx
            else:
                idx = range(0, len(self.xs[i]))
            x_ = self.xs[i][idx][:n_points[i]]
            y_ = self.ys[i].numpy()[idx][:n_points[i]]

            fun_no_ = np.ones((len(x_), 1)) * i
            self.X = np.concatenate([self.X, x_])
            self.y = np.concatenate([self.y, y_])
            self.fun_no = np.concatenate([self.fun_no, fun_no_])

        return self.X, self.fun_no, self.y

    def function_with_noise(self, fun, x, noise):
        """evaluate a function with noise
        :param fun: function to evaluate
        :param x: input values
        :param noise: noise
        :return: function values with noise"""
        mean, variance = fun.predict_f(x)
        return mean + np.random.normal(0, noise, (len(x), 1))

    def plot_data(self):
        """plot the true data"""

        if self.observed_dims == 1:
            for i, fun in enumerate(self.functions):
                ys_mean, ys_var = fun.predict_y(self.x_full)
                plt.plot(self.x_full, ys_mean.numpy(), label=f'function {i + 1}', alpha=0.5)
                idx = np.where(self.fun_no == i)
                x_ = self.xs[i]
                y_ = self.ys[i]
                plt.scatter(x_, y_, label=f'data  {i + 1}')

            plt.title('Data')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel('x')
            plt.ylabel('y')
            plot_lims = [np.min(np.concatenate(self.ys)) - 0.25, np.max(np.concatenate(self.ys)) + 0.25]
            plt.ylim(plot_lims[0], plot_lims[1])
            return plot_lims

        if self.observed_dims == 2:

            fig, axs = plt.subplots(ncols=int(np.ceil(self.n_fun / 2)), nrows=2,
                                    figsize=(np.ceil(self.n_fun / 2) * 3, 5))
            ax = axs.flatten()

            for i, fun in enumerate(self.functions):
                ys_mean, ys_var = fun.predict_y(self.x_full)
                contour = ax[i].contourf(self.xs_full[0], self.xs_full[1],
                                         ys_mean.numpy().reshape(self.xs_full[1].shape).T,
                                         label=f'sample {i + 1}')

                # idx = np.where(self.fun_no == i)
                x_ = self.xs[i][:, 0]
                y_ = self.xs[i][:, 1]
                ax[i].scatter(x_, y_, label=f'data', color='k', marker='x')
                ax[i].set_title(f'function {i + 1}')
                ax[i].set_xlabel('x1')
                ax[i].set_ylabel('x2')
                cbar = plt.colorbar(contour, ax=ax[i])

            plt.suptitle('Data')
            plt.tight_layout()
            return None

    def plot_data_seperate_plots(self, n_points=None):
        """plot the data in separate plots for each function"""

        fig, axs = plt.subplots(ncols=self.n_fun, figsize=(3 * self.n_fun, 3))

        ax = axs.flatten()

        if self.observed_dims == 1:
            for i, fun in enumerate(self.functions):
                ys_mean, ys_var = fun.predict_y(self.x_full)
                ax[i].plot(self.x_full, ys_mean.numpy(), label=f'function {i + 1}', alpha=0.5)
                idx = np.where(self.fun_no == i)
                if n_points:
                    x_ = self.xs[idx][:n_points[i]]
                    y_ = self.ys[idx][:n_points[i]]
                else:
                    x_ = self.xs[idx]
                    y_ = self.ys[idx]
                ax[i].scatter(x_, y_, label=f'data  {i + 1}')
                ax[i].set_xlabel('x')
                ax[i].set_ylabel('y')

            plt.suptitle('Data')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()

            plot_lims = [np.min(np.concatenate(self.ys)) - 0.25, np.max(np.concatenate(self.ys)) + 0.25]
            plt.ylim(plot_lims[0], plot_lims[1])
            return plot_lims

        if self.observed_dims == 2:

            fig, axs = plt.subplots(ncols=int(np.ceil(self.n_fun / 2)), nrows=2,
                                    figsize=(np.ceil(self.n_fun / 2) * 3, 5))
            ax = axs.flatten()

            for i, fun in enumerate(self.functions):
                ys_mean, ys_var = fun.predict_y(self.x_full)
                contour = ax[i].contourf(self.xs_full[0], self.xs_full[1],
                                         ys_mean.numpy().reshape(self.xs_full[1].shape).T,
                                         label=f'sample {i + 1}')

                # idx = np.where(self.fun_no == i)
                x_ = self.xs[i][:, 0]
                y_ = self.xs[i][:, 1]
                ax[i].scatter(x_, y_, label=f'data', color='k', marker='x')
                ax[i].set_title(f'function {i + 1}')
                ax[i].set_xlabel('x1')
                ax[i].set_ylabel('x2')
                cbar = plt.colorbar(contour, ax=ax[i])

            plt.suptitle('Data')
            plt.tight_layout()
            return None


class TestFunUncorrelated(_TestFun):

    def __init__(self, domain, seed, n_fun, observed_dims, latent_dims=2, max_points=100, noise=0.1,
                 n_grid_points=100, same_points=False, lengthscales_X=None):
        super().__init__(domain, seed, n_fun, observed_dims, latent_dims, max_points, noise,
                         n_grid_points, same_points, lengthscales_X)

    def create_functions(self):
        """Creates uncorrelated test functions. Each function is an independent sample from a Gaussian process
        prior, a new GP with the correct kernel is then fitted to each function to allow them to be evaluated anywhere.
        :returns functions: a list of the data generating functions"""

        lengthscales_X = self.lengthscales_X[:self.observed_dims]

        kernel_X = gpflow.kernels.RBF(lengthscales=tf.convert_to_tensor(lengthscales_X,
                                                                        dtype=default_float()), variance=2,
                                      active_dims=list(range(self.observed_dims)))

        kernel = kernel_X
        X = np.array([[-1e6] * (self.observed_dims)])  # a point far away from training data so GP will resort to
        # prior at domain
        x_new = self.x_full

        Y = np.array([[0.]])
        gp = gpflow.models.GPR(data=(tf.convert_to_tensor(X, dtype=default_float()),
                                     tf.convert_to_tensor(Y, dtype=default_float())), kernel=kernel)

        samples = gp.predict_f_samples(tf.convert_to_tensor(x_new, dtype=default_float()), self.n_fun)

        functions = []
        for sample in samples:
            gp = gpflow.models.GPR(data=(tf.convert_to_tensor(self.x_full, dtype=default_float()),
                                         tf.convert_to_tensor(sample, dtype=default_float())), kernel=kernel_X)
            functions.append(gp)

        self.functions = functions

        return self.functions


class TestFunLinearCorrelation(_TestFun):

    def __init__(self, domain, seed, n_fun, observed_dims, latent_dims=2, max_points=100, noise=0.1,
                 n_grid_points=100, same_points=False, lengthscales_X=None):
        super().__init__(domain, seed, n_fun, observed_dims, latent_dims, max_points, noise,
                         n_grid_points, same_points, lengthscales_X)

    def create_functions(self):
        """Creates uncorrelated test functions. Each function is an independent sample from a Gaussian process
        prior, a new GP with the correct kernel is then fitted to each function to allow them to be evaluated anywhere.
        :returns functions: a list of the data generating functions"""

        # create a GP with one observed data point far away from the domain of interest
        lengthscales_X = self.lengthscales_X[:self.observed_dims]

        kernel = gpflow.kernels.RBF(lengthscales=tf.convert_to_tensor(lengthscales_X,
                                                                      dtype=default_float()), variance=2,
                                    active_dims=list(range(self.observed_dims)))

        X = np.array([[-1e6] * (self.observed_dims)])
        x_new = self.x_full

        Y = np.array([[0.]])
        gp = gpflow.models.GPR(data=(tf.convert_to_tensor(X, dtype=default_float()),
                                     tf.convert_to_tensor(Y, dtype=default_float())), kernel=kernel)

        # generate one sample for each latent function
        samples = gp.predict_f_samples(tf.convert_to_tensor(x_new, dtype=default_float()), self.latent_dims)

        self.x_new = x_new
        self.latents = samples
        functions = []

        # generate coefficients for linear combination of latent functions
        c_new = np.random.uniform(-1, 1, (self.n_fun, self.latent_dims))
        self.c_new = c_new

        # create each function as a unique linear combination of latent functions
        for i in range(self.n_fun):
            y = samples[0] * self.c_new[i, 0] + samples[1] * self.c_new[i, 1]  # + samples2[i]
            gp = gpflow.models.GPR(data=(tf.convert_to_tensor(self.x_full, dtype=default_float()),
                                         tf.convert_to_tensor(y,
                                                              dtype=default_float())), kernel=kernel)
            functions.append(gp)

        self.functions = functions
        return self.functions


class TestFunNonLinearCorrelation(_TestFun):

    def __init__(self, domain, seed, n_fun, observed_dims, latent_dims=2, max_points=100, noise=0.1,
                 n_grid_points=100, same_points=False, lengthscales_X=None, constants=None):
        super().__init__(domain, seed, n_fun, observed_dims, latent_dims, max_points, noise,
                         n_grid_points, same_points, lengthscales_X)

        if constants is None:
            self.constants = np.random.uniform(4, 12, n_fun)
        else:
            self.constants = constants

    def create_functions(self):
        """Creates non-linearly correlated test functions. Each function is a GP fitted to arandomly offset sigmoid
        function.
        :returns functions: a list of the data generating functions"""

        # create points to fit the GP to following a sigmoid

        x = np.linspace(self.domain[0] - 1, self.domain[1] + 1, 100).reshape(100, 1)

        ys = [1 / (1 + np.exp(-8 * x + n)).reshape(100, 1) for n in self.constants]

        kernel_X = gpflow.kernels.RBF(lengthscales=tf.convert_to_tensor(1,
                                                                        dtype=default_float()), variance=2,
                                      active_dims=list(range(self.observed_dims)))
        functions = []
        for y in ys:
            gp = gpflow.models.GPR(data=(tf.convert_to_tensor(x, dtype=default_float()),
                                         tf.convert_to_tensor(y, dtype=default_float())), kernel=kernel_X)

            maxiter = 1000
            _ = gpflow.optimizers.Scipy().minimize(
                gp.training_loss, gp.trainable_variables, options=dict(maxiter=maxiter), method="L-BFGS-B")

            functions.append(gp)

        self.functions = functions

        return self.functions
