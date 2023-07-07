"""
Cookbook: Analysis
==================

This cookbook provides an overview of how to use and extend `Analysis` objects in **PyAutoFit**.

It has the following sections:

It first covers standard options available for all non-linear searches, in the following sections:

 - Example Fit: A simple example of a non-linear search to remind us how it works.
 - Output To Hard-Disk: Output results to hard-disk so they can be inspected and used to restart a crashed search.
 - Unique Identifier: Ensure results are output in unique folders, so tthey do not overwrite each other.
 - Iterations Per Update: Control how often non-linear searches output results to hard-disk.
 - Parallelization: Use parallel processing to speed up the sampling of parameter space.
 - Plots: Perform non-linear search specific visualization using their in-built visualization tools.

It then provides example code for using every search, in the following sections:

 - Emcee (MCMC): The Emcee ensemble sampler MCMC.
 - Zeus (MCMC): The Zeus ensemble sampler MCMC.
 - DynestyDynamic (Nested Sampling): The Dynesty dynamic nested sampler.
 - DynestyStatic (Nested Sampling): The Dynesty static nested sampler.
 - UltraNest (Nested Sampling): The UltraNest nested sampler.
 - PySwarmsGlobal (Particle Swarm Optimization): The global PySwarms particle swarm optimization
 - PySwarmsLocal (Particle Swarm Optimization): The local PySwarms particle swarm optimization.
 - LBFGS: The L-BFGS scipy optimization.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path

import autofit as af
import autofit.plot as aplt

"""
__Example__

An example of how to write a simple `Analysis` class, to remind ourselves of the basic structure and inputs.
"""
class Analysis(af.Analysis):
    def __init__(self, data: np.ndarray, noise_map: np.ndarray):
        """
        The `Analysis` class acts as an interface between the data and model in **PyAutoFit**.

        Its `log_likelihood_function` defines how the model is fitted to the data and it is called many times by
        the non-linear search fitting algorithm.

        In this example the `Analysis` `__init__` constructor only contains the `data` and `noise-map`, but it can be
        easily extended to include other quantities.

        Parameters
        ----------
        data
            A 1D numpy array containing the data (e.g. a noisy 1D signal) fitted in the workspace examples.
        noise_map
            A 1D numpy array containing the noise values of the data, used for computing the goodness of fit
            metric, the log likelihood.
        """
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance) -> float:
        """
        Returns the log likelihood of a fit of a 1D Gaussian to the dataset.

        The data is fitted using an `instance` of the `Gaussian` class where its `model_data_1d_via_xvalues_from`
        is called in order to create a model data representation of the Gaussian that is fitted to the data.
        """

        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)

        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * self.noise_map ** 2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

"""
An instance of the analysis class is created as follows.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

analysis = Analysis(data=data, noise_map=noise_map)

"""
__Analysis Customization__

The `Analysis` class can be fully customized to be suitable for your model-fit.

For example, additional inputs can be included in the `__init__` constructor and used in the `log_likelihood_function`.
if they are required for your `log_likelihood_function` to work.

The example below includes three additional inputs:

 - Instead of inputting a `noise_map`, a `noise_covariance_matrix` is input, which means that corrrlated noise is 
   accounted for in the `log_likelihood_function`.
 
 - A `mask` is input which masks the data such that certain data points are omitted from the log likelihood
 
 - A `kernel` is input which can account for certain blurring operations during data acquisition.
"""
class Analysis(af.Analysis):
    def __init__(
            self,
            data: np.ndarray,
            noise_covariance_matrix: np.ndarray,
            mask: np.ndarray,
            kernel: np.ndarray
    ):
        """
        The `Analysis` class which has had its inputs edited for a different model-fit.

        Parameters
        ----------
        data
            A 1D numpy array containing the data (e.g. a noisy 1D signal) fitted in the workspace examples.
        noise_covariance_matrix
            A 2D numpy array containing the noise values and their covariances for the data, used for computing the
            goodness of fit whilst accounting for correlated noise.
        mask
            A 1D numpy array containing a mask, where `True` values mean a data point is masked and is omitted from
            the log likelihood.
        kernel
            A 1D numpy array containing the blurring kernel of the data, used for creating the model data.
        """
        super().__init__()

        self.data = data
        self.noise_covariance_matrix = noise_covariance_matrix
        self.mask = mask
        self.kernel = kernel

    def log_likelihood_function(self, instance) -> float:
        """
        The `log_likelihood_function` now has access to the  `noise_covariance_matrix`, `mask` and `kernel`
        input above.
        """
        print(self.noise_covariance_matrix)
        print(self.mask)
        print(self.kernel)

        """
        We do not provide a specific example of how to use these inputs in the `log_likelihood_function` as they are
        specific to your model fitting problem.
        
        The key point is that any inputs required to compute the log likelihood can be passed into the `__init__`
        constructor of the `Analysis` class and used in the `log_likelihood_function`.
        """

        log_likelihood = None

        return log_likelihood

"""
An instance of the analysis class is created as follows.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))

noise_covariance_matrix = np.ones(shape=(data.shape[0], data.shape[0]))
mask = np.full(fill_value=False, shape=data.shape)
kernel = np.full(fill_value=1.0, shape=data.shape)

analysis = Analysis(data=data, noise_covariance_matrix=noise_covariance_matrix, mask=mask, kernel=kernel)

"""
__Visualization__

If a `name` is input into a non-linear search, all results are output to hard-disk in a folder.

By extending the `Analysis` class with a `visualize` function, model specific visualization will also be output
into an `image` folder, for example as `.png` files.

This uses the maximum log likelihood model of the model-fit inferred so far.

Visualization of the results of the search, such as the corner plot of what is called the "Probability Density 
Function", are also automatically output during the model-fit on the fly.
"""
class Analysis(af.Analysis):
    def __init__(self, data, noise_map):
        """
        We use the simpler Analysis class above for this example.
        """
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        """
        The `log_likelihood_function` is identical to the example above
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

    def visualize(self, paths, instance, during_analysis):
        """
        During a model-fit, the `visualize` method is called throughout the non-linear search.

        The `instance` passed into the visualize method is maximum log likelihood solution obtained by the model-fit
        so far and it can be used to provide on-the-fly images showing how the model-fit is going.

        The `paths` object contains the path to the folder where the visualization should be output, which is determined
        by the non-linear search `name` and other inputs.
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)
        residual_map = self.data - model_data

        """
        The visualizer now outputs images of the best-fit results to hard-disk (checkout `visualizer.py`).
        """
        import matplotlib.pyplot as plt

        plt.errorbar(
            x=xvalues,
            y=self.data,
            yerr=self.noise_map,
            color="k",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
        plt.plot(xvalues, model_data, color="r")
        plt.title("Maximum Likelihood Fit")
        plt.xlabel("x value of profile")
        plt.ylabel("Profile Normalization")
        plt.savefig(path.join(paths.image_path, f"model_fit.png"))
        plt.clf()

        plt.errorbar(
            x=xvalues,
            y=residual_map,
            yerr=self.noise_map,
            color="k",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
        plt.title("Residuals of Maximum Likelihood Fit")
        plt.xlabel("x value of profile")
        plt.ylabel("Residual")
        plt.savefig(path.join(paths.image_path, f"model_fit.png"))
        plt.clf()

"""
Finish.
"""