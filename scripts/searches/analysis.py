import autofit as af

import os
from os import path
import matplotlib.pyplot as plt
import numpy as np

"""
The `analysis.py` module contains the dataset and log likelihood function which given a model instance (set up by
the non-linear search) fits the dataset and returns the log likelihood of that model.
"""


class Analysis(af.Analysis):

    """
    In this example the Analysis only contains the data and noise-map. It can be easily extended however, for more
    complex data-sets and model fitting problems.
    """

    def __init__(self, data, noise_map):

        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        """
        Determine the log likelihood of a fit of a `Gaussian` to the dataset, using a model instance of the Gaussian.

        Parameters
        ----------
        instance : model.Gaussian
            The `Gaussian` model instance.

        Returnsn
        -------
        fit : Fit.log_likelihood
            The log likelihood value indicating how well this model fit the dataset.

        The `instance` that comes into this method is an instance of the `Gaussian` class. To convince yourself of this,
        go ahead and uncomment the lines below and run the non-linear search.
        """
        # print("Gaussian Instance:")
        # print("Centre = ", instance.centre)
        # print("Intensity = ", instance.intensity)
        # print("Sigma = ", instance.sigma)

        """Get the range of x-values the data is defined on, to evaluate the model of the Gaussian."""
        xvalues = np.arange(self.data.shape[0])

        """Use these xvalues to create model data of our Gaussian."""
        model_data = instance.profile_from_xvalues(xvalues=xvalues)

        """Fit the model gaussian line data to the observed data, computing the residuals and chi-squareds."""
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        log_likelihood = -0.5 * sum(chi_squared_map)

        return log_likelihood

    def visualize(self, paths, instance, during_analysis):

        """
        During a model-fit, the `visualize` method is called throughout the non-linear search. The `instance` passed
        into the visualize method is maximum log likelihood solution obtained by the model-fit so far and it can be
        used to provide on-the-fly images showing how the model-fit is going.
        """

        xvalues = np.arange(self.data.shape[0])

        model_data = instance.profile_from_xvalues(xvalues=xvalues)

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
        plt.title("Model fit to 1D Gaussian dataset.")
        plt.xlabel("x values of profile")
        plt.ylabel("Profile intensity")

        os.makedirs(paths.image_path, exist_ok=True)
        plt.savefig(path.join(paths.image_path, "model_fit.png"))
        plt.clf()
