import autofit as af

import numpy as np
import matplotlib.pyplot as plt
from os import path
import os

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

    """
    In the log_likelihood_function function below, `instance` is an instance of our model, which in this example is
    an instance of the `Gaussian` class and Exponential class in `model.py`. Their parameters are set via the
    non-linear search. This gives us the instance of the model we need to fit our data!
    """

    def log_likelihood_function(self, instance):
        """
        Determine the log likelihood of a fit of multiple profiles to the dataset.

        Parameters
        ----------
        instance : af.CollectionPriorModel
            The model instances of the profiles.

        Returnsn
        -------
        fit : Fit.log_likelihood
            The log likelihood value indicating how well this model fit the dataset.

        The `instance` that comes into this method is a CollectionPriorModel. It contains instances of every class
        we instantiated it with, where each instance is named following the names given to the CollectionPriorModel,
        which in this example is a `Gaussian` (with name `gaussian) and Exponential (with name `exponential`):
        """
        # print("Gaussian Instance:")
        # print("Centre = ", instance.gaussian.centre)
        # print("Intensity = ", instance.gaussian.intensity)
        # print("Sigma = ", instance.gaussian.sigma)

        # print("Exponential Instance:")
        # print("Centre = ", instance.exponential.centre)
        # print("Intensity = ", instance.exponential.intensity)
        # print("Rate = ", instance.exponential.rate)

        """Get the range of x-values the data is defined on, to evaluate the model of the profiles."""

        xvalues = np.arange(self.data.shape[0])

        """
        The simplest way to create the summed profile is to add the profile of each model component. If we
        know we are going to fit a `Gaussian` + Exponential we can do the following:

            model_data_gaussian = instance.gaussian.profile_from_xvalues(xvalues=xvalues)
            model_data_exponential = instance.exponential.profile_from_xvalues(xvalues=xvalues)
            model_data = model_data_gaussian + model_data_exponential

        However, this does not work if we change our model components. However, the *instance* variable is a list of
        our model components. We can iterate over this list, calling their profile_from_xvalues and summing the result
        to compute the summed profile of any model.
        
        Use these xvalues to create model data of our profiles.
        
        """
        model_data = sum(
            [line.profile_from_xvalues(xvalues=xvalues) for line in instance]
        )

        """Fit the model profile data to the observed data, computing the residuals and chi-squareds."""
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

        model_datas = [line.profile_from_xvalues(xvalues=xvalues) for line in instance]
        model_data = sum(model_datas)

        plt.errorbar(
            x=xvalues,
            y=self.data,
            yerr=self.noise_map,
            color="k",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
        plt.plot(range(self.data.shape[0]), model_data, color="r")
        for model_data_individual in model_datas:
            plt.plot(range(self.data.shape[0]), model_data_individual, "--")
        plt.title("Dynesty model fit to 1D Gaussian + Exponential dataset.")
        plt.xlabel("x values of profile")
        plt.ylabel("Profile intensity")

        os.makedirs(paths.image_path, exist_ok=True)
        plt.savefig(path.join(paths.image_path, "model_fit.png"))
        plt.clf()
