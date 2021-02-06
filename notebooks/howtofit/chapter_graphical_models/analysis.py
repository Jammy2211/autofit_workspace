import autofit as af
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_line(
    xvalues,
    line,
    title=None,
    ylabel=None,
    errors=None,
    color="k",
    output_path=None,
    output_filename=None,
):
    plt.errorbar(
        x=xvalues, y=line, yerr=errors, color=color, ecolor="k", elinewidth=1, capsize=2
    )
    plt.title(title)
    plt.xlabel("x value of profile")
    plt.ylabel(ylabel)

    if not path.exists(output_path):
        os.makedirs(output_path)

    print(output_path, f"{output_filename}.png")

    plt.savefig(path.join(output_path, f"{output_filename}.png"))
    plt.clf()


class Analysis(af.Analysis):
    def __init__(self, data, noise_map):

        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        model_data = self.model_data_from_instance(instance=instance)

        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * self.noise_map ** 2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

    def model_data_from_instance(self, instance):
        """
        To create the summed profile of all individual profiles in an instance, we can use a dictionary comprehension
        to iterate over all profiles in the instance.
        """
        xvalues = np.arange(self.data.shape[0])

        return sum(
            [profile.profile_from_xvalues(xvalues=xvalues) for profile in instance]
        )

    def visualize(self, paths, instance, during_analysis):
        """
        This method is identical to the previous tutorial, except it now uses the `model_data_from_instance` method
        to create the profile.
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = self.model_data_from_instance(instance=instance)

        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0

        """The visualizer now outputs images of the best-fit results to hard-disk (checkout `visualizer.py`)."""

        plot_line(
            xvalues=xvalues,
            line=self.data,
            title="Data",
            ylabel="Data Values",
            color="k",
            output_path=paths.image_path,
            output_filename="data",
        )

        plot_line(
            xvalues=xvalues,
            line=model_data,
            title="Model Data",
            ylabel="Model Data Values",
            color="k",
            output_path=paths.image_path,
            output_filename="model_data",
        )

        plot_line(
            xvalues=xvalues,
            line=residual_map,
            title="Residual Map",
            ylabel="Residuals",
            color="k",
            output_path=paths.image_path,
            output_filename="residual_map",
        )

        plot_line(
            xvalues=xvalues,
            line=chi_squared_map,
            title="Chi-Squared Map",
            ylabel="Chi-Squareds",
            color="k",
            output_path=paths.image_path,
            output_filename="chi_squared_map",
        )

    def save_attributes_for_aggregator(self, paths):
        """
        Save files like the data and noise-map as pickle files so they can be loaded in the `Aggregator`.
        """

        # These functions save the objects we will later access using the aggregator. They are saved via the `pickle`
        # module in Python, which serializes the data on to the hard-disk.

        with open(path.join(f"{paths.pickle_path}", "data.pickle"), "wb") as f:
            pickle.dump(self.data, f)

        with open(path.join(f"{paths.pickle_path}", "noise_map.pickle"), "wb") as f:
            pickle.dump(self.noise_map, f)
