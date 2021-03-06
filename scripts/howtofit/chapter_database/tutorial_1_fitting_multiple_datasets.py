"""
Tutorial 1: Fitting Multiple Datasets
=====================================

The default behaviour of **PyAutoFit** is for model-fitting results to be output to hard-disc in folders, which are
straight forward to navigate and manually check the model-fitting results and visualization. For small model-fitting
tasks this is sufficient, however many users have a need to perform many model fits to very large datasets, making
the manual inspection of results time consuming.

PyAutoFit's database feature outputs all model-fitting results as a
sqlite3 (https://docs.python.org/3/library/sqlite3.html) relational database, such that all results
can be efficiently loaded into a Jupyter notebook or Python script for inspection, analysis and interpretation. This
database supports advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset) can be
loaded.

In this tutorial, we fit multiple dataset's with a `NonLinearSearch`, producing multiple sets of results on our
hard-disc. In the following tutorials, we load these results using the database into our Jupyter notebook and
interpret, inspect and plot the results.

we'll fit 3 different dataset's, each with a single `Gaussian` model.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
import pickle

"""
We'll reuse the `plot_line` and `Analysis` classes of the previous tutorial.

Note that the `Analysis` class has a new method, `save_attributes_for_aggregator`. This method specifies which properties of the
fit are output to hard-disc so that we can load them using the `Aggregator` in the next tutorial.
"""
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
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map ** 2.0))
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

        """
        The visualizer now outputs images of the best-fit results to hard-disk (checkout `visualizer.py`).
        """
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
        Save files like the data and noise-map as pickle files so they can be loaded in the `Aggregator`
        """

        # These functions save the objects we will later access using the aggregator. They are saved via the `pickle`
        # module in Python, which serializes the data on to the hard-disk.

        with open(path.join(f"{paths.pickle_path}", "data.pickle"), "wb") as f:
            pickle.dump(self.data, f)

        with open(path.join(f"{paths.pickle_path}", "noise_map.pickle"), "wb") as f:
            pickle.dump(self.noise_map, f)


"""
We'll fit the single `Gaussian` model used in chapter 1 of **HowToFit**.
"""
import profiles as p

model = af.CollectionPriorModel(gaussian=p.Gaussian)

model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.intensity = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.gaussian.sigma = af.GaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

"""
For each dataset we load it from hard-disc, set up its `Analysis` class and fit it with a non-linear search. 

The 3 datasets are in the `autofit_workspace/dataset/example_1d` folder.

We want each results to be stored in the database with an entry specific to the dataset. We'll use the `Dataset`'s name 
string to do this, so lets create a list of the 3 dataset names.
"""
dataset_names = ["gaussian_x1_0", "gaussian_x1_1", "gaussian_x1_2"]

"""
We can also attach information to the model-fit, by setting up an info dictionary. 

Information about our model-fit (e.g. the dataset) that isn't part of the model-fit is made accessible to the 
database. For example, below we write info on the dataset`s (hypothetical) data of observation and exposure time.
"""
info = {"date_of_observation": "01-02-18", "exposure_time": 1000.0}

"""
This for loop runs over every dataset, checkout the comments below for how we set up the path structure.
"""
for dataset_name in dataset_names:

    """
    The code below loads the dataset and sets up the Analysis class.
    """
    dataset_path = path.join("dataset", "example_1d", dataset_name)

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    analysis = Analysis(data=data, noise_map=noise_map)

    """
    In all examples so far, results have gone to the default output path, which was the `autofit_workspace/output` 
    folder and a folder named after the non linear search. In this example, we will repeat this process and then load
    these results into the database and a `database.sqlite` file.

    However, results can be written directly to the `database.sqlite` file omitted hard-disc output entirely, which
    can be important for performing large model-fitting tasks on high performance computing facilities where there
    may be limits on the number of files allowed. The commented out code below shows how one would perform
    direct output to the `.sqlite` file. 

    [NOTE: direct writing to .sqlite not supported yet, so this fit currently outputs to hard-disc as per usual and
    these outputs will be used to make the database.]
    """
    emcee = af.DynestyStatic(
        path_prefix=path.join("howtofit", "database", dataset_name),
    )

    print(
        f"Emcee has begun running, checkout \n"
        f"autofit_workspace/output/howtofit/database/{dataset_name}/tutorial_7_multi folder for live \n"
        f"output of the results. This Jupyter notebook cell with progress once Emcee has completed, this could take a \n"
        f"few minutes!"
    )

    emcee.fit(model=model, analysis=analysis, info=info)

"""
Checkout the output folder, you should see three new sets of results corresponding to our 3 `Gaussian` datasets.

This completes tutorial 1, which was less of a tutorial and more a quick exercise in getting the results of three 
model-fits onto our hard-disc to demonstrate **PyAutoFit**'s database feature!
"""
