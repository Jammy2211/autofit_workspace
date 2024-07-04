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

from os import path
import numpy as np

import autofit as af

"""
__Example Source Code (`af.ex`)__

The **PyAutoFit** source code has the following example objects (accessed via `af.ex`) used in this tutorial:

 - `Gaussian`: a model component representing a 1D Gaussian profile.

 - `plot_profile_1d`: a function for plotting 1D profile datasets including their noise.

These are functionally identical to the `Gaussian` and `plot_profile_1d` objects and functions you 
have seen and used elsewhere throughout the workspace.

__Analysis__

The `Analysis` class below has a new method, `save_attributes`, which specifies the properties of the 
fit that are output to hard-disc so that we can load them using the database in the next tutorials.

In particular, note how we output the `data` and `noise_map`, these will be loaded in tutorial 4.
"""


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
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

    def model_data_from_instance(self, instance):
        """
        To create the summed profile of all individual profiles in an instance, we can use a dictionary comprehension
        to iterate over all profiles in the instance.
        """
        xvalues = np.arange(self.data.shape[0])

        return sum([profile.model_data_from(xvalues=xvalues) for profile in instance])

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
        af.ex.plot_profile_1d(
            xvalues=xvalues,
            profile_1d=self.data,
            title="Data",
            ylabel="Data Values",
            color="k",
            output_path=paths.image_path,
            output_filename="data",
        )

        af.ex.plot_profile_1d(
            xvalues=xvalues,
            profile_1d=model_data,
            title="Model Data",
            ylabel="Model Data Values",
            color="k",
            output_path=paths.image_path,
            output_filename="model_data",
        )

        af.ex.plot_profile_1d(
            xvalues=xvalues,
            profile_1d=residual_map,
            title="Residual Map",
            ylabel="Residuals",
            color="k",
            output_path=paths.image_path,
            output_filename="residual_map",
        )

        af.ex.plot_profile_1d(
            xvalues=xvalues,
            profile_1d=chi_squared_map,
            title="Chi-Squared Map",
            ylabel="Chi-Squareds",
            color="k",
            output_path=paths.image_path,
            output_filename="chi_squared_map",
        )

    def save_attributes(self, paths):
        """
        Save files like the data and noise-map as pickle files so they can be loaded in the `Aggregator`
        """

        # These functions save the objects we will later access using the aggregator. They are saved via the `pickle`
        # module in Python, which serializes the data on to the hard-disk.

        paths.save_object("data", self.data)

        paths.save_object("noise_map", self.noise_map)


"""
__Dataset__

We'll fit the single `Gaussian` model used in chapter 1 of **HowToFit**.
"""
model = af.Collection(gaussian=af.ex.Gaussian)

model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.gaussian.sigma = af.GaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

"""
For each dataset we load it from hard-disc, set up its `Analysis` class and fit it with a non-linear search. 

The 3 datasets are in the `autofit_workspace/dataset/example_1d` folder.

We want each results to be stored in the database with an entry specific to the dataset. We'll use the `Dataset`'s name 
string to do this, so lets create a list of the 3 dataset names.
"""
dataset_name_list = ["gaussian_x1_0", "gaussian_x1_1", "gaussian_x1_2"]

"""
__Info__

We can also attach information to the model-fit, by setting up an info dictionary. 

Information about our model-fit (e.g. the dataset) that isn't part of the model-fit is made accessible to the 
database. For example, below we write info on the dataset`s (hypothetical) data of observation and exposure time.
"""
info = {"date_of_observation": "01-02-18", "exposure_time": 1000.0}

"""
__Results From Hard Disk__

We now perform a simple model-fit to 3 datasets, where the results are written to hard-disk using the standard 
output directory structure and we will then build the database from these results. This behaviour is governed 
by us inputting `session=None`.

If you have existing results you wish to build a database for, you can therefore adapt this example you to do this.

Later in this example we show how results can also also be output directly to an .sqlite database, saving on hard-disk 
space. This will be acheived by setting `session` to something that is not `None`.
"""
session = None

"""
__Model Fit__

This for loop runs over every dataset, checkout the comments below for how we set up the path structure.

Note how the `session` is passed to the `DynestyStatic` search.
"""
for dataset_name in dataset_name_list:
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
    In all examples so far, results were wrriten to the `autofit_workspace/output` folder with a path and folder 
    named after the non-linear search.

    In this example, results are written directly to the `database.sqlite` file after the model-fit is complete and 
    only stored in the output folder during the model-fit. This can be important for performing large model-fitting 
    tasks on high performance computing facilities where there may be limits on the number of files allowed, or there
    are too many results to make navigating the output folder manually feasible.

    The `unique_tag` uses the `dataset_name` name below to generate the unique identifier, which in other examples we 
    have seen is also generated depending on the search settings and model. In this example, all three model fits
    use an identical search and model, so this `unique_tag` is key in ensuring 3 separate sets of results for each
    model-fit are stored in the output folder and written to the .sqlite database. 
    """
    search = af.DynestyStatic(
        name="database_example",
        path_prefix=path.join("howtofit", "chapter_database", dataset_name),
        unique_tag=dataset_name,  # This makes the unique identifier use the dataset name
        session=session,  # This can instruct the search to write to the .sqlite database.
    )

    print(
        f"The non-linear search has begun running, checkout \n"
        f"autofit_workspace/output/howtofit/database/{dataset_name}/tutorial_1_fitting_multiple_datasets folder for live \n"
        f"output of the results. This Jupyter notebook cell with progress once the search has completed, this could take a \n"
        f"few minutes!"
    )

    search.fit(model=model, analysis=analysis, info=info)


"""
__Building a Database File From an Output Folder__

The fits above wrote the results to hard-disk in folders, not as an .sqlite database file. 

We build the database below, where the `database_name` corresponds to the name of your output folder and is also the 
name of the `.sqlite` database file that is created.

If you are fitting a relatively small number of datasets (e.g. 10-100) having all results written to hard-disk (e.g. 
for quick visual inspection) and using the database for sample wide analysis is beneficial.

We can optionally only include completed model-fits but setting `completed_only=True`.

If you inspect the `output` folder, you will see a `database.sqlite` file which contains the results.
"""
database_name = "chapter_database"

agg = af.Aggregator.from_database(
    filename=f"{database_name}.sqlite", completed_only=False
)

agg.add_directory(directory=path.join("output", "howtofit", database_name))

"""
__Writing Directly To Database__

Results can be written directly to the .sqlite database file, skipping output to hard-disk entirely, by creating
a session and passing this to the non-linear search.

The code below shows how to do this, but it is commented out to avoid rerunning the non-linear searches.

This is ideal for tasks where model-fits to hundreds or thousands of datasets are performed, as it becomes unfeasible
to inspect the results of all fits on the hard-disk. 

Our recommended workflow is to set up database analysis scripts using ~10 model-fits, and then scaling these up
to large samples by writing directly to the database.
"""
# session = af.db.open_database("chapter_database.sqlite")
#
# search = af.Nautilus(
#     path_prefix=path.join("database"),
#     name="database_example",
#     unique_tag=dataset_name,  # This makes the unique identifier use the dataset name
#     session=session,  # This can instruct the search to write to the .sqlite database.
#     n_live=100,
# )

"""
If you run the above code and inspect the `output` folder, you will see a `database.sqlite` file which contains 
the results.

We can load the database using the `Aggregator` as we did above.
"""
# agg = af.Aggregator.from_database("chapter_database.sqlite")

"""
__Wrap Up__

Checkout the output folder, during the model-fits you should see three new sets of results corresponding to 
our 3 `Gaussian` datasets. If the model-fits are already complete, you will only see the .sqlite database file.

This completes tutorial 1, which was less of a tutorial and more a quick exercise in getting the results of three 
model-fits onto our hard-disc to demonstrate **PyAutoFit**'s database feature!
"""
