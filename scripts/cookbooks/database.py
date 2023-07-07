"""
Feature: Database
=================

The default behaviour of **PyAutoFit** is for model-fitting results to be output to hard-disc in folders, which are
straight forward to navigate and manually check. For small model-fitting tasks this is sufficient, however many users
have a need to perform many model fits to very large datasets, making manual inspection of results time consuming.

PyAutoFit's database feature outputs all model-fitting results as a
sqlite3 (https://docs.python.org/3/library/sqlite3.html) relational database, such that all results
can be efficiently loaded into a Jupyter notebook or Python script for inspection, analysis and interpretation. This
database supports advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset) can be
loaded.

This example extends our example of fitting a 1D `Gaussian` profile and fits 3 independent datasets each containing a
1D Gaussian. The results will be written to a `.sqlite` database, which we will load to demonstrate the database.

A full description of PyAutoFit's database tools is provided in the database chapter of the `HowToFit` lectures.

__Example Source Code (`af.ex`)__

The **PyAutoFit** source code has the following example objects (accessed via `af.ex`) used in this tutorial:

 - `Analysis`: an analysis object which fits noisy 1D datasets, including `log_likelihood_function` and
 `visualize` functions.

 - `Gaussian`: a model component representing a 1D Gaussian profile.

These are functionally identical to the `Analysis` and `Gaussian` objects you have seen elsewhere in the workspace.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import numpy as np

import autofit as af
import autofit.plot as aplt

"""
__Dataset Names__

For each dataset we load it from hard-disc, set up its `Analysis` class and fit it with a non-linear search. 

The 3 datasets are in the `autofit_workspace/dataset/example_1d` folder.

We want each results to be stored in the database with an entry specific to the dataset, by generating the unique
identifier using the name of the dataset. We'll use the `dataset_name` string to do this, so lets create a list of 
the 3 dataset names.
"""
dataset_name_list = ["gaussian_x1_0", "gaussian_x1_1", "gaussian_x1_2"]

"""
__Info__

Information about our model-fit that isn't part of the model-fit can be made accessible to the database, by passing 
an `info` dictionary. 

For example, below we write info on the dataset`s (hypothetical) data of observation and exposure time, which the
database will be able to access.
"""
info = {"date_of_observation": "01-02-18", "exposure_time": 1000.0}

"""
__Model__

Next, we create our model, which again corresponds to a single `Gaussian` with manual priors.
"""
model = af.Collection(gaussian=af.ex.Gaussian)

model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.gaussian.sigma = af.GaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

"""
___Session__

To output results directly to the database, we start a session, which includes the name of the database `.sqlite` file
where results are stored.
"""
session = af.db.open_database("database.sqlite")

"""
This for loop runs over every dataset, checkout the comments below for how we set up the database entry of each fit.

Note how the `session` is passed to the `Dynesty` search.
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

    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    """
    In all examples so far, results were written to the `autofit_workspace/output` folder with a path and folder 
    named after a unique identifier, which was derived from the non-linear search and model. This unique identifier
    plays a vital role in the database: it is used to ensure every entry in the database is unique. 
    
    In this example, results are written directly to the `database.sqlite` file after the model-fit is complete and 
    only stored in the output folder during the model-fit. This can be important for performing large model-fitting 
    tasks on high performance computing facilities where there may be limits on the number of files allowed, or there
    are too many results to make navigating the output folder manually feasible.
    
    The `unique_tag` below uses the `dataset_name` to alter the unique identifier, which as we have seen is also 
    generated depending on the search settings and model. In this example, all three model fits use an identical 
    search and model, so this `unique_tag` is key for ensuring 3 separate sets of results for each model-fit are 
    stored in the output folder and written to the .sqlite database. 
    """
    search = af.DynestyStatic(
        name="database_example",
        path_prefix=path.join("features", "database"),
        unique_tag=dataset_name,  # This makes the unique identifier use the dataset name
        session=session,  # This instructs the search to write to the .sqlite database.
        nlive=50,
    )

    print(
        f"The non-linear search has begun running. This Jupyter notebook cell with progress once Dynesty has completed, this could take a "
        f"few minutes!"
    )

    result = search.fit(model=model, analysis=analysis, info=info)

print("Dynesty has finished run - you may now continue the notebook.")

"""
The results are not contained in the `output` folder after each search completes. Instead, they are
contained in the `database.sqlite` file, which we can load using the `Aggregator`.
"""
agg = af.Aggregator.from_database("database.sqlite")

"""
__Generators__

Before using the aggregator to inspect results, let me quickly cover Python generators. A generator is an object that 
iterates over a function when it is called. The aggregator creates all of the objects that it loads from the database 
as generators (as opposed to a list, or dictionary, or other Python type).

Why? Because lists and dictionaries store every entry in memory simultaneously. If you fit many datasets, this will use 
a lot of memory and crash your laptop! On the other hand, a generator only stores the object in memory when it is used; 
Python is then free to overwrite it afterwards. Thus, your laptop won't crash!

There are two things to bare in mind with generators:

 1) A generator has no length and to determine how many entries it contains you first must turn it into a list.

 2) Once we use a generator, we cannot use it again and need to remake it. For this reason, we typically avoid 
 storing the generator as a variable and instead use the aggregator to create them on use.

We can now create a `samples` generator of every fit. The `results` example scripts show how , an instance of 
the `Samples` class acts as an interface to the results of the non-linear search.
"""
samples_gen = agg.values("samples")

"""
__Samples__

When we convert this generator to a list and it, the outputs are 3 different SamplesMCMC instances. These correspond to 
the 3 model-fits performed above.
"""
print("Dynesty Samples:\n")
print(samples_gen)
print("Total Samples Objects = ", len(agg), "\n")

"""
The `Samples` class is described in the `result.py` example script. Using the `Aggregator` we can access all of the 
attributes described in that example, for example the value of every parameter.

Refer to `result.py` for all the properties that are accessible via the `Aggregator`.
"""
for samples in agg.values("samples"):
    print("All parameters of the very first sample")
    print(samples.parameter_lists[0])
    print("The tenth sample`s third parameter")
    print(samples.parameter_lists[9][2], "\n")

"""
__Querying__

We can use the `Aggregator`'s to query the database and return only specific fits that we are interested in. We first 
do this, using the `info` object, whereby we can query any of its entries, for example the `dataset_name` string we 
input into the model-fit above. 

By querying using the string `gaussian_x1_1` the model-fit to only the second `Gaussian` dataset is returned:
"""
unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "gaussian_x1_1")

"""
As expected, this list now has only 1 SamplesMCMC corresponding to the second dataset.
"""
print(agg_query.values("samples"))
print("Total Samples Objects via dataset_name Query = ", len(agg_query), "\n")

"""
__Unique Tag__

If we query using an incorrect dataset name we get no results:
"""
unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "incorrect_name")
samples_gen = agg_query.values("samples")

"""
__Search Name__

We can also use the `name` of the search used to fit to the model as a query. 

In this example, all three fits used the same search, which had the `name` `database_example`. Thus, using it as a 
query in this example is somewhat pointless. However, querying based on the search name is very useful for model-fits
which use search chaining (see chapter 3 **HowToLens**), where the results of a particular fit in the chain can be
instantly loaded.

As expected, this query contains all 3 results.
"""
name = agg.search.name
agg_query = agg.query(name == "database_example")

print(agg_query.values("samples"))
print("Total Samples Objects via name Query = ", len(agg_query), "\n")

"""
__Model__

We can also query based on the model fitted. 

For example, we can load all results which fitted a `Gaussian` model-component, which in this simple example is all
3 model-fits.
 
The ability to query via the model is extremely powerful. It enables a user to perform many model-fits with many 
different model parameterizations to large datasets and efficiently load and inspect the results. 

[Note: the code `agg.model.gaussian` corresponds to the fact that in the `Collection` above, we named the model
component `gaussian`. If this `Collection` had used a different name the code below would change 
correspondingly. Models with multiple model components (e.g., `gaussian` and `exponential`) are therefore also easily 
accessed via the database.]
"""
gaussian = agg.model.gaussian
agg_query = agg.query(gaussian == af.ex.Gaussian)
print("Total Samples Objects via `Gaussian` model query = ", len(agg_query), "\n")

"""
Queries using the results of model-fitting are also supported. Below, we query the database to find all fits where the 
inferred value of `sigma` for the `Gaussian` is less than 3.0 (which returns only the first of the three model-fits).
"""
gaussian = agg.model.gaussian
agg_query = agg.query(gaussian.sigma < 3.0)
print("Total Samples Objects In Query `gaussian.sigma < 3.0` = ", len(agg_query), "\n")

"""
Advanced queries can be constructed using logic, for example we below we combine the two queries above to find all
results which fitted a `Gaussian` AND (using the & symbol) inferred a value of sigma less than 3.0. 

The OR logical clause is also supported via the symbol |.
"""
gaussian = agg.model.gaussian
agg_query = agg.query((gaussian == af.ex.Gaussian) & (gaussian.sigma < 3.0))
print(
    "Total Samples Objects In Query `Gaussian & sigma < 3.0` = ", len(agg_query), "\n"
)

"""
The Probability Density Functions of the results can be plotted using Dynesty's in-built visualization tools, 
which are wrapped via the `DynestyPlotter` object.
"""
for samples in agg.values("samples"):
    search_plotter = aplt.DynestyPlotter(samples=samples)
    search_plotter.cornerplot()
    search_plotter.runplot()

"""
The API for querying is fairly self explanatory. Through the combination of info based queries, model based
queries and result based queries a user has all the tools they need to fit extremely large datasets with many different
models and load only the results they are interested in for inspection and analysis.

The Database chapter of the **HowToFit** Jupyter notebooks give a full description of the database feature, including 
examples of advanced queries and how to load and plot the results of a model-fit in more detail.
"""
