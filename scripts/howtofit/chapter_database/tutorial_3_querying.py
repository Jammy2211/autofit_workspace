"""
Tutorial 3: Querying
====================

In the previous tutorial, the `Aggregator` loaded all of the results of all 3 fits.

However, imagine we want the results of a fit to 1 specific data or the model-fits with certain properties. In this
tutorial, we'll learn how query the database and load only the results that we want.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import profiles as p

"""
We begin by loading the database via the `.sqlite` file as we did in the previous tutorial. 

Below, we also filter results to only include completed results  by including the `completed_only` bool. If any 
results were present in the database that were in the middle of an unfinished `NonLinearSearch` they would be omitted 
and not loaded, albeit for this tutorial all 3 of our model-fits had completed anyway!
"""
from autofit.database.aggregator import Aggregator

agg = Aggregator.from_database(
    path.join("output", "howtofit", "database", "database.sqlite"),
 #   completed_only=True
)

"""
First, lets print the number of `Samples` objects the `Aggregator` finds. As in the previous tutorial, we should find 
there are 3 results:
"""
print("Emcee Samples:\n")
print("Total Samples Objects = ", len(list(agg.values("samples"))), "\n")

"""
We can use the `Aggregator`'s to query the database and return only specific fits that we are interested in. We first 
do this, using the `info` object, whereby we can query any of its entries, for example the `dataset_name` string we 
input into the model-fit above. 

By querying using the string `gaussian_x1_1` the model-fit to only the second `Gaussian` dataset is returned:
"""
# FEature Missing

# agg_query = agg.query(agg.directory.contains("gaussian_x1_1"))
# samples_gen = agg_query.values("samples")
samples_gen = agg.values("samples")

print(samples_gen)
print("Total Samples Objects via dataset_name Query = ", len(list(samples_gen)), "\n")

"""
We can also filter based on the model fitted. 

For example, we can load all results which fitted a `Gaussian` model-component, which in this simple example is all
3 model-fits.
 
The ability to query via the model is extremely powerful. It enalbes a user to perform many model-fits with many 
different model parameterizations to large datasets and efficiently load and inspect the results. 

[Note: the code `agg.gaussian` corresponds to the fact that in the `CollectionPriorModel` above, we named the model
component `gaussian`. If this `CollectionPriorModel` had used a different name the code below would change 
correspondingly. Models with multiple model components (e.g., `gaussian` and `exponential`) are therefore also easily 
accessed via the database.]
"""
gaussian = agg.gaussian
agg_query = agg.query(gaussian == p.Gaussian)
samples_gen = agg_query.values("samples")
print("Total Samples Objects via `Gaussian` model query = ", len(list(samples_gen)), "\n")

"""
We can also query based on the result of the model that is fitted. Below, we query to the database to find all fits 
where the inferred value of `sigma` for the `Gaussian` is less than 3.0 (which returns only the first of the
three model-fits).
"""
gaussian = agg.gaussian
agg_query = agg.query(gaussian.sigma < 3.0)
samples_gen = agg_query.values("samples")
print("Total Samples Objects In Query `gaussian.sigma < 3.0` = ", len(list(samples_gen)), "\n")

"""
Advanced queries can be constructed using logic, for example we below we combine the two queries above to find all
results which fitted a `Gaussian` AND (using the & symbol) inferred a value of sigma less than 3.0. 

The OR logical clause is also supported via the symbol |.
"""
gaussian = agg.gaussian
agg_query = agg.query((gaussian == p.Gaussian) & (gaussian.sigma < 3.0))
samples_gen = agg_query.values("samples")
print("Total Samples Objects In Query `Gaussian & sigma < 3.0` = ", len(list(samples_gen)), "\n")

"""
Tutorial 3 complete! 

The API for querying is fairly self explanatory. Through the combination of info based queries, model based
queries and result based queries a user has all the tools they need to fit extremely large datasets with many different
models and load only the results they are interested in for inspection and analysis.
"""
