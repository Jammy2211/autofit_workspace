"""
Cookbook 1: Model Composition Basics
====================================

This cookbook provides an overview of the basic model composition tools, specifically the `Model` object.

Examples using different PyAutoFit API’s for model composition are provided, which produce more concise and readable
code for different use-cases.

__Python Class Template__

A model component is written as a Python class using the following format:

 - The name of the class is the name of the model component, in this case, “Gaussian”.

 - The input arguments of the constructor are the parameters of the mode (here centre, normalization and sigma).

 - The default values of the input arguments tell PyAutoFit whether a parameter is a single-valued float or a
 multi-valued tuple.

Below, we define a 1D Gaussian model component, which is used throughout the **PyAutoFit** workspace to perform
example model fits.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import json
import os
from os import path

import autofit as af


class Gaussian:
    def __init__(
        self,
        centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization=1.0,  # <- are the Gaussian``s model parameters.
        sigma=5.0,
    ):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma


"""
__Model Composition__

We can instantiate a Python class as a model component using `af.Model()`.
"""
model = af.Model(Gaussian)

"""
We can check the model has a `prior_count` of 3, meaning the 3 parameters defined above (`centre`, `normalization` and
`sigma`) all have priors associated with them .

This also means each parameter is fitted for if we fitted the model to data via a non-linear search.
"""
print(f"Model Prior Count = {model.prior_count}")

"""
If we print the `info` attribute of the model we get information on all of the parameters and their priors.
"""
print(model.info)

"""
__Instances__

We can create an instance of the `Gaussian` class using this model.

Below, we create an `instance` of the `Gaussian` class via the model where `centre=30.0`, `normalization=2.0` and
`sigma=3.0`.
"""
instance = model.instance_from_vector(vector=[30.0, 2.0, 3.0])

print("Model Instance: \n")
print(instance)

print("Instance Parameters \n")
print("centre = ", instance.centre)
print("normalization = ", instance.normalization)
print("sigma = ", instance.sigma)

"""
__Model Customization__

We can overwrite the priors of one or more parameters from the default value assumed via configuration files:
"""
model = af.Model(Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-4, upper_limit=1e4)
model.sigma = af.GaussianPrior(mean=0.0, sigma=1.0, lower_limit=0.0, upper_limit=1e5)

"""
We can create an instance of the `Gaussian` by inputting unit values (e.g. between 0.0 and 1.0) which are mapped to
physical values via the priors defined above.

The inputs 0.5 below are mapped as follows:

 - `centre`: goes to 0.5 because this is the midpoint of the `UniformPrior`'s `lower_limit=0.0` and `upper_limit=1.0`.
 
 - `normalization` goes to > because this is the midpoint of 
 the `LogUniformPrior`'s `lower_limit=1e-4` and `upper_limit=1e4` in log10 space.
 
 - `sigma`: goes to 0.5 because this is the mean of its `GaussianPrior`.
"""
instance = model.instance_from_unit_vector(unit_vector=[0.5, 0.5, 0.5])

print("Model Instance: \n")
print(instance)

print("Instance Parameters \n")
print("centre = ", instance.centre)
print("normalization = ", instance.normalization)
print("sigma = ", instance.sigma)

"""
We can fix a free parameter to a specific value (reducing the dimensionality of parameter space by 1):
"""
model = af.Model(Gaussian)
model.centre = 0.0

print(f"\n Model Prior Count After Fixing Centre = {model.prior_count}")

"""
We can link two parameters together such they always assume the same value (reducing the dimensionality of 
parameter space by 1):
"""
model = af.Model(Gaussian)
model.centre = model.normalization

print(f"\n Model Prior Count After Linking Parameters = {model.prior_count}")

"""
Offsets between linked parameters or with certain values are possible:
"""
model = af.Model(Gaussian)
model.centre = model.normalization + model.sigma

print(f"Model Prior Count After Linking Parameters = {model.prior_count}")

"""
Assertions remove regions of parameter space:
"""
model = af.Model(Gaussian)
model.add_assertion(model.sigma > 5.0)
model.add_assertion(model.centre > model.normalization)

"""
__Instance Methods__

We can create instances of the `Gaussian` using the median value of the prior of every parameter.
"""
instance = model.instance_from_prior_medians()

print("Instance Parameters \n")
print("centre = ", instance.centre)
print("normalization = ", instance.normalization)
print("sigma = ", instance.sigma)

"""
We can create a random instance, where the random values are unit values drawn between 0.0 and 1.0.
 
This means the parameter values of this instance are randomly drawn from the priors.
"""
model = af.Model(Gaussian)
instance = model.random_instance()

"""
__Alternative API__

The overwriting of priors shown above can be achieved via the following alternative API:
"""
model = af.Model(
    Gaussian,
    centre=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
    normalization=af.LogUniformPrior(lower_limit=1e-4, upper_limit=1e4),
    sigma=af.GaussianPrior(mean=0.0, sigma=1.0),
)

"""
This API can also be used for fixing a parameter to a certain value:
"""
model = af.Model(Gaussian, centre=0.0)

"""
__Model Dictionary__

A model has a `dict` attribute, which express all information about the model as a Python .

By printing this dictionary we can therefore get a concise summary of the model.
"""
model = af.Model(Gaussian)

print(model.dict())

"""
__JSon Outputs__

Python dictionaries can easily be saved to hard disk as a `.json` file.

This means we can save any **PyAutoFit** model to hard-disk.

Checkout the file `autofit_workspace/*/model/jsons/model.json` to see the model written as a .json.
"""
model_path = path.join("scripts", "model", "jsons")

os.makedirs(model_path, exist_ok=True)

model_file = path.join(model_path, "model.json")

with open(model_file, "w+") as f:
    json.dump(model.dict(), f, indent=4)

"""
We can load the model from its `.json` file.

This means in **PyAutoFit** one can easily writen a model, save it to hard disk and load it elsewhere.
"""
model = af.Model.from_json(file=model_file)

print(f"\n Model via Json Prior Count = {model.prior_count}")

"""
__Wrap Up__

This cookbook shows how to compose simple models using the `af.Model()` object.

The next cookbook describes how to compose models from multiple model components using a `af.Collection()`.
"""
