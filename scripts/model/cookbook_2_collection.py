"""
Cookbook 2: Model Collections
=============================

This cookbook provides an overview of the basic model composition tools, specifically the `Collection` object,
whuich groups together multiple `Model()` components in order to compose complex models.

Examples using different PyAutoFit API’s for model composition are provided, which produce more concise and readable
code for different use-cases.

__Python Class Template__

A model component is written as a Python class using the following format:

 - The name of the class is the name of the model component, in this case, “Gaussian”.

 - The input arguments of the constructor are the parameters of the mode (here centre, normalization and sigma).

 - The default values of the input arguments tell PyAutoFit whether a parameter is a single-valued float or a
 multi-valued tuple.

 Below, we define a 1D Gaussian and 1D Exponential model components, which are used throughout the **PyAutoFit**
 workspace to perform example model fits.
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


class Exponential:
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments are the model
        normalization=0.1,  # <- parameters of the Exponential.
        rate=0.01,
    ):
        self.centre = centre
        self.normalization = normalization
        self.rate = rate


"""
__Model Composition__

To instantiate multiple Python classes into a combined model component we combine the `af.Collection()` and `af.Model()` 
objects.

By passing the key word arguments `gaussian` and `exponential` below, these are used as the names of the attributes of 
instances created using this model (which is illustrated clearly below).
"""
gaussian = af.Model(Gaussian)
exponential = af.Model(Exponential)

model = af.Collection(gaussian=gaussian, exponential=exponential)

"""
We can check the model has a `prior_count` of 6, meaning the 3 parameters defined above (`centre`, `normalization`, 
`sigma` and `rate`) for both the `Gaussian` and `Exponential` classes all have priors associated with them .

This also means each parameter is fitted for if we fitted the model to data via a non-linear search.
"""
print(f"Model Prior Count = {model.prior_count}")

"""
Printing the `info` attribute of the model gives us information on all of the parameters, their priors and the 
structure of the model collection.
"""
print(model.info)

"""
__Instances__

We can create an instance of collection containing both the `Gaussian` and `Exponential` classes using this model.

Below, we create an `instance` where: 

- The `Gaussian` class has`centre=30.0`, `normalization=2.0` and `sigma=3.0`.
- The `Exponential` class has`centre=60.0`, `normalization=4.0` and `rate=1.0``.
"""
instance = model.instance_from_vector(vector=[30.0, 2.0, 3.0, 60.0, 4.0, 1.0])

"""
Because we passed the key word arguments `gaussian` and `exponential` above, these are the names of the attributes of 
instances created using this model (e.g. this is why we write `instance.gaussian`):
"""

print("Model Instance: \n")
print(instance)

print("Instance Parameters \n")
print("centre (Gaussian) = ", instance.gaussian.centre)
print("normalization (Gaussian)  = ", instance.gaussian.normalization)
print("sigma (Gaussian)  = ", instance.gaussian.sigma)
print("centre (Exponential) = ", instance.exponential.centre)
print("normalization (Exponential) = ", instance.exponential.normalization)
print("sigma (Exponential) = ", instance.exponential.rate)

"""
Alternatively, the instance's variables can also be accessed as a list, whereby instead of using attribute names
(e.g. `gaussian_0`) we input the list index.

Note that the order of the instance model components is derived by the order the components are input into the model.

For example, for the line `af.Collection(gaussian=gaussian, exponential=exponential)`, the first entry in the list
is the gaussian because it is the first input to the `Collection`.
"""
print("centre (Gaussian) = ", instance[0].centre)
print("normalization (Gaussian)  = ", instance[0].normalization)
print("sigma (Gaussian)  = ", instance[0].sigma)
print("centre (Gaussian) = ", instance[1].centre)
print("normalization (Gaussian) = ", instance[1].normalization)
print("sigma (Exponential) = ", instance[1].rate)

"""
__Implicit Model__

When creating a model via a `Collection`, there is no need to actually pass the python classes as an `af.Model()`
because **PyAutoFit** implicitly assumes they are to be created as a `Model()`..

This enables more concise code, whereby the following code:
"""

gaussian = af.Model(Gaussian)
exponential = af.Model(Exponential)

model = af.Collection(gaussian=gaussian, exponential=exponential)

"""
Can instead be written as:
"""
model = af.Collection(gaussian=Gaussian, exponential=Exponential)


"""
__Model Customization__

By setting up each Model first the model can be customized using either of the af.Model API’s shown above:
"""
gaussian = af.Model(Gaussian)
gaussian.normalization = 1.0
gaussian.sigma = af.GaussianPrior(mean=0.0, sigma=1.0)

exponential = af.Model(Exponential)
exponential.centre = 50.0
exponential.add_assertion(exponential.rate > 5.0)

model = af.Collection(gaussian=gaussian, exponential=exponential)

print(f"Model Prior Count After Customization = {model.prior_count}")

"""
Below is an alternative API that can be used to create the same model as above:
"""
gaussian = af.Model(
    Gaussian, normalization=1.0, sigma=af.GaussianPrior(mean=0.0, sigma=1.0)
)
exponential = af.Model(Exponential, centre=50.0)
exponential.add_assertion(exponential.rate > 5.0)

model = af.Collection(gaussian=gaussian, exponential=exponential)

print(f"Model Prior Count After Customization = {model.prior_count}")

"""
__Model Customization After Collection__

After creating the model as a `Collection` we can customize it afterwards:
"""
model = af.Collection(gaussian=Gaussian, exponential=Exponential)

model.gaussian.normalization = 1.0
model.gaussian.sigma = af.GaussianPrior(mean=0.0, sigma=1.0)

model.exponential.centre = 50.0
model.exponential.add_assertion(exponential.rate > 5.0)

print(f"Model Prior Count After Customization = {model.prior_count}")

"""
__Many Components__

There is no limit to the number of components we can use to set up a model via a `Collection`.
"""
model = af.Collection(
    gaussian_0=Gaussian,
    gaussian_1=Gaussian,
    exponential_0=Exponential,
    exponential_1=Exponential,
    exponential_2=Exponential,
)

print(f"Model Prior Count = {model.prior_count}")

"""
__Model Composition via Dictionaries__

A model can be created via `af.Collection()` where a dictionary of `af.Model()` objects are passed to it.

The two models created below are identical - one uses the API detailed above whereas the second uses a dictionary.
"""
model = af.Collection(gaussian_0=Gaussian, gaussian_1=Gaussian)
print(f"Model Prior Count = {model.prior_count}")


model_dict = {"gaussian_0": Gaussian, "gaussian_1": Gaussian}
model = af.Collection(**model_dict)
print(f"Model Prior Count = {model.prior_count}")

"""
The keys of the dictionary passed to the model (e.g. `gaussian_0` and `gaussian_1` above) are used to create the
names of the attributes of instances of the model.
"""
instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

print("Model Instance: \n")
print(instance)

print("Instance Parameters \n")
print("centre (Gaussian) = ", instance.gaussian_0.centre)
print("normalization (Gaussian)  = ", instance.gaussian_0.normalization)
print("sigma (Gaussian)  = ", instance.gaussian_0.sigma)
print("centre (Gaussian) = ", instance.gaussian_1.centre)
print("normalization (Gaussian) = ", instance.gaussian_1.normalization)
print("sigma (Gaussian) = ", instance.gaussian_1.sigma)

"""
__Model Composition via Lists__

A list of model components can also be passed to an `af.Collection` to create a model:
"""
model = af.Collection([Gaussian, Gaussian])

print(model.info)

"""
When a list is used, there is no string with which to name the model components (e.g. we do not input `gaussian_0`
and `gaussian_1` anywhere.

The `instance` therefore can only be accessed via list indexing.
"""
instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

print("Model Instance: \n")
print(instance)

print("Instance Parameters \n")
print("centre (Gaussian) = ", instance[0].centre)
print("normalization (Gaussian)  = ", instance[0].normalization)
print("sigma (Gaussian)  = ", instance[0].sigma)
print("centre (Gaussian) = ", instance[1].centre)
print("normalization (Gaussian) = ", instance[1].normalization)
print("sigma (Gaussian) = ", instance[1].sigma)

"""
__Model Dictionary__

A `Collection` has a `dict` attribute, which express all information about the model as a Python dictionary.

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

model_file = path.join(model_path, "collection.json")

with open(model_file, "w+") as f:
    json.dump(model.dict(), f, indent=4)

"""
We can load the model from its `.json` file.

This means in **PyAutoFit** one can easily writen a model, save it to hard disk and load it else where.
"""
model = af.Model.from_json(file=model_file)

print(f"\n Model via Json Prior Count = {model.prior_count}")

"""
__Wrap Up__

This cookbook shows how to compose models consisting of multiple components using the `af.Collection()` object.

The next cookbook describes how **PyAutoFit**'s model composition tools can be used to customize models which 
fit multiple datasets simultaneously.
"""
