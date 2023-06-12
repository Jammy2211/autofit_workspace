"""
Cookbook 4: Multi-level Models
==============================

A multi-level model component is written as a Python class where input arguments include one or more optional lists of
Python classes that themselves are instantiated as model components.

For example, the multi-level model below is a Python class that consists of a collection of 1D Gaussian's but has
all of their centres as its own higher level parameter:
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
        normalization=1.0,  # <- **PyAutoFit** recognises these constructor arguments
        sigma=5.0,  # <- are the Gaussian``s model parameters.
    ):
        self.normalization = normalization
        self.sigma = sigma


class MultiLevelGaussians:
    def __init__(
        self,
        higher_level_centre=50.0,  # This is the centre of all Gaussians in this multi level component.
        gaussian_list=None,  # This will contain a list of ``af.Model(Gaussian)``'s
    ):
        self.higher_level_centre = higher_level_centre

        self.gaussian_list = gaussian_list


"""
__Composition__

The multi-level model is instantiated via the af.Model() command, which is passed one or more Gaussian components:
"""
model = af.Model(
    MultiLevelGaussians, gaussian_list=[af.Model(Gaussian), af.Model(Gaussian)]
)

"""
The multi-level model consists of two `Gaussian`'s, however their centres are now shared as a high level parameter.

Thus, the total number of parameters is N=5 (x2 `normalizations`, `x2 `sigma`'s and x1 `higher_level_centre`.
"""
print(f"Multi-level Model Prior Count = {model.prior_count}")

"""
Printing the `info` attribute of the model gives us information on all of the parameters, their priors and the 
structure of the multi level model.
"""

"""
__Instances__

When we create an instance via a multi-level model.

Its attributes are structured in a slightly different way to the `Collection` seen in previous cookbooks. 
"""
instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0, 5.0])

print("Model Instance: \n")
print(instance)

print("Instance Parameters \n")
print("Normalization (Gaussian 0) = ", instance.gaussian_list[0].normalization)
print("Sigma (Gaussian 0) = ", instance.gaussian_list[0].sigma)
print("Normalization (Gaussian 0) = ", instance.gaussian_list[1].normalization)
print("Sigma (Gaussian 0) = ", instance.gaussian_list[1].sigma)
print("Higher Level Centre= ", instance.higher_level_centre)

"""
__Collection Equivalent__

An identical model in terms of functionality could of been created via the `Collection` object as follows:
"""


class GaussianCentre:
    def __init__(
        self,
        centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization=1.0,  # <- are the Gaussian``s model parameters.
        sigma=5.0,
    ):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma


model = af.Collection(gaussian_0=GaussianCentre, gaussian_1=GaussianCentre)

model.gaussian_0.centre = model.gaussian_1.centre

"""
__When to Use a Multi Level Model?__

This raises the question of when to use a `Collection` and when to use multi-level models.

The answer depends on the structure of the models you are composing and fitting. It is common for many models to 
have a natural multi-level structure. 

For example, imagine we had a dataset with 3 groups of 1D `Gaussian`'s with shared centres, where each group had 3 
`Gaussian`'s. 

This model is concise and easy to define using the multi-level API:
"""
multi_0 = af.Model(MultiLevelGaussians, gaussian_list=3 * [Gaussian])

multi_1 = af.Model(MultiLevelGaussians, gaussian_list=3 * [Gaussian])

multi_2 = af.Model(MultiLevelGaussians, gaussian_list=3 * [Gaussian])

model = af.Collection(multi_0=multi_0, multi_1=multi_1, multi_2=multi_2)

"""
Composing the same model without the multi-level model is less concise, less readable and prone to error:
"""
multi_0 = af.Collection(
    gaussian_0=GaussianCentre, gaussian_1=GaussianCentre, gaussian_2=GaussianCentre
)

multi_0.gaussian_0.centre = multi_0.gaussian_1.centre
multi_0.gaussian_0.centre = multi_0.gaussian_2.centre
multi_0.gaussian_1.centre = multi_0.gaussian_2.centre

multi_1 = af.Collection(
    gaussian_0=GaussianCentre, gaussian_1=GaussianCentre, gaussian_2=GaussianCentre
)

multi_1.gaussian_0.centre = multi_1.gaussian_1.centre
multi_1.gaussian_0.centre = multi_1.gaussian_2.centre
multi_1.gaussian_1.centre = multi_1.gaussian_2.centre

multi_2 = af.Collection(
    gaussian_0=GaussianCentre, gaussian_1=GaussianCentre, gaussian_2=GaussianCentre
)

multi_2.gaussian_0.centre = multi_2.gaussian_1.centre
multi_2.gaussian_0.centre = multi_2.gaussian_2.centre
multi_2.gaussian_1.centre = multi_2.gaussian_2.centre

model = af.Collection(multi_0=multi_0, multi_1=multi_1, multi_2=multi_2)

"""
The multi-level model API is more **extensible**. 

For example, if I wanted to compose a model with more `Gaussians`, `Exponential`'s and other 1D profiles I would simply 
write:


multi = af.Model(
    MultiLevelGaussians, gaussian_list=[Gaussian, Gaussian, Exponential, YourProfileHere]
)

Composing the same model using just a `Model` and `Collection` is again possible, but would be even more cumbersome,
less readable and is not an API that is anywhere near as extensible as the multi-level model API.

__Multi Level Model Customization__

To customize the higher level parameters of a multi-level the usual Model API is used:
"""
multi = af.Model(MultiLevelGaussians, gaussian_list=[Gaussian, Gaussian])

multi.higher_level_centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)

"""
To customize a multi-level model instantiated via lists, each model component is accessed via its index:
"""
multi = af.Model(MultiLevelGaussians, gaussian_list=[Gaussian, Gaussian])

multi_level = af.Model(MultiLevelGaussians, gaussian_list=[Gaussian, Gaussian])

multi_level.gaussian_list[0].normalization = multi_level.gaussian_list[1].normalization

"""
Any combination of the API’s shown above can be used for customizing this model:
"""
gaussian_0 = af.Model(Gaussian)
gaussian_1 = af.Model(Gaussian)

gaussian_0.normalization = gaussian_1.normalization

multi_level = af.Model(
    MultiLevelGaussians, gaussian_list=[gaussian_0, gaussian_1, af.Model(Gaussian)]
)

multi_level.higher_level_centre = 1.0
multi_level.gaussian_list[2].normalization = multi_level.gaussian_list[1].normalization

"""
__Alternative API__

A multi-level model can be instantiated where each model sub-component is setup using a name (as opposed to a list).

This means no list input parameter is required in the Python class of the model component, but we do need to include
the `**kwargs` input.
"""


class MultiLevelGaussians:
    def __init__(self, higher_level_centre=1.0, **kwargs):
        self.higher_level_centre = higher_level_centre


model = af.Model(
    MultiLevelGaussians, gaussian_0=af.Model(Gaussian), gaussian_1=af.Model(Gaussian)
)

print(f"Multi-level Model Prior Count = {model.prior_count}")

instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0, 5.0])

print("Instance Parameters \n")
print("Normalization (Gaussian 0) = ", instance.gaussian_0.normalization)
print("Sigma (Gaussian 0) = ", instance.gaussian_0.sigma)
print("Normalization (Gaussian 0) = ", instance.gaussian_1.normalization)
print("Sigma (Gaussian 0) = ", instance.gaussian_1.sigma)
print("Higher Level Centre= ", instance.higher_level_centre)

"""
The use of Python dictionaries illustrated in previous cookbooks can also be used with multi-level models.
"""

model_dict = {"gaussian_0": Gaussian, "gaussian_1": Gaussian}

model = af.Model(MultiLevelGaussians, **model_dict)

print(f"Multi-level Model Prior Count = {model.prior_count}")

instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0, 5.0])

print("Instance Parameters \n")
print("Normalization (Gaussian 0) = ", instance.gaussian_0.normalization)
print("Sigma (Gaussian 0) = ", instance.gaussian_0.sigma)
print("Normalization (Gaussian 0) = ", instance.gaussian_1.normalization)
print("Sigma (Gaussian 0) = ", instance.gaussian_1.sigma)
print("Higher Level Centre= ", instance.higher_level_centre)

"""
__Model Dictionary__

Multi level models also have a `dict` attribute, which express all information about the model as a Python dictionary.

By printing this dictionary we can therefore get a concise summary of the model.
"""
model = af.Model(Gaussian)

print(model.dict())

"""
__JSon Outputs__

This allows us to output and load multi-level models from hard-disk as .json files, as we did for `Model` 
and `Collection` objects in the previous cookbooks.
"""
model_path = path.join("scripts", "model", "jsons")

os.makedirs(model_path, exist_ok=True)

model_file = path.join(model_path, "multi_level.json")

with open(model_file, "w+") as f:
    json.dump(model.dict(), f, indent=4)

"""
We can load the model from its `.json` file.

This means in **PyAutoFit** one can easily writen a model, save it to hard disk and load it else where.
"""
model = af.Model.from_json(file=model_file)

print(f"\n Multi Level Model via Json Prior Count = {model.prior_count}")

"""
__Wrap Up__

This cookbook shows how to compose multi-level models from hierarchies of Python classes.

This is a compelling means by which to compose concise, readable and extendable models, if your modeling problem is
multi-level in its structure.
"""
