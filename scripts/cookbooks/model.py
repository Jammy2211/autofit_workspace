"""
Cookbook: Models
================

Model composition is the process of defining a probabilistic model as a collection of model components, which are
ultimate fitted to a dataset via a non-linear search.

This cookbook provides an overview of basic model composition tools.

__Contents__

**Models:**

If first describes how to use the `af.Model` object to define models with a single model component from single
Python classes, with the following sections:

 - Python Class Template: The template of a model component written as a Python class.
 - Model Composition (Model): Creating a model via `af.Model()`.
 - Priors (Model): How the default priors of a model are set and how to customize them.
 - Instances (Model): Creating an instance of a model via input parameters.
 - Model Customization (Model): Customizing a model (e.g. fixing parameters or linking them to one another).
 - Tuple Parameters (Model): Defining model components with parameters that are tuples.
 - Json Output (Model): Output a model in human readable text via a .json file and loading it back again.

**Collections:**

It then describes how to use the `af.Collection` object to define models with many model components from multiple
Python classes, with the following sections:

 - Model Composition (Collection): Creating a model via `af.Collection()`.
 - Priors (Collection): How the default priors of a collection are set and how to customize them.
 - Instances (Collection): Create an instance of a collection via input parameters.
 - Model Customization (Collection): Customize a collection (e.g. fixing parameters or linking them to one another).
 - Json Output (Collection): Output a collection in human readable text via a .json file and loading it back again.
 - Extensible Models (Collection): Using collections to extend models with new model components, including the use
   of Python dictionaries and lists.

**Arrays:**

The cookbook next describes using NumPy arrays via tbe `af.Array` object to compose models, where each entry of the
array is a free parameters, therefore offering maximum flexibility with the number of free parameter. This has
the following sections:

 - Model Composition (af.Array): Composing models using NumPy arrays and `af.Array`().
 - Prior Customization (af.Array): How to customize the priors of a numpy array model.
 - Instances (af.Array): Create an instance of a numpy array model via input parameters.
 - Model Customization (af.Array): Customize a numpy array model (e.g. fixing parameters or linking them to one another).
 - Json Output (af.Array): Output a numpy array model in human readable text via a .json file and loading it back again.
 - Extensible Models (af.Array): Using numpy arrays to compose models with a flexible number of parameters.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import json
import os
from os import path
from typing import Tuple

import autofit as af

"""
__Python Class Template__

A model component is written as a Python class using the following format:

 - The name of the class is the name of the model component, in this case, “Gaussian”.

 - The input arguments of the constructor are the parameters of the mode (here `centre`, `normalization` and `sigma`).

 - The default values and typing of the input arguments tell PyAutoFit whether a parameter is a single-valued float or 
   a multi-valued tuple.

We define a 1D Gaussian model component to illustrate model composition in PyAutoFit.
"""


class Gaussian:
    def __init__(
        self,
        centre: float = 30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization: float = 1.0,  # <- are the Gaussian``s model parameters.
        sigma: float = 5.0,
    ):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma


"""
__Model Composition (Model)__

We can instantiate a Python class as a model component using `af.Model()`.
"""
model = af.Model(Gaussian)

"""
The model has 3 free parameters, corresponding to the 3 parameters defined above (`centre`, `normalization` 
and `sigma`).

Each parameter has a prior associated with it, meaning they are fitted for if the model is passed to a non-linear 
search.
"""
print(f"Model Total Free Parameters = {model.total_free_parameters}")

"""
If we print the `info` attribute of the model we get information on all of the parameters and their priors.
"""
print(model.info)

"""
__Priors (Model)__

The model has a set of default priors, which have been loaded from a config file in the PyAutoFit workspace.

The config cookbook describes how to setup config files in order to produce custom priors, which means that you do not
need to manually specify priors in your Python code every time you compose a model.

If you do not setup config files, all priors must be manually specified before you fit the model, as shown below.
"""
model = af.Model(Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-4, upper_limit=1e4)
model.sigma = af.TruncatedGaussianPrior(
    mean=0.0, sigma=1.0, lower_limit=0.0, upper_limit=1e5
)

"""
__Instances (Model)__

Instances of the model components above (created via `af.Model`) can be created, where an input `vector` of
parameters is mapped to create an instance of the Python class of the model.

We first need to know the order of parameters in the model, so we know how to define the input `vector`. This
information is contained in the models `paths` attribute.
"""
print(model.paths)

"""
We create an `instance` of the `Gaussian` class via the model where `centre=30.0`, `normalization=2.0` and `sigma=3.0`.
"""
instance = model.instance_from_vector(vector=[30.0, 2.0, 3.0])

print("\nModel Instance:")
print(instance)

print("\nInstance Parameters:\n")
print("centre = ", instance.centre)
print("normalization = ", instance.normalization)
print("sigma = ", instance.sigma)

"""
We can create an `instance` by inputting unit values (e.g. between 0.0 and 1.0) which are mapped to the input values 
via the priors.

The inputs of 0.5 below are mapped as follows:

 - `centre`: goes to 0.5 because this is the midpoint of a `UniformPrior` with `lower_limit=0.0` and `upper_limit=1.0`.

 - `normalization` goes to 1.0 because this is the midpoint of the `LogUniformPrior`' with `lower_limit=1e-4` 
 and `upper_limit=1e4`, corresponding to log10 space.

 - `sigma`: goes to 0.0 because this is the `mean` of the `TruncatedGaussianPrior`.
"""
instance = model.instance_from_unit_vector(unit_vector=[0.5, 0.5, 0.5])

print("\nModel Instance:")
print(instance)

print("\nInstance Parameters:\n")
print("centre = ", instance.centre)
print("normalization = ", instance.normalization)
print("sigma = ", instance.sigma)

"""
We can create instances of the `Gaussian` using the median value of the prior of every parameter.
"""
instance = model.instance_from_prior_medians()

print("\nInstance Parameters:\n")
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
__Model Customization (Model)__

We can fix a free parameter to a specific value (reducing the dimensionality of parameter space by 1):
"""
model = af.Model(Gaussian)
model.centre = 0.0

print(
    f"\n Model Total Free Parameters After Fixing Centre = {model.total_free_parameters}"
)

"""
We can link two parameters together such they always assume the same value (reducing the dimensionality of 
parameter space by 1):
"""
model.centre = model.normalization

print(
    f"\n Model Total Free Parameters After Linking Parameters = {model.total_free_parameters}"
)

"""
Offsets between linked parameters or with certain values are possible:
"""
model.centre = model.normalization + model.sigma

print(
    f"Model Total Free Parameters After Linking Parameters = {model.total_free_parameters}"
)

"""
Assertions remove regions of parameter space (but do not reduce the dimensionality of parameter space):
"""
model.add_assertion(model.sigma > 5.0)
model.add_assertion(model.centre > model.normalization)

"""
The customized model can be inspected by printing its `info` attribute.
"""
print(model.info)

"""
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
__Tuple Parameters__

The `Gaussian` model component above only has parameters that are single-valued floats. 

Parameters can also be tuples, which is useful for defining model components where certain parameters are naturally
grouped together.

For example, we can define a 2D Gaussian with a center that has two coordinates and therefore free parameters, (x, y), 
using a tuple.
"""


class Gaussian2D:
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        normalization: float = 0.1,
        sigma: float = 1.0,
    ):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma


"""
The model's `total_free_parameters` attribute now includes 4 free parameters, as the tuple `centre` parameter accounts
for 2 free parameters.
"""
model = af.Model(Gaussian2D)

print(f"Model Total Free Parameters = {model.total_free_parameters}")

"""
This information is again displayed in the `info` attribute:
"""
print("\nInfo:")
print(model.info)

"""
Here are examples of how model customization can be applied to a model with tuple parameters:
"""
model = af.Model(Gaussian2D)
model.centre = (0.0, 0.0)

model.centre_0 = model.normalization

model.centre_1 = model.normalization + model.sigma

model.add_assertion(model.centre_0 > model.normalization)

"""
__Json Outputs (Model)__

A model has a `dict` attribute, which expresses all information about the model as a Python dictionary.

By printing this dictionary we can therefore get a concise summary of the model.
"""
model = af.Model(Gaussian)

print(model.dict())

"""
The dictionary representation printed above can be saved to hard disk as a `.json` file.

This means we can save any **PyAutoFit** model to hard-disk in a human readable format.

Checkout the file `autofit_workspace/*/cookbooks/jsons/collection.json` to see the model written as a .json.
"""
model_path = path.join("scripts", "cookbooks", "jsons")

os.makedirs(model_path, exist_ok=True)

model_file = path.join(model_path, "model.json")

with open(model_file, "w+") as f:
    json.dump(model.dict(), f, indent=4)

"""
We can load the model from its `.json` file, meaning that one can easily save a model to hard disk and load it 
elsewhere.
"""
model = af.Model.from_json(file=model_file)

print(model.info)

"""
__Model Composition (Collection)__

To illustrate `Collection` objects we define a second model component, representing a `Exponential` profile.
"""


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
To instantiate multiple Python classes into a combined model component we combine the `af.Collection()` and `af.Model()` 
objects.

By passing the key word arguments `gaussian` and `exponential` below, these are used as the names of the attributes of 
instances created using this model (which is illustrated clearly below).
"""
model = af.Collection(gaussian=af.Model(Gaussian), exponential=af.Model(Exponential))

"""
We can check the model has a `total_free_parameters` of 6, meaning the 3 parameters defined 
above (`centre`, `normalization`, `sigma` and `rate`) for both the `Gaussian` and `Exponential` classes all have 
priors associated with them .

This also means each parameter is fitted for if we fitted the model to data via a non-linear search.
"""
print(f"Model Total Free Parameters = {model.total_free_parameters}")

"""
Printing the `info` attribute of the model gives us information on all of the parameters. 
"""
print(model.info)

"""
__Priors (Collection)__

The model has a set of default priors, which have been loaded from a config file in the PyAutoFit workspace.

The configs cookbook describes how to setup config files in order to produce custom priors, which means that you do not
need to manually specify priors in your Python code every time you compose a model.

If you do not setup config files, all priors must be manually specified before you fit the model, as shown below.
"""
model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
model.exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.exponential.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
model.exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

"""
When creating a model via a `Collection`, there is no need to actually pass the python classes as an `af.Model()`
because **PyAutoFit** implicitly assumes they are to be created as a `Model()`.

This enables more concise code, whereby the following code:
"""
model = af.Collection(gaussian=af.Model(Gaussian), exponential=af.Model(Exponential))

"""
Can instead be written as:
"""
model = af.Collection(gaussian=Gaussian, exponential=Exponential)

"""
__Instances (Collection)__

We can create an instance of collection containing both the `Gaussian` and `Exponential` classes using this model.

Below, we create an `instance` where: 

- The `Gaussian` class has `centre=30.0`, `normalization=2.0` and `sigma=3.0`.
- The `Exponential` class has `centre=60.0`, `normalization=4.0` and `rate=1.0``.
"""
instance = model.instance_from_vector(vector=[30.0, 2.0, 3.0, 60.0, 4.0, 1.0])

"""
Because we passed the key word arguments `gaussian` and `exponential` above, these are the names of the attributes of 
instances created using this model (e.g. this is why we write `instance.gaussian`):
"""

print("\nModel Instance:")
print(instance)

print("\nInstance Parameters:\n")
print("centre (Gaussian) = ", instance.gaussian.centre)
print("normalization (Gaussian)  = ", instance.gaussian.normalization)
print("sigma (Gaussian)  = ", instance.gaussian.sigma)
print("centre (Exponential) = ", instance.exponential.centre)
print("normalization (Exponential) = ", instance.exponential.normalization)
print("rate (Exponential) = ", instance.exponential.rate)

"""
Alternatively, the instance's variables can also be accessed as a list, whereby instead of using attribute names
(e.g. `gaussian_0`) we input the list index.

Note that the order of the instance model components is determined from the order the components are input into the 
`Collection`.

For example, for the line `af.Collection(gaussian=gaussian, exponential=exponential)`, the first entry in the list
is the gaussian because it is the first input to the `Collection`.
"""
print("centre (Gaussian) = ", instance[0].centre)
print("normalization (Gaussian)  = ", instance[0].normalization)
print("sigma (Gaussian)  = ", instance[0].sigma)
print("centre (Gaussian) = ", instance[1].centre)
print("normalization (Gaussian) = ", instance[1].normalization)
print("rate (Exponential) = ", instance[1].rate)

"""
__Model Customization (Collection)__

By setting up each Model first the model can be customized using either of the API’s shown above:
"""
gaussian = af.Model(Gaussian)
gaussian.normalization = 1.0
gaussian.sigma = af.GaussianPrior(mean=0.0, sigma=1.0)

exponential = af.Model(Exponential)
exponential.centre = 50.0
exponential.add_assertion(exponential.rate > 5.0)

model = af.Collection(gaussian=gaussian, exponential=exponential)

print(model.info)

"""
Below is an alternative API that can be used to create the same model as above.

Which API is used is up to the user and which they find most intuitive.
"""
gaussian = af.Model(
    Gaussian, normalization=1.0, sigma=af.GaussianPrior(mean=0.0, sigma=1.0)
)
exponential = af.Model(Exponential, centre=50.0)
exponential.add_assertion(exponential.rate > 5.0)

model = af.Collection(gaussian=gaussian, exponential=exponential)

print(model.info)

"""
After creating the model as a `Collection` we can customize it afterwards:
"""
model = af.Collection(gaussian=Gaussian, exponential=Exponential)

model.gaussian.normalization = 1.0
model.gaussian.sigma = af.GaussianPrior(mean=0.0, sigma=1.0)

model.exponential.centre = 50.0
model.exponential.add_assertion(exponential.rate > 5.0)

print(model.info)

"""
__JSon Outputs (Collection)__

A `Collection` has a `dict` attribute, which express all information about the model as a Python dictionary.

By printing this dictionary we can therefore get a concise summary of the model.
"""
print(model.dict())

"""
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
We can load the model from its `.json` file, meaning that one can easily save a model to hard disk and load it 
elsewhere.
"""
model = af.Model.from_json(file=model_file)

print(f"\n Model via Json Prior Count = {model.prior_count}")


"""
__Extensible Models (Collection)__

There is no limit to the number of components we can use to set up a model via a `Collection`.
"""
model = af.Collection(
    gaussian_0=Gaussian,
    gaussian_1=Gaussian,
    exponential_0=Exponential,
    exponential_1=Exponential,
    exponential_2=Exponential,
)

print(model.info)

"""
A model can be created via `af.Collection()` where a dictionary of `af.Model()` objects are passed to it.

The two models created below are identical - one uses the API detailed above whereas the second uses a dictionary.
"""
model = af.Collection(gaussian_0=Gaussian, gaussian_1=Gaussian)
print(model.info)

model_dict = {"gaussian_0": Gaussian, "gaussian_1": Gaussian}
model = af.Collection(**model_dict)
print(model.info)

"""
The keys of the dictionary passed to the model (e.g. `gaussian_0` and `gaussian_1` above) are used to create the
names of the attributes of instances of the model.
"""
instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

print("\nModel Instance:")
print(instance)

print("\nInstance Parameters:\n")
print("centre (Gaussian) = ", instance.gaussian_0.centre)
print("normalization (Gaussian)  = ", instance.gaussian_0.normalization)
print("sigma (Gaussian)  = ", instance.gaussian_0.sigma)
print("centre (Gaussian) = ", instance.gaussian_1.centre)
print("normalization (Gaussian) = ", instance.gaussian_1.normalization)
print("sigma (Gaussian) = ", instance.gaussian_1.sigma)

"""
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

print("\nModel Instance:")
print(instance)

print("\nInstance Parameters:\n")
print("centre (Gaussian) = ", instance[0].centre)
print("normalization (Gaussian)  = ", instance[0].normalization)
print("sigma (Gaussian)  = ", instance[0].sigma)
print("centre (Gaussian) = ", instance[1].centre)
print("normalization (Gaussian) = ", instance[1].normalization)
print("sigma (Gaussian) = ", instance[1].sigma)

"""
__Model Composition (af.Array)__

Models can be composed using NumPy arrays, where each element of the array is a free parameter. 

This offers a lot more flexibility than using `Model` and `Collection` objects, as the number of parameters in the 
model is chosen on initialization via the input of the `shape` attribute. 

For many use cases, this flexibility is key to ensuring model composition is as easy as possible, for example when 
a part of the model being fitted is a matrix of parameters which may change shape depending on the dataset being
fitted.

To compose models using NumPy arrays, we use the `af.Array` object.
"""
model = af.Array(
    shape=(2, 2),
    prior=af.GaussianPrior(mean=0.0, sigma=1.0),
)

"""
Each element of the array is a free parameter, which for `shape=(2,2)` means the model has 4 free parameters.
"""
print(f"Model Total Free Parameters = {model.total_free_parameters}")

"""
The `info` attribute of the model gives information on all of the parameters and their priors.
"""
print(model.info)

"""
__Prior Customization (af.Array)__

The prior of every parameter in the array is set via the `prior` input above.

NumPy array models do not currently support default priors via config files, so all priors must be manually specified.

The prior of every parameter in the array can be customized by normal NumPy array indexing:
"""
model = af.Array(shape=(2, 2), prior=af.GaussianPrior(mean=0.0, sigma=1.0))

model[0, 0] = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
model[0, 1] = af.LogUniformPrior(lower_limit=1e-4, upper_limit=1e4)
model[1, 0] = af.GaussianPrior(mean=0.0, sigma=2.0)

"""
The `info` attribute shows the customized priors.
"""
print(model.info)

"""
__Instances (af.Array)__

Instances of numpy array model components can be created, where an input `vector` of parameters is mapped to create 
an instance of the Python class of the model.

If the priors of the numpy array are not customized, ordering of parameters goes from element [0,0] to [0,1] to [1,0],
as shown by the `paths` attribute.
"""
model = af.Array(
    shape=(2, 2),
    prior=af.GaussianPrior(mean=0.0, sigma=1.0),
)

print(model.paths)

"""
An instance can then be created by passing a vector of parameters to the model via the `instance_from_vector` method.

The `instance` created is a NumPy array, where each element is the value passed in the vector.
"""
instance = model.instance_from_vector(vector=[0.0, 1.0, 2.0, 3.0])

print("\nModel Instance:")
print(instance)

"""
Prior customization changes the order of the parameters, therefore if you customize the priors of the numpy 
array you must check the ordering of the parameters in the `paths` attribute before passing a vector to 
the `instance_from_vector`
"""
model[0, 0] = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
model[0, 1] = af.LogUniformPrior(lower_limit=1e-4, upper_limit=1e4)
model[1, 0] = af.GaussianPrior(mean=0.0, sigma=2.0)

print(model.paths)

instance = model.instance_from_vector(vector=[0.0, 1.0, 2.0, 3.0])

print("\nModel Instance:")
print(instance)

"""
__Model Customization (af.Array)__

The model customization API for numpy array models is the same as for `af.Model` and `af.Collection` objects.
"""
model = af.Array(
    shape=(2, 2),
    prior=af.GaussianPrior(mean=0.0, sigma=1.0),
)

model[0, 0] = 50.0
model[0, 1] = model[1, 0]
model.add_assertion(model[1, 1] > 0.0)

print(model.info)

"""
__JSon Outputs (af.Array)__

An `Array` has a `dict` attribute, which express all information about the model as a Python dictionary.

By printing this dictionary we can therefore get a concise summary of the model.
"""
model = af.Array(
    shape=(2, 2),
    prior=af.GaussianPrior(mean=0.0, sigma=1.0),
)

print(model.dict())

"""
Python dictionaries can easily be saved to hard disk as a `.json` file.

This means we can save any **PyAutoFit** model to hard-disk.

Checkout the file `autofit_workspace/*/model/jsons/array.json` to see the model written as a .json.
"""
model_path = path.join("scripts", "model", "jsons")

os.makedirs(model_path, exist_ok=True)

model_file = path.join(model_path, "array.json")

with open(model_file, "w+") as f:
    json.dump(model.dict(), f, indent=4)

"""
We can load the model from its `.json` file, meaning that one can easily save a model to hard disk and load it 
elsewhere.
"""
model = af.Array.from_json(file=model_file)

print(f"\n Model via Json Prior Count = {model.prior_count}")

"""
__Extensible Models (af.Array)__

For `Model` objects, the number of parameters is fixed to those listed in the input Python class when the model is 
created.

For `Collection` objects, the use of dictionaries and lists allows for the number of parameters to be extended, but it
was still tied to the input Python classes when the model was created.

For `Array` objects, the number of parameters is fully customizable, you choose the shape of the array and therefore
the number of parameters in the model when you create it.

This makes `Array` objects the most extensible and flexible way to compose models.

You can also combine `Array` objects with `Collection` objects to create models with a mix of fixed and extensible
parameters.
"""
model = af.Collection(
    gaussian=Gaussian,
    array=af.Array(shape=(3, 2), prior=af.GaussianPrior(mean=0.0, sigma=1.0)),
)

model.gaussian.sigma = 2.0
model.array[0, 0] = 1.0

print(model.info)

"""
__Wrap Up__

This cookbook shows how to compose models consisting of multiple components using the `af.Model()` 
and `af.Collection()` object.

Advanced model composition uses multi-level models, which compose models from hierarchies of Python classes. This is
described in the multi-level model cookbook. 
"""
