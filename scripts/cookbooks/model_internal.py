"""
Cookbook: Model Internals
=========================

The model composition cookbooks show how to compose models and fit them to data using the high level **PyAutoFit** API
(e.g. ``af.Model``, ``af.Collection``, ``af.Array``).

Under the hood, **PyAutoFit** uses a recursive tree structure to map parameter vectors to Python class instances. This
cookbook exposes that internal machinery, which is useful when:

 - You need to programmatically inspect or manipulate models at a lower level than the standard API.
 - You are building custom non-linear search integrations or analysis pipelines.
 - You want to pass information between model-fitting stages (prior passing, model replacement).
 - You want to understand how **PyAutoFit** works under the hood so you can debug and extend it.

__Contents__

 - Prior Identity And Shared Priors: How linked parameters work via Python object identity.
 - Prior Tuples And Ordering: The canonical ordering of priors that defines the parameter vector layout.
 - Argument Dictionaries: Building ``{Prior: value}`` dictionaries and using ``instance_for_arguments`` directly.
 - Model Tree Navigation: Using ``paths``, ``object_for_path``, and ``path_instance_tuples_for_class`` to traverse models.
 - Creating New Models From Existing Ones: ``mapper_from_prior_arguments``, ``take_attributes``, ``from_instance``.
 - Model Subsetting: ``with_paths`` and ``without_paths`` to extract sub-models.
 - Freezing For Performance: Caching repeated lookups during fitting.
 - Serialization Round Trip: ``dict()`` and ``from_dict()`` for JSON persistence.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import numpy as np

"""
__Setup__

We define two simple model component classes to use throughout this cookbook.
"""


class Gaussian:
    def __init__(
        self,
        centre: float = 30.0,
        normalization: float = 1.0,
        sigma: float = 5.0,
    ):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma


class Exponential:
    def __init__(
        self,
        centre: float = 30.0,
        normalization: float = 1.0,
        rate: float = 0.01,
    ):
        self.centre = centre
        self.normalization = normalization
        self.rate = rate


"""
__Prior Identity And Shared Priors__

In **PyAutoFit**, every free parameter is represented by a ``Prior`` object. When you create a model, each constructor
argument gets its own ``Prior`` instance with a unique ``id``.

The key insight is that **linked parameters share the same ``Prior`` object** (same Python reference). This means the
non-linear search treats them as a single dimension.
"""
model = af.Model(Gaussian)

"""
Each parameter is a ``Prior`` with a unique id:
"""
print("centre prior id:", model.centre.id)
print("normalization prior id:", model.normalization.id)
print("sigma prior id:", model.sigma.id)

"""
Now link two parameters by assigning one to the other:
"""
model.centre = model.normalization

"""
After linking, ``centre`` and ``normalization`` are the exact same ``Prior`` object:
"""
print("\nAfter linking:")
print("centre is normalization:", model.centre is model.normalization)
print("centre prior id:", model.centre.id)
print("normalization prior id:", model.normalization.id)

"""
This reduces the dimensionality of the model by 1:
"""
print("Free parameters:", model.total_free_parameters)  # 2, not 3

"""
When the model is instantiated, both ``centre`` and ``normalization`` receive the same value from the parameter vector.

Let's verify this directly:
"""
instance = model.instance_from_vector(vector=[10.0, 5.0])
print("\ncentre =", instance.centre)
print("normalization =", instance.normalization)
print("sigma =", instance.sigma)

"""
Both ``centre`` and ``normalization`` are 10.0 because they share the same prior, and that prior was mapped to the
first element of the vector.

__Prior Tuples And Ordering__

The model maintains several views of its priors as lists of ``(name, prior)`` tuples. Understanding these is essential
for working with the internal API.

Let's create a fresh model to explore:
"""
model = af.Collection(gaussian=Gaussian, exponential=Exponential)

"""
``prior_tuples`` returns ALL ``(name, Prior)`` pairs, recursively traversing the model tree. If a prior appears in
multiple places (linked), it appears multiple times here:
"""
print("prior_tuples:")
for name, prior in model.prior_tuples:
    print(f"  {name}: id={prior.id}")

"""
``unique_prior_tuples`` deduplicates by prior identity. For an unlinked model, this is the same as ``prior_tuples``:
"""
print(f"\nunique_prior_tuples count: {len(model.unique_prior_tuples)}")

"""
``prior_tuples_ordered_by_id`` is the **canonical ordering** that defines the layout of the parameter vector. Priors
are sorted by their ``id`` attribute (an auto-incrementing integer assigned at creation time):
"""
print("\nprior_tuples_ordered_by_id (defines vector layout):")
for i, (name, prior) in enumerate(model.prior_tuples_ordered_by_id):
    print(f"  vector[{i}] -> {name} (prior id={prior.id})")

"""
The ``paths`` property gives the same ordering but as tuple-paths through the model tree:
"""
print("\npaths:")
for path in model.paths:
    print(f"  {path}")

"""
The distinction between ``direct_prior_tuples`` and ``prior_tuples`` is important:

 - ``direct_prior_tuples``: Only priors that are direct attributes of THIS model level (not nested).
 - ``prior_tuples``: All priors recursively, including those inside child Models and Collections.
"""
print(f"\nCollection direct_prior_tuples: {len(model.direct_prior_tuples)}")
print(f"Collection prior_tuples: {len(model.prior_tuples)}")
print(f"Gaussian direct_prior_tuples: {len(model.gaussian.direct_prior_tuples)}")

"""
__Argument Dictionaries__

The core of the model mapping system is the ``{Prior: value}`` dictionary, called ``arguments``. Every method
that creates an instance ultimately works through this dictionary.

You can build one manually and call ``instance_for_arguments`` directly. This is useful when you need fine-grained
control over how instances are created.
"""
model = af.Model(Gaussian)

"""
Step 1: Get the priors. Each prior object is a key in the arguments dictionary.
"""
centre_prior = model.centre
normalization_prior = model.normalization
sigma_prior = model.sigma

"""
Step 2: Build the arguments dictionary mapping each Prior to a physical value.
"""
arguments = {
    centre_prior: 50.0,
    normalization_prior: 3.0,
    sigma_prior: 10.0,
}

"""
Step 3: Call ``instance_for_arguments`` to create the instance.
"""
instance = model.instance_for_arguments(arguments)

print("Instance from arguments:")
print(f"  centre = {instance.centre}")
print(f"  normalization = {instance.normalization}")
print(f"  sigma = {instance.sigma}")

"""
This is exactly what ``instance_from_vector`` does internally. Here is the equivalent using a vector:
"""
instance_from_vec = model.instance_from_vector(vector=[50.0, 3.0, 10.0])

"""
For a ``Collection``, the same arguments dictionary is shared across all child models. Each child
picks out its own priors from the shared dictionary:
"""
model = af.Collection(gaussian=Gaussian, exponential=Exponential)

arguments = {}
for name, prior in model.prior_tuples_ordered_by_id:
    arguments[prior] = 1.0  # Set all parameters to 1.0

instance = model.instance_for_arguments(arguments)

print("\nCollection instance (all params = 1.0):")
print(f"  gaussian.centre = {instance.gaussian.centre}")
print(f"  exponential.rate = {instance.exponential.rate}")

"""
__Unit Vector Mapping__

Non-linear searches work in "unit space" where each parameter ranges from 0 to 1. The ``instance_from_unit_vector``
method transforms unit values to physical values via each prior's ``value_for`` method.
"""
model = af.Model(Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

"""
A unit vector of [0.5, 0.5, 0.5] maps to the midpoint of each prior:
"""
instance = model.instance_from_unit_vector(unit_vector=[0.5, 0.5, 0.5])

print("Instance from unit vector [0.5, 0.5, 0.5]:")
print(f"  centre = {instance.centre}")  # 50.0
print(f"  normalization = {instance.normalization}")  # 5.0
print(f"  sigma = {instance.sigma}")  # 25.0

"""
You can also convert a unit vector to a physical vector without creating an instance:
"""
physical_vector = model.vector_from_unit_vector(unit_vector=[0.0, 1.0, 0.5])
print(f"\nPhysical vector: {physical_vector}")  # [0.0, 10.0, 25.0]

"""
__Model Tree Navigation__

Models form a tree. **PyAutoFit** provides several methods to navigate this tree.

``object_for_path`` retrieves any object in the model by its tuple-path:
"""
model = af.Collection(gaussian=af.Model(Gaussian), exponential=af.Model(Exponential))

"""
Retrieve a child model:
"""
gaussian_model = model.object_for_path(("gaussian",))
print(f"Object at ('gaussian',): {gaussian_model}")

"""
Retrieve a specific prior:
"""
centre_prior = model.object_for_path(("gaussian", "centre"))
print(f"Object at ('gaussian', 'centre'): {centre_prior}")

"""
``path_instance_tuples_for_class`` finds all objects of a given type in the tree, returning their paths:
"""
from autofit.mapper.prior.abstract import Prior

print("\nAll Prior locations:")
for path, prior in model.path_instance_tuples_for_class(Prior):
    print(f"  path={path}, prior id={prior.id}")

"""
``path_for_prior`` and ``name_for_prior`` go the other direction, finding where a prior lives:
"""
prior = model.gaussian.centre
path = model.path_for_prior(prior)
name = model.name_for_prior(prior)
print(f"\nPath for gaussian.centre prior: {path}")
print(f"Name for gaussian.centre prior: {name}")

"""
__Creating New Models From Existing Ones__

A common workflow is to take the results of one fit and use them to initialize the next. **PyAutoFit** provides
several methods for this.

``mapper_from_prior_arguments`` creates a new model where specified priors are replaced:
"""
model = af.Model(Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

"""
``mapper_from_partial_prior_arguments`` creates a new model where specified priors are replaced and
all other priors are kept unchanged:
"""
new_centre_prior = af.GaussianPrior(mean=50.0, sigma=5.0)
new_model = model.mapper_from_partial_prior_arguments({model.centre: new_centre_prior})

print("Original centre prior:", model.centre)
print("New centre prior:", new_model.centre)
print("Normalization unchanged:", type(new_model.normalization).__name__)
print(f"New model prior count: {new_model.prior_count}")

"""
``mapper_from_prior_arguments`` requires ALL priors to be mapped. It is the lower-level version:
"""
new_model = model.mapper_from_prior_arguments(
    {
        model.centre: af.GaussianPrior(mean=50.0, sigma=2.0),
        model.normalization: model.normalization,  # Keep unchanged
        model.sigma: model.sigma,  # Keep unchanged
    }
)
print(f"\nFull replacement prior count: {new_model.prior_count}")

"""
``take_attributes`` copies matching attributes from another model or instance. This is the mechanism
behind "prior passing" between search stages:
"""
source_instance = Gaussian(centre=50.0, normalization=3.0, sigma=10.0)

new_model = af.Model(Gaussian)
new_model.take_attributes(source_instance)

print("\nAfter take_attributes from instance:")
print(f"  centre = {new_model.centre}")  # Now fixed at 50.0
print(f"  normalization = {new_model.normalization}")  # Now fixed at 3.0
print(f"  sigma = {new_model.sigma}")  # Now fixed at 10.0
print(f"  Free parameters: {new_model.total_free_parameters}")  # 0

"""
``from_instance`` converts a concrete instance back into a prior model. Parameters become fixed values
unless the instance type is in ``model_classes``, in which case they become free priors:
"""
instance = Gaussian(centre=50.0, normalization=3.0, sigma=10.0)

"""
All parameters fixed (instance conversion):
"""
fixed_model = af.AbstractPriorModel.from_instance(instance)
print(f"\nfrom_instance (no model_classes): prior_count = {fixed_model.prior_count}")

"""
With model_classes, matching types get free parameters:
"""
free_model = af.AbstractPriorModel.from_instance(instance, model_classes=(Gaussian,))
print(f"from_instance (Gaussian as model_class): prior_count = {free_model.prior_count}")

"""
__Model Subsetting__

You can create sub-models that contain only a subset of the original model's parameters using ``with_paths`` and
``without_paths``.
"""
model = af.Collection(gaussian=Gaussian, exponential=Exponential)

"""
Keep only the gaussian component:
"""
gaussian_only = model.with_paths([("gaussian",)])
print(f"\ngaussian_only prior count: {gaussian_only.prior_count}")

"""
Remove the gaussian component, keeping everything else:
"""
without_gaussian = model.without_paths([("gaussian",)])
print(f"without_gaussian prior count: {without_gaussian.prior_count}")

"""
You can also subset at the parameter level:
"""
centre_only = model.with_paths([("gaussian", "centre"), ("exponential", "centre")])
print(f"centres only prior count: {centre_only.prior_count}")

"""
__Freezing For Performance__

During a non-linear search, the model structure does not change between iterations but properties
like ``prior_tuples_ordered_by_id`` are recomputed each time. Freezing a model caches these lookups.
"""
model = af.Collection(gaussian=Gaussian, exponential=Exponential)

"""
Freeze the model (caches results of decorated methods):
"""
model.freeze()

"""
Repeated calls now return cached results:
"""
_ = model.prior_tuples_ordered_by_id  # Computed
_ = model.prior_tuples_ordered_by_id  # Cached

"""
A frozen model cannot be modified:
"""
try:
    model.gaussian.centre = 1.0
except AssertionError as e:
    print(f"\nCannot modify frozen model: {e}")

"""
Unfreeze to allow modification again:
"""
model.unfreeze()
model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
print(f"After unfreeze and modification, prior count: {model.prior_count}")

"""
__Serialization Round Trip__

Models can be serialized to dictionaries and JSON for persistence. The ``dict()`` method produces a
Python dictionary, and ``from_dict()`` reconstructs the model.
"""
import json
from os import path
import os

model = af.Collection(gaussian=Gaussian, exponential=Exponential)
model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.normalization = 5.0  # Fix this parameter

"""
Convert to a dictionary:
"""
model_dict = model.dict()
print("\nModel dict keys:", list(model_dict.keys()))

"""
Save to JSON:
"""
json_path = path.join("scripts", "cookbooks", "jsons")
os.makedirs(json_path, exist_ok=True)

json_file = path.join(json_path, "model_internal.json")
with open(json_file, "w+") as f:
    json.dump(model_dict, f, indent=4)

"""
Load from JSON:
"""
loaded_model = af.AbstractPriorModel.from_json(file=json_file)
print(f"Loaded model prior count: {loaded_model.prior_count}")

"""
Linked priors are preserved through serialization. If two parameters shared the same prior before saving,
they will share the same prior after loading.
"""

"""
__Log Prior Computation__

During MCMC sampling (e.g. with emcee), the log prior probability for a parameter vector is needed. The model
provides this directly:
"""
model = af.Model(Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

log_priors = model.log_prior_list_from_vector(vector=[50.0, 5.0, 25.0])
print(f"\nLog priors for [50.0, 5.0, 25.0]: {log_priors}")

"""
Values outside the prior bounds return -inf:
"""
log_priors_oob = model.log_prior_list_from_vector(vector=[150.0, 5.0, 25.0])
print(f"Log priors for [150.0, 5.0, 25.0]: {log_priors_oob}")

"""
__Instance From Path And Name Arguments__

Instead of building a ``{Prior: value}`` dictionary or passing a flat vector, you can create instances using
path-based or name-based arguments. This is useful for programmatic workflows:
"""
model = af.Collection(gaussian=Gaussian, exponential=Exponential)

"""
Using path arguments (tuple paths):
"""
instance = model.instance_from_path_arguments(
    {
        ("gaussian", "centre"): 50.0,
        ("gaussian", "normalization"): 3.0,
        ("gaussian", "sigma"): 10.0,
        ("exponential", "centre"): 60.0,
        ("exponential", "normalization"): 2.0,
        ("exponential", "rate"): 0.5,
    }
)

print(f"\nPath arguments instance:")
print(f"  gaussian.centre = {instance.gaussian.centre}")
print(f"  exponential.rate = {instance.exponential.rate}")

"""
Using name arguments (underscore-joined names):
"""
instance = model.instance_from_prior_name_arguments(
    {
        "gaussian_centre": 50.0,
        "gaussian_normalization": 3.0,
        "gaussian_sigma": 10.0,
        "exponential_centre": 60.0,
        "exponential_normalization": 2.0,
        "exponential_rate": 0.5,
    }
)

print(f"\nName arguments instance:")
print(f"  gaussian.centre = {instance.gaussian.centre}")
print(f"  exponential.rate = {instance.exponential.rate}")

"""
__Assertions__

Assertions constrain the parameter space without reducing dimensionality. They are checked during instance creation
and cause a ``FitException`` if violated, prompting the non-linear search to resample.
"""
model = af.Model(Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)

"""
Add an assertion that sigma must be less than centre:
"""
model.add_assertion(model.sigma < model.centre, name="sigma_less_than_centre")

"""
Check the parameter ordering so we know which vector index maps to which parameter:
"""
print("\nParameter ordering:")
for i, path in enumerate(model.paths):
    print(f"  vector[{i}] -> {path}")

"""
This works when the assertion holds (centre > sigma):
"""
from autofit import exc

try:
    instance = model.instance_from_vector(vector=[50.0, 3.0, 10.0])
    print(f"\nValid instance: centre={instance.centre}, sigma={instance.sigma}")
except exc.FitException:
    print("Unexpected assertion failure - check parameter ordering")

"""
And raises ``FitException`` when it does not (sigma > centre):
"""
try:
    instance = model.instance_from_vector(vector=[10.0, 3.0, 50.0])
except exc.FitException as e:
    print(f"Assertion failed (as expected): {e}")

"""
__Wrap Up__

This cookbook covered the internal model mapping machinery of **PyAutoFit**:

 - How prior identity enables linked parameters.
 - The canonical prior ordering that defines the parameter vector layout.
 - Building and using ``{Prior: value}`` argument dictionaries directly.
 - Navigating the model tree with paths and type queries.
 - Creating new models from existing ones for multi-stage pipelines.
 - Subsetting, freezing, and serializing models.

These tools give you fine-grained control over model composition and are the foundation that the high-level API
is built upon. They are particularly useful when building custom search integrations, analysis pipelines, or
debugging model-fitting workflows.
"""
