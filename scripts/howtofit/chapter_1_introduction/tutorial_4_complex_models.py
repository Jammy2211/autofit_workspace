"""
Tutorial 4: Complex Models
==========================

In this tutorial, we will fix more complex models with N=10, N=20 and more parameters. We will consider the following:

 - Why more complex model are more difficult to fit, and may lead the non-linear search to incorrectly infer
   models with significantly lower likelihoods than the true maximum likelihood model.

 - Strategies for ensuring the non-linear search correctly estimates the maximum likelihood model.

 - What drives the run-times of a model-fit, and how one must carefully balance run-times with model complexity.
for mitigating this:

WHAT I NEED TO WRITE:

- Example which fits an N=15 model and gets an incorrect result, concepts like "local maxima", model complexity,
using composition API to simplify model, etc, using priors to do this.

- Sections on run times.

- Sections on non-linear search settings.

Can rewrite and borrow from HowToLens.

In this example, every fit to the noisy 1D signal was a good fit, based on the fit looking visually close to the data.

For modeling in general, however, things are not always so simple. It is common for the model-fit to provide a bad fit to the data.
Furthermore, it can be difficult to determine if this is because the model is genuinely a poor fit or because the non-linear search (e.g. `emcee`)
failed sample parameter space robustly enough to locate the highest likelihood regions of parameter space. The next session will illustrate an example of this.

When a non-linear search infers a lower likelihood solution than the highest likelihood solutions that exist in the parameter space, called
the "global maximum likelihood", it is said to have become trapped by a "local maximum". There is no simple way to determine if a non-linear has
done this. The process typically involves visually inspecting the results, fitting the model many times (ideally with different models, non-linear searches and settings) and building up intuition for your modeling problem as to how things behave and when they work / do not work.

Owing to the model-specific nature of this problem, these lectures will only briefly illustrate model-fitting failures and how one might overcome them.
If you embark on your own model-fitting endeavours, this will be the aspect of model-fitting you will have to learn about yourself!
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

"""
__Data__

We first load the dataset we will fit, which is a new `dataset` where the underlying signal is a sum of two  `Gaussian` 
profiles which share the same centre
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x2")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
Plotting the data shows the noisy signal is more complicated than just a 1D Gaussian.

Note that both Gaussians are centred at the same point (x = 50). We will compose a model that reflects this.
"""
xvalues = np.arange(data.shape[0])
plt.errorbar(
    xvalues, data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.title("1D Gaussian dataset with errors from the noise-map.")
plt.xlabel("x values of profile")
plt.ylabel("Signal Value")
plt.show()

"""
__Models__

We create the `Gaussian` class which will form our model components using the standard **PyAutoFit** format.
"""


class Gaussian:
    def __init__(
        self,
        centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization=1.0,  # <- are the Gaussian`s model parameters.
        sigma=5.0,
    ):
        """
        Represents a 1D Gaussian profile.

        This is a model-component of example models in the **HowToFit** lectures and is used to fit example datasets
        via a non-linear search.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization of the profile.
        sigma
            The sigma value controlling the size of the Gaussian.
        """
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    def model_data_1d_via_xvalues_from(self, xvalues: np.ndarray):
        """

        Returns a 1D Gaussian on an input list of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, via its `centre`.

        The output is referred to as the `model_data` to signify that it is a representation of the data from the
        model.

        Parameters
        ----------
        xvalues
            The x coordinates in the original reference frame of the data.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return np.multiply(
            np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )


"""
__Analysis__

We now define the  `Analysis` class for this model-fit. 

The `log_likelihood_function` of this analysis now assumes that the `instance` that is input into it will contain
multiple 1D profiles.
 
 The way the `model_data` is computed is updating accordingly (the sum of each individual Gaussian's `model_data`).
"""


class Analysis(af.Analysis):
    def __init__(self, data, noise_map):
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        """
        Returns the log likelihood of the fit of an `instance` containing many 1D
        Profiles (e.g. Gaussians) to the dataset, using a model instance.

        Parameters
        ----------
        instance
            A list of 1D profiles with parameters set via the non-linear search.

        Returns
        -------
        float
            The log likelihood value indicating how well this model fit the `MaskedDataset`.
        """

        """
        In the previous tutorial the instance was a single `Gaussian` profile, meaning we could create the model data 
        using the line:

            model_data = instance.gaussian.model_data_1d_via_xvalues_from(xvalues=self.data.xvalues)

        In this tutorial our instance is comprised of multiple 1D Gaussians, because we will use a `Collection` to
        compose the model:

            model = Collection(gaussian_0=Gaussian, gaussian_1=Gaussian).

        By using a Collection, this means the instance parameter input into the fit function is a
        dictionary where individual profiles (and their parameters) can be accessed as followed:

            print(instance.gaussian_0)
            print(instance.gaussian_1)
            print(instance.gaussian_0.centre)

        In this tutorial, the `model_data` is therefore the summed `model_data` of all individual Gaussians in the 
        model. The function `model_data_from_instance` performs this summation. 
        """
        model_data = self.model_data_from_instance(instance=instance)

        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

    def model_data_from_instance(self, instance):
        """
        To create the summed profile of all individual profiles, we use a list comprehension to iterate over
        all profiles in the instance.

        The key point to understand is that the `instance` has the properties of a Python `iterator` and therefore
        can be looped over using the standard Python for syntax (e.g. `for profile in instance`).

        __Alternative Syntax__

        For those not familiar with list comprehensions, the code below shows how to use the instance to create the
        summed profile using a more simple for loop.

        model_data = np.zeros(shape=self.data.xvalues.shape[0])

        for profile in instance:
            model_data += profile.model_data_1d_via_xvalues_from(xvalues=self.data.xvalues)

        return model_data
        """
        xvalues = np.arange(self.data.shape[0])

        return sum(
            [
                profile.model_data_1d_via_xvalues_from(xvalues=xvalues)
                for profile in instance
            ]
        )


"""
__Collection__

Use a `Collection` to compose the model we fit, consisting of two `Gaussian`'s.
"""
model = af.Collection(gaussian_0=Gaussian, gaussian_1=Gaussian)

"""
__Model Customization__

We can fully customize the model that we fit. 

First, lets align the centres of the two `Gaussian`'s (given we know they are aligned in the data). Note that
doing so reduces the number of free parameters in the model by 1, from N=6 to N=5.

Lets suppose we have a `dataset` that consists of three `Gaussian` 
profiles, but we also know the following information about the dataset:

- The 2 `Gaussian`'s are centrally aligned.
- The `sigma` of one `Gaussian` is equal to 1.0.
- The sigma of another `Gaussian` is above 3.0.

We can edit the `Model` components we pass into the `Collection` to meet these constraints accordingly.

Lets first create the model `Gaussian`'s as we did in the previous tutorial.
"""
gaussian_0 = af.Model(Gaussian)
gaussian_1 = af.Model(Gaussian)

"""
We can centrally align the two `Gaussian`'s by setting the `centre` of the first `Gaussian` to the `centre` of the
second `Gaussian`.

This removes a free parameter from the model reducing the dimensionality by 1 (from N=6 to N=5).
"""
gaussian_0.centre = gaussian_1.centre

"""
We can follow the same API to set the `sigma` of the first `Gaussian` to 1.0.

This again removes another free parameter from the model (from N=5 to N=4).
"""
gaussian_0.sigma = 1.0

"""
We can add assertions, for example requiring that  the `sigma` value of the second `Gaussian` is above 2.0.

Assertions do not change the dimensionality of the model, because we are not fixing or removing any free parameters.
"""
gaussian_1.add_assertion(gaussian_1.sigma > 3.0)

"""
We again input these newly customized model components into the `Collection`.
"""
model = af.Collection(
    gaussian_0=gaussian_0,
    gaussian_1=gaussian_1,
)

"""
The customized model can be printed via the `info` attribute, where the customizes discussed above can be seen.
"""
print(model.info)

"""
__Model Fit__

Lets now perform the fit using our model which is composed of two profile's in a non-linear parameter space of
dimensionality N=4.
"""
analysis = Analysis(data=data, noise_map=noise_map)

search = af.Emcee()

print(
    "Emcee has begun running. \n"
    "Checkout the autofit_workspace/output/howtofit/tutorial_5__gaussian_x1__exponential_x1 \n"
    "folder for live output of the results.\n"
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
)

result = search.fit(model=model, analysis=analysis)

print("Emcee has finished run - you may now continue the notebook.")

"""
__Result__

The `info` attribute shows the result in a readable format, which contains informaiton on the full collection
of model components.
"""
print(result.info)

"""
__Cookbooks__

This tutorial illustrates how to compose model out of multiple components, using a `Collection`.

**PyAutoFit** has many advanced model composition tools, which offer more customization of `Collection` objects,
allow models to be composed and fitted to multiple datasets and for multi-level models to be created out of
hierarchies of Python classes.

Checkout the `autofit_workspace/*/model` package for these cookbooks with give a full run through of all of
**PyAutoFit**'s model composition tools, or read them on the readthedocs:

 - `cookbook 1: Basics  <https://pyautofit.readthedocs.io/en/latest/cookbooks/cookbook_1_basics.html>`_

 - `cookbook 2: Collections  <https://pyautofit.readthedocs.io/en/latest/cookbooks/cookbook_2_collections.html>`_

__Wrap Up__

And with that, we are complete. In this tutorial, we learned how to compose and fit complex models in **PyAutoFit**.
 
To end, you should think again in more detail about your model fitting problem:

 Are there many different model components you may wish to define and fit?

 Is your data the super position of many different model components, like the profiles in this tutorial?

 In this tutorial, all components of our model did the same thing, represent a 1D profile. In your model, you may
have model components that represent different parts of your model, which need to be combined in more complicated ways
in order to create your model-fit. You now have all the tools you need to define, compose and fit very complex models!
"""
