"""
Tutorial 5: Complex Models
==========================

Up to now, we've fitted a very simple model, a 1D `Gaussian` with 3 free parameters. In this tutorial, we'll look at
how **PyAutoFit** allows us to compose and fit models of arbitrary complexity.

To begin, you should check out the module `autofit_workspace/howtofit/chapter_1_introduction/profiles.py`.

In previous tutorials we used the module `gaussian.py` which contained only the `Gaussian` class. The `profiles.py`
includes a second profile, `Exponential`, which like the `Gaussian` class is a model-component that can be fitted to
data.

Up to now, our data has always been generated using a single `Gaussian` profile. Thus, we have only needed to fit
it with a single `Gaussian`. In this tutorial, our `dataset` is now a superpositions of multiple profiles. The models
we compose and fit are therefore composed of multiple profiles, such that when we generate the model-data we
generate it as the sum of all individual profiles in our model.
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

Lets quickly recap tutorial 1, where using `Models` we created a `Gaussian` as a model component and used it to 
map a list of parameters to a model `instance`.
"""
import profiles as p

model = af.Model(p.Gaussian)

print("Model `Gaussian` object: \n")
print(model)

instance = model.instance_from_vector(vector=[0.1, 0.2, 0.3])

print("Model Instance: \n")
print(instance)

print("Instance Parameters \n")
print("x = ", instance.centre)
print("normalization = ", instance.normalization)
print("sigma = ", instance.sigma)

print()
print(model.info)

"""
__Collection__

Defining a model using multiple model components is straight forward in **PyAutoFit**, using a `Collection`
object.
"""
model = af.Collection(
    gaussian=af.Model(p.Gaussian), exponential=af.Model(p.Exponential)
)

"""
A `Collection` behaves like a `Model` but contains a collection of model components. For example, it
creates a model instance by mapping a list of parameters, which in this case is 6 (3 for the `Gaussian` (centre,
normalization, sigma) and 3 for the `Exponential` (centre, normalization, rate)).
"""
instance = model.instance_from_vector(vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.01])

"""
This `instance` contains each of the model components we defined above, using the input argument name of the
`Collection` to define the attributes in the `instance`:
"""
print("Instance Parameters \n")
print("x (Gaussian) = ", instance.gaussian.centre)
print("normalization (Gaussian) = ", instance.gaussian.normalization)
print("sigma (Gaussian) = ", instance.gaussian.sigma)
print("x (Exponential) = ", instance.exponential.centre)
print("normalization (Exponential) = ", instance.exponential.normalization)
print("sigma (Exponential) = ", instance.exponential.rate)

"""
All of the information about the collection can be printed at once using its `info` attribute:
"""
print(model.info)

"""
We can call the components of a `Collection` whatever we want, and the mapped `instance` will use those names.
"""
model_custom_names = af.Collection(
    custom_name=af.Model(p.Gaussian), another_custom_name=af.Model(p.Exponential)
)

instance = model_custom_names.instance_from_vector(
    vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.01]
)

print("Instance Parameters \n")
print("x (Gaussian) = ", instance.custom_name.centre)
print("normalization (Gaussian) = ", instance.custom_name.normalization)
print("sigma (Gaussian) = ", instance.custom_name.sigma)
print("x (Exponential) = ", instance.another_custom_name.centre)
print("normalization (Exponential) = ", instance.another_custom_name.normalization)
print("sigma (Exponential) = ", instance.another_custom_name.rate)

"""
These names are also seen in the `info` attribute:
"""
print(model.info)

"""
__Plot Function__

To perform visualization we'll again use the plot_profile_1d function.
"""


def plot_profile_1d(
    xvalues,
    profile_1d,
    title=None,
    ylabel=None,
    errors=None,
    color="k",
    output_path=None,
    output_filename=None,
):
    """
    Plot 1D data on a plot of x versus y, where the x-axis is the x coordinate of the profile and the y-axis
    is the normalization of the profile at that coordinate.

    The function include options to output the image to the hard-disk as a .png.

    Parameters
    ----------
    xvalues
        The x-coordinates the profile is defined on.
    profile_1d
        The normalization values of the profile which are plotted.
    ylabel
        The y-label of the plot.
    output_path
        The path the image is to be output to hard-disk as a .png.
    output_filename
        The filename of the file if it is output as a .png.
    output_format
        Determines where the plot is displayed on your screen ("show") or output to the hard-disk as a png ("png").
    """
    plt.errorbar(
        x=xvalues,
        y=profile_1d,
        yerr=errors,
        color=color,
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.title(title)
    plt.xlabel("x value of profile")
    plt.ylabel(ylabel)
    if not path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(path.join(output_path, f"{output_filename}.png"))
    plt.clf()


"""
__Analysis__

Now we can create a model composed of multiple components we need to fit it to our data. To do this, we use updated 
`Analysis` class that creates the `model_data` as a super position of all of the model's individual `Profile`'s. For 
example, in the model above, the `model_data` is the sum of the `Gaussian`'s  individual profile and `Exponential`'s 
individual profile.
"""


class Analysis(af.Analysis):
    def __init__(self, data, noise_map):
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        """
        Returns the log likelihood of a list of Profiles (Gaussians, Exponentials, etc.) to the dataset, using a
        model instance.

        Parameters
        ----------
        instance
            The list of Profile model instance (e.g. the Gaussians, Exponentials, etc.).

        Returns
        -------
        float
            The log likelihood value indicating how well this model fit the `MaskedDataset`.

        In tutorials 3 & 4, the instance was an instance of a single `Gaussian` profile. PyAutoFit knew this instance
        would contain just one Gaussian, because when the model was created we used a Model object in PyAutoFit
        to make the Gaussian. This meant we could create the model data using the line:

            model_data = instance.gaussian.model_data_1d_via_xvalues_from(xvalues=self.masked_dataset.xvalues)

        In this tutorial our instance is comprised of multiple Profile objects, because we used a Collection:

            model = Collection(gaussian=profiles.Gaussian, exponential=profiles.Exponential).

        By using a Collection, this means the instance parameter input into the fit function is a
        dictionary where individual profiles (and their parameters) can be accessed as followed:

            print(instance.gaussian)
            print(instance.exponential)
            print(instance.exponential.centre)

        The names of the attributes of the instance correspond to what we input into the Collection. Lets
        look at a second example:

        model = Collection(
                      gaussian_0=af.Model(profiles.Gaussian),
                      gaussian_1=af.Model(profiles.Gaussian),
                      whatever_i_want=af.Model(profiles.Exponential)
                 ).

        print(instance.gaussian_0)
        print(instance.gaussian_1)
        print(instance.whatever_i_want.centre)

        A Collection allows us to name our model components whatever we want!

        In this tutorial, we want our `fit` function to fit the data with a profile which is the summed profile
        of all individual profiles in the model. Look at `model_data_from_instance` to see how we do this.
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
        To create the summed profile of all individual profiles in an instance, we can use a dictionary comprehension
        to iterate over all profiles in the instance.
        """
        xvalues = np.arange(self.data.shape[0])

        return sum(
            [
                profile.model_data_1d_via_xvalues_from(xvalues=xvalues)
                for profile in instance
            ]
        )

        """
        For those not familiar with dictionary comprehensions, below I've included how one would use the instance to 
        create the summed profile using a more simple for loop.

        model_data = np.zeros(shape=self.masked_dataset.xvalues.shape[0])

        for profile in instance:
            model_data += profile.model_data_1d_via_xvalues_from(xvalues=self.masked_dataset.xvalues)

        return model_data
        """

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
        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=self.data,
            title="Data",
            ylabel="Data Values",
            color="k",
            output_path=paths.image_path,
            output_filename="data",
        )

        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=model_data,
            title="Model Data",
            ylabel="Model Data Values",
            color="k",
            output_path=paths.image_path,
            output_filename="model_data",
        )

        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=residual_map,
            title="Residual Map",
            ylabel="Residuals",
            color="k",
            output_path=paths.image_path,
            output_filename="residual_map",
        )

        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=chi_squared_map,
            title="Chi-Squared Map",
            ylabel="Chi-Squareds",
            color="k",
            output_path=paths.image_path,
            output_filename="chi_squared_map",
        )


"""
__Data (Complex)__

Load the dataset from the `autofit_workspace/dataset` folder. This uses a new `dataset` that is a sum of a 
`Gaussian` and `Exponential` profile.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1__exponential_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
__Model Fit__

Lets now perform the fit using our model which is composed of two profile's. You'll note that the `Emcee`
dimensionality has increased from N=3 to N=6, given that we are now fitting two `Profile`'s each with 3 free parameters.
"""
analysis = Analysis(data=data, noise_map=noise_map)

search = af.Emcee(
    path_prefix=path.join("howtofit", "chapter_1"),
    name="tutorial_5__gaussian_x1__exponential_x1",
)

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

The `info` attribute shows the result in a readable format.
"""
print(result.info)

"""
Inspect the results of the fit by going to the folder 

`autofit_workspace/output/howtofit/tutorial_5__gaussian_x1__exponential_x1`. The fit takes longer to run than 
the fits performed in previous tutorials, because the dimensionality of the model we fit increases from 3 to 6.

__Triple Profile Fit__

With the `Collection`, **PyAutoFit** provides all the tools needed to compose and fit any model imaginable!
Lets fit a model composed of two `Gaussian`. and and an `Exponential`, which will have a dimensionality of N=9.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x2__exponential_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

analysis = Analysis(data=data, noise_map=noise_map)

model = af.Collection(
    gaussian_0=af.Model(p.Gaussian),
    gaussian_1=af.Model(p.Gaussian),
    exponential=af.Model(p.Exponential),
)

search = af.Emcee(
    name="tutorial_5__gaussian_x2__exponential_x1",
    path_prefix=path.join("howtofit", "chapter_1"),
)

print(
    "Emcee has begun running.\n"
    "checkout the autofit_workspace/output/howtofit/tutorial_5__gaussian_x2__exponential_x1\n"
    " folder for live output of the results.\n"
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
)

result = search.fit(model=model, analysis=analysis)

print("Emcee has finished run - you may now continue the notebook.")

"""
__Model Customization__

We can fully customize the model that we fit. Lets suppose we have a dataset` that consists of three `Gaussian` 
profiles, but we also know the following information about the dataset:

- All 3 `Gaussian`'s are centrally aligned.
- The `sigma` of one `Gaussian` is equal to 1.0.
- The sigma of another `Gaussian` is above 3.0.

We can edit the `Model` components we pass into the `Collection` to meet these constraints accordingly.
"""
gaussian_0 = af.Model(p.Gaussian)
gaussian_1 = af.Model(p.Gaussian)
gaussian_2 = af.Model(p.Gaussian)

"""
This aligns the `centre`'s of the 3 `Gaussian`'s reducing the dimensionality of the model from N=9 to N=7.
"""
gaussian_0.centre = gaussian_1.centre
gaussian_1.centre = gaussian_2.centre

"""
This fixes the `sigma` value of one `Gaussian` to 1.0, further reducing the dimensionality from N=7 to N=6.
"""
gaussian_0.sigma = 1.0

"""
This assertion forces all values of the `sigma` value of the third `Gaussian` to  be above 3.0.
"""
gaussian_2.add_assertion(gaussian_2.sigma > 3.0)

"""
We now input these model components into the `Collection`.
"""
model = af.Collection(
    gaussian_0=gaussian_0, gaussian_1=gaussian_1, gaussian_2=gaussian_2
)

"""
We can now fit this model as per usual.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x3")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

analysis = Analysis(data=data, noise_map=noise_map)

search = af.Emcee(
    path_prefix=path.join("howtofit", "chapter_1"), name="tutorial_5__gaussian_x3"
)

print(
    "Emcee has begun running. "
    "Checkout the autofit_workspace/output/howtofit/tutorial_5__gaussian_x3"
    " folder for live output of the results."
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
)

result = search.fit(model=model, analysis=analysis)

print("Emcee has finished run - you may now continue the notebook.")

"""
We can again quickly inspect the results.
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
