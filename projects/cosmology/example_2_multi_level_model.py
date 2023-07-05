"""
Project: Cosmology
==================

This project uses the astrophysical phenomena of Strong Gravitational Lensing to illustrate basic and advanced model
composition and fitting with **PyAutoFit**. The first tutorial described what a strong gravitational lens is and how
we build and fit a model of one.

In this example, we use **PyAutoFit**'s multi-level models to compose a strong lens model consisting of a lens and
source galaxy, and fit it to the data on SDSSJ2303+1422.

__Config Path__

We first set up the path to this projects config files, which is located at `autofit_workspace/projects/cosmology/config`.

This includes the default priors for the lens model, check it out!
"""
import os
from os import path
from autoconf import conf

cwd = os.getcwd()
config_path = path.join(cwd, "projects", "cosmology", "config")
conf.instance.push(new_path=config_path)

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import src as cosmo
import matplotlib.pyplot as plt
import numpy as np

"""
__Plot__

First, lets again define the plotting convenience functions we used in the previous example.
"""


def plot_array(array, title=None, norm=None):
    plt.imshow(array, norm=norm)
    plt.colorbar()
    plt.title(title)
    plt.show()
    plt.close()


def plot_grid(grid, title=None):
    plt.scatter(x=grid[:, :, 0], y=grid[:, :, 1], s=1)
    plt.title(title)
    plt.show()
    plt.close()


"""
__Data__

Now lets load and plot Hubble Space Telescope imaging data of the strong gravitational lens SDSSJ2303+1422.
"""
dataset_path = path.join("projects", "cosmology", "dataset")

data = np.load(file=path.join(dataset_path, "data.npy"))
plot_array(array=data, title="Image of Strong Lens SDSSJ2303+1422")

noise_map = np.load(file=path.join(dataset_path, "noise_map.npy"))
plot_array(array=noise_map, title="Noise Map of Strong Lens SDSSJ2303+1422")

psf = np.load(file=path.join(dataset_path, "psf.npy"))
plot_array(array=psf, title="Point Spread Function of Strong Lens SDSSJ2303+1422")

grid = np.load(file=path.join(dataset_path, "grid.npy"))

plot_grid(
    grid=grid,
    title="Cartesian grid of (x,y) coordinates aligned with strong lens dataset",
)

"""
__Multi-level Model__

In the previous example, we saw that we can use instances of the light profiles, mass profiles and galaxy objects to
perform strong lens ray-tracing calculations:
"""
light_profile = cosmo.lp.LightDeVaucouleurs(
    centre=(0.01, 0.01), axis_ratio=0.7, angle=45.0, intensity=1.0, effective_radius=2.0
)
mass_profile = cosmo.mp.MassIsothermal(
    centre=(0.01, 0.01), axis_ratio=0.7, angle=45.0, mass=0.5
)
galaxy = cosmo.Galaxy(
    redshift=0.5, light_profile_list=[light_profile], mass_profile_list=[mass_profile]
)

"""
In this example, we want to perform a model-fit using a non-linear search, where the `Galaxy` is a `Model`, but it
contains model subcomponents that are its individual light and mass profiles. 

Here is a pictoral representation of the model:

![Strong Lens Model](https://github.com/rhayes777/PyAutoFit/blob/main/docs/overview/image/lens_model.png?raw=true "cluster")

__Model Composition__

How do we compose a strong lens model where a `Galaxy` is a `Model`, but it contains the light and mass profiles 
as `Model` themselves?

We use **PyAutoFit**'s multi-level model composition:
"""
lens = af.Model(
    cls=cosmo.Galaxy,  # The overall model object uses this input.
    redshift=0.5,
    light_profile_list=[
        af.Model(cosmo.lp.LightDeVaucouleurs)
    ],  # These will be subcomponents of the model.
    mass_profile_list=[
        af.Model(cosmo.mp.MassIsothermal)
    ],  # These will be subcomponents of the model.
)

print(lens.info)

"""
Lets consider what is going on here:

 1) We use a `Model` to create the overall model component. The `cls` input is the `Galaxy` class, therefore the 
    overall model that is created is a `Galaxy`.
  
 2) **PyAutoFit** next inspects whether the key word argument inputs to the `Model` match any of the `__init__` 
    constructor arguments of the `Galaxy` class. This determine if these inputs are to be composed as model 
    subcomponents of the overall `Galaxy` model. 

 3) **PyAutoFit** matches the `light_profile_list` and  `mass_profile_list` inputs, noting they are passed as separate 
    lists  containing the `LightDeVaucouleurs` and `MassIsothermal` class. They are both created as subcomponents of 
    the overall `Galaxy` model.
  
 4) It also matches the `redshift` input, making it a fixed value of 0.5 for the model and not treating it as a 
    free parameter.
 
We can confirm this by printing the `total_free_parameters` of the lens, and noting it is 11 (6 parameters for 
the `LightDeVaucouleurs` and 5 for the `MassIsothermal`).
"""
print(lens.total_free_parameters)
print(lens.light_profile_list[0].total_free_parameters)
print(lens.mass_profile_list[0].total_free_parameters)

"""
The `lens` behaves exactly like the model-components we are used to previously. For example, we can unpack its 
individual parameters to customize the model, where below we:

 1) Fix the light and mass profiles to the centre (0.0, 0.0).
 2) Customize the prior on the light profile `axis_ratio`.
 3) Fix the `axis_ratio` of the mass profile to 0.8.
"""

lens.light_profile_list[0].centre = (0.0, 0.0)
lens.light_profile_list[0].axis_ratio = af.UniformPrior(
    lower_limit=0.7, upper_limit=0.9
)
lens.light_profile_list[0].angle = af.UniformPrior(lower_limit=0.0, upper_limit=180.0)
lens.light_profile_list[0].intensity = af.LogUniformPrior(
    lower_limit=1e-4, upper_limit=1e4
)
lens.light_profile_list[0].effective_radius = af.UniformPrior(
    lower_limit=0.0, upper_limit=5.0
)

lens.mass_profile_list[0].centre = (0.0, 0.0)
lens.mass_profile_list[0].axis_ratio = 0.8
lens.mass_profile_list[0].angle = af.UniformPrior(lower_limit=0.0, upper_limit=180.0)
lens.mass_profile_list[0].mass = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

print(lens.info)

"""
__Alternative API__

We can create the `Galaxy` model component with the exact same customization by creating each profile as a `Model` and
passing these to the galaxy `Model`. 
"""
light = af.Model(cosmo.lp.LightDeVaucouleurs)

light.centre = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
light.axis_ratio = af.UniformPrior(lower_limit=0.7, upper_limit=0.9)
light.angle = af.UniformPrior(lower_limit=0.0, upper_limit=180.0)
light.intensity = af.LogUniformPrior(lower_limit=1e-4, upper_limit=1e4)
light.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=5.0)


mass = af.Model(cosmo.mp.MassIsothermal)

mass.centre = (0.0, 0.0)
mass.axis_ratio = af.UniformPrior(lower_limit=0.7, upper_limit=1.0)
mass.angle = af.UniformPrior(lower_limit=0.0, upper_limit=180.0)
mass.mass = af.UniformPrior(lower_limit=0.0, upper_limit=4.0)

lens = af.Model(
    cosmo.Galaxy, redshift=0.5, light_profile_list=[light], mass_profile_list=[mass]
)

print(lens.info)

"""
We can now create a model of our source galaxy using the same API.
"""
light = af.Model(cosmo.lp.LightExponential)

light.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
light.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
light.axis_ratio = af.UniformPrior(lower_limit=0.7, upper_limit=1.0)
light.angle = af.UniformPrior(lower_limit=0.0, upper_limit=180.0)
light.intensity = af.LogUniformPrior(lower_limit=1e-4, upper_limit=1e4)
light.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

source = af.Model(cosmo.Galaxy, redshift=1.0, light_profile_list=[light])

print(source.info)

"""
We can now create our overall strong lens model, using a `Collection` in the same way we have seen previously.
"""
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

"""
The model contains both galaxies in the strong lens, alongside all of their light and mass profiles.

For every iteration of the non-linear search **PyAutoFit** generates an instance of this model, where all of the
`LightDeVaucouleurs`, `MassIsothermal` and `Galaxy` parameters of the are determined via their priors. 

An example instance is show below:
"""
instance = model.instance_from_prior_medians()

print("Strong Lens Model Instance:")
print("Lens Galaxy = ", instance.galaxies.lens)
print("Lens Galaxy Light = ", instance.galaxies.lens.light_profile_list)
print(
    "Lens Galaxy Light Centre = ", instance.galaxies.lens.light_profile_list[0].centre
)
print("Lens Galaxy Mass Centre = ", instance.galaxies.lens.mass_profile_list[0].centre)
print("Source Galaxy = ", instance.galaxies.source)

"""
We have successfully composed a multi-level model, which we can fit via a non-linear search.

At this point, you should check out the `Analysis` class of this example project, in the 
module `projects/cosmology/src/analysis.py`. This class serves the same purpose that we have seen in the Gaussian 1D 
examples, with the `log_likelihood_function` implementing the calculation we showed in the first tutorial.

The `path_prefix1 and `name` inputs below sepciify the path and folder where the results of the model-fit are stored
in the output folder `autolens_workspace/output`. Results for this tutorial are writtent to hard-disk, due to the 
longer run-times of the model-fit.
"""

search = af.DynestyStatic(
    path_prefix=path.join("projects", "cosmology"),
    name="multi_level",
    nlive=50,
    iterations_per_update=2500,
)

analysis = cosmo.Analysis(data=data, noise_map=noise_map, psf=psf, grid=grid)

"""
If you comment out the code below, you will perform a lens model fit using the model and analysis class for 
this project. However, this model-fit is slow to run, and it isn't paramount that you run it yourself.

The animation below shows a slide-show of the lens modeling procedure. Many lens models are fitted to the data over
and over, gradually improving the quality of the fit to the data and looking more and more like the observed image.

![Lens Modeling Animation](https://github.com/Jammy2211/auto_files/blob/main/lensmodel.gif?raw=true "model")
"""

result = search.fit(model=model, analysis=analysis)

"""
__Extensibility__

This example project highlights how multi-level models can make certain model-fitting problem fully extensible. For 
example:

 1) A `Galaxy` class can be created using any combination of light and mass profiles, because it can wrap their
 `image_from_grid` and `deflections_from_grid` methods as the sum of the individual profiles.
 
 2) The overall strong lens model can contain any number of `Galaxy`'s, as their methods are used 
    to implement the lensing calculations in the `Analysis` class and `log_likelihood_function`.
  
For problems of this nature, we can design and write code in a way that fully utilizes **PyAutoFit**'s multi-level
modeling features to compose and fits models of arbitrary complexity and dimensionality. 

__Galaxy Clusters__

To illustrate this further, consider the following dataset which is called a "strong lens galaxy cluster":

![Strong Lens Cluster](https://github.com/rhayes777/PyAutoFit/blob/main/docs/overview/image/cluster_example.jpg?raw=true "cluster")

For this strong lens, there are many tens of strong lens galaxies as well as multiple background source galaxies. 

However, despite it being a significantly more complex system than the single-galaxy strong lens we modeled above,
our use of multi-level models ensures that we can model such datasets without any additional code development, for
example:

The lensing calculations in the source code `Analysis` object did not properly account for multiple galaxies 
(called multi-plane ray tracing). This would need to be updated to properly model a galaxy cluster, but this
tutorial shows how a model can be composed for such a system.
"""
lens_0 = af.Model(
    cosmo.Galaxy,
    redshift=0.5,
    light_profile_list=[cosmo.lp.LightDeVaucouleurs],
    mass_profile_list=[cosmo.mp.MassIsothermal],
)

lens_1 = af.Model(
    cosmo.Galaxy,
    redshift=0.5,
    light_profile_list=[cosmo.lp.LightDeVaucouleurs],
    mass_profile_list=[cosmo.mp.MassIsothermal],
)

source_0 = af.Model(
    cosmo.Galaxy, redshift=1.0, light_profile_list=[af.Model(cosmo.lp.LightExponential)]
)

# ... repeat for desired model complexity ...

model = af.Collection(
    galaxies=af.Collection(
        lens_0=lens_0,
        lens_1=lens_1,
        source_0=source_0,
        # ... repeat for desired model complexity ...
    )
)

print(model.info)

"""
Here is a pictoral representation of a strong lens cluster as a multi-level model:

![Strong Lens Cluster Model](https://github.com/rhayes777/PyAutoFit/blob/main/docs/overview/image/lens_model_cluster.png?raw=true "cluster")

__Wrap Up__

Strong gravitational lensing is a great example of a problem that can be approached using multi-level models. 

At the core of this is how there are many different models one could imagine defining which describe the light or mass 
of a galaxy. However, all of these models must derive the same fundamental property in order to fit the data, for
example the image of a light profile or the deflection angles of the mass profile.

The multi-level nature of strong lensing is not unique, and is commonly found in my Astronomy problems and the 
scientific literature in general. For example Astronomy problems:

 - Studies of galaxy structure, which represent the surface brightness distributions of galaxies as sums of Sersic
 profiles (or other parametric equations) to quantify whether they are bulge-like or disk-like.
 
 - Studies of galaxy dynamics, which represent the mass distribution of galaxies as sums of profiles like the Isothermal
 profile.
 
 - Studies of the activate galactic nuclei (AGN) of galaxies, where the different components of the AGN are represented
 as different model components.
"""
