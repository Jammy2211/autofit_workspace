# %%
"""
__Galaxy Morphology__

In this chapter, we illustrated PyAutoFit with the same model through - fitting 1D line profiles consisting primarily
of Gaussians, but sometimes Exponentials.
#
For this, the final tutorial, I've created a template project of based around a real-world modeling problem from
Astrophysics; fitting the morphology of galaxies. Lets look at an image of a galaxy, by using the project's dataset
and plot packages.

The questions we use our model-fitting software to ask are things like:

- What is the total brightness of a galaxy? Can we quantify how much more luminous one galaxy is relative to another?

- How is this galaxy's light varying radially? How much light is there in the centre relative to the outskirts? How
  does this compare across a sample of galaxies?

- Is the galaxy's light best represented as one continuous structure or multiple structures? What morphology are
  those stuctures, for example are they more disk-like or bulge-like?

And many more! In this tutorial, we won't go into too much detail about answering these questions. However, in
chapter 2 we are going to introduce many new statistical methods, techniques and tools and we'll use our galaxy
morphology project (as well as the 1D line fitting) to illustrate these. So, if you plan to continue on to chapte 2
it is worth being familar with why we fit galaxy light!


To begin, you should take a look around the source code in the folder:

'autofit_workspace/howtofit/chapter_1_introduction/tutorial_8_galaxy_morphology'.

In terms of PyAutoFit there is nothing new here you haven't seen in the previous 7 tutorials. However, the switch to
fitting images of galaxys changes many model-specifc aspects of the project, for example:

- The dataset, masks, models, fit, etc. packages and modules are now all in 2D.

- The dataset contains a new attribute in addition to the 'data' and 'noise_map' called the 'psf' or Point-Spread
  Function. This accounts for blurring effects that occur when we observe the galaxy using a telescope.

- The model package, which now contains a 'light_profiles.py' module has many more profiles than the 2 we saw
  previously. This is because galaxy morphologies can be fitted with many different models.

- There are new phase-settings (which were introduced in tutorial 7) that customize the dataset and model-fit in
  different ways. These tag phase names in the same way we saw before.

After looking through the project, the main thing I want you to take from it is that we have been able to set up a
new project which is very very different to our example of fitting 1D Gaussians using *the exactly same project
format*. The project format we have introduced in this chapter is a template that can easily generalize to any
project, and I strongly recommend you build your PyAutoFit project adhering to this template.

Lets quickly use the project to perform a fit. As I said above, we will mostly use this project in chapter 2,
but I'm sure you're eager to fit the morphology of a galaxy!
"""
