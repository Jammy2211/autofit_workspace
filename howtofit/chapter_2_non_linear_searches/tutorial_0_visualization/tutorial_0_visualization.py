import autoarray as aa
import autoarray.plot as aplt

# In this tutorial, we'll cover visualization in PyAutoArray and make sure images display properly on your computer.

# First, lets load an image of a two dimensional Gaussian. In these tutorials we're going to adopt 2D Gaussian fitting
# as our baseline toy model, to illustrate model-fitting in PyAutoFit.

# Don't worry too much about what the code below is doing as it will be covered in a later tutorial.

# First you need to change the path below to the chapter 1 directory so we can load the dataset.
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_1_introduction/"

# The dataset path specifies where the dataset is located, this time in the directory 'chapter_path/dataset'. We'll
# load the example dataset containing one Gaussian.

dataset_path = chapter_path + "dataset/gaussian_x1/"

# We now load this dataset from .fits files and create an instance of an 'dataset' object.
dataset = aa.dataset.from_fits(
    image_path=dataset_path + "image.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    psf_path=dataset_path + "psf.fits",
    pixel_scales=0.1,
)

# We can plot an image as follows.

aplt.dataset.image(dataset=dataset)

# Does the figure display correctly on your computer screen?

# If not, you can customize a number of matplotlib setup options using a Plotter object in PyAutoArray.

plotter = aplt.Plotter(
    figure=aplt.Figure(figsize=(7, 7)),
    ticks=aplt.Ticks(ysize=8, xsize=8),
    labels=aplt.Labels(ysize=6, xsize=6, titlesize=12),
)

aplt.dataset.image(dataset=dataset, plotter=plotter)

# Many matplotlib setup options can be customized, but for now we're only concerned with making sure figures display
# cleanly in your Jupter Notebooks. However, for future reference, a description of all options can be found in the file
# 'autofit_workspace/plot/mat_objs.py'.

# Ideally, we wouldn't need to specify a new plotter every time we plot an image, especially as you'll be changing
# the same option to the same value over and over again (e.g. the figsize). Fortunately, the default values used by
# PyAutoArray can be fully customized.

# Checkout the the file 'autofit_workspace/config/visualize/figures.ini'.

# All default matplotlib values used by PyAutoArray are here. There's lots, so lets only focus on whats important for
# displaying figures correctly:

# [figures] -> figsize
# [labels] -> titlesize, ysize, xsize
# [ticks] -> ysize, xsize

# Don't worry about all the other options listed in this file for now, as they'll make a lot more sense once you
# are familiar with PyAutoArray.

# (Note that you will need to reset your Juypter notebook server for these changes to take effect, so make sure
# you have the right values using the function above beforehand!)

# In addition to individual 'figures' which use a 'plotter' to plot them, PyAutoArray also plots 'subplots' using a
# 'sub_plotter'. Lets plot a subplot of our dataset data:

aplt.dataset.subplot_dataset(dataset=dataset)

# Again, we can customize this subplot using a SubPlotter.

# (The '.sub' ensures we load the setting values from the config file 'autofit_workspace/config/visualize/subplots.ini'

sub_plotter = aplt.SubPlotter(
    figure=aplt.Figure.sub(figsize=(7, 7)),
    ticks=aplt.Ticks.sub(ysize=8, xsize=8),
    labels=aplt.Labels.sub(ysize=6, xsize=6, titlesize=12),
)

aplt.dataset.subplot_dataset(dataset=dataset, sub_plotter=sub_plotter)

# Again, you can customize the default appearance of subplots by editing the config file
# autofit_workspace/config/visualize/subplots.ini'.

# The other thing we can do with figures is choose what we include in the plot. For example, we can choose whether to
# include the origin of the coordinate system on our plot of the image:

aplt.dataset.image(dataset=dataset, plotter=plotter, include=aplt.Include(origin=True))

aplt.dataset.image(dataset=dataset, plotter=plotter, include=aplt.Include(origin=False))

# Throughout the HowToFit lecture series you'll see lots more objects that can include on figures.

# Just like the matplotlib setup, you can customize what does and does not appear on figures by default using the
# config file 'autofit_workspace/config/visualize/general.ini'

# Great! Hopefully, visualization in PyAutoArray is displaying nicely for us to get on with the HowToFit lecture
# series.
