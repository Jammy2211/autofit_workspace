"""
Plots: NestPlotter
=======================

This example illustrates how to plot visualization summarizing the results of a ultranest non-linear search using
a `NestPlotter`.

Installation
------------

Because UltraNest is an optional library, you will likely have to install it manually via the command:

`pip install ultranest`
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path

import autofit as af
import autofit.plot as aplt

"""
First, lets create a result via ultranest by repeating the simple model-fit that is performed in 
the `overview/simple/fit.py` example.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

model = af.Model(af.ex.Gaussian)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

search = af.UltraNest(path_prefix="plot", name="NestPlotter", max_ncalls=10)

result = search.fit(model=model, analysis=analysis)

samples = result.samples

"""
We now pass the samples to a `NestPlotter` which will allow us to use ultranest's in-built plotting libraries to 
make figures.

The ultranest readthedocs describes fully all of the methods used below 

 - https://johannesbuchner.github.io/UltraNest/readme.html
 - https://johannesbuchner.github.io/UltraNest/ultranest.html#module-ultranest.plot
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
ultranest_plotter = aplt.NestPlotter(samples=samples)

"""
The `corner` method produces a triangle of 1D and 2D PDF's of every parameter using the library `corner.py`.
"""
ultranest_plotter.corner()
"""
Finish.
"""
