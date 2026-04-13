"""
Plots: DynestyPlotter
=======================

This example illustrates how to plot visualization summarizing the results of a ultranest non-linear search using
the `autofit.plot` module-level functions.

Installation
------------

Because UltraNest is an optional library, you will likely have to install it manually via the command:

`pip install ultranest`

__Contents__

This script is split into the following sections:

- **Notation**: How parameter labels and superscripts are customized for plots.
- **Plotting**: Using the plot functions to visualize UltraNest search results.
- **Search Specific Visualization**: Accessing the native UltraNest sampler for custom visualizations.
- **Plots**: Producing UltraNest-specific diagnostic plots.
"""

# from autoconf import setup_notebook; setup_notebook()

from os import path

import autofit as af
import autofit.plot as aplt

"""
First, lets create a result via ultranest by repeating the simple model-fit that is performed in 
the `overview/simple/fit.py` example.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not path.exists(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulators/simulators.py"],
        check=True,
    )

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
__Notation__

Plot are labeled with short hand parameter names (e.g. the `centre` parameters are plotted using an `x`). 

The mappings of every parameter to its shorthand symbol for plots is specified in the `config/notation.yaml` file 
and can be customized.

Each label also has a superscript corresponding to the model component the parameter originates from. For example,
Gaussians are given the superscript `g`. This can also be customized in the `config/notation.yaml` file.

__Plotting__

We now use the `autofit.plot` module-level functions to visualize the results.

The ultranest readthedocs describes fully all of the methods used below 

 - https://johannesbuchner.github.io/UltraNest/readme.html
 - https://johannesbuchner.github.io/UltraNest/ultranest.html#module-ultranest.plot
 
In all the examples below, we use the `kwargs` of this function to pass in any of the input parameters that are 
described in the API docs.
"""
"""
The `corner_anesthetic` function produces a triangle of 1D and 2D PDF's of every parameter using the library `anesthetic`.
"""
aplt.corner_anesthetic(samples=samples)

"""
The `corner_cornerpy` function produces a triangle of 1D and 2D PDF's of every parameter using the library `corner.py`.
"""
aplt.corner_cornerpy(samples=samples)

"""
__Search Specific Visualization__

The internal sampler can be used to plot the results of the non-linear search. 

We do this using the `search_internal` attribute which contains the sampler in its native form.

The first time you run a search, the `search_internal` attribute will be available because it is passed ot the
result via memory. 

If you rerun the fit on a completed result, it will not be available in memory, and therefore
will be loaded from the `files/search_internal` folder. The `search_internal` entry of the `output.yaml` must be true 
for this to be possible.
"""
search_internal = result.search_internal

"""
__Plots__

UltraNest example plots are not shown explicitly below, so checkout their docs for examples!
"""
