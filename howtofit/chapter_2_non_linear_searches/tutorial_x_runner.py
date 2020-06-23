# %%
"""
__Pipelines__
"""

# %%
#%matplotlib inline

# %%
from autoconf import conf
import numpy as np

from howtofit.chapter_2_non_linear_searches.src.dataset import dataset as ds

# %%
"""
You need to change the path below to the chapter 2 directory so we can load the dataset
"""

# %%
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autofit_workspace/howtofit/chapter_2_non_linear_searches"

# %%
"""
Setup the configs as we did in the previous tutorial, as well as the output folder for our non-linear search.
"""

# %%
conf.instance = conf.Config(
    config_path=f"{workspace_path}/config",
    output_path=f"{workspace_path}/output/howtofit",
)

# %%
"""
Lets load the dataset, create a mask and perform the fit.
"""

# %%
dataset_path = f"{chapter_path}/dataset/gaussian_x2_split/"

dataset = ds.Dataset.from_fits(
    data_path=f"{dataset_path}/data.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
)

# %%
"""
Lets use a plot function to plot our data.

Note how - describe its x2 peaks.
"""

from howtofit.chapter_2_non_linear_searches.src.plot import dataset_plots

dataset_plots.data(dataset=dataset)

mask = np.full(fill_value=False, shape=dataset.data.shape)

print(
    "Pipeline has begun running - checkout the autofit_workspace/howtofit/chapter_2_non_linear_searches/output/pipeline__x2_gaussians"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once the Pipeline has completed - this could take a few minutes!"
)

from howtofit.chapter_2_non_linear_searches import tutoial_x_pipeline

pipeline = tutoial_x_pipeline.make_pipeline()

pipeline.run(dataset=dataset, mask=mask)
