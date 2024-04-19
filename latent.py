# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autofit.plot as aplt

import matplotlib.pyplot as plt
import numpy as np
import os
from os import path

dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

model = af.Model(af.ex.Gaussian)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

search = af.DynestyStatic(
    name="latent",
    nlive=100,
    sample="rwalk",
    number_of_cores=1,
)

result = search.fit(model=model, analysis=analysis)

print(result.latent_variables)
