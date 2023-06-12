"""
__Simulators__

These scripts simulates many 1D Gaussian datasets with a low signal to noise ratio, which are used to demonstrate
model-fitting.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path

import autofit as af
import util

"""
__Gaussian x1 low snr (centre fixed to 50.0)__

This is used for demonstrating expectation propagation, whereby a shared `centre` parameter is inferred from a sample 
of `total_datasets` 1D Gaussian datasets.
"""
total_datasets = 50

for i in range(total_datasets):
    dataset_path = path.join(
        "dataset", "example_1d", f"gaussian_x1__low_snr", f"dataset_{i}"
    )
    gaussian = af.ex.Gaussian(centre=50.0, normalization=0.5, sigma=5.0)
    util.simulate_dataset_1d_via_gaussian_from(
        gaussian=gaussian, dataset_path=dataset_path
    )

"""
__Gaussian x1 low snr (centre drawn from parent Gaussian distribution to 50.0)__

This is used for demonstrating expectation propagation and hierachical modeling, whereby a the `centre` parameters 
of a sample of `total_datasets` 1D Gaussian datasets are drawn from a Gaussian distribution.
"""

total_datasets = 10

gaussian_parent_model = af.Model(
    af.ex.Gaussian,
    centre=af.GaussianPrior(mean=50.0, sigma=10.0, lower_limit=0.0, upper_limit=100.0),
    normalization=0.5,
    sigma=5.0,
)

for i in range(total_datasets):
    dataset_path = path.join(
        "dataset", "example_1d", f"gaussian_x1__hierarchical", f"dataset_{i}"
    )

    gaussian = gaussian_parent_model.random_instance()

    util.simulate_dataset_1d_via_gaussian_from(
        gaussian=gaussian, dataset_path=dataset_path
    )


"""
__Gaussian x2 offset centre__

This is used for demonstrating the benefits of graphical models over fitting one-by-one, because it creates a 
degeneracy in the offset of the centres of the two Gaussians.
"""
total_datasets = 10

for i in range(total_datasets):
    dataset_path = path.join(
        "dataset", "example_1d", f"gaussian_x2__offset_centres", f"dataset_{i}"
    )

    sigma_0_prior = af.GaussianPrior(
        lower_limit=0.0, upper_limit=20.0, mean=10.0, sigma=10.0
    )
    while True:
        try:
            sigma_0_value = sigma_0_prior.value_for(unit=np.random.random(1))
            break
        except af.exc.PriorLimitException:
            continue

    sigma_1_prior = af.GaussianPrior(
        lower_limit=0.0, upper_limit=20.0, mean=10.0, sigma=10.0
    )
    while True:
        try:
            sigma_1_value = sigma_1_prior.value_for(unit=np.random.random(1))
            break
        except af.exc.PriorLimitException:
            continue

    gaussian_0 = af.ex.Gaussian(centre=40.0, normalization=1.0, sigma=sigma_0_value)
    gaussian_1 = af.ex.Gaussian(centre=60.0, normalization=1.0, sigma=sigma_1_value)

    util.simulate_dataset_1d_via_profile_1d_list_from(
        profile_1d_list=[gaussian_0, gaussian_1], dataset_path=dataset_path
    )
