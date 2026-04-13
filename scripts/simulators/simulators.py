"""
__Simulators__

These scripts simulate the 1D Gaussian datasets used to demonstrate model-fitting.

__Contents__

This script is split into the following sections:

- **Gaussian x1**: Simulate a single 1D Gaussian dataset.
- **Gaussian x1 (0)**: Simulate a single Gaussian with sigma=1.0.
- **Gaussian x1 (1)**: Simulate a single Gaussian with sigma=5.0.
- **Gaussian x1 (2)**: Simulate a single Gaussian with sigma=10.0.
- **Gaussian x1 (Identical 0)**: Simulate an identical single Gaussian dataset (copy 0).
- **Gaussian x1 (Identical 1)**: Simulate an identical single Gaussian dataset (copy 1).
- **Gaussian x1 (Identical 2)**: Simulate an identical single Gaussian dataset (copy 2).
- **Gaussian x1 + Exponential x1**: Simulate a dataset with one Gaussian and one Exponential.
- **Gaussian x2 + Exponential x1**: Simulate a dataset with two Gaussians and one Exponential.
- **Gaussian x2**: Simulate a dataset with two Gaussians.
- **Gaussian x3**: Simulate a dataset with three Gaussians.
- **Gaussian x5**: Simulate a dataset with five Gaussians.
- **Gaussian x1 unconvolved**: Simulate a single Gaussian without convolution.
- **Gaussian x1 convolved**: Simulate a single Gaussian with kernel convolution.
- **Gaussian x1 with feature**: Simulate a Gaussian with a small feature bump.
- **Gaussian x2 split**: Simulate two separated Gaussians.
- **Gaussian x1 time**: Simulate time-varying Gaussian datasets.
"""

# from autoconf import setup_notebook; setup_notebook()

import util
from os import path

import autofit as af

"""
__Gaussian x1__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 (0)__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_0")
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=1.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 (1)__
"""
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=5.0)
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_1")
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 (2)__
"""
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_2")
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 (Identical 0)__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_identical_0")
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 (Identical 1)__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_identical_1")
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 (Identical 2)__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_identical_2")
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 + Exponential x1__
"""
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
exponential = af.ex.Exponential(centre=50.0, normalization=40.0, rate=0.05)
dataset_path = path.join("dataset", "example_1d", "gaussian_x1__exponential_x1")
util.simulate_dataset_1d_via_profile_1d_list_from(
    profile_1d_list=[gaussian, exponential], dataset_path=dataset_path
)

"""
__Gaussian x2 + Exponential x1__
"""
gaussian_0 = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
gaussian_1 = af.ex.Gaussian(centre=20.0, normalization=30.0, sigma=5.0)
exponential = af.ex.Exponential(centre=70.0, normalization=40.0, rate=0.005)
dataset_path = path.join("dataset", "example_1d", "gaussian_x2__exponential_x1")
util.simulate_dataset_1d_via_profile_1d_list_from(
    profile_1d_list=[gaussian_0, gaussian_1, exponential], dataset_path=dataset_path
)

"""
__Gaussian x2__
"""
gaussian_0 = af.ex.Gaussian(centre=50.0, normalization=20.0, sigma=1.0)
gaussian_1 = af.ex.Gaussian(centre=50.0, normalization=40.0, sigma=5.0)
dataset_path = path.join("dataset", "example_1d", "gaussian_x2")
util.simulate_dataset_1d_via_profile_1d_list_from(
    profile_1d_list=[gaussian_0, gaussian_1], dataset_path=dataset_path
)

"""
__Gaussian x3__
"""
gaussian_0 = af.ex.Gaussian(centre=50.0, normalization=20.0, sigma=1.0)
gaussian_1 = af.ex.Gaussian(centre=50.0, normalization=40.0, sigma=5.0)
gaussian_2 = af.ex.Gaussian(centre=50.0, normalization=60.0, sigma=10.0)
dataset_path = path.join("dataset", "example_1d", "gaussian_x3")
util.simulate_dataset_1d_via_profile_1d_list_from(
    profile_1d_list=[gaussian_0, gaussian_1, gaussian_2], dataset_path=dataset_path
)

"""
__Gaussian x5__
"""
gaussian_0 = af.ex.Gaussian(centre=50.0, normalization=20.0, sigma=1.0)
gaussian_1 = af.ex.Gaussian(centre=50.0, normalization=40.0, sigma=5.0)
gaussian_2 = af.ex.Gaussian(centre=50.0, normalization=60.0, sigma=10.0)
gaussian_3 = af.ex.Gaussian(centre=50.0, normalization=80.0, sigma=15.0)
gaussian_4 = af.ex.Gaussian(centre=50.0, normalization=100.0, sigma=20.0)
dataset_path = path.join("dataset", "example_1d", "gaussian_x5")
util.simulate_dataset_1d_via_profile_1d_list_from(
    profile_1d_list=[gaussian_0, gaussian_1, gaussian_2, gaussian_3, gaussian_4],
    dataset_path=dataset_path,
)

"""
__Gaussian x1 unconvolved__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_unconvolved")
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=3.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 convolved__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_convolved")
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=3.0)
util.simulate_data_1d_with_kernel_via_gaussian_from(
    gaussian=gaussian, dataset_path=dataset_path
)

"""
__Gaussian x1 with feature__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_with_feature")
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
gaussian_feature = af.ex.Gaussian(centre=70.0, normalization=0.3, sigma=0.5)
util.simulate_dataset_1d_via_profile_1d_list_from(
    profile_1d_list=[gaussian, gaussian_feature], dataset_path=dataset_path
)

"""
__Gaussian x2 split__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x2_split")
gaussian_0 = af.ex.Gaussian(centre=25.0, normalization=50.0, sigma=12.5)
gaussian_1 = af.ex.Gaussian(centre=75.0, normalization=50.0, sigma=12.5)
util.simulate_dataset_1d_via_profile_1d_list_from(
    profile_1d_list=[gaussian_0, gaussian_1], dataset_path=dataset_path
)


"""
__Gaussian x1 time__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_time", "time_0")
gaussian = af.ex.Gaussian(centre=40.0, normalization=50.0, sigma=20.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

dataset_path = path.join("dataset", "example_1d", "gaussian_x1_time", "time_1")
gaussian = af.ex.Gaussian(centre=50.0, normalization=50.0, sigma=20.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

dataset_path = path.join("dataset", "example_1d", "gaussian_x1_time", "time_2")
gaussian = af.ex.Gaussian(centre=60.0, normalization=50.0, sigma=20.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)


"""
__Gaussian x1 time__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_variable", "sigma_0")
gaussian = af.ex.Gaussian(centre=50.0, normalization=50.0, sigma=10.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

dataset_path = path.join("dataset", "example_1d", "gaussian_x1_variable", "sigma_1")
gaussian = af.ex.Gaussian(centre=50.0, normalization=50.0, sigma=20.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

dataset_path = path.join("dataset", "example_1d", "gaussian_x1_variable", "sigma_2")
gaussian = af.ex.Gaussian(centre=50.0, normalization=50.0, sigma=30.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

"""
Finish.
"""
