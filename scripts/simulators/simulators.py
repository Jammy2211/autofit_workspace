"""
__Simulators__

These scripts simulate the 1D Gaussian datasets used to demonstrate model-fitting.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

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
