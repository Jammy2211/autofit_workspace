"""
__Simulators__

These scripts simulate the 1D Gaussian datasets used to demonstrate model-fitting.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import profiles
import util
from os import path

"""
__Gaussian x1__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=10.0)
util.simulate_line_from_gaussian(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 (0)__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_0")
gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=1.0)
util.simulate_line_from_gaussian(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 (1)__
"""
gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=5.0)
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_1")
util.simulate_line_from_gaussian(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 (2)__
"""
gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=10.0)
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_2")
util.simulate_line_from_gaussian(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 + Exponential x1__
"""
gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=10.0)
exponential = profiles.Exponential(centre=50.0, intensity=40.0, rate=0.05)
dataset_path = path.join("dataset", "example_1d", "gaussian_x1__exponential_x1")
util.simulate_line_from_profiles(
    profiles=[gaussian, exponential], dataset_path=dataset_path
)

"""
__Gaussian x2 + Exponential x1__
"""
gaussian_0 = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=10.0)
gaussian_1 = profiles.Gaussian(centre=20.0, intensity=30.0, sigma=5.0)
exponential = profiles.Exponential(centre=70.0, intensity=40.0, rate=0.005)
dataset_path = path.join("dataset", "example_1d", "gaussian_x2__exponential_x1")
util.simulate_line_from_profiles(
    profiles=[gaussian_0, gaussian_1, exponential], dataset_path=dataset_path
)

"""
__Gaussian x3__
"""
gaussian_0 = profiles.Gaussian(centre=50.0, intensity=20.0, sigma=1.0)
gaussian_1 = profiles.Gaussian(centre=50.0, intensity=40.0, sigma=5.0)
gaussian_2 = profiles.Gaussian(centre=50.0, intensity=60.0, sigma=10.0)
dataset_path = path.join("dataset", "example_1d", "gaussian_x3")
util.simulate_line_from_profiles(
    profiles=[gaussian_0, gaussian_1, gaussian_2], dataset_path=dataset_path
)

"""
__Gaussian x1 unconvolved__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_unconvolved")
gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=3.0)
util.simulate_line_from_gaussian(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 convolved__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_convolved")
gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=3.0)
util.simulate_line_with_kernel_from_gaussian(
    gaussian=gaussian, dataset_path=dataset_path
)

"""
__Gaussian x1 with feature__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_with_feature")
gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=10.0)
gaussian_feature = profiles.Gaussian(centre=70.0, intensity=0.3, sigma=0.5)
util.simulate_line_from_profiles(
    profiles=[gaussian, gaussian_feature], dataset_path=dataset_path
)

"""
__Gaussian x2 split__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x2_split")
gaussian_0 = profiles.Gaussian(centre=25.0, intensity=50.0, sigma=12.5)
gaussian_1 = profiles.Gaussian(centre=75.0, intensity=50.0, sigma=12.5)
util.simulate_line_from_profiles(
    profiles=[gaussian_0, gaussian_1], dataset_path=dataset_path
)

"""
__Gaussian x1 low snr (0)__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_0__low_snr")
gaussian = profiles.Gaussian(centre=50.0, intensity=1.5, sigma=6.0)
util.simulate_line_from_gaussian(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 low snr (1)__
"""
gaussian = profiles.Gaussian(centre=50.0, intensity=2.0, sigma=8.0)
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_1__low_snr")
util.simulate_line_from_gaussian(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 low snr (2)__
"""
gaussian = profiles.Gaussian(centre=50.0, intensity=5.0, sigma=15.0)
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_2__low_snr")
util.simulate_line_from_gaussian(gaussian=gaussian, dataset_path=dataset_path)

"""
Finish.
"""
