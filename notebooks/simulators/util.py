import autofit as af
import profiles

from os import path
import numpy as np
import matplotlib.pyplot as plt


def simulate_line_from_gaussian(gaussian, dataset_path):

    """
    Specify the number of pixels used to create the xvalues on which the 1D line of the profile is generated using and
    thus defining the number of data-points in our data.
    """
    pixels = 100
    xvalues = np.arange(pixels)

    """Evaluate this `Gaussian` model instance at every xvalues to create its model profile."""
    model_line = gaussian.profile_from_xvalues(xvalues=xvalues)

    """Determine the noise (at a specified signal to noise level) in every pixel of our model profile."""
    signal_to_noise_ratio = 25.0
    noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

    """
    Add this noise to the model line to create the line data that is fitted, using the signal-to-noise ratio to compute
    noise-map of our data which is required when evaluating the chi-squared value of the likelihood.
    """
    data = model_line + noise
    noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

    """
    Output the data and noise-map to the `autofit_workspace/dataset` folder so they can be loaded and used 
    in other example scripts.
    """
    af.util.numpy_array_to_json(
        array=data, file_path=path.join(dataset_path, "data.json"), overwrite=True
    )
    af.util.numpy_array_to_json(
        array=noise_map,
        file_path=path.join(dataset_path, "noise_map.json"),
        overwrite=True,
    )
    plt.errorbar(
        x=xvalues,
        y=data,
        yerr=noise_map,
        color="k",
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.title("1D Gaussian Dataset.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile intensity")
    plt.savefig(path.join(dataset_path, "image.png"))
    plt.close()


def simulate_line_with_kernel_from_gaussian(gaussian, dataset_path):

    """
    Specify the number of pixels used to create the xvalues on which the 1D line of the profile is generated using and
    thus defining the number of data-points in our data.
    """
    pixels = 100
    xvalues = np.arange(pixels)

    """Evaluate this `Gaussian` model instance at every xvalues to create its model profile."""
    model_line = gaussian.profile_from_xvalues(xvalues=xvalues)

    """Determine the noise (at a specified signal to noise level) in every pixel of our model profile."""
    kernel_pixels = 21
    kernel_xvalues = np.arange(kernel_pixels)
    kernel_sigma = 5.0
    kernel_centre = 10.0
    kernel_xvalues = np.subtract(kernel_xvalues, kernel_centre)
    kernel = np.multiply(
        np.divide(1.0, kernel_sigma * np.sqrt(2.0 * np.pi)),
        np.exp(-0.5 * np.square(np.divide(kernel_xvalues, kernel_sigma))),
    )
    kernel = kernel / np.sum(kernel)

    "Convolve the model line with this kernel."
    blurred_model_line = np.convolve(model_line, kernel, mode="same")

    """Create a Gaussian kernel which the model line will be convolved with."""
    signal_to_noise_ratio = 25.0
    noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

    """
    Add this noise to the model line to create the line data that is fitted, using the signal-to-noise ratio to compute
    noise-map of our data which is required when evaluating the chi-squared value of the likelihood.
    """
    data = blurred_model_line + noise
    noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

    """
    Output the data and noise-map to the `autofit_workspace/dataset` folder so they can be loaded and used 
    in other example scripts.
    """
    af.util.numpy_array_to_json(
        array=data, file_path=path.join(dataset_path, "data.json"), overwrite=True
    )
    af.util.numpy_array_to_json(
        array=noise_map,
        file_path=path.join(dataset_path, "noise_map.json"),
        overwrite=True,
    )
    af.util.numpy_array_to_json(
        array=kernel, file_path=path.join(dataset_path, "kernel.json"), overwrite=True
    )
    plt.errorbar(
        x=xvalues,
        y=data,
        yerr=noise_map,
        color="k",
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.title("1D Gaussian Dataset with Kernel2D Blurring.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile intensity")
    plt.savefig(path.join(dataset_path, "image.png"))
    plt.close()


def simulate_line_from_profiles(profiles, dataset_path):
    """
    Specify the number of pixels used to create the xvalues on which the 1D line of the profile is generated using and
    thus defining the number of data-points in our data.
    """
    pixels = 100
    xvalues = np.arange(pixels)

    """
    Evaluate the `Gaussian` and Exponential model instances at every xvalues to create their model profile and sum
    them together to create the overall model profile.
    """
    model_line = np.zeros(shape=pixels)

    for profile in profiles:

        model_line += profile.profile_from_xvalues(xvalues=xvalues)

    """Determine the noise (at a specified signal to noise level) in every pixel of our model profile."""
    signal_to_noise_ratio = 25.0
    noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

    """
    Add this noise to the model line to create the line data that is fitted, using the signal-to-noise ratio to compute
    noise-map of our data which is required when evaluating the chi-squared value of the likelihood.
    """
    data = model_line + noise
    noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

    """
    Output the data and noise-map to the `autofit_workspace/dataset` folder so they can be loaded and used 
    in other example scripts.
    """
    af.util.numpy_array_to_json(
        array=data, file_path=path.join(dataset_path, "data.json"), overwrite=True
    )
    af.util.numpy_array_to_json(
        array=noise_map,
        file_path=path.join(dataset_path, "noise_map.json"),
        overwrite=True,
    )
    plt.errorbar(
        x=xvalues,
        y=data,
        yerr=noise_map,
        color="k",
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.title("1D Profiles Dataset.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile intensity")
    plt.savefig(path.join(dataset_path, "image.png"))
    plt.close()


def simulate_line_with_kernel_from_profiles(profiles, dataset_path):

    """
    Specify the number of pixels used to create the xvalues on which the 1D line of the profile is generated using and
    thus defining the number of data-points in our data.
    """
    pixels = 100
    xvalues = np.arange(pixels)

    """
    Evaluate the `Gaussian` and Exponential model instances at every xvalues to create their model profile and sum
    them together to create the overall model profile.
    """
    model_line = np.zeros(shape=pixels)

    for profile in profiles:

        model_line += profile.profile_from_xvalues(xvalues=xvalues)

    """Create a Gaussian kernel which the model line will be convolved with."""
    kernel_pixels = 21
    kernel_xvalues = np.arange(kernel_pixels)
    kernel_sigma = 5.0
    kernel_centre = 10.0
    kernel_xvalues = np.subtract(kernel_xvalues, kernel_centre)
    kernel = np.multiply(
        np.divide(1.0, kernel_sigma * np.sqrt(2.0 * np.pi)),
        np.exp(-0.5 * np.square(np.divide(kernel_xvalues, kernel_sigma))),
    )
    kernel = kernel / np.sum(kernel)

    "Convolve the model line with this kernel."

    blurred_model_line = np.convolve(model_line, kernel, mode="same")

    """Determine the noise (at a specified signal to noise level) in every pixel of our model profile."""
    signal_to_noise_ratio = 25.0
    noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

    """
    Add this noise to the model line to create the line data that is fitted, using the signal-to-noise ratio to compute
    noise-map of our data which is required when evaluating the chi-squared value of the likelihood.
    """
    data = blurred_model_line + noise
    noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

    """
    Output the data and noise-map to the `autofit_workspace/dataset` folder so they can be loaded and used 
    in other example scripts.
    """
    af.util.numpy_array_to_json(
        array=data, file_path=path.join(dataset_path, "data.json"), overwrite=True
    )
    af.util.numpy_array_to_json(
        array=noise_map,
        file_path=path.join(dataset_path, "noise_map.json"),
        overwrite=True,
    )
    af.util.numpy_array_to_json(
        array=kernel, file_path=path.join(dataset_path, "kernel.json"), overwrite=True
    )

    plt.errorbar(
        x=xvalues,
        y=data,
        yerr=noise_map,
        color="k",
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.title("1D Profiles Dataset with Kernel2D Blurring.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile intensity")
    plt.savefig(path.join(dataset_path, "image.png"))
    plt.close()
