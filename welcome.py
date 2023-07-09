from os import path

input(
    "\n"
    "############################################\n"
    "### AUTOFIT WORKSPACE WORKING DIRECTORY ###\n"
    "############################################\n\n"
    """
    PyAutoFit scripts assume that the `autofit_workspace` directory is the Python working directory. 
    This means  that, when you run an example script, you should run it from the `autofit_workspace` 
    as follows:


    cd path/to/autofit_workspace (if you are not already in the autofit_workspace).
    python3 scripts/overview/simple/fit.py

    The reasons for this are so that PyAutoFit can:

    - Load configuration settings from config files in the `autofit_workspace/config` folder.
    - Load example data from the `autofit_workspace/dataset` folder.
    - Output the results of models fits to your hard-disk to the `autofit/output` folder. 

    Jupyter notebooks update the current working directory to the `autofit_workspace` directory via a magicmethod.

    If you have any errors relating to importing modules, loading data or outputting results it is likely because you
    are not running the script with the `autofit_workspace` as the working directory!

    [Press Enter to continue]"""
)


import autofit as af
import matplotlib.pyplot as plt

dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

plt.plot(range(data.shape[0]), data)
plt.show()
plt.close()

input(
    "\n"
    "###############################\n"
    "## EXAMPLE NOISY 1D GAUSIAN ###\n"
    "###############################\n\n"
    """
    The image displayed on your screen shows noisy 1D data of a Gaussian.
    
    The example of fitting noisy 1D profiles is used throughout the autofit_workspace to illustrate the PyAutoFit
    API and the different statistical techniques and methods available.
    
    [Press Enter to continue]
    """
)

input(
    ""
    "\n"
    "###############################\n"
    "######## WORKSPACE TOUR #######\n"
    "###############################\n\n"
    """
    PyAutoFit is now set up and you can begin exploring the workspace.  We recommend new users begin by following the
    'overview_1_the_basics' notebook, which gives an overview of PyAutoFit and the workspace.

    Examples are provided as both Jupyter notebooks in the 'notebooks' folder and Python scripts in the 'scripts'
    folder. It is up to you how you would prefer to use PyAutoFit. With these folders, you can find the following
    packages:
    
    - howtofit: Jupyter notebook providing new users with a step-by-step guide on how
    to compose a model, fit it, and analyse the results.
    
    - overview: An overview of how to quickly compose and fit a model in PyAutoFit.
    
    - features: Examples of PyAutoFit's advanced modeling features such as search chaining and sensitivity mapping.
    
    Once you a familiar with the PyAutoFit API you should be ready to use it to compose and fit models for your
    model-fitting problem!
    """
)
