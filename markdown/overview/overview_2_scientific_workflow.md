> ✏️ **This page is auto-generated from [`scripts/overview/overview_2_scientific_workflow.py`](../../scripts/overview/overview_2_scientific_workflow.py) — do not edit it directly.**
> It shows the example fully executed, with its real output images.
> Run it yourself via the [Python script](../../scripts/overview/overview_2_scientific_workflow.py) or the [Jupyter notebook](../../notebooks/overview/overview_2_scientific_workflow.ipynb).

Overview: Scientific Workflow
=============================

A scientific workflow comprises the tasks you perform to conduct a scientific study. This includes fitting models to
datasets, interpreting the results, and gaining insights into your scientific problem.

Different problems require different scientific workflows, depending on factors such as model complexity, dataset size,
and computational run times. For example, some problems involve fitting a single dataset with many models to gain
scientific insights, while others involve fitting thousands of datasets with a single model for large-scale studies.

The **PyAutoFit** API is flexible, customizable, and extensible, enabling users to develop scientific workflows
tailored to their specific problems.

This overview covers the key features of **PyAutoFit** that support the development of effective scientific workflows:

- **On The Fly**: Display results immediately (e.g., in Jupyter notebooks) to provide instant feedback for adapting your workflow.
- **Hard Disk Output**: Output results to hard disk with high customization, allowing quick and detailed inspection of fits to many datasets.
- **Visualization**: Generate model-specific visualizations to create custom plots that streamline result inspection.
- **Loading Results**: Load results from the hard disk to inspect and interpret the outcomes of a model fit.
- **Result Customization**: Customize the returned results to simplify scientific interpretation.
- **Model Composition**: Extensible model composition makes it easy to fit many models with different parameterizations and assumptions.
- **Searches**: Support for various non-linear searches (e.g., nested sampling, MCMC), including gradient based fitting using JAX, to find the right method for your problem.
- **Configs**: Configuration files that set default model, fitting, and visualization behaviors, streamlining model fitting.
- **Database**: Store results in a relational SQLite3 database, enabling efficient management of large modeling results.
- **Scaling Up**: Guidance on scaling up your scientific workflow from small to large datasets.

__Contents__

This overview is split into the following sections:

- **Data**: Load the 1D Gaussian data from disk to illustrate the scientific workflow.
- **On The Fly**: Display intermediate results during model fitting for instant feedback.
- **Hard Disk Output**: Enable persistent saving of search results with customizable output structure.
- **Visualization**: Generate model-specific visualizations saved to disk during fitting.
- **Loading Results**: Use the Aggregator API to load and inspect results from hard disk.
- **Result Customization**: Extend the Result class with custom properties specific to the model-fitting problem.
- **Model Composition**: Construct diverse models with parameter assignments and complex hierarchies.
- **Searches**: Select and customize non-linear search methods appropriate for the problem.
- **Configs**: Use configuration files to define default model priors and search settings.
- **Database**: Store and query results in a SQLite3 relational database.
- **Scaling Up**: Guidance on expanding workflows from small to large datasets.
- **Wrap Up**: Summary of scientific workflow features in PyAutoFit.


```python

from autoconf import setup_notebook; setup_notebook()

import numpy as np
from typing import Optional
from os import path

import autofit as af
```

    Working Directory has been set to `autofit_workspace`


__Data__

To illustrate a few aspects of the scientific workflow, we'll fit a 1D Gaussian profile to data, which
we load from hard-disk.


```python
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
```

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.


```python
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

```

__On The Fly__

The on-the-fly feature described below is not implemented yet, we are working on it currently.
The best way to get on-the-fly output is to output to hard-disk, which is described in the next section.
This feature is fully implemented and provides on-the-fly output of results to hard-disk.

When a model fit is running, information about the fit is displayed at user-specified intervals.

The frequency of this on-the-fly output is controlled by a search's `iterations_per_full_update` parameter, which
specifies how often this information is output. The example code below outputs on-the-fly information every 1000 
iterations:


```python
search = af.DynestyStatic(iterations_per_full_update=1000)
```

In a Jupyter notebook, the default behavior is for this information to appear in the cell being run and to include:

- Text displaying the maximum likelihood model inferred so far and related information.
- A visual showing how the search has sampled parameter space so far, providing intuition on how the search is 
performing.

Here is an image of how this looks:

![Example On-the-Fly Output](path/to/image.png)

The most valuable on-the-fly output is often specific to the model and dataset you are fitting. For instance, it
might be a ``matplotlib`` subplot showing the maximum likelihood model's fit to the dataset, complete with residuals
and other diagnostic information.

The on-the-fly output can be fully customized by extending the ``on_the_fly_output`` method of the ``Analysis``
class being used to fit the model.

The example below shows how this is done for the simple case of fitting a 1D Gaussian profile:


```python


class Analysis(af.Analysis):
    def __init__(self, data: np.ndarray, noise_map: np.ndarray):
        """
        Example Analysis class illustrating how to customize the on-the-fly output of a model-fit.
        """
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def on_the_fly_output(self, instance):
        """
        During a model-fit, the `on_the_fly_output` method is called throughout the non-linear search.

        The `instance` passed into the method is maximum log likelihood solution obtained by the model-fit so far and it can be
        used to provide on-the-fly output showing how the model-fit is going.
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_from(xvalues=xvalues)

        """
        The visualizer now outputs images of the best-fit results to hard-disk (checkout `visualizer.py`).
        """
        import matplotlib.pyplot as plt

        plt.errorbar(
            x=xvalues,
            y=self.data,
            yerr=self.noise_map,
            linestyle="",
            color="k",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
        plt.plot(xvalues, model_data, color="r")
        plt.title("Maximum Likelihood Fit")
        plt.xlabel("x value of profile")
        plt.ylabel("Profile Normalization")
        plt.show()  # By using `plt.show()` the plot will be displayed in the Jupyter notebook.

```

Here's how the visuals appear in a Jupyter Notebook:

![Example On-the-Fly Output](path/to/image.png)

In the early stages of setting up a scientific workflow, on-the-fly output is invaluable. It provides immediate
feedback on how your model fitting is performing, which is often crucial at the beginning of a project when things
might not be going well. It also encourages you to prioritize visualizing your fit and diagnosing whether the process
is working correctly.

We highly recommend users starting a new model-fitting problem begin by setting up on-the-fly output!

__Hard Disk Output__

By default, a non-linear search does not save its results to the hard disk; the results can only be inspected in 
a Jupyter Notebook or Python script via the returned `result`.

However, you can enable the output of non-linear search results to the hard disk by specifying 
the `name` and/or `path_prefix` attributes. These attributes determine how files are named and where results 
are saved on your hard disk.

Benefits of saving results to the hard disk include:

- More efficient inspection of results for multiple datasets compared to using a Jupyter Notebook.
- Results are saved on-the-fly, allowing you to check the progress of a fit midway.
- Additional information about a fit, such as visualizations, can be saved (see below).
- Unfinished runs can be resumed from where they left off if they are terminated.
- On high-performance supercomputers, results often need to be saved in this manner.

Here's how to enable the output of results to the hard disk:


```python
search = af.Emcee(path_prefix=path.join("folder_0", "folder_1"), name="my_search_name")
```

The screenshot below shows the output folder where all output is enabled:

.. image:: https://raw.githubusercontent.com/PyAutoLabs/PyAutoFit/main/docs/overview/image/output_example.png
  :width: 400
  :alt: Alternative text

Let's break down the output folder generated by **PyAutoFit**:

- **Unique Identifier**: Results are saved in a folder named with a unique identifier composed of random characters. 
  This identifier is automatically generated based on the specific model fit. For scientific workflows involving 
  numerous model fits, this ensures that each fit is uniquely identified without requiring manual updates to output paths.

- **Info Files**: These files contain valuable information about the fit. For instance, `model.info` provides the 
  complete model composition used in the fit, while `search.summary` details how long the search has been running 
  and other relevant search-specific information.

- **Files Folder**: Within the output folder, the `files` directory contains detailed information saved as `.json` 
  files. For example, `model.json` stores the model configuration used in the fit. This enables researchers to 
  revisit the results later and review how the fit was performed.

**PyAutoFit** offers extensive tools for customizing hard-disk output. This includes using configuration files to 
control what information is saved, which helps manage disk space utilization. Additionally, specific `.json` files 
tailored to different models can be utilized for more detailed output.

For many scientific workflows, having detailed output for each fit is crucial for thorough inspection and accurate
interpretation of results. However, in scenarios where the volume of output data might overwhelm users or impede
scientific study, this feature can be easily disabled by omitting the `name` or `path prefix` when initiating the search.

__Visualization__

When search hard-disk output is enabled in **PyAutoFit**, the visualization of model fits can also be saved directly
to disk. This capability is crucial for many scientific workflows as it allows for quick and effective assessment of
fit quality.

To accomplish this, you can customize the `Visualizer` object of an `Analysis` class with a custom `Visualizer` class.
This custom class is responsible for generating and saving visual representations of the model fits. By leveraging
this approach, scientists can efficiently visualize and analyze the outcomes of model fitting processes.


```python


class Visualizer(af.Visualizer):
    @staticmethod
    def visualize_before_fit(
        analysis, paths: af.DirectoryPaths, model: af.AbstractPriorModel
    ):
        """
        Before a model-fit, the `visualize_before_fit` method is called to perform visualization.

        The function receives as input an instance of the `Analysis` class which is being used to perform the fit,
        which is used to perform the visualization (e.g. it contains the data and noise map which are plotted).

        This can output visualization of quantities which do not change during the model-fit, for example the
        data and noise-map.

        The `paths` object contains the path to the folder where the visualization should be output, which is determined
        by the non-linear search `name` and other inputs.
        """

        import matplotlib.pyplot as plt

        xvalues = np.arange(analysis.data.shape[0])

        plt.errorbar(
            x=xvalues,
            y=analysis.data,
            yerr=analysis.noise_map,
            linestyle="",
            color="k",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
        plt.title("Maximum Likelihood Fit")
        plt.xlabel("x value of profile")
        plt.ylabel("Profile Normalization")
        plt.savefig(path.join(paths.image_path, f"data.png"))
        plt.clf()

    @staticmethod
    def visualize(analysis, paths: af.DirectoryPaths, instance, during_analysis):
        """
        During a model-fit, the `visualize` method is called throughout the non-linear search.

        The function receives as input an instance of the `Analysis` class which is being used to perform the fit,
        which is used to perform the visualization (e.g. it generates the model data which is plotted).

        The `instance` passed into the visualize method is maximum log likelihood solution obtained by the model-fit
        so far and it can be used to provide on-the-fly images showing how the model-fit is going.

        The `paths` object contains the path to the folder where the visualization should be output, which is determined
        by the non-linear search `name` and other inputs.
        """
        xvalues = np.arange(analysis.data.shape[0])

        model_data = instance.model_data_from(xvalues=xvalues)
        residual_map = analysis.data - model_data

        """
        The visualizer now outputs images of the best-fit results to hard-disk (checkout `visualizer.py`).
        """
        import matplotlib.pyplot as plt

        plt.errorbar(
            x=xvalues,
            y=analysis.data,
            yerr=analysis.noise_map,
            linestyle="",
            color="k",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
        plt.plot(xvalues, model_data, color="r")
        plt.title("Maximum Likelihood Fit")
        plt.xlabel("x value of profile")
        plt.ylabel("Profile Normalization")
        plt.savefig(path.join(paths.image_path, f"model_fit.png"))
        plt.clf()

        plt.errorbar(
            x=xvalues,
            y=residual_map,
            yerr=analysis.noise_map,
            linestyle="",
            color="k",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
        plt.title("Residuals of Maximum Likelihood Fit")
        plt.xlabel("x value of profile")
        plt.ylabel("Residual")
        plt.savefig(path.join(paths.image_path, f"model_fit.png"))
        plt.clf()

```

The ``Analysis`` class is defined following the same API as before, but now with its `Visualizer` class attribute
overwritten with the ``Visualizer`` class above.


```python


class Analysis(af.Analysis):
    """
    This over-write means the `Visualizer` class is used for visualization throughout the model-fit.

    This `VisualizerExample` object is in the `autofit.example.visualize` module and is used to customize the
    plots output during the model-fit.

    It has been extended with visualize methods that output visuals specific to the fitting of `1D` data.
    """

    Visualizer = Visualizer

    def __init__(self, data, noise_map):
        """
        An Analysis class which illustrates visualization.
        """
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        """
        The `log_likelihood_function` is identical to the example above
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_from(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

```

Visualization of the results of the non-linear search, for example the "Probability Density
Function", are also automatically output during the model-fit on the fly.

We now perform a quick fit, outputting the results to the hard disk and visualizing the model-fit,
so you can see how the results are output and the visualizations produced.


```python
analysis = Analysis(data=data, noise_map=noise_map)

model = af.Model(af.ex.Gaussian)

search = af.DynestyStatic(
    path_prefix=path.join("result_folder"), name="overview_2_scientific_workflow"
)

result = search.fit(model=model, analysis=analysis)
```

    2026-07-10 18:09:38,284 - autofit.non_linear.search.abstract_search - INFO - Starting non-linear search with 1 cores.


    2026-07-10 18:09:38,321 - overview_2_scientific_workflow - INFO - The output path of this fit is autofit_workspace/output/result_folder/overview_2_scientific_workflow/014dd38ae16c1cc2473db87d25577c42


    2026-07-10 18:09:38,324 - overview_2_scientific_workflow - INFO - Outputting pre-fit files (e.g. model.info, visualization).


    2026-07-10 18:09:38,782 - overview_2_scientific_workflow - INFO - Starting new Dynesty non-linear search (no previous samples found).


    2026-07-10 18:09:39,196 - autofit.non_linear.initializer - INFO - Generating initial samples of model using JAX LH Function cores


    2026-07-10 18:09:39,237 - autofit.non_linear.initializer - INFO - Initial samples generated, starting non-linear search


    /usr/lib/python3.12/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      self.pid = os.fork()


    0it [00:00, ?it/s]

    15it [00:00, 142.74it/s, bound: 0 | nc: 4 | ncall: 70 | eff(%): 21.429 | loglstar:   -inf < -157997.480 <    inf | logz: -158002.382 +/-  0.312 | dlogz: 203044.632 >  0.059]

    30it [00:00, 132.00it/s, bound: 0 | nc: 4 | ncall: 99 | eff(%): 30.303 | loglstar:   -inf < -5243.037 <    inf | logz: -5245.470 +/-  0.219 | dlogz: 2304.990 >  0.059]      

    44it [00:00, 92.33it/s, bound: 0 | nc: 7 | ncall: 134 | eff(%): 32.836 | loglstar:   -inf < -5243.023 <    inf | logz: -5244.543 +/-  0.173 | dlogz: 2303.690 >  0.059]

    55it [00:00, 84.69it/s, bound: 0 | nc: 1 | ncall: 170 | eff(%): 32.353 | loglstar:   -inf < -5242.911 <    inf | logz: -5244.214 +/-  0.154 | dlogz: 2303.129 >  0.059]

    65it [00:00, 57.13it/s, bound: 0 | nc: 5 | ncall: 226 | eff(%): 28.761 | loglstar:   -inf < -5242.568 <    inf | logz: -5243.979 +/-  0.142 | dlogz: 2302.694 >  0.059]

    72it [00:01, 35.28it/s, bound: 0 | nc: 10 | ncall: 269 | eff(%): 26.766 | loglstar:   -inf < -5242.035 <    inf | logz: -5243.793 +/-  0.138 | dlogz: 2351.478 >  0.059]

    79it [00:01, 39.99it/s, bound: 0 | nc: 4 | ncall: 300 | eff(%): 26.333 | loglstar:   -inf < -5240.816 <    inf | logz: -5243.472 +/-  0.147 | dlogz: 2351.051 >  0.059] 

    87it [00:01, 44.80it/s, bound: 0 | nc: 11 | ncall: 343 | eff(%): 25.364 | loglstar:   -inf < -5239.097 <    inf | logz: -5242.705 +/-  0.186 | dlogz: 4335.725 >  0.059]

    93it [00:01, 47.07it/s, bound: 0 | nc: 4 | ncall: 368 | eff(%): 25.272 | loglstar:   -inf < -5235.086 <    inf | logz: -5240.314 +/-  0.278 | dlogz: 4333.630 >  0.059] 

    100it [00:01, 50.74it/s, bound: 0 | nc: 6 | ncall: 405 | eff(%): 24.691 | loglstar:   -inf < -5232.453 <    inf | logz: -5237.544 +/-  0.279 | dlogz: 4330.579 >  0.059]

    106it [00:01, 52.24it/s, bound: 0 | nc: 2 | ncall: 437 | eff(%): 24.256 | loglstar:   -inf < -5227.422 <    inf | logz: -5233.272 +/-  0.316 | dlogz: 4326.806 >  0.059]

    112it [00:02, 46.38it/s, bound: 0 | nc: 2 | ncall: 470 | eff(%): 23.830 | loglstar:   -inf < -5217.605 <    inf | logz: -5223.095 +/-  0.312 | dlogz: 4316.040 >  0.059]

    118it [00:02, 30.58it/s, bound: 0 | nc: 9 | ncall: 505 | eff(%): 23.366 | loglstar:   -inf < -5207.866 <    inf | logz: -5213.798 +/-  0.332 | dlogz: 4307.109 >  0.059]

    123it [00:02, 27.69it/s, bound: 0 | nc: 8 | ncall: 548 | eff(%): 22.445 | loglstar:   -inf < -5191.425 <    inf | logz: -5198.417 +/-  0.367 | dlogz: 4294.311 >  0.059]

    127it [00:02, 24.89it/s, bound: 0 | nc: 26 | ncall: 591 | eff(%): 21.489 | loglstar:   -inf < -5179.194 <    inf | logz: -5186.312 +/-  0.375 | dlogz: 4285.462 >  0.059]

    131it [00:03, 19.59it/s, bound: 0 | nc: 11 | ncall: 633 | eff(%): 20.695 | loglstar:   -inf < -5173.249 <    inf | logz: -5179.841 +/-  0.345 | dlogz: 4273.235 >  0.059]

    134it [00:03, 18.36it/s, bound: 0 | nc: 35 | ncall: 684 | eff(%): 19.591 | loglstar:   -inf < -5167.272 <    inf | logz: -5174.385 +/-  0.364 | dlogz: 4268.926 >  0.059]

    141it [00:03, 21.80it/s, bound: 0 | nc: 32 | ncall: 741 | eff(%): 19.028 | loglstar:   -inf < -5141.912 <    inf | logz: -5149.307 +/-  0.382 | dlogz: 4415.692 >  0.059]

    144it [00:03, 20.14it/s, bound: 0 | nc: 7 | ncall: 791 | eff(%): 18.205 | loglstar:   -inf < -5130.026 <    inf | logz: -5136.567 +/-  0.348 | dlogz: 4397.119 >  0.059] 

    147it [00:04, 19.93it/s, bound: 0 | nc: 4 | ncall: 831 | eff(%): 17.690 | loglstar:   -inf < -5112.045 <    inf | logz: -5119.488 +/-  0.378 | dlogz: 4540.450 >  0.059]

    150it [00:04, 18.87it/s, bound: 0 | nc: 23 | ncall: 890 | eff(%): 16.854 | loglstar:   -inf < -5084.863 <    inf | logz: -5092.428 +/-  0.386 | dlogz: 4515.225 >  0.059]

    153it [00:04, 19.16it/s, bound: 0 | nc: 12 | ncall: 920 | eff(%): 16.630 | loglstar:   -inf < -5051.112 <    inf | logz: -5058.651 +/-  0.379 | dlogz: 4479.231 >  0.059]

    155it [00:04, 16.44it/s, bound: 0 | nc: 42 | ncall: 963 | eff(%): 16.096 | loglstar:   -inf < -5034.564 <    inf | logz: -5042.236 +/-  0.389 | dlogz: 4466.199 >  0.059]

    157it [00:04, 15.26it/s, bound: 0 | nc: 8 | ncall: 1007 | eff(%): 15.591 | loglstar:   -inf < -5013.187 <    inf | logz: -5020.901 +/-  0.391 | dlogz: 4453.329 >  0.059]

    159it [00:04, 15.60it/s, bound: 0 | nc: 16 | ncall: 1039 | eff(%): 15.303 | loglstar:   -inf < -4992.975 <    inf | logz: -5000.475 +/-  0.374 | dlogz: 4420.037 >  0.059]

    161it [00:04, 16.25it/s, bound: 0 | nc: 3 | ncall: 1066 | eff(%): 15.103 | loglstar:   -inf < -4932.144 <    inf | logz: -4939.938 +/-  0.393 | dlogz: 4396.625 >  0.059] 

    163it [00:05, 10.37it/s, bound: 0 | nc: 73 | ncall: 1186 | eff(%): 13.744 | loglstar:   -inf < -4908.345 <    inf | logz: -4916.178 +/-  0.394 | dlogz: 4352.588 >  0.059]

    165it [00:05, 10.70it/s, bound: 0 | nc: 7 | ncall: 1226 | eff(%): 13.458 | loglstar:   -inf < -4894.144 <    inf | logz: -4902.016 +/-  0.395 | dlogz: 4329.352 >  0.059] 

    167it [00:05, 11.88it/s, bound: 0 | nc: 26 | ncall: 1263 | eff(%): 13.222 | loglstar:   -inf < -4831.562 <    inf | logz: -4839.474 +/-  0.396 | dlogz: 4303.195 >  0.059]

    171it [00:05, 16.66it/s, bound: 0 | nc: 7 | ncall: 1291 | eff(%): 13.246 | loglstar:   -inf < -4745.146 <    inf | logz: -4753.137 +/-  0.398 | dlogz: 4187.421 >  0.059] 

    174it [00:06, 13.45it/s, bound: 0 | nc: 58 | ncall: 1369 | eff(%): 12.710 | loglstar:   -inf < -4686.061 <    inf | logz: -4694.112 +/-  0.399 | dlogz: 4134.590 >  0.059]

    176it [00:06, 14.10it/s, bound: 0 | nc: 21 | ncall: 1409 | eff(%): 12.491 | loglstar:   -inf < -4670.819 <    inf | logz: -4678.909 +/-  0.400 | dlogz: 4800.400 >  0.059]

    178it [00:06, 12.14it/s, bound: 0 | nc: 4 | ncall: 1463 | eff(%): 12.167 | loglstar:   -inf < -4627.449 <    inf | logz: -4635.579 +/-  0.401 | dlogz: 4759.342 >  0.059] 

    181it [00:06, 14.31it/s, bound: 0 | nc: 6 | ncall: 1484 | eff(%): 12.197 | loglstar:   -inf < -4574.655 <    inf | logz: -4582.844 +/-  0.403 | dlogz: 4703.374 >  0.059]

    184it [00:06, 14.89it/s, bound: 0 | nc: 32 | ncall: 1528 | eff(%): 12.042 | loglstar:   -inf < -4484.922 <    inf | logz: -4493.170 +/-  0.404 | dlogz: 4670.896 >  0.059]

    187it [00:06, 15.14it/s, bound: 0 | nc: 26 | ncall: 1573 | eff(%): 11.888 | loglstar:   -inf < -4472.417 <    inf | logz: -4480.704 +/-  0.403 | dlogz: 4596.843 >  0.059]

    189it [00:07, 10.42it/s, bound: 0 | nc: 78 | ncall: 1658 | eff(%): 11.399 | loglstar:   -inf < -4405.502 <    inf | logz: -4413.850 +/-  0.407 | dlogz: 4589.845 >  0.059]

    191it [00:07, 10.77it/s, bound: 0 | nc: 45 | ncall: 1708 | eff(%): 11.183 | loglstar:   -inf < -4299.741 <    inf | logz: -4308.129 +/-  0.408 | dlogz: 4473.963 >  0.059]

    193it [00:07, 11.78it/s, bound: 0 | nc: 22 | ncall: 1749 | eff(%): 11.035 | loglstar:   -inf < -4227.762 <    inf | logz: -4236.189 +/-  0.409 | dlogz: 4412.550 >  0.059]

    196it [00:07, 10.22it/s, bound: 0 | nc: 104 | ncall: 1885 | eff(%): 10.398 | loglstar:   -inf < -4186.515 <    inf | logz: -4195.001 +/-  0.410 | dlogz: 4315.885 >  0.059]

    198it [00:08, 10.71it/s, bound: 0 | nc: 44 | ncall: 1952 | eff(%): 10.143 | loglstar:   -inf < -4122.162 <    inf | logz: -4130.688 +/-  0.411 | dlogz: 4266.029 >  0.059] 

    200it [00:08, 10.40it/s, bound: 0 | nc: 64 | ncall: 2018 | eff(%):  9.911 | loglstar:   -inf < -4029.276 <    inf | logz: -4037.841 +/-  0.412 | dlogz: 4186.987 >  0.059]

    205it [00:08, 16.97it/s, bound: 1 | nc: 3 | ncall: 2032 | eff(%): 10.089 | loglstar:   -inf < -3622.038 <    inf | logz: -3630.703 +/-  0.414 | dlogz: 3815.145 >  0.059] 

    210it [00:08, 23.28it/s, bound: 1 | nc: 9 | ncall: 2065 | eff(%): 10.169 | loglstar:   -inf < -3505.835 <    inf | logz: -3514.598 +/-  0.416 | dlogz: 3634.050 >  0.059]

    217it [00:08, 26.28it/s, bound: 2 | nc: 31 | ncall: 2117 | eff(%): 10.250 | loglstar:   -inf < -3290.372 <    inf | logz: -3299.274 +/-  0.420 | dlogz: 3426.989 >  0.059]

    225it [00:08, 35.10it/s, bound: 2 | nc: 8 | ncall: 2151 | eff(%): 10.460 | loglstar:   -inf < -3132.414 <    inf | logz: -3141.475 +/-  0.424 | dlogz: 3311.839 >  0.059] 

    230it [00:08, 38.16it/s, bound: 2 | nc: 2 | ncall: 2166 | eff(%): 10.619 | loglstar:   -inf < -2979.841 <    inf | logz: -2989.001 +/-  0.426 | dlogz: 3123.844 >  0.059]

    235it [00:09, 38.93it/s, bound: 3 | nc: 6 | ncall: 2179 | eff(%): 10.785 | loglstar:   -inf < -2890.936 <    inf | logz: -2900.195 +/-  0.428 | dlogz: 3031.254 >  0.059]

    246it [00:09, 55.37it/s, bound: 3 | nc: 7 | ncall: 2203 | eff(%): 11.167 | loglstar:   -inf < -2598.394 <    inf | logz: -2607.871 +/-  0.433 | dlogz: 2732.715 >  0.059]

    255it [00:09, 54.82it/s, bound: 3 | nc: 15 | ncall: 2243 | eff(%): 11.369 | loglstar:   -inf < -2393.041 <    inf | logz: -2402.695 +/-  0.437 | dlogz: 2538.259 >  0.059]

    264it [00:09, 63.00it/s, bound: 4 | nc: 1 | ncall: 2256 | eff(%): 11.702 | loglstar:   -inf < -2202.915 <    inf | logz: -2212.748 +/-  0.441 | dlogz: 2344.522 >  0.059] 

    272it [00:09, 63.60it/s, bound: 4 | nc: 3 | ncall: 2273 | eff(%): 11.967 | loglstar:   -inf < -2017.486 <    inf | logz: -2027.478 +/-  0.445 | dlogz: 2186.811 >  0.059]

    279it [00:09, 59.95it/s, bound: 4 | nc: 8 | ncall: 2303 | eff(%): 12.115 | loglstar:   -inf < -1945.827 <    inf | logz: -1955.957 +/-  0.448 | dlogz: 2079.147 >  0.059]

    286it [00:09, 50.19it/s, bound: 5 | nc: 2 | ncall: 2324 | eff(%): 12.306 | loglstar:   -inf < -1767.471 <    inf | logz: -1777.733 +/-  0.450 | dlogz: 1893.155 >  0.059]

    298it [00:10, 64.26it/s, bound: 5 | nc: 1 | ncall: 2345 | eff(%): 12.708 | loglstar:   -inf < -1384.187 <    inf | logz: -1394.292 +/-  0.437 | dlogz: 1505.508 >  0.059]

    306it [00:10, 61.59it/s, bound: 5 | nc: 1 | ncall: 2372 | eff(%): 12.901 | loglstar:   -inf < -1311.423 <    inf | logz: -1321.108 +/-  0.435 | dlogz: 1431.527 >  0.059]

    313it [00:10, 58.32it/s, bound: 6 | nc: 3 | ncall: 2394 | eff(%): 13.074 | loglstar:   -inf < -1245.725 <    inf | logz: -1256.528 +/-  0.462 | dlogz: 1374.686 >  0.059]

    320it [00:10, 51.23it/s, bound: 6 | nc: 6 | ncall: 2433 | eff(%): 13.152 | loglstar:   -inf < -1149.043 <    inf | logz: -1158.886 +/-  0.441 | dlogz: 1268.962 >  0.059]

    330it [00:10, 54.88it/s, bound: 6 | nc: 11 | ncall: 2465 | eff(%): 13.387 | loglstar:   -inf < -985.554 <    inf | logz: -996.694 +/-  0.470 | dlogz: 1116.332 >  0.059] 

    336it [00:10, 44.42it/s, bound: 7 | nc: 3 | ncall: 2484 | eff(%): 13.527 | loglstar:   -inf < -881.158 <    inf | logz: -892.416 +/-  0.472 | dlogz: 1015.231 >  0.059] 

    341it [00:11, 38.88it/s, bound: 7 | nc: 9 | ncall: 2513 | eff(%): 13.569 | loglstar:   -inf < -777.807 <    inf | logz: -789.165 +/-  0.474 | dlogz: 915.123 >  0.059] 

    347it [00:11, 42.83it/s, bound: 7 | nc: 4 | ncall: 2535 | eff(%): 13.688 | loglstar:   -inf < -750.217 <    inf | logz: -760.869 +/-  0.454 | dlogz: 870.583 >  0.059]

    352it [00:11, 40.20it/s, bound: 8 | nc: 3 | ncall: 2550 | eff(%): 13.804 | loglstar:   -inf < -675.108 <    inf | logz: -686.665 +/-  0.477 | dlogz: 799.698 >  0.059]

    365it [00:11, 57.49it/s, bound: 8 | nc: 8 | ncall: 2581 | eff(%): 14.142 | loglstar:   -inf < -547.755 <    inf | logz: -559.588 +/-  0.484 | dlogz: 686.725 >  0.059]

    374it [00:11, 64.65it/s, bound: 8 | nc: 4 | ncall: 2605 | eff(%): 14.357 | loglstar:   -inf < -493.766 <    inf | logz: -505.430 +/-  0.468 | dlogz: 623.491 >  0.059]

    382it [00:11, 56.45it/s, bound: 9 | nc: 1 | ncall: 2625 | eff(%): 14.552 | loglstar:   -inf < -448.073 <    inf | logz: -459.012 +/-  0.458 | dlogz: 575.920 >  0.059]

    389it [00:11, 55.01it/s, bound: 9 | nc: 6 | ncall: 2645 | eff(%): 14.707 | loglstar:   -inf < -377.028 <    inf | logz: -389.008 +/-  0.476 | dlogz: 506.917 >  0.059]

    398it [00:11, 61.27it/s, bound: 9 | nc: 3 | ncall: 2670 | eff(%): 14.906 | loglstar:   -inf < -332.106 <    inf | logz: -344.210 +/-  0.480 | dlogz: 461.901 >  0.059]

    407it [00:12, 61.93it/s, bound: 10 | nc: 4 | ncall: 2697 | eff(%): 15.091 | loglstar:   -inf < -285.994 <    inf | logz: -298.659 +/-  0.501 | dlogz: 460.598 >  0.059]

    423it [00:12, 84.16it/s, bound: 10 | nc: 4 | ncall: 2726 | eff(%): 15.517 | loglstar:   -inf < -200.092 <    inf | logz: -213.074 +/-  0.507 | dlogz: 374.194 >  0.059]

    433it [00:12, 87.62it/s, bound: 10 | nc: 1 | ncall: 2748 | eff(%): 15.757 | loglstar:   -inf < -153.152 <    inf | logz: -166.332 +/-  0.511 | dlogz: 332.117 >  0.059]

    443it [00:12, 60.77it/s, bound: 11 | nc: 3 | ncall: 2776 | eff(%): 15.958 | loglstar:   -inf < -104.802 <    inf | logz: -118.122 +/-  0.510 | dlogz: 273.361 >  0.059]

    451it [00:12, 59.41it/s, bound: 11 | nc: 3 | ncall: 2798 | eff(%): 16.119 | loglstar:   -inf < -71.743 <    inf | logz: -85.172 +/-  0.510 | dlogz: 239.694 >  0.059]  

    458it [00:12, 56.75it/s, bound: 11 | nc: 7 | ncall: 2820 | eff(%): 16.241 | loglstar:   -inf < -42.430 <    inf | logz: -55.985 +/-  0.511 | dlogz: 210.083 >  0.059]

    465it [00:12, 59.15it/s, bound: 11 | nc: 4 | ncall: 2841 | eff(%): 16.367 | loglstar:   -inf < -23.054 <    inf | logz: -36.571 +/-  0.503 | dlogz: 213.229 >  0.059]

    472it [00:13, 59.27it/s, bound: 12 | nc: 6 | ncall: 2858 | eff(%): 16.515 | loglstar:   -inf < 17.111 <    inf | logz:  3.175 +/-  0.524 | dlogz: 176.608 >  0.059]  

    486it [00:13, 78.42it/s, bound: 12 | nc: 1 | ncall: 2897 | eff(%): 16.776 | loglstar:   -inf < 43.509 <    inf | logz: 29.553 +/-  0.516 | dlogz: 147.151 >  0.059]

    495it [00:13, 69.40it/s, bound: 13 | nc: 2 | ncall: 2925 | eff(%): 16.923 | loglstar:   -inf < 54.128 <    inf | logz: 40.367 +/-  0.503 | dlogz: 136.882 >  0.059]

    513it [00:13, 92.32it/s, bound: 13 | nc: 4 | ncall: 2956 | eff(%): 17.355 | loglstar:   -inf < 86.319 <    inf | logz: 71.619 +/-  0.535 | dlogz: 107.497 >  0.059]

    527it [00:13, 103.30it/s, bound: 13 | nc: 2 | ncall: 2981 | eff(%): 17.679 | loglstar:   -inf < 104.703 <    inf | logz: 90.867 +/-  0.509 | dlogz: 85.415 >  0.059]

    539it [00:13, 88.64it/s, bound: 14 | nc: 4 | ncall: 3010 | eff(%): 17.907 | loglstar:   -inf < 117.157 <    inf | logz: 102.737 +/-  0.514 | dlogz: 73.351 >  0.059]

    549it [00:13, 88.47it/s, bound: 14 | nc: 1 | ncall: 3036 | eff(%): 18.083 | loglstar:   -inf < 125.120 <    inf | logz: 110.805 +/-  0.518 | dlogz: 65.065 >  0.059]

    559it [00:14, 88.76it/s, bound: 14 | nc: 2 | ncall: 3062 | eff(%): 18.256 | loglstar:   -inf < 128.420 <    inf | logz: 114.854 +/-  0.502 | dlogz: 60.335 >  0.059]

    569it [00:14, 83.17it/s, bound: 15 | nc: 3 | ncall: 3082 | eff(%): 18.462 | loglstar:   -inf < 139.797 <    inf | logz: 125.168 +/-  0.532 | dlogz: 50.401 >  0.059]

    581it [00:14, 92.20it/s, bound: 15 | nc: 1 | ncall: 3104 | eff(%): 18.718 | loglstar:   -inf < 146.990 <    inf | logz: 131.597 +/-  0.532 | dlogz: 43.860 >  0.059]

    591it [00:14, 90.76it/s, bound: 15 | nc: 1 | ncall: 3132 | eff(%): 18.870 | loglstar:   -inf < 151.896 <    inf | logz: 137.150 +/-  0.528 | dlogz: 38.066 >  0.059]

    601it [00:14, 79.08it/s, bound: 16 | nc: 2 | ncall: 3158 | eff(%): 19.031 | loglstar:   -inf < 155.840 <    inf | logz: 140.947 +/-  0.527 | dlogz: 34.020 >  0.059]

    610it [00:14, 74.20it/s, bound: 16 | nc: 2 | ncall: 3185 | eff(%): 19.152 | loglstar:   -inf < 159.479 <    inf | logz: 144.736 +/-  0.528 | dlogz: 29.904 >  0.059]

    618it [00:14, 56.42it/s, bound: 17 | nc: 3 | ncall: 3225 | eff(%): 19.163 | loglstar:   -inf < 162.010 <    inf | logz: 147.155 +/-  0.525 | dlogz: 27.276 >  0.059]

    627it [00:15, 61.86it/s, bound: 17 | nc: 3 | ncall: 3246 | eff(%): 19.316 | loglstar:   -inf < 164.746 <    inf | logz: 149.870 +/-  0.526 | dlogz: 24.356 >  0.059]

    637it [00:15, 69.17it/s, bound: 17 | nc: 3 | ncall: 3268 | eff(%): 19.492 | loglstar:   -inf < 168.677 <    inf | logz: 153.185 +/-  0.539 | dlogz: 20.999 >  0.059]

    645it [00:15, 59.45it/s, bound: 18 | nc: 2 | ncall: 3297 | eff(%): 19.563 | loglstar:   -inf < 170.951 <    inf | logz: 156.059 +/-  0.535 | dlogz: 17.874 >  0.059]

    654it [00:15, 65.89it/s, bound: 18 | nc: 1 | ncall: 3324 | eff(%): 19.675 | loglstar:   -inf < 172.532 <    inf | logz: 157.611 +/-  0.529 | dlogz: 17.142 >  0.059]

    666it [00:15, 78.52it/s, bound: 18 | nc: 2 | ncall: 3354 | eff(%): 19.857 | loglstar:   -inf < 176.152 <    inf | logz: 160.686 +/-  0.538 | dlogz: 15.004 >  0.059]

    675it [00:15, 48.54it/s, bound: 19 | nc: 15 | ncall: 3404 | eff(%): 19.830 | loglstar:   -inf < 177.683 <    inf | logz: 162.451 +/-  0.537 | dlogz: 13.009 >  0.059]

    682it [00:16, 44.52it/s, bound: 19 | nc: 3 | ncall: 3441 | eff(%): 19.820 | loglstar:   -inf < 178.879 <    inf | logz: 163.425 +/-  0.535 | dlogz: 11.888 >  0.059] 

    688it [00:16, 38.67it/s, bound: 20 | nc: 4 | ncall: 3464 | eff(%): 19.861 | loglstar:   -inf < 179.353 <    inf | logz: 164.116 +/-  0.534 | dlogz: 11.057 >  0.059]

    693it [00:16, 40.05it/s, bound: 20 | nc: 1 | ncall: 3479 | eff(%): 19.920 | loglstar:   -inf < 180.020 <    inf | logz: 164.600 +/-  0.533 | dlogz: 10.469 >  0.059]

    699it [00:16, 43.46it/s, bound: 20 | nc: 3 | ncall: 3504 | eff(%): 19.949 | loglstar:   -inf < 180.741 <    inf | logz: 165.232 +/-  0.534 | dlogz:  9.722 >  0.059]

    705it [00:16, 46.36it/s, bound: 21 | nc: 1 | ncall: 3520 | eff(%): 20.028 | loglstar:   -inf < 181.754 <    inf | logz: 165.989 +/-  0.538 | dlogz:  8.871 >  0.059]

    716it [00:16, 60.49it/s, bound: 21 | nc: 2 | ncall: 3540 | eff(%): 20.226 | loglstar:   -inf < 182.298 <    inf | logz: 166.931 +/-  0.537 | dlogz:  7.647 >  0.059]

    725it [00:16, 66.88it/s, bound: 21 | nc: 3 | ncall: 3560 | eff(%): 20.365 | loglstar:   -inf < 183.339 <    inf | logz: 167.579 +/-  0.537 | dlogz:  6.834 >  0.059]

    733it [00:17, 61.69it/s, bound: 21 | nc: 2 | ncall: 3579 | eff(%): 20.481 | loglstar:   -inf < 184.265 <    inf | logz: 168.241 +/-  0.540 | dlogz:  6.072 >  0.059]

    740it [00:17, 56.83it/s, bound: 22 | nc: 2 | ncall: 3597 | eff(%): 20.573 | loglstar:   -inf < 184.906 <    inf | logz: 168.876 +/-  0.543 | dlogz:  5.297 >  0.059]

    750it [00:17, 64.94it/s, bound: 22 | nc: 5 | ncall: 3631 | eff(%): 20.655 | loglstar:   -inf < 185.534 <    inf | logz: 169.612 +/-  0.545 | dlogz:  4.348 >  0.059]

    758it [00:17, 65.62it/s, bound: 22 | nc: 7 | ncall: 3667 | eff(%): 20.671 | loglstar:   -inf < 185.935 <    inf | logz: 170.024 +/-  0.544 | dlogz:  3.903 >  0.059]

    765it [00:17, 64.02it/s, bound: 23 | nc: 1 | ncall: 3679 | eff(%): 20.794 | loglstar:   -inf < 186.131 <    inf | logz: 170.320 +/-  0.544 | dlogz:  3.470 >  0.059]

    777it [00:17, 65.90it/s, bound: 23 | nc: 2 | ncall: 3707 | eff(%): 20.960 | loglstar:   -inf < 186.568 <    inf | logz: 170.734 +/-  0.543 | dlogz:  3.483 >  0.059]

    786it [00:17, 69.58it/s, bound: 23 | nc: 2 | ncall: 3724 | eff(%): 21.106 | loglstar:   -inf < 186.944 <    inf | logz: 171.013 +/-  0.543 | dlogz:  3.041 >  0.059]

    794it [00:18, 40.16it/s, bound: 24 | nc: 1 | ncall: 3745 | eff(%): 21.202 | loglstar:   -inf < 187.166 <    inf | logz: 171.216 +/-  0.543 | dlogz:  2.697 >  0.059]

    800it [00:18, 28.97it/s, bound: 24 | nc: 2 | ncall: 3758 | eff(%): 21.288 | loglstar:   -inf < 187.334 <    inf | logz: 171.363 +/-  0.543 | dlogz:  2.450 >  0.059]

    805it [00:18, 30.20it/s, bound: 24 | nc: 1 | ncall: 3765 | eff(%): 21.381 | loglstar:   -inf < 187.418 <    inf | logz: 171.468 +/-  0.543 | dlogz:  2.262 >  0.059]

    810it [00:19, 15.67it/s, bound: 24 | nc: 1 | ncall: 3774 | eff(%): 21.463 | loglstar:   -inf < 187.556 <    inf | logz: 171.566 +/-  0.544 | dlogz:  2.087 >  0.059]

    814it [00:20, 13.10it/s, bound: 24 | nc: 1 | ncall: 3796 | eff(%): 21.444 | loglstar:   -inf < 187.706 <    inf | logz: 171.640 +/-  0.544 | dlogz:  1.953 >  0.059]

    817it [00:20, 12.83it/s, bound: 24 | nc: 5 | ncall: 3806 | eff(%): 21.466 | loglstar:   -inf < 187.730 <    inf | logz: 171.695 +/-  0.544 | dlogz:  1.856 >  0.059]

    820it [00:20, 14.20it/s, bound: 24 | nc: 2 | ncall: 3814 | eff(%): 21.500 | loglstar:   -inf < 187.911 <    inf | logz: 171.749 +/-  0.544 | dlogz:  1.761 >  0.059]

    823it [00:20, 15.80it/s, bound: 25 | nc: 3 | ncall: 3821 | eff(%): 21.539 | loglstar:   -inf < 187.976 <    inf | logz: 171.803 +/-  0.544 | dlogz:  1.668 >  0.059]

    836it [00:20, 31.76it/s, bound: 25 | nc: 2 | ncall: 3837 | eff(%): 21.788 | loglstar:   -inf < 188.156 <    inf | logz: 172.003 +/-  0.545 | dlogz:  1.336 >  0.059]

    846it [00:20, 41.31it/s, bound: 25 | nc: 11 | ncall: 3864 | eff(%): 21.894 | loglstar:   -inf < 188.371 <    inf | logz: 172.129 +/-  0.545 | dlogz:  1.107 >  0.059]

    858it [00:20, 54.85it/s, bound: 25 | nc: 3 | ncall: 3888 | eff(%): 22.068 | loglstar:   -inf < 188.582 <    inf | logz: 172.260 +/-  0.546 | dlogz:  0.875 >  0.059] 

    866it [00:21, 36.92it/s, bound: 26 | nc: 1 | ncall: 3903 | eff(%): 22.188 | loglstar:   -inf < 188.711 <    inf | logz: 172.337 +/-  0.546 | dlogz:  0.743 >  0.059]

    873it [00:21, 41.25it/s, bound: 26 | nc: 1 | ncall: 3914 | eff(%): 22.305 | loglstar:   -inf < 188.778 <    inf | logz: 172.396 +/-  0.547 | dlogz:  0.667 >  0.059]

    879it [00:21, 41.10it/s, bound: 26 | nc: 3 | ncall: 3940 | eff(%): 22.310 | loglstar:   -inf < 188.854 <    inf | logz: 172.442 +/-  0.547 | dlogz:  0.627 >  0.059]

    885it [00:22, 27.89it/s, bound: 27 | nc: 3 | ncall: 3975 | eff(%): 22.264 | loglstar:   -inf < 188.903 <    inf | logz: 172.484 +/-  0.547 | dlogz:  0.555 >  0.059]

    893it [00:22, 33.62it/s, bound: 27 | nc: 13 | ncall: 4006 | eff(%): 22.292 | loglstar:   -inf < 188.969 <    inf | logz: 172.533 +/-  0.547 | dlogz:  0.471 >  0.059]

    900it [00:22, 39.14it/s, bound: 27 | nc: 7 | ncall: 4032 | eff(%): 22.321 | loglstar:   -inf < 189.052 <    inf | logz: 172.571 +/-  0.548 | dlogz:  0.408 >  0.059] 

    906it [00:22, 35.84it/s, bound: 28 | nc: 7 | ncall: 4060 | eff(%): 22.315 | loglstar:   -inf < 189.117 <    inf | logz: 172.601 +/-  0.548 | dlogz:  0.361 >  0.059]

    916it [00:22, 45.82it/s, bound: 28 | nc: 6 | ncall: 4089 | eff(%): 22.402 | loglstar:   -inf < 189.200 <    inf | logz: 172.645 +/-  0.548 | dlogz:  0.293 >  0.059]

    922it [00:22, 41.35it/s, bound: 29 | nc: 3 | ncall: 4122 | eff(%): 22.368 | loglstar:   -inf < 189.230 <    inf | logz: 172.668 +/-  0.548 | dlogz:  0.259 >  0.059]

    932it [00:22, 52.77it/s, bound: 29 | nc: 1 | ncall: 4142 | eff(%): 22.501 | loglstar:   -inf < 189.286 <    inf | logz: 172.702 +/-  0.549 | dlogz:  0.211 >  0.059]

    942it [00:22, 61.48it/s, bound: 29 | nc: 4 | ncall: 4165 | eff(%): 22.617 | loglstar:   -inf < 189.334 <    inf | logz: 172.729 +/-  0.549 | dlogz:  0.171 >  0.059]

    952it [00:23, 69.53it/s, bound: 29 | nc: 1 | ncall: 4191 | eff(%): 22.715 | loglstar:   -inf < 189.354 <    inf | logz: 172.752 +/-  0.549 | dlogz:  0.140 >  0.059]

    960it [00:23, 68.74it/s, bound: 30 | nc: 1 | ncall: 4206 | eff(%): 22.825 | loglstar:   -inf < 189.389 <    inf | logz: 172.768 +/-  0.549 | dlogz:  0.118 >  0.059]

    974it [00:23, 86.18it/s, bound: 30 | nc: 1 | ncall: 4233 | eff(%): 23.010 | loglstar:   -inf < 189.419 <    inf | logz: 172.790 +/-  0.549 | dlogz:  0.090 >  0.059]

    984it [00:23, 73.10it/s, bound: 30 | nc: 3 | ncall: 4266 | eff(%): 23.066 | loglstar:   -inf < 189.454 <    inf | logz: 172.803 +/-  0.549 | dlogz:  0.074 >  0.059]

    993it [00:23, 65.19it/s, bound: 31 | nc: 2 | ncall: 4287 | eff(%): 23.163 | loglstar:   -inf < 189.491 <    inf | logz: 172.813 +/-  0.549 | dlogz:  0.061 >  0.059]

    994it [00:23, 41.86it/s, +50 | bound: 31 | nc: 1 | ncall: 4338 | eff(%): 24.347 | loglstar:   -inf < 189.695 <    inf | logz: 172.866 +/-  0.551 | dlogz:  0.001 >  0.059]

    2026-07-10 18:10:03,027 - overview_2_scientific_workflow - INFO - Fit Running: Updating results (see output folder).


    


    /usr/lib/python3.12/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      self.pid = os.fork()


    0it [00:00, ?it/s]

    994it [00:00, 10961.00it/s, +50 | bound: 31 | nc: 1 | ncall: 4338 | eff(%): 24.347 | loglstar:   -inf < 189.695 <    inf | logz: 172.866 +/-  0.540 | dlogz:  0.001 >  0.059]

    2026-07-10 18:10:09,397 - overview_2_scientific_workflow - INFO - Fit Running: Updating results (see output folder).


    


    2026-07-10 18:10:09,635 - autofit.non_linear.samples.samples - INFO - Samples with weight less than 1e-10 removed from samples.csv.


    2026-07-10 18:10:09,664 - autofit.non_linear.search.updater - INFO - Creating latent samples by drawing 100 from the PDF.


    2026-07-10 18:10:14,351 - overview_2_scientific_workflow - INFO - Removing all files except for .zip file


    2026-07-10 18:10:14,855 - overview_2_scientific_workflow - INFO - Search complete, returning result



    <Figure size 640x480 with 0 Axes>


__Loading Results__

In your scientific workflow, you'll likely conduct numerous model fits, each generating outputs stored in individual
folders on your hard disk.

To efficiently work with these results in Python scripts or Jupyter notebooks, **PyAutoFit** provides
the `aggregator` API. This tool simplifies the process of loading results from hard disk into Python variables.
By pointing the aggregator at the folder containing your results, it automatically loads all relevant information
from each model fit.

This capability streamlines the workflow by enabling easy manipulation and inspection of model-fit results directly
within your Python environment. It's particularly useful for managing and analyzing large-scale studies where
handling multiple model fits and their associated outputs is essential.


```python
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=path.join("result_folder"),
)
```

    Aggregator loading search_outputs... could take some time.
    
    No search_outputs found in result_folder
    


The ``values`` method is used to specify the information that is loaded from the hard-disk, for example the
``samples`` of the model-fit.

The for loop below iterates over all results in the folder passed to the aggregator above.


```python
for samples in agg.values("samples"):
    print(samples.parameter_lists[0])
```

Result loading uses Python generators to ensure that memory use is minimized, meaning that even when loading
thousands of results from hard-disk the memory use of your machine is not exceeded.

The `result cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/model.html>`_ gives a full run-through of
the tools that allow results to be loaded and inspected.

__Result Customization__

An effective scientific workflow ensures that this object contains all information a user needs to quickly inspect
the quality of a model-fit and undertake scientific interpretation.

The result can be can be customized to include additional information about the model-fit that is specific to your
model-fitting problem.

For example, for fitting 1D profiles, the ``Result`` could include the maximum log likelihood model 1D data,
which would enable the following code to be used after the model-fit:

print(result.max_log_likelihood_model_data_1d)

To do this we use the custom result API, where we first define a custom ``Result`` class which includes the
property ``max_log_likelihood_model_data_1d``:


```python


class ResultExample(af.Result):
    @property
    def max_log_likelihood_model_data_1d(self) -> np.ndarray:
        """
        Returns the maximum log likelihood model's 1D model data.

        This is an example of how we can pass the `Analysis` class a custom `Result` object and extend this result
        object with new properties that are specific to the model-fit we are performing.
        """
        xvalues = np.arange(self.analysis.data.shape[0])

        return self.instance.model_data_from(xvalues=xvalues)

```

The custom result has access to the analysis class, meaning that we can use any of its methods or properties to
compute custom result properties.

To make it so that the ``ResultExample`` object above is returned by the search we overwrite the ``Result`` class attribute
of the ``Analysis`` and define a ``make_result`` object describing what we want it to contain:


```python


class Analysis(af.Analysis):
    """
    This overwrite means the `ResultExample` class is returned after the model-fit.
    """

    Result = ResultExample

    def __init__(self, data, noise_map):
        """
        An Analysis class which illustrates custom results.
        """
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        """
        The `log_likelihood_function` is identical to the example above
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_from(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

    def make_result(
        self,
        samples_summary: af.SamplesSummary,
        paths: af.AbstractPaths,
        samples: Optional[af.SamplesPDF] = None,
        search_internal: Optional[object] = None,
        analysis: Optional[object] = None,
    ) -> Result:
        """
        Returns the `Result` of the non-linear search after it is completed.

        The result type is defined as a class variable in the `Analysis` class (see top of code under the python code
        `class Analysis(af.Analysis)`.

        The result can be manually overwritten by a user to return a user-defined result object, which can be extended
        with additional methods and attribute specific to the model-fit.

        This example class does example this, whereby the analysis result has been overwritten with the `ResultExample`
        class, which contains a property `max_log_likelihood_model_data_1d` that returns the model data of the
        best-fit model. This API means you can customize your result object to include whatever attributes you want
        and therefore make a result object specific to your model-fit and model-fitting problem.

        The `Result` object you return can be customized to include:

        - The samples summary, which contains the maximum log likelihood instance and median PDF model.

        - The paths of the search, which are used for loading the samples and search internal below when a search
        is resumed.

        - The samples of the non-linear search (e.g. MCMC chains) also stored in `samples.csv`.

        - The non-linear search used for the fit in its internal representation, which is used for resuming a search
        and making bespoke visualization using the search's internal results.

        - The analysis used to fit the model (default disabled to save memory, but option may be useful for certain
        projects).

        Parameters
        ----------
        samples_summary
            The summary of the samples of the non-linear search, which include the maximum log likelihood instance and
            median PDF model.
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        samples
            The samples of the non-linear search, for example the chains of an MCMC run.
        search_internal
            The internal representation of the non-linear search used to perform the model-fit.
        analysis
            The analysis used to fit the model.

        Returns
        -------
        Result
            The result of the non-linear search, which is defined as a class variable in the `Analysis` class.
        """
        return self.Result(
            samples_summary=samples_summary,
            paths=paths,
            samples=samples,
            search_internal=search_internal,
            analysis=self,
        )

```

By repeating the model-fit above, the `Result` object returned by the search will be an instance of the `ResultExample`
class, which includes the property `max_log_likelihood_model_data_1d`.


```python
analysis = Analysis(data=data, noise_map=noise_map)

model = af.Model(af.ex.Gaussian)

search = af.DynestyStatic(
    path_prefix=path.join("output", "result_folder"),
    name="overview_2_scientific_workflow",
)

result = search.fit(model=model, analysis=analysis)

print(result.max_log_likelihood_model_data_1d)
```

    2026-07-10 18:10:14,987 - autofit.non_linear.search.abstract_search - INFO - Starting non-linear search with 1 cores.


    2026-07-10 18:10:15,045 - overview_2_scientific_workflow - INFO - The output path of this fit is autofit_workspace/output/output/result_folder/overview_2_scientific_workflow/014dd38ae16c1cc2473db87d25577c42


    2026-07-10 18:10:15,047 - overview_2_scientific_workflow - INFO - Outputting pre-fit files (e.g. model.info, visualization).


    2026-07-10 18:10:15,060 - overview_2_scientific_workflow - INFO - Starting new Dynesty non-linear search (no previous samples found).


    2026-07-10 18:10:15,084 - autofit.non_linear.initializer - INFO - Generating initial samples of model using JAX LH Function cores


    2026-07-10 18:10:15,160 - autofit.non_linear.initializer - INFO - Initial samples generated, starting non-linear search


    /usr/lib/python3.12/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      self.pid = os.fork()


    0it [00:00, ?it/s]

    10it [00:00, 99.81it/s, bound: 0 | nc: 1 | ncall: 61 | eff(%): 16.393 | loglstar:   -inf <   -inf <    inf | logz:   -inf +/-  0.305 | dlogz:    inf >  0.059]

    20it [00:00, 94.54it/s, bound: 0 | nc: 1 | ncall: 73 | eff(%): 27.397 | loglstar:   -inf < -93865.771 <    inf | logz: -93870.772 +/-  0.315 | dlogz: 172834.275 >  0.059]

    33it [00:00, 107.49it/s, bound: 0 | nc: 3 | ncall: 90 | eff(%): 36.667 | loglstar:   -inf < -5243.037 <    inf | logz: -5246.317 +/-  0.254 | dlogz: 4240.698 >  0.059]   

    44it [00:00, 87.89it/s, bound: 0 | nc: 1 | ncall: 114 | eff(%): 38.596 | loglstar:   -inf < -5243.032 <    inf | logz: -5245.006 +/-  0.197 | dlogz: 4238.907 >  0.059]

    54it [00:00, 84.97it/s, bound: 0 | nc: 2 | ncall: 128 | eff(%): 42.188 | loglstar:   -inf < -5243.017 <    inf | logz: -5244.571 +/-  0.174 | dlogz: 4238.246 >  0.059]

    63it [00:00, 74.68it/s, bound: 0 | nc: 2 | ncall: 157 | eff(%): 40.127 | loglstar:   -inf < -5242.708 <    inf | logz: -5244.299 +/-  0.161 | dlogz: 4237.791 >  0.059]

    71it [00:00, 71.42it/s, bound: 0 | nc: 1 | ncall: 178 | eff(%): 39.888 | loglstar:   -inf < -5241.233 <    inf | logz: -5243.956 +/-  0.159 | dlogz: 4237.325 >  0.059]

    79it [00:01, 37.77it/s, bound: 0 | nc: 7 | ncall: 248 | eff(%): 31.855 | loglstar:   -inf < -5236.076 <    inf | logz: -5241.836 +/-  0.287 | dlogz: 4236.160 >  0.059]

    85it [00:01, 31.83it/s, bound: 0 | nc: 6 | ncall: 288 | eff(%): 29.514 | loglstar:   -inf < -5233.202 <    inf | logz: -5237.984 +/-  0.283 | dlogz: 4231.539 >  0.059]

    90it [00:01, 26.47it/s, bound: 0 | nc: 4 | ncall: 336 | eff(%): 26.786 | loglstar:   -inf < -5229.691 <    inf | logz: -5235.275 +/-  0.295 | dlogz: 4229.069 >  0.059]

    95it [00:02, 29.70it/s, bound: 0 | nc: 2 | ncall: 354 | eff(%): 26.836 | loglstar:   -inf < -5224.943 <    inf | logz: -5231.097 +/-  0.323 | dlogz: 4225.406 >  0.059]

    99it [00:02, 28.11it/s, bound: 0 | nc: 1 | ncall: 384 | eff(%): 25.781 | loglstar:   -inf < -5216.099 <    inf | logz: -5222.555 +/-  0.348 | dlogz: 4218.170 >  0.059]

    103it [00:02, 25.36it/s, bound: 0 | nc: 9 | ncall: 422 | eff(%): 24.408 | loglstar:   -inf < -5201.016 <    inf | logz: -5207.658 +/-  0.362 | dlogz: 4206.673 >  0.059]

    108it [00:02, 28.93it/s, bound: 0 | nc: 5 | ncall: 439 | eff(%): 24.601 | loglstar:   -inf < -5184.287 <    inf | logz: -5190.997 +/-  0.361 | dlogz: 4187.582 >  0.059]

    112it [00:02, 25.06it/s, bound: 0 | nc: 3 | ncall: 476 | eff(%): 23.529 | loglstar:   -inf < -5156.703 <    inf | logz: -5163.523 +/-  0.367 | dlogz: 4162.673 >  0.059]

    116it [00:02, 27.21it/s, bound: 0 | nc: 5 | ncall: 489 | eff(%): 23.722 | loglstar:   -inf < -5127.854 <    inf | logz: -5134.756 +/-  0.370 | dlogz: 4136.453 >  0.059]

    121it [00:03, 28.57it/s, bound: 0 | nc: 15 | ncall: 518 | eff(%): 23.359 | loglstar:   -inf < -5075.040 <    inf | logz: -5082.041 +/-  0.372 | dlogz: 4103.014 >  0.059]

    125it [00:03, 29.20it/s, bound: 0 | nc: 1 | ncall: 541 | eff(%): 23.105 | loglstar:   -inf < -5052.681 <    inf | logz: -5059.750 +/-  0.372 | dlogz: 4056.985 >  0.059] 

    129it [00:03, 25.80it/s, bound: 0 | nc: 7 | ncall: 593 | eff(%): 21.754 | loglstar:   -inf < -5006.216 <    inf | logz: -5013.376 +/-  0.377 | dlogz: 4021.027 >  0.059]

    132it [00:04, 11.93it/s, bound: 0 | nc: 56 | ncall: 715 | eff(%): 18.462 | loglstar:   -inf < -4974.647 <    inf | logz: -4981.866 +/-  0.378 | dlogz: 3999.394 >  0.059]

    135it [00:04, 11.15it/s, bound: 0 | nc: 23 | ncall: 784 | eff(%): 17.219 | loglstar:   -inf < -4935.629 <    inf | logz: -4942.904 +/-  0.379 | dlogz: 3941.577 >  0.059]

    138it [00:04, 11.96it/s, bound: 0 | nc: 33 | ncall: 835 | eff(%): 16.527 | loglstar:   -inf < -4926.457 <    inf | logz: -4933.421 +/-  0.359 | dlogz: 3927.246 >  0.059]

    141it [00:04, 14.00it/s, bound: 0 | nc: 4 | ncall: 863 | eff(%): 16.338 | loglstar:   -inf < -4872.362 <    inf | logz: -4879.360 +/-  0.360 | dlogz: 3923.892 >  0.059] 

    143it [00:04, 13.25it/s, bound: 0 | nc: 30 | ncall: 900 | eff(%): 15.889 | loglstar:   -inf < -4805.763 <    inf | logz: -4813.200 +/-  0.384 | dlogz: 3872.286 >  0.059]

    145it [00:05, 12.00it/s, bound: 0 | nc: 11 | ncall: 945 | eff(%): 15.344 | loglstar:   -inf < -4779.948 <    inf | logz: -4787.425 +/-  0.385 | dlogz: 3843.964 >  0.059]

    147it [00:05, 12.89it/s, bound: 0 | nc: 14 | ncall: 971 | eff(%): 15.139 | loglstar:   -inf < -4775.136 <    inf | logz: -4782.565 +/-  0.376 | dlogz: 3828.181 >  0.059]

    149it [00:05, 12.32it/s, bound: 0 | nc: 5 | ncall: 1021 | eff(%): 14.594 | loglstar:   -inf < -4769.538 <    inf | logz: -4776.998 +/-  0.377 | dlogz: 3822.580 >  0.059]

    151it [00:05, 12.80it/s, bound: 0 | nc: 13 | ncall: 1053 | eff(%): 14.340 | loglstar:   -inf < -4723.884 <    inf | logz: -4731.480 +/-  0.388 | dlogz: 3808.407 >  0.059]

    153it [00:05, 12.61it/s, bound: 0 | nc: 32 | ncall: 1094 | eff(%): 13.985 | loglstar:   -inf < -4668.778 <    inf | logz: -4676.413 +/-  0.389 | dlogz: 3763.733 >  0.059]

    155it [00:05, 11.73it/s, bound: 0 | nc: 29 | ncall: 1156 | eff(%): 13.408 | loglstar:   -inf < -4614.018 <    inf | logz: -4621.693 +/-  0.390 | dlogz: 3676.848 >  0.059]

    157it [00:06, 10.11it/s, bound: 0 | nc: 13 | ncall: 1228 | eff(%): 12.785 | loglstar:   -inf < -4583.844 <    inf | logz: -4591.558 +/-  0.391 | dlogz: 3656.968 >  0.059]

    159it [00:06,  8.44it/s, bound: 0 | nc: 111 | ncall: 1341 | eff(%): 11.857 | loglstar:   -inf < -4523.673 <    inf | logz: -4531.427 +/-  0.392 | dlogz: 3627.065 >  0.059]

    160it [00:06,  7.07it/s, bound: 0 | nc: 45 | ncall: 1386 | eff(%): 11.544 | loglstar:   -inf < -4522.862 <    inf | logz: -4529.995 +/-  0.366 | dlogz: 3573.789 >  0.059] 

    162it [00:06,  8.49it/s, bound: 0 | nc: 11 | ncall: 1413 | eff(%): 11.465 | loglstar:   -inf < -4445.669 <    inf | logz: -4453.482 +/-  0.393 | dlogz: 3549.855 >  0.059]

    165it [00:07, 10.47it/s, bound: 0 | nc: 28 | ncall: 1463 | eff(%): 11.278 | loglstar:   -inf < -4407.101 <    inf | logz: -4414.974 +/-  0.395 | dlogz: 3467.362 >  0.059]

    167it [00:07, 11.14it/s, bound: 0 | nc: 35 | ncall: 1506 | eff(%): 11.089 | loglstar:   -inf < -4369.389 <    inf | logz: -4377.301 +/-  0.396 | dlogz: 3433.272 >  0.059]

    169it [00:07, 11.65it/s, bound: 0 | nc: 14 | ncall: 1548 | eff(%): 10.917 | loglstar:   -inf < -4317.109 <    inf | logz: -4325.060 +/-  0.397 | dlogz: 3398.875 >  0.059]

    171it [00:07,  7.75it/s, bound: 0 | nc: 36 | ncall: 1687 | eff(%): 10.136 | loglstar:   -inf < -4263.239 <    inf | logz: -4271.155 +/-  0.390 | dlogz: 3316.605 >  0.059]

    173it [00:08,  8.30it/s, bound: 0 | nc: 61 | ncall: 1752 | eff(%):  9.874 | loglstar:   -inf < -4117.603 <    inf | logz: -4125.634 +/-  0.399 | dlogz: 3209.966 >  0.059]

    175it [00:08,  9.69it/s, bound: 1 | nc: 17 | ncall: 1775 | eff(%):  9.859 | loglstar:   -inf < -4060.660 <    inf | logz: -4068.731 +/-  0.400 | dlogz: 3163.223 >  0.059]

    178it [00:08, 13.13it/s, bound: 1 | nc: 2 | ncall: 1802 | eff(%):  9.878 | loglstar:   -inf < -3915.117 <    inf | logz: -3923.247 +/-  0.401 | dlogz: 3799.346 >  0.059] 

    180it [00:08, 11.09it/s, bound: 2 | nc: 5 | ncall: 1851 | eff(%):  9.724 | loglstar:   -inf < -3851.672 <    inf | logz: -3859.841 +/-  0.402 | dlogz: 3732.310 >  0.059]

    184it [00:08, 15.48it/s, bound: 2 | nc: 12 | ncall: 1886 | eff(%):  9.756 | loglstar:   -inf < -3771.059 <    inf | logz: -3779.308 +/-  0.404 | dlogz: 3694.902 >  0.059]

    187it [00:08, 15.64it/s, bound: 3 | nc: 8 | ncall: 1908 | eff(%):  9.801 | loglstar:   -inf < -3636.470 <    inf | logz: -3644.778 +/-  0.406 | dlogz: 3595.854 >  0.059] 

    193it [00:08, 23.72it/s, bound: 3 | nc: 6 | ncall: 1930 | eff(%): 10.000 | loglstar:   -inf < -3474.629 <    inf | logz: -3483.057 +/-  0.409 | dlogz: 3393.005 >  0.059]

    200it [00:09, 33.51it/s, bound: 3 | nc: 1 | ncall: 1947 | eff(%): 10.272 | loglstar:   -inf < -3363.816 <    inf | logz: -3372.382 +/-  0.412 | dlogz: 3243.765 >  0.059]

    212it [00:09, 53.47it/s, bound: 3 | nc: 1 | ncall: 1968 | eff(%): 10.772 | loglstar:   -inf < -2984.359 <    inf | logz: -2993.163 +/-  0.418 | dlogz: 2866.307 >  0.059]

    219it [00:09, 47.70it/s, bound: 4 | nc: 2 | ncall: 1995 | eff(%): 10.977 | loglstar:   -inf < -2755.194 <    inf | logz: -2764.136 +/-  0.421 | dlogz: 2730.724 >  0.059]

    229it [00:09, 59.48it/s, bound: 4 | nc: 4 | ncall: 2013 | eff(%): 11.376 | loglstar:   -inf < -2517.179 <    inf | logz: -2526.319 +/-  0.425 | dlogz: 2401.917 >  0.059]

    237it [00:09, 60.85it/s, bound: 4 | nc: 2 | ncall: 2029 | eff(%): 11.681 | loglstar:   -inf < -2386.942 <    inf | logz: -2396.238 +/-  0.429 | dlogz: 2260.275 >  0.059]

    245it [00:09, 64.83it/s, bound: 4 | nc: 3 | ncall: 2047 | eff(%): 11.969 | loglstar:   -inf < -2190.538 <    inf | logz: -2199.995 +/-  0.433 | dlogz: 2069.725 >  0.059]

    253it [00:09, 53.65it/s, bound: 5 | nc: 4 | ncall: 2069 | eff(%): 12.228 | loglstar:   -inf < -2035.591 <    inf | logz: -2045.207 +/-  0.436 | dlogz: 1920.841 >  0.059]

    265it [00:09, 67.02it/s, bound: 5 | nc: 3 | ncall: 2093 | eff(%): 12.661 | loglstar:   -inf < -1808.411 <    inf | logz: -1817.918 +/-  0.423 | dlogz: 1676.717 >  0.059]

    274it [00:10, 68.90it/s, bound: 5 | nc: 9 | ncall: 2121 | eff(%): 12.918 | loglstar:   -inf < -1650.217 <    inf | logz: -1660.248 +/-  0.446 | dlogz: 1563.085 >  0.059]

    282it [00:10, 70.08it/s, bound: 6 | nc: 2 | ncall: 2143 | eff(%): 13.159 | loglstar:   -inf < -1468.522 <    inf | logz: -1478.711 +/-  0.449 | dlogz: 1348.146 >  0.059]

    290it [00:10, 60.80it/s, bound: 6 | nc: 15 | ncall: 2181 | eff(%): 13.297 | loglstar:   -inf < -1330.984 <    inf | logz: -1341.330 +/-  0.452 | dlogz: 1205.145 >  0.059]

    297it [00:10, 47.29it/s, bound: 7 | nc: 1 | ncall: 2215 | eff(%): 13.409 | loglstar:   -inf < -1226.387 <    inf | logz: -1236.874 +/-  0.456 | dlogz: 1353.342 >  0.059] 

    303it [00:10, 41.67it/s, bound: 7 | nc: 1 | ncall: 2250 | eff(%): 13.467 | loglstar:   -inf < -1204.765 <    inf | logz: -1214.634 +/-  0.434 | dlogz: 1318.600 >  0.059]

    309it [00:10, 44.28it/s, bound: 7 | nc: 1 | ncall: 2261 | eff(%): 13.667 | loglstar:   -inf < -1163.542 <    inf | logz: -1174.266 +/-  0.461 | dlogz: 1286.781 >  0.059]

    315it [00:11, 40.80it/s, bound: 8 | nc: 6 | ncall: 2279 | eff(%): 13.822 | loglstar:   -inf < -1101.648 <    inf | logz: -1112.491 +/-  0.463 | dlogz: 1224.588 >  0.059]

    323it [00:11, 48.65it/s, bound: 8 | nc: 2 | ncall: 2296 | eff(%): 14.068 | loglstar:   -inf < -998.747 <    inf | logz: -1009.745 +/-  0.466 | dlogz: 1118.540 >  0.059] 

    330it [00:11, 50.35it/s, bound: 8 | nc: 2 | ncall: 2313 | eff(%): 14.267 | loglstar:   -inf < -942.088 <    inf | logz: -953.228 +/-  0.470 | dlogz: 1067.519 >  0.059] 

    336it [00:11, 47.32it/s, bound: 8 | nc: 1 | ncall: 2338 | eff(%): 14.371 | loglstar:   -inf < -909.193 <    inf | logz: -920.449 +/-  0.472 | dlogz: 1029.083 >  0.059]

    342it [00:11, 43.08it/s, bound: 9 | nc: 6 | ncall: 2355 | eff(%): 14.522 | loglstar:   -inf < -856.804 <    inf | logz: -868.182 +/-  0.475 | dlogz: 982.928 >  0.059] 

    349it [00:11, 47.68it/s, bound: 9 | nc: 5 | ncall: 2372 | eff(%): 14.713 | loglstar:   -inf < -754.761 <    inf | logz: -766.277 +/-  0.478 | dlogz: 913.668 >  0.059]

    360it [00:11, 61.22it/s, bound: 9 | nc: 3 | ncall: 2394 | eff(%): 15.038 | loglstar:   -inf < -608.990 <    inf | logz: -620.724 +/-  0.482 | dlogz: 745.512 >  0.059]

    368it [00:12, 62.33it/s, bound: 9 | nc: 8 | ncall: 2417 | eff(%): 15.225 | loglstar:   -inf < -575.609 <    inf | logz: -587.404 +/-  0.477 | dlogz: 707.041 >  0.059]

    375it [00:12, 54.41it/s, bound: 10 | nc: 4 | ncall: 2438 | eff(%): 15.381 | loglstar:   -inf < -525.151 <    inf | logz: -537.183 +/-  0.488 | dlogz: 668.735 >  0.059]

    385it [00:12, 64.38it/s, bound: 10 | nc: 4 | ncall: 2463 | eff(%): 15.631 | loglstar:   -inf < -481.743 <    inf | logz: -493.850 +/-  0.482 | dlogz: 612.847 >  0.059]

    393it [00:12, 66.69it/s, bound: 10 | nc: 2 | ncall: 2482 | eff(%): 15.834 | loglstar:   -inf < -435.992 <    inf | logz: -448.375 +/-  0.495 | dlogz: 570.377 >  0.059]

    401it [00:12, 59.38it/s, bound: 11 | nc: 1 | ncall: 2507 | eff(%): 15.995 | loglstar:   -inf < -366.279 <    inf | logz: -378.740 +/-  0.492 | dlogz: 514.184 >  0.059]

    412it [00:12, 71.34it/s, bound: 11 | nc: 2 | ncall: 2535 | eff(%): 16.252 | loglstar:   -inf < -290.813 <    inf | logz: -303.577 +/-  0.503 | dlogz: 445.252 >  0.059]

    422it [00:12, 78.52it/s, bound: 11 | nc: 2 | ncall: 2563 | eff(%): 16.465 | loglstar:   -inf < -206.619 <    inf | logz: -219.581 +/-  0.507 | dlogz: 364.632 >  0.059]

    431it [00:12, 63.85it/s, bound: 12 | nc: 2 | ncall: 2588 | eff(%): 16.654 | loglstar:   -inf < -164.562 <    inf | logz: -177.499 +/-  0.498 | dlogz: 318.621 >  0.059]

    439it [00:13, 60.45it/s, bound: 12 | nc: 4 | ncall: 2617 | eff(%): 16.775 | loglstar:   -inf < -137.150 <    inf | logz: -149.520 +/-  0.482 | dlogz: 289.013 >  0.059]

    446it [00:13, 49.97it/s, bound: 13 | nc: 2 | ncall: 2656 | eff(%): 16.792 | loglstar:   -inf < -107.316 <    inf | logz: -119.949 +/-  0.495 | dlogz: 259.701 >  0.059]

    465it [00:13, 75.77it/s, bound: 13 | nc: 9 | ncall: 2695 | eff(%): 17.254 | loglstar:   -inf < -44.528 <    inf | logz: -58.309 +/-  0.520 | dlogz: 202.464 >  0.059]  

    474it [00:13, 65.31it/s, bound: 14 | nc: 2 | ncall: 2729 | eff(%): 17.369 | loglstar:   -inf < -28.435 <    inf | logz: -41.449 +/-  0.492 | dlogz: 189.774 >  0.059]

    483it [00:13, 69.41it/s, bound: 14 | nc: 2 | ncall: 2757 | eff(%): 17.519 | loglstar:   -inf <  8.810 <    inf | logz: -4.166 +/-  0.502 | dlogz: 152.460 >  0.059]  

    491it [00:13, 63.95it/s, bound: 14 | nc: 2 | ncall: 2782 | eff(%): 17.649 | loglstar:   -inf < 44.377 <    inf | logz: 30.434 +/-  0.517 | dlogz: 137.367 >  0.059]

    498it [00:14, 54.69it/s, bound: 15 | nc: 1 | ncall: 2808 | eff(%): 17.735 | loglstar:   -inf < 56.849 <    inf | logz: 43.565 +/-  0.503 | dlogz: 122.963 >  0.059]

    508it [00:14, 63.82it/s, bound: 15 | nc: 1 | ncall: 2828 | eff(%): 17.963 | loglstar:   -inf < 73.493 <    inf | logz: 59.703 +/-  0.512 | dlogz: 113.720 >  0.059]

    523it [00:14, 83.55it/s, bound: 15 | nc: 1 | ncall: 2851 | eff(%): 18.344 | loglstar:   -inf < 89.825 <    inf | logz: 74.929 +/-  0.538 | dlogz: 102.761 >  0.059]

    538it [00:14, 82.44it/s, bound: 16 | nc: 4 | ncall: 2881 | eff(%): 18.674 | loglstar:   -inf < 108.626 <    inf | logz: 93.656 +/-  0.530 | dlogz: 85.245 >  0.059]

    548it [00:14, 75.89it/s, bound: 16 | nc: 13 | ncall: 2929 | eff(%): 18.709 | loglstar:   -inf < 121.114 <    inf | logz: 106.520 +/-  0.521 | dlogz: 71.379 >  0.059]

    557it [00:14, 66.83it/s, bound: 17 | nc: 1 | ncall: 2961 | eff(%): 18.811 | loglstar:   -inf < 129.975 <    inf | logz: 114.816 +/-  0.535 | dlogz: 63.471 >  0.059] 

    567it [00:14, 71.46it/s, bound: 17 | nc: 5 | ncall: 2984 | eff(%): 19.001 | loglstar:   -inf < 139.251 <    inf | logz: 124.684 +/-  0.523 | dlogz: 52.624 >  0.059]

    579it [00:15, 81.83it/s, bound: 17 | nc: 2 | ncall: 3011 | eff(%): 19.229 | loglstar:   -inf < 145.186 <    inf | logz: 130.966 +/-  0.517 | dlogz: 45.789 >  0.059]

    588it [00:15, 75.24it/s, bound: 18 | nc: 1 | ncall: 3038 | eff(%): 19.355 | loglstar:   -inf < 149.085 <    inf | logz: 134.485 +/-  0.521 | dlogz: 43.113 >  0.059]

    598it [00:15, 80.11it/s, bound: 18 | nc: 5 | ncall: 3067 | eff(%): 19.498 | loglstar:   -inf < 153.108 <    inf | logz: 137.903 +/-  0.523 | dlogz: 39.582 >  0.059]

    607it [00:15, 80.95it/s, bound: 18 | nc: 6 | ncall: 3096 | eff(%): 19.606 | loglstar:   -inf < 158.859 <    inf | logz: 143.505 +/-  0.533 | dlogz: 33.884 >  0.059]

    616it [00:15, 80.19it/s, bound: 19 | nc: 2 | ncall: 3114 | eff(%): 19.782 | loglstar:   -inf < 161.979 <    inf | logz: 146.579 +/-  0.528 | dlogz: 30.485 >  0.059]

    625it [00:15, 79.44it/s, bound: 19 | nc: 6 | ncall: 3145 | eff(%): 19.873 | loglstar:   -inf < 166.562 <    inf | logz: 151.338 +/-  0.533 | dlogz: 25.444 >  0.059]

    634it [00:15, 78.21it/s, bound: 19 | nc: 8 | ncall: 3173 | eff(%): 19.981 | loglstar:   -inf < 170.088 <    inf | logz: 154.858 +/-  0.535 | dlogz: 21.737 >  0.059]

    642it [00:15, 70.53it/s, bound: 20 | nc: 1 | ncall: 3189 | eff(%): 20.132 | loglstar:   -inf < 171.710 <    inf | logz: 156.753 +/-  0.530 | dlogz: 19.578 >  0.059]

    656it [00:16, 86.79it/s, bound: 20 | nc: 3 | ncall: 3216 | eff(%): 20.398 | loglstar:   -inf < 174.573 <    inf | logz: 159.772 +/-  0.529 | dlogz: 16.202 >  0.059]

    667it [00:16, 91.79it/s, bound: 20 | nc: 2 | ncall: 3243 | eff(%): 20.567 | loglstar:   -inf < 176.912 <    inf | logz: 161.531 +/-  0.530 | dlogz: 14.258 >  0.059]

    677it [00:16, 85.99it/s, bound: 21 | nc: 3 | ncall: 3266 | eff(%): 20.729 | loglstar:   -inf < 177.925 <    inf | logz: 162.934 +/-  0.530 | dlogz: 12.592 >  0.059]

    697it [00:16, 115.11it/s, bound: 21 | nc: 4 | ncall: 3299 | eff(%): 21.128 | loglstar:   -inf < 181.089 <    inf | logz: 165.432 +/-  0.537 | dlogz:  9.735 >  0.059]

    710it [00:16, 100.38it/s, bound: 22 | nc: 5 | ncall: 3336 | eff(%): 21.283 | loglstar:   -inf < 182.378 <    inf | logz: 166.737 +/-  0.537 | dlogz:  8.736 >  0.059]

    721it [00:16, 95.98it/s, bound: 22 | nc: 1 | ncall: 3370 | eff(%): 21.395 | loglstar:   -inf < 183.354 <    inf | logz: 167.704 +/-  0.538 | dlogz:  7.538 >  0.059] 

    732it [00:16, 79.03it/s, bound: 23 | nc: 1 | ncall: 3422 | eff(%): 21.391 | loglstar:   -inf < 184.504 <    inf | logz: 168.566 +/-  0.540 | dlogz:  6.463 >  0.059]

    741it [00:16, 77.55it/s, bound: 23 | nc: 1 | ncall: 3454 | eff(%): 21.453 | loglstar:   -inf < 185.010 <    inf | logz: 169.215 +/-  0.541 | dlogz:  5.617 >  0.059]

    750it [00:17, 69.35it/s, bound: 24 | nc: 1 | ncall: 3481 | eff(%): 21.546 | loglstar:   -inf < 185.407 <    inf | logz: 169.688 +/-  0.541 | dlogz:  4.953 >  0.059]

    764it [00:17, 84.14it/s, bound: 24 | nc: 1 | ncall: 3507 | eff(%): 21.785 | loglstar:   -inf < 186.184 <    inf | logz: 170.325 +/-  0.541 | dlogz:  4.045 >  0.059]

    777it [00:17, 92.83it/s, bound: 24 | nc: 7 | ncall: 3539 | eff(%): 21.955 | loglstar:   -inf < 186.543 <    inf | logz: 170.772 +/-  0.541 | dlogz:  3.393 >  0.059]

    787it [00:17, 86.26it/s, bound: 25 | nc: 3 | ncall: 3570 | eff(%): 22.045 | loglstar:   -inf < 186.870 <    inf | logz: 171.038 +/-  0.541 | dlogz:  2.949 >  0.059]

    798it [00:17, 92.04it/s, bound: 25 | nc: 1 | ncall: 3605 | eff(%): 22.136 | loglstar:   -inf < 187.148 <    inf | logz: 171.291 +/-  0.541 | dlogz:  2.506 >  0.059]

    808it [00:17, 82.80it/s, bound: 26 | nc: 3 | ncall: 3643 | eff(%): 22.180 | loglstar:   -inf < 187.428 <    inf | logz: 171.488 +/-  0.542 | dlogz:  2.148 >  0.059]

    822it [00:17, 96.30it/s, bound: 26 | nc: 1 | ncall: 3670 | eff(%): 22.398 | loglstar:   -inf < 187.821 <    inf | logz: 171.739 +/-  0.543 | dlogz:  1.696 >  0.059]

    834it [00:18, 89.91it/s, bound: 27 | nc: 4 | ncall: 3703 | eff(%): 22.522 | loglstar:   -inf < 188.184 <    inf | logz: 171.926 +/-  0.544 | dlogz:  1.363 >  0.059]

    846it [00:18, 96.99it/s, bound: 27 | nc: 3 | ncall: 3731 | eff(%): 22.675 | loglstar:   -inf < 188.406 <    inf | logz: 172.096 +/-  0.545 | dlogz:  1.178 >  0.059]

    857it [00:18, 73.75it/s, bound: 28 | nc: 2 | ncall: 3798 | eff(%): 22.565 | loglstar:   -inf < 188.727 <    inf | logz: 172.238 +/-  0.546 | dlogz:  0.960 >  0.059]

    867it [00:18, 74.37it/s, bound: 28 | nc: 12 | ncall: 3830 | eff(%): 22.637 | loglstar:   -inf < 188.819 <    inf | logz: 172.350 +/-  0.547 | dlogz:  0.779 >  0.059]

    876it [00:18, 65.27it/s, bound: 29 | nc: 2 | ncall: 3860 | eff(%): 22.694 | loglstar:   -inf < 188.969 <    inf | logz: 172.435 +/-  0.547 | dlogz:  0.644 >  0.059] 

    889it [00:18, 78.35it/s, bound: 29 | nc: 3 | ncall: 3885 | eff(%): 22.883 | loglstar:   -inf < 189.078 <    inf | logz: 172.536 +/-  0.548 | dlogz:  0.489 >  0.059]

    901it [00:18, 86.45it/s, bound: 29 | nc: 3 | ncall: 3908 | eff(%): 23.055 | loglstar:   -inf < 189.219 <    inf | logz: 172.610 +/-  0.548 | dlogz:  0.380 >  0.059]

    911it [00:19, 82.34it/s, bound: 30 | nc: 2 | ncall: 3937 | eff(%): 23.139 | loglstar:   -inf < 189.294 <    inf | logz: 172.663 +/-  0.549 | dlogz:  0.307 >  0.059]

    931it [00:19, 109.66it/s, bound: 30 | nc: 2 | ncall: 3972 | eff(%): 23.439 | loglstar:   -inf < 189.368 <    inf | logz: 172.743 +/-  0.549 | dlogz:  0.210 >  0.059]

    943it [00:19, 92.35it/s, bound: 31 | nc: 8 | ncall: 4021 | eff(%): 23.452 | loglstar:   -inf < 189.428 <    inf | logz: 172.778 +/-  0.550 | dlogz:  0.164 >  0.059] 

    958it [00:19, 104.15it/s, bound: 31 | nc: 3 | ncall: 4061 | eff(%): 23.590 | loglstar:   -inf < 189.471 <    inf | logz: 172.812 +/-  0.550 | dlogz:  0.120 >  0.059]

    970it [00:19, 97.24it/s, bound: 32 | nc: 2 | ncall: 4097 | eff(%): 23.676 | loglstar:   -inf < 189.502 <    inf | logz: 172.833 +/-  0.550 | dlogz:  0.094 >  0.059] 

    984it [00:19, 103.22it/s, bound: 32 | nc: 8 | ncall: 4140 | eff(%): 23.768 | loglstar:   -inf < 189.531 <    inf | logz: 172.852 +/-  0.550 | dlogz:  0.071 >  0.059]

    992it [00:19, 50.03it/s, +50 | bound: 33 | nc: 1 | ncall: 4210 | eff(%): 25.048 | loglstar:   -inf < 189.702 <    inf | logz: 172.914 +/-  0.552 | dlogz:  0.001 >  0.059]

    2026-07-10 18:10:35,039 - overview_2_scientific_workflow - INFO - Fit Running: Updating results (see output folder).


    


    /usr/lib/python3.12/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      self.pid = os.fork()


    0it [00:00, ?it/s]

    992it [00:00, 18890.09it/s, +50 | bound: 33 | nc: 1 | ncall: 4210 | eff(%): 25.048 | loglstar:   -inf < 189.702 <    inf | logz: 172.914 +/-  0.540 | dlogz:  0.001 >  0.059]

    2026-07-10 18:10:37,539 - overview_2_scientific_workflow - INFO - Fit Running: Updating results (see output folder).


    


    2026-07-10 18:10:37,658 - autofit.non_linear.samples.samples - INFO - Samples with weight less than 1e-10 removed from samples.csv.


    2026-07-10 18:10:37,679 - autofit.non_linear.search.updater - INFO - Creating latent samples by drawing 100 from the PDF.


    2026-07-10 18:10:40,592 - overview_2_scientific_workflow - INFO - Removing all files except for .zip file


    2026-07-10 18:10:40,993 - overview_2_scientific_workflow - INFO - Search complete, returning result


    [3.21714436e-06 5.30994715e-06 8.67578787e-06 1.40322311e-05
     2.24669235e-05 3.56089822e-05 5.58694752e-05 8.67738167e-05
     1.33414147e-04 2.03055213e-04 3.05932405e-04 4.56284687e-04
     6.73667170e-04 9.84586664e-04 1.42449717e-03 2.04017917e-03
     2.89250490e-03 4.05956009e-03 5.64004958e-03 7.75685986e-03
     1.05605867e-02 1.42327631e-02 1.89884482e-02 2.50777635e-02
     3.27859074e-02 4.24311390e-02 5.43602253e-02 6.89408894e-02
     8.65509008e-02 1.07563612e-01 1.32329979e-01 1.61157374e-01
     1.94285862e-01 2.31862914e-01 2.73917928e-01 3.20338180e-01
     3.70848071e-01 4.24993631e-01 4.82134152e-01 5.41442629e-01
     6.01916256e-01 6.62397638e-01 7.21606695e-01 7.78182409e-01
     8.30732767e-01 8.77890500e-01 9.18371576e-01 9.51033016e-01
     9.74926401e-01 9.89343591e-01 9.93851584e-01 9.88314120e-01
     9.72898520e-01 9.48067287e-01 9.14555059e-01 8.73332511e-01
     8.25559682e-01 7.72531860e-01 7.15621532e-01 6.56220021e-01
     5.95682209e-01 5.35277340e-01 4.76148219e-01 4.19280385e-01
     3.65482010e-01 3.15374474e-01 2.69392903e-01 2.27795341e-01
     1.90678884e-01 1.58000856e-01 1.29603090e-01 1.05237458e-01
     8.45910504e-02 6.73096858e-02 5.30187866e-02 4.13410100e-02
     3.19103418e-02 2.43826501e-02 1.84429094e-02 1.38094708e-02
     1.02358454e-02 7.51051075e-03 5.45524556e-03 3.92245719e-03
     2.79190861e-03 1.96717619e-03 1.37209568e-03 9.47380867e-04
     6.47535973e-04 4.38129268e-04 2.93453781e-04 1.94570156e-04
     1.27706153e-04 8.29748536e-05 5.33679159e-05 3.39791936e-05
     2.14163259e-05 1.33621378e-05 8.25288889e-06 5.04585920e-06]


Result customization has full support for **latent variables**, which are parameters that are not sampled by the non-linear
search but are computed from the sampled parameters.

They are often integral to assessing and interpreting the results of a model-fit, as they present information
on the model in a different way to the sampled parameters.

The `result cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/result.html>`_ gives a full run-through of
all the different ways the result can be customized.

__Model Composition__

In many scientific workflows, there's often a need to construct and fit a variety of different models. This
could range from making minor adjustments to a model's parameters to handling complex models with thousands of parameters and multiple components.

For simpler scenarios, adjustments might include:

- **Parameter Assignment**: Setting specific values for certain parameters or linking parameters together so they share the same value.
- **Parameter Assertions**: Imposing constraints on model parameters, such as requiring one parameter to be greater than another.
- **Model Arithmetic**: Defining relationships between parameters using arithmetic operations, such as defining a 
  linear relationship like `y = mx + c`, where `m` and `c` are model parameters.

In more intricate cases, models might involve numerous parameters and complex compositions of multiple model components.

**PyAutoFit** offers a sophisticated model composition API designed to handle these complexities. It provides
tools for constructing elaborate models using lists of Python classes, NumPy arrays and hierarchical structures of Python classes.

For a detailed exploration of these capabilities, you can refer to
the `model cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/model.html>`_, which provides comprehensive
guidance on using the model composition API. This resource covers everything from basic parameter assignments to
constructing complex models with hierarchical structures.

__Searches__

Different model-fitting problems often require different approaches to fitting the model effectively.

The choice of the most suitable search method depends on several factors:

- **Model Dimensions**: How many parameters constitute the model and its non-linear parameter space?
- **Model Complexity**: Different models exhibit varying degrees of parameter degeneracy, which necessitates different 
  non-linear search techniques.
- **Run Times**: How efficiently can the likelihood function be evaluated and the model-fit performed?
- **Gradients**: If your likelihood function is differentiable, leveraging JAX and using a search that exploits 
  gradient information can be advantageous.

**PyAutoFit** provides support for a wide range of non-linear searches, ensuring that users can select the method
best suited to their specific problem.

During the initial stages of setting up your scientific workflow, it's beneficial to experiment with different
searches. This process helps identify which methods reliably infer maximum likelihood fits to the data and assess
their efficiency in terms of computational time.

For a comprehensive exploration of available search methods and customization options, refer to
the `search cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/search.html>`_. This resource covers
detailed guides on all non-linear searches supported by PyAutoFit and provides insights into how to tailor them to your 
needs.

There are currently no documentation guiding reads on what search might be appropriate for their problem and how to
profile and experiment with different methods. Writing such documentation is on the to do list and will appear
in the future. However, you can make progress now simply using visuals output by PyAutoFit and the ``search.summary` file.

__Configs__

As you refine your scientific workflow, you'll often find yourself repeatedly setting up models with identical priors
and using the same non-linear search configurations. This repetition can result in lengthy Python scripts with
redundant inputs.

To streamline this process, configuration files can be utilized to define default values. This approach eliminates
the need to specify identical prior inputs and search settings in every script, leading to more concise and
readable Python code. Moreover, it reduces the cognitive load associated with performing model-fitting tasks.

For a comprehensive guide on setting up and utilizing configuration files effectively, refer
to the `configs cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/configs.html>`_. This resource provides
detailed instructions on configuring and optimizing your PyAutoFit workflow through the use of configuration files.

__Database__

By default, model-fitting results are written to folders on hard-disk, which is straightforward for navigating and
manual inspection. However, this approach becomes impractical for large datasets or extensive scientific workflows,
where manually checking each result can be time-consuming.

To address this challenge, all results can be stored in an sqlite3 relational database. This enables loading results
directly into Jupyter notebooks or Python scripts for inspection, analysis, and interpretation. The database
supports advanced querying capabilities, allowing users to retrieve specific model-fits based on criteria such
as the fitted model or dataset.

For a comprehensive guide on using the database functionality within PyAutoFit, refer to
the `database cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/multiple_datasets.html>`. This resource
provides detailed instructions on leveraging the database to manage and analyze model-fitting results efficiently.

__Scaling Up__

Regardless of your final scientific objective, it's crucial to consider scalability in your scientific workflow and
ensure it remains flexible to accommodate varying scales of complexity.

Initially, scientific studies often begin with a small number of datasets (e.g., tens of datasets). During this phase,
researchers iteratively refine their models and gain insights through trial and error. This involves fitting numerous
models to datasets and manually inspecting results to evaluate model performance. A flexible workflow is essential
here, allowing rapid iteration and outputting results in a format that facilitates quick inspection and interpretation.

As the study progresses, researchers may scale up to larger datasets (e.g., thousands of datasets). Manual inspection
of individual results becomes impractical, necessitating a more automated approach to model fitting and interpretation.
Additionally, analyses may transition to high-performance computing environments, requiring output formats suitable for 
these setups.

**PyAutoFit** is designed to enable the development of effective scientific workflows for both small and large datasets.

__Wrap Up__

This overview has provided a comprehensive guide to the key features of **PyAutoFit** that support the development of
effective scientific workflows. By leveraging these tools, researchers can tailor their workflows to specific problems,
streamline model fitting, and gain valuable insights into their scientific studies.

The final aspect of core functionality, described in the next overview, is the wide variety of statistical
inference methods available in **PyAutoFit**. These methods include graphical models, hierarchical models,
Bayesian model comparison and many more.


```python

```
