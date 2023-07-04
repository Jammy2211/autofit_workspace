"""
Feature: Model Comparison
=========================

Common questions when fitting a model to data are: what model should I use? How many parameters should the model have? 
Is the model too complex or too simple?

Model comparison answers to these questions. It amounts to composing and fitting many different models to the data 
and comparing how well they fit the data. 

This example illustrates model comparison using the noisy 1D Gaussian example. We fit a dataset consisting of two 
Gaussians and fit it with three models comprised of 1, 2 and 3 Gaussian's respectively. Using the Bayesian evidence to 
compare the models, we favour the model with 2 Gaussians, which is the "correct" model given that it was the model used 
to simulate the dataset in the first place.

__Metrics__

Different metrics can be used compare models and quantify their goodness-of-fit.

In this example we show the results of using two different metrics:

 - `log_likelihood`: The value returned by the `log_likelihood_function` of an `Analysis` object. which is directly 
   related to the sum of the residuals squared (e.g. the `chi_squared`). The log likelihood does not change when more 
   or less parameters are included in the model, therefore it does not account for over-fitting and will often favour 
   more complex models irrespective of whether they fit the data better.
 
 - `log_evidence`: The Bayesian evidence, which is closely related to the log likelihood but utilizes additional
   information which penalizes models based on their complexity. The Bayesian evidence will therefore favour simpler
   models over more complex models, unless the more complex model provides a much better fit to the data.
   
__Example Source Code (`af.ex`)__

The **PyAutoFit** source code has the following example objects (accessed via `af.ex`) used in this tutorial:

 - `Analysis`: an analysis object which fits noisy 1D datasets, including `log_likelihood_function` and 
 `visualize` functions.
 
 - `Gaussian`: a model component representing a 1D Gaussian profile.

These are functionally identical to the `Analysis` and `Gaussian` objects you have seen elsewhere in the workspace.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from os import path

import autofit as af

"""
__Data__

Load data of a 1D Gaussian from a .json file in the directory `autofit_workspace/dataset/gaussian_x2`.

This 1D data was created using two 1D Gaussians, therefore model comparison should favor a model with two Gaussians over 
a models with 1 or 3 Gaussians.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x2")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
Plot the data. 
"""
xvalues = range(data.shape[0])

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", linestyle=" ", elinewidth=1, capsize=2
)
plt.title("1D Gaussian Dataset Used For Model Comparison.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Model x1 Gaussian__

Create a model to fit the data, starting with a model where the data is fitted with 1 Gaussian.
"""
model = af.Collection(gaussian_0=af.ex.Gaussian)

"""
The `info` attribute shows the model in a readable format, showing it contains one `Gaussian`.
"""
print(model.info)

"""
Create the analysis which fits the model to the data.
"""
analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
Fit the data using a non-linear search, to determine the goodness of fit of this model.

We use the nested sampling algorithm Dynesty, noting that the Bayesian evidence (`log_evidence`) of a model can only
be estimated using a nested sampling algorithm.
"""
search = af.DynestyStatic(
    path_prefix=path.join("features", "model_comparison"),
    name="gaussian_x1",
    nlive=50,
    iterations_per_update=3000
)

"""
Perform the fit.
"""
result_x1_gaussian = search.fit(model=model, analysis=analysis)
ddd
"""
The results are concisely summarised using the `result.info` property.

These show that the parameters of the Gaussian are well constrained, with small errors on their inferred values.
However, it does not inform us of whether the model provides a good fit to the data overall.
"""
print(result_x1_gaussian.info)

"""
The maximum log likelihood model is used to visualize the fit.

For 1 Gaussian, residuals are visible, whereby the model Gaussian cannot fit the highest central data-point and 
there is a mismatch at the edges of the profile around pixels 40 and 60.

Based on visual inspection, the model therefore provides a poor fit to the data.
"""
instance = result_x1_gaussian.max_log_likelihood_instance

gaussian_0 = instance.gaussian_0.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
model_data = gaussian_0

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", linestyle=" ", elinewidth=1, capsize=2
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.plot(range(data.shape[0]), gaussian_0, "--")
plt.title("Model fit using 1 Gaussian.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
Print the `log_likelihood` and `log_evidence` of this model-fit, which we will compare to more complex models in order 
to determine which model provides the best fit to the data.
"""
print("1 Gaussian:")
print(f"Log Likelihood: {result_x1_gaussian.samples.max_log_likelihood()}")
print(f"Log Evidence: {result_x1_gaussian.samples.log_evidence}")

"""
__Model x2 Gaussian__

We now create a model to fit the data which consists of 2 Gaussians.
"""
model = af.Collection(gaussian_0=af.ex.Gaussian, gaussian_1=af.ex.Gaussian)

"""
The `info` attribute shows the model now consists of two `Gaussian`'s.
"""
print(model.info)

"""
We repeat the steps above to create the non-linear search and perform the model-fit.
"""
search = af.DynestyStatic(
    path_prefix=path.join("features", "model_comparison"),
    name="gaussian_x2",
    nlive=50,
    iterations_per_update=3000
)

result_x2_gaussian = search.fit(model=model, analysis=analysis)

"""
The results show that two Gaussians have now been fitted to the data.
"""
print(result_x2_gaussian.info)

"""
Visualizing the fit, we see that the problems with the previous fit have been addressed. The central data-point at the 
highest normalization is fitted correctly and the residuals at the edges of the profile around pixels 40 and 60 are 
significantly reduced.

There are effectively no residuals, indicating that the model provides a good fit to the data.

The residuals are so small that they are consistent with noise in the data. One therefore should not expect that 
a more complex model than one with 2 Gaussians can provide a better fit.
"""
instance = result_x2_gaussian.max_log_likelihood_instance

gaussian_0 = instance.gaussian_0.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
gaussian_1 = instance.gaussian_0.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
model_data = gaussian_0 + gaussian_1

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", linestyle=" ", elinewidth=1, capsize=2
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.plot(range(data.shape[0]), gaussian_0, "--")
plt.plot(range(data.shape[0]), gaussian_1, "--")
plt.title("Model fit using 2 Gaussian.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
Print the `log_likelihood` and `log_evidence` of this model-fit, and compare these values to the previous model-fit
which used 1 Gaussian.
"""
print("1 Gaussian:")
print(f"Log Likelihood: {max(result_x1_gaussian.samples.log_likelihood_list)}")
print(f"Log Evidence: {result_x1_gaussian.samples.log_evidence}")

print("2 Gaussians:")
print(f"Log Likelihood: {max(result_x2_gaussian.samples.log_likelihood_list)}")
print(f"Log Evidence: {result_x2_gaussian.samples.log_evidence}")

"""
Both the `log_likelihood` and `log_evidence` have increased significantly, indicating that the model with 2 Gaussians
is favored over the model with 1 Gaussian.

This is expected, as we know the data was generated using 2 Gaussians!

__Model x3 Gaussian__

We now create a model to fit the data which consists of 3 Gaussians.
"""
model = af.Collection(gaussian_0=af.ex.Gaussian, gaussian_1=af.ex.Gaussian, gaussian_2=af.ex.Gaussian)

"""
The `info` attribute shows the model consists of three `Gaussian`'s.
"""
print(model.info)

"""
We repeat the steps above to create the non-linear search and perform the model-fit.
"""
search = af.DynestyStatic(
    path_prefix=path.join("features", "model_comparison"),
    name="gaussian_x3",
    nlive=50,
    iterations_per_update=3000
)

result_x3_gaussian = search.fit(model=model, analysis=analysis)

"""
The results show that three Gaussians have now been fitted to the data.
"""
print(result_x3_gaussian.info)

"""
Visualizing the fit, we see that there are effectively no residuals, indicating that the model provides a good fit.

By eye, this fit is as good as the 2 Gaussian model above.
"""
instance = result_x3_gaussian.max_log_likelihood_instance

gaussian_0 = instance.gaussian_0.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
gaussian_1 = instance.gaussian_0.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
gaussian_2 = instance.gaussian_0.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
model_data = gaussian_0 + gaussian_1 + gaussian_2

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", linestyle=" ", elinewidth=1, capsize=2
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.plot(range(data.shape[0]), gaussian_0, "--")
plt.plot(range(data.shape[0]), gaussian_1, "--")
plt.plot(range(data.shape[0]), gaussian_2, "--")
plt.title("Model fit using 3 Gaussian.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
We print the `log_likelihood` and `log_evidence` of this model-fit, and compare these values to the previous model-fit
which used 1 and 2 Gaussian.
"""
print("1 Gaussian:")
print(f"Log Likelihood: {max(result_x1_gaussian.samples.log_likelihood_list)}")
print(f"Log Evidence: {result_x1_gaussian.samples.log_evidence}")

print("2 Gaussians:")
print(f"Log Likelihood: {max(result_x2_gaussian.samples.log_likelihood_list)}")
print(f"Log Evidence: {result_x2_gaussian.samples.log_evidence}")

print("3 Gaussians:")
print(f"Log Likelihood: {max(result_x3_gaussian.samples.log_likelihood_list)}")
print(f"Log Evidence: {result_x3_gaussian.samples.log_evidence}")

"""
We now see an interesting result. The `log_likelihood` of the 3 Gaussian model is higher than the 2 Gaussian model
(albeit, only slightly higher). However, the `log_evidence` is lower than the 2 Gaussian model.

This confirms the behavior discussed at the start of the tutorial. The Bayesian evidence penalizes models with more 
freedom to fit the data, unless they provide a significantly better fit to the data. Using the evidence we favor the
2 Gaussian model over the 3 Gaussian model for this reason, whereas using the likelihood we favor the 3 Gaussian model.

__Wrap Up__

Discuss Priors. Discuss unique id and benefits of autofit / science workflow.
"""