# In the previous example, we used a non-linear search to infer the best-fit lens model of imaging-imaging of a strong lens.
# In this example, we'll get a deeper intuition of how a non-linear search works.

# First, I want to develop the idea of a 'parameter space'. Lets think of a function, like the simple function below:

# f(x) = x^2

# In this function, when we input a parameter x, it returns a value f(x). The mappings between different values of x
# and f(x) define a parameter space (and if you remember your high school math classes, you'll remember this parameter
# space is a parabola).

# A function can of course have multiple parameters:

# f(x, y, z) = x + y^2 - z^3

# This function has 3 parameters, x, y and z. The mappings between x, y and z and f(x, y, z) again define a parameter
# space, albeit now in 3 dimensions. Nevertheless, one could still picture this parameter space as some 3 dimensional
# curved surface.

# The process of computing a likelihood in PyAutoLens can be visualized in exactly the same way. We have a set of lens
# model parameters, which we input into PyAutoLens's 'likelihood function'. Now, this likelihood function isn't something
# that we can write down analytically and its inherently non-linear. But, nevertheless, it is a function;
# if we put the same set of lens model parameters into it, we'll compute the same likelihood.

# We can write our likelihood function as follows (using x_mp, y_mp, I_lp etc. as short-hand notation for the
# mass-profile and light-profile parameters):

# f(x_mp, y_mp, R_mp, x_lp, y_lp, I_lp, R_lp) = a likelihood from PyAutoLens's tracer and fit.

# The point is, like we did for the simple functions above, we again have a parameter space! It can't be written down
# analytically and its undoubtedly very complex and non-linear. Fortunately, we've already learnt how to search it, and
# find the solutions which maximize our likelihood function!

# Lets inspect the results of the last tutorial's non-linear search. We're going to look at what are called 'probably
# density functions' or PDF's for short. These represent where the highest likelihood regions of parameter space
# were found for each parameter.
#
# Navigate to the folder 'autolens_workspace/howtolens/chapter_2_lens_modeling/output_optimizer/t1_non_linear_search/image' and
# open the 'pdf_triangle.png' figure. The Gaussian shaped lines running down the diagonal of this triangle represent 1D
# estimates of the highest likelihood regions that were found in parameter space for each parameter.

# The remaining figures, which look like contour-maps, show the maximum likelihood regions in 2D between every parameter
# pair. We often see that two parameters are 'degenerate', whereby increasing one and decreasing the other leads to a
# similar likelihood value. The 2D PDF between the source galaxy's light-profile's intensity (I_l4) and effective
# radius (R_l4) shows such a degeneracy. This makes sense - making the source galaxy brighter and smaller is similar to
# making it fainter and bigger!

# So, how does PyAutoLens know where to look in parameter space? A parameter, say, the Einstein Radius, could in
# principle take any value between negative and positive infinity. AutoLens must of told it to only search regions of
# parameter space with 'reasonable' values (i.e. Einstein radii of around 1"-3").

# These are our 'priors' - which define where we tell the non-linear search to search parameter space. PyAutoLens uses
# two types of prior:

# 1) UniformPrior - The values of a parameter are randomly drawn between a lower and upper limit. For example, the
#                   orientation angle phi of a profile typically assumes a uniform prior between 0.0 and 180.0 degrees.

# 2) GaussianPrior - The values of a parameter are randomly drawn from a Gaussian distribution with a mean value
#                    and a width sigma. For example, an Einstein radius might assume a mean value of 1.0" and width
#                    of sigma = 1.0".
#

# The default priors on all parameters can be found by navigating to the 'config/priors/default' folder, and inspecting
# config files like light_profiles.ini. The convention is as follow:

# [EllipticalSersic]          # These are the priors used for an EllipticalSersic profile.
# effective_radius=u,0.0,2.0  # Its effective radius uses a UniformPrior with lower_limit=0.0, upper_limit=2.0
# sersic_index=g,4.0,2.0      # Its Sersic index uses a GaussianPrior with mean=4.0 and sigma=2.0
