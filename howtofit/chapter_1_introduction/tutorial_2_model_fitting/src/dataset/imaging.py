from autoarray.dataset import imaging

# This module is somewhat superflous - we are inheriting from an existing Python class in PyAutoArray and not extending
# the class with any new functionality.
#
# The only reason this module is here is so that you are aware that when you apply PyAutoFit to your model-fitting
# problem you will need a module which specifies your data-set, specifically the data (e.g. an image) and its noise-map.
# However, there may also be other aspects of the data you need when you fit the model (e.g. to generate model images of
# our Gaussians we also use the the grid the data is defined on).

# If you arn't familiar with Python classes and are unsure what the 'super' method does, just ignore it - as stated
# above you shouldn't care about what this module does, only take note that it exists.


class Dataset(imaging.Imaging):
    def __init__(self, image, noise_map):

        super(Dataset, self).__init__(image=image, noise_map=noise_map)
