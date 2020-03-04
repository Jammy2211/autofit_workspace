from autoarray.dataset import imaging

# This module is identical to tutorial_2_model_fitting.


class Dataset(imaging.Imaging):
    def __init__(self, image, noise_map):

        super(Dataset, self).__init__(image=image, noise_map=noise_map)
