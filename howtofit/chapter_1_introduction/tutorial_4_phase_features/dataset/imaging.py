from autoarray.dataset import imaging

# This module is identical to tutorial_2_model_fitting.


class Imaging(imaging.Imaging):
    def __init__(self, image, noise_map):

        super(Imaging, self).__init__(image=image, noise_map=noise_map)
