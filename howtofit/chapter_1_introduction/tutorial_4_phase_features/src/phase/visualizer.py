import autoarray.plot as aplt

# The visualizer is used by a phase to output results to the hard-disk. It is called both during the model-fit,
# enabling the best-fit model of a non-linear search to be output on-the-fly (e.g. whilst it is still running) and
# at the end.

# The 'image_path' specifies the path where images are output. By default, this is the image_path of the optimizer,
# so the folder 'output/phase_name/phase_tag/image'.


class AbstractVisualizer:
    def __init__(self, image_path):

        self.image_path = image_path


class Visualizer(AbstractVisualizer):
    def __init__(self, masked_dataset, image_path):

        # When the Visualizer is instantiated, the masked dataset is passed to it and visualized.

        super().__init__(image_path)

        self.masked_dataset = masked_dataset

        aplt.imaging.subplot_imaging(
            imaging=masked_dataset.dataset,
            sub_plotter=aplt.SubPlotter(
                output=aplt.Output(
                    filename="subplot_dataset", path=self.image_path, format="png"
                )
            ),
        )

        # For visualizing your own dataset, you should put your own methods here. You could either call a function that
        # directs PyAutoFit to your visualization methods:

        # from project import visualization_methods

        # visualization_method.plot_image(image=self.masked_dataset.image, image_path=self.image_path)

        # Or you can import matplotlib in this module and output figures directly here.

        # import matplotlib.pyplot as plt

        # plt.imshow(self.masked_dataset.image)
        # plt.savefig(self.image_path + "my_image" + ".png")
        # plt.imshow(image2)
        # plt.savefig(self.image_path + "my_image2" + ".png")

    def visualize_fit(self, fit, during_analysis):

        # The fit is visualized during the model-fit, thus it requires its own method which is called by the non-linear
        # search every set number of intervals. Below, we use PyAutoArray to output a fit subplot.

        aplt.fit_imaging.subplot_fit_imaging(
            fit=fit,
            sub_plotter=aplt.SubPlotter(
                output=aplt.Output(
                    filename="subplot_fit_imaging", path=self.image_path, format="png"
                )
            ),
        )

        # Like the dataset above, we can visualize a fit by pointing to methods in your source code or calling
        # matplotlib here.

        # from project import visualization_methods

        # visualization_method.plot_fit(fit=fit, image_path=self.image_path)

        # Or you can import matplotlib in this module and output figures directly here.

        # import matplotlib.pyplot as plt

        # plt.imshow(image)
        # plt.savefig(self.image_path + "my_image" + ".png")
        # plt.imshow(image2)
        # plt.savefig(self.image_path + "my_image2" + ".png")

        if not during_analysis:

            # If this function is called during an analysis, the during_analysis bool will be 'True'. If there are
            # images you only want to output at the end of the analysis, you can thus save them for this if clause only.

            # For example, this phase only visualizes individual images of the fit's residual map and
            # chi-squared map after the model fit has finished.

            aplt.fit_imaging.residual_map(
                fit=fit,
                plotter=aplt.Plotter(
                    output=aplt.Output(
                        filename="residual_map", path=self.image_path, format="png"
                    )
                ),
            )

            aplt.fit_imaging.chi_squared_map(
                fit=fit,
                plotter=aplt.Plotter(
                    output=aplt.Output(
                        filename="chi_squared_map", path=self.image_path, format="png"
                    )
                ),
            )
