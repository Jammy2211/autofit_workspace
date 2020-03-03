import autoarray as aa
import autofit as af
from autoarray.plot import mat_objs
from gaussian.src.plot import gaussian_plotters
from autoarray.plot import fit_imaging_plots


def setting(section, name):
    return af.conf.instance.visualize_plots.get(section, name, bool)


def plot_setting(section, name):
    return setting(section, name)


class AbstractVisualizer:
    def __init__(self, image_path):

        self.plotter = gaussian_plotters.Plotter(
            output=mat_objs.Output(path=image_path, format="png")
        )
        self.sub_plotter = gaussian_plotters.SubPlotter(
            output=mat_objs.Output(path=image_path + "subplots/", format="png")
        )
        self.include = gaussian_plotters.Include()


class DatasetVisualizer(AbstractVisualizer):
    def __init__(self, masked_dataset, image_path):
        super().__init__(image_path)
        self.masked_dataset = masked_dataset

        self.plot_subplot_dataset = plot_setting("dataset", "subplot_dataset")
        self.plot_dataset_data = plot_setting("dataset", "data")
        self.plot_dataset_noise_map = plot_setting("dataset", "noise_map")
        self.plot_dataset_psf = plot_setting("dataset", "psf")

        self.plot_dataset_signal_to_noise_map = plot_setting(
            "dataset", "signal_to_noise_map"
        )
        self.plot_dataset_absolute_signal_to_noise_map = plot_setting(
            "dataset", "absolute_signal_to_noise_map"
        )
        self.plot_dataset_potential_chi_squared_map = plot_setting(
            "dataset", "potential_chi_squared_map"
        )

        self.plot_fit_all_at_end_png = plot_setting("fit", "all_at_end_png")
        self.plot_fit_all_at_end_fits = plot_setting("fit", "all_at_end_fits")

        self.plot_subplot_fit = plot_setting("fit", "subplot_fit")
        self.plot_fit_data = plot_setting("fit", "data")
        self.plot_fit_noise_map = plot_setting("fit", "noise_map")
        self.plot_fit_signal_to_noise_map = plot_setting("fit", "signal_to_noise_map")
        self.plot_fit_model_data = plot_setting("fit", "model_data")
        self.plot_fit_residual_map = plot_setting("fit", "residual_map")
        self.plot_fit_normalized_residual_map = plot_setting(
            "fit", "normalized_residual_map"
        )
        self.plot_fit_chi_squared_map = plot_setting("fit", "chi_squared_map")

        self.visualize_imaging()

    @property
    def masked_imaging(self):
        return self.masked_dataset

    def visualize_imaging(self):

        plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "imaging/"
        )

        if self.plot_subplot_dataset:

            aa.plot.imaging.subplot_imaging(
                imaging=self.masked_imaging.imaging,
                mask=self.include.mask_from_masked_dataset(
                    masked_dataset=self.masked_dataset
                ),
                include=self.include,
                sub_plotter=self.sub_plotter,
            )

        aa.plot.imaging.individual(
            imaging=self.masked_imaging.imaging,
            mask=self.include.mask_from_masked_dataset(
                masked_dataset=self.masked_dataset
            ),
            plot_image=self.plot_dataset_data,
            plot_noise_map=self.plot_dataset_noise_map,
            plot_psf=self.plot_dataset_psf,
            plot_signal_to_noise_map=self.plot_dataset_signal_to_noise_map,
            plot_absolute_signal_to_noise_map=self.plot_dataset_absolute_signal_to_noise_map,
            plot_potential_chi_squared_map=self.plot_dataset_potential_chi_squared_map,
            include=self.include,
            plotter=plotter,
        )

    def visualize_fit(self, fit, during_analysis):

        plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "fit_imaging/"
        )

        if self.plot_subplot_fit:
            fit_imaging_plots.subplot_fit_imaging(
                fit=fit, include=self.include, sub_plotter=self.sub_plotter
            )

        fit_imaging_plots.individuals(
            fit=fit,
            plot_image=self.plot_fit_data,
            plot_noise_map=self.plot_fit_noise_map,
            plot_signal_to_noise_map=self.plot_fit_signal_to_noise_map,
            plot_model_image=self.plot_fit_model_data,
            plot_residual_map=self.plot_fit_residual_map,
            plot_chi_squared_map=self.plot_fit_chi_squared_map,
            plot_normalized_residual_map=self.plot_fit_normalized_residual_map,
            include=self.include,
            plotter=plotter,
        )

        if not during_analysis:

            if self.plot_fit_all_at_end_png:
                fit_imaging_plots.individuals(
                    fit=fit,
                    plot_image=True,
                    plot_noise_map=True,
                    plot_signal_to_noise_map=True,
                    plot_model_image=True,
                    plot_residual_map=True,
                    plot_normalized_residual_map=True,
                    plot_chi_squared_map=True,
                    include=self.include,
                    plotter=plotter,
                )

            if self.plot_fit_all_at_end_fits:

                self.visualize_fit_in_fits(fit=fit)

    def visualize_fit_in_fits(self, fit):

        fits_plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "fit_imaging/fits/", format="fits"
        )

        fit_imaging_plots.individuals(
            fit=fit,
            plot_image=True,
            plot_noise_map=True,
            plot_signal_to_noise_map=True,
            plot_model_image=True,
            plot_residual_map=True,
            plot_normalized_residual_map=True,
            plot_chi_squared_map=True,
            include=self.include,
            plotter=fits_plotter,
        )
