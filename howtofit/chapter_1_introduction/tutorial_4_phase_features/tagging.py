# This module create tags for phases settings that customize the analysis. We tag these phases for two reasons:

# 1) Tags describes the analysis settings, making it explicit what analysis was used to create the results.

# 2) Tags create a unique output path, ensuring that if you run multiple phases on the same data but with different
#    settings each non-linear search (e.g. MultiNest) won't inadvertantly use results generated via a different analysis
#    method.

def phase_tag_from_phase_settings(
    signal_to_noise_limit=None,
):

    signal_to_noise_limit_tag = signal_to_noise_limit_tag_from_signal_to_noise_limit(
        signal_to_noise_limit=signal_to_noise_limit
    )

    # You may well have many more tag which appear here.

    return (
        "phase_tag" # For every tag you add, you'll add it to this return statement
        + signal_to_noise_limit_tag
 # e.g. + your_own_tag
    )

# This function generates a string we'll use to 'tag' a phase which uses this setting, thus ensuring results are
# output to a unique path.

def signal_to_noise_limit_tag_from_signal_to_noise_limit(signal_to_noise_limit):
    """Generate a signal to noise limit tag, to customize phase names based on limiting the signal to noise ratio of
    the dataset being fitted.

    This changes the phase name 'phase_name' as follows:

    signal_to_noise_limit = None -> phase_name
    signal_to_noise_limit = 2 -> phase_name__snr_2
    signal_to_noise_limit = 10 -> phase_name__snr_10
    """
    if signal_to_noise_limit is None:
        return ""
    else:
        return "__snr_" + str(signal_to_noise_limit)
