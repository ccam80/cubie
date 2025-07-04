from warnings import warn

def check_requested_timing_possible(dt_min: float, dt_max: float, dt_save: float, dt_summarise: float) -> None:
    """Check if the requested timing parameters make sense - you can't save more frequently than you step,
    and you can't summarise more often than you save (that's not really a summary).

    Args:
        dt_min (float): The minimum internal step size of the loop.
        dt_max (float): The maximum internal step size of the loop.
        dt_save (float): The time interval between saves.
        dt_summarise (float): The time interval between summaries.

    Raises:
        ValueError: If the requested timing parameters are not compatible with the internal step size.
    """
    if dt_max < dt_min:
        raise ValueError("dt_max must be greater than or equal to dt_min. You've requested a maximum step size of "
                         f"{dt_max}s, but your minimum step size is {dt_min}s.")
    if dt_save < dt_min:
        raise ValueError("dt_save must be greater than or equal to the minimum internal step size (dt_min). "
                         f"you've requested to save every {dt_save}s, but your minimum loop step size is {dt_min}s.")
    if dt_summarise < dt_save:
        raise ValueError("dt_summarise must be greater than or equal to dt_save. You've requested to summarise every "
                         f"{dt_summarise}s, but your save interval is {dt_save}s.")

    if dt_max > dt_save:
        warn(f"dt_max ({dt_max}s) is greater than dt_save ({dt_save}s). The loop will never be able to step"
             f"that far before stopping to save, so dt_max is redundant.", UserWarning)


def convert_times_to_fixed_steps(internal_step_size: float, dt_save: float, dt_summarise: float) -> tuple[
    int, int, float, float]:
    """Convert time intervals to fixed step counts based on the internal step size.

    Args:
        internal_step_size (float): The internal step size of the loop.
        dt_save (float): The time interval between saves.
        dt_summarise (float): The time interval between summaries.

    Returns:
        tuple: A tuple containing:
            - n_steps_save: Number of internal steps between saves
            - n_steps_summarise: Number of save steps between summaries
            - actual_dt_save: The actual save interval (may differ from requested)
            - actual_dt_summarise: The actual summary interval (may differ from requested)
    """
    n_steps_save = int(dt_save / internal_step_size)
    n_steps_summarise = int(dt_summarise / dt_save)

    actual_dt_save = n_steps_save * internal_step_size
    actual_dt_summarise = n_steps_summarise * actual_dt_save

    return n_steps_save, n_steps_summarise, actual_dt_save, actual_dt_summarise