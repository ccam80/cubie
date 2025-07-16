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
