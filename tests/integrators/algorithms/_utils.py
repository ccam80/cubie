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