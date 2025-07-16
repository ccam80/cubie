from numba import cuda
from attrs import define, field
from Typing import Sequence

@define
class OutputSizes:
    """Data class to hold the offsets for state and observable summaries."""
    _n_peaks: int = field(default=0, converter=int)
    mean: int = field(default=1, init=False)
    peaks: int = field(default=1, init=False)
    max: int = field(default=1, init=False)
    rms: int = field(default=1, init=False)

    def __attrs_post_init__(self):
        """Calculate the offsets based on the number of peaks."""
        self.peaks = self._n_peaks

    @property
    def unpacked_sizes(self) -> tuple[int, int, int, int]:
        """Unpack the offsets into a tuple."""
        return (self.mean, self.peaks, self.max, self.rms)

    @classmethod
    def unpack_sizes_from_n_peaks(cls, n_peaks: int) -> tuple[int, int, int, int]:
        """Create a TempOffsets instance from the number of peaks."""
        return cls(n_peaks=n_peaks).unpacked_sizes

@define
class TempArrayOffsets:
    """Represents the temporary memory offsets for each summary type."""
    _n_peaks: int = field(default=0, converter=int)
    mean_size: int = field(default=1, init=False)
    peaks_size: int = field(default=3, init=False)
    max_size: int = field(default=1, init=False)
    rms_size: int = field(default=1, init=False)

    def __attrs_post_init__(self):
        """Post-initialization to set the peaks offset based on the number of peaks."""
        self.peaks += self._n_peaks

    @property
    def unpacked_sizes(self) -> tuple[int, int, int, int]:
        """Unpack the offsets into a tuple."""
        return (self.mean, self.peaks, self.max, self.rms)



    @classmethod
    def unpack_sizes_from_n_peaks(cls, n_peaks: int) -> tuple[int, int, int, int]:
        """Deliver a tuple of offsets from an output list without faffing about with the class."""
        return cls(n_peaks=n_peaks).unpacked_sizes

@define
class SummaryFlags:
    """Data class to hold flags for summary calculations."""
    summarise_mean: bool = False
    summarise_peaks: bool = False
    summarise_max: bool = False
    summarise_rms: bool = False

    def __attrs_post_init__(self):
        """ Set combined "summarise" flag """
        self.summarise = self.summarise_mean or self.summarise_peaks or self.summarise_max or self.summarise_rms

    @classmethod
    def from_list(cls, summaries_list: Sequence[str]) -> 'SummaryFlags':
        summarise_mean = False
        summarise_peaks = False
        summarise_max = False
        summarise_rms = False
        if "mean" in summaries_list:
            summarise_mean = True
        if "peaks" in summaries_list:
            summarise_peaks = True
        if "max" in summaries_list:
            summarise_max = True
        if "rms" in summaries_list:
            summarise_rms = True
        return cls(summarise_mean, summarise_peaks, summarise_max, summarise_rms)

    @property
    def unpacked(self) -> tuple[bool, bool, bool, bool, bool]:
        """Unpack the flags into a tuple."""
        return (self.summarise, self.summarise_mean, self.summarise_peaks, self.summarise_max, self.summarise_rms)

    @classmethod
    def unpack_from_list(cls, output_types: list[str]) -> tuple[int, int, int, int]:
        """Deliver a tuple of flags from an output list without faffing about with the class."""
        return cls.from_list(output_types).unpacked

#***********************************************************************************************************************
# Device functions for updating running summaries
#***********************************************************************************************************************

@cuda.jit(device=True, inline=True)
def update_mean(value,
                 temp_array,
                 temp_array_start_index,
                 ):
    """Update running sum - 1 temp memory slot required per state"""
    temp_array[temp_array_start_index] += value


@cuda.jit(device=True, inline=True)
def save_mean(value,
                 temp_array,
                 temp_array_start_index,
                 ):
    """Update running sum - 1 temp memory slot required per state"""
    temp_array[temp_array_start_index] += value

@cuda.jit(device=True, inline=True)
def running_max(value,
                temp_array,
                temp_array_start_index,
                ):
    """Update the maximum value in the temporary array."""
    if value > temp_array[temp_array_start_index]:
        temp_array[temp_array_start_index] = value


@cuda.jit(device=True, inline=True)
def running_peaks(value,
                  temp_array,
                  temp_array_start_index,
                  current_index,
                  npeaks,
                  ):
    """Update the peak value in the temporary array.
    Per-state temporary array requirements:
    [prev, prev_prev, peak_counter, peak1, peak2, ..., peakN]
    total requirement: 3 + npeaks slots per state"""

    prev = temp_array[temp_array_start_index + 0]
    prev_prev = temp_array[temp_array_start_index + 1]
    peak_counter = int(temp_array[temp_array_start_index + 2])

    #only check if we have enough points and we haven't already maxed out the counter, and that we're not working with
    # a 0.0 value (at the start of the run, for example). This assumes no natural 0.0 values, which seems realistic
    # for many systems. A more robust implementation would check if we're within 3 samples of summarise_every, probably.
    if (current_index >= 2) and (peak_counter < npeaks) and (prev_prev != 0.0):
        if prev > value and prev_prev < prev:
            #Bingo
            temp_array[temp_array_start_index + 3 + peak_counter] = float(current_index - 1)
            temp_array[temp_array_start_index + 2] = float(int(temp_array[temp_array_start_index + 2]) + 1)

    temp_array[temp_array_start_index + 0] = value  # Update previous value
    temp_array[temp_array_start_index + 1] = prev  # Update previous previous value


@cuda.jit(device=True, inline=True)
def running_rms(value,
                temp_array,
                temp_array_start_index,
                current_step,
                ):
    """Update the RMS value in the temporary array.
    Per-state temporary array requirements:
    [running_meansquare]
    total requirement: 1 slots per state
    We keep a running square, and mean/square root when saving.
    this accumulates some numerical error in the case of widely varying input scales, and tests have been
    relaxed to rtol=1e-3 for float32, 1e-9 for float64 to accomodate.
    """
    sum_of_squares = temp_array[temp_array_start_index]
    if current_step == 0:
        sum_of_squares = 0.0
    sum_of_squares += (value * value)
    temp_array[temp_array_start_index] = sum_of_squares