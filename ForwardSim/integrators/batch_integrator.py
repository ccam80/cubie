# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:08:11 2024

@author: cca79
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:31:43 2024

@author: cca79
"""

# -*- coding: utf-8 -*-
import os

if __name__ == "__main__":
    os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
    os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"
    os.environ["NUMBA_OPT"] = "1"


import numpy as np
from cupy import asarray, asnumpy, ascontiguousarray, get_default_memory_pool
from numba import cuda, from_dtype
from numba import float32, float64, int32, int64, void, int16
from numba.extending import as_numba_type
from _utils import timing


# from systems import  thermal_cantilever_ax_b # For testing code only, do not touch otherwise
# from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64, xoroshiro128p_dtype
from warnings import catch_warnings, filterwarnings
with catch_warnings():
    filterwarnings("ignore", category=FutureWarning)
    from cupyx.scipy.signal import firwin, welch
from cupyx.scipy.fft import rfftfreq, rfft
# from _utils import get_noise_64, get_noise_32, timing


# xoro_type = from_dtype(xoroshiro128p_dtype)


class genericSolver(object):

    def __init__(self,
                 precision = np.float32,
                 system = None
                 ):

        self.precision = precision
        self.numba_precision = from_dtype(precision)

        if system:
            self.load_system(system)


    def load_system(self, diffeq_system):
        self.build_kernel(diffeq_system)


    def build_kernel(self, system):
        dxdtfunc = system.dxdtfunc

        global zero
        zero = 0
        global nstates
        nstates = system.num_states
        # global constants_length
        # constants_length = len(system.constants_array)

        precision = self.numba_precision

        # if precision == float32:
        #     get_noise = get_noise_32
        # else:
        #     get_noise = get_noise_64

        #Junk kernel loaded just to set structure.
        @cuda.jit(opt=True, lineinfo=True) # Lazy compilation allows for literalisation of shared mem params.
        def eulermaruyamakernel(xblocksize,
                                output,
                                grid_values,
                                # constants,
                                inits,
                                step_size,
                                duration,
                                output_fs,
                                filtercoeffs,
                                # RNG,
                                # noise_sigmas,
                                warmup_time):


            #Figure out where we are on the chip
            tx = int16(cuda.threadIdx.x)
            block_index = int32(cuda.blockIdx.x)
            l_param_set = int32(xblocksize * block_index + tx)




        self.integratorKernel = eulermaruyamakernel

    # @timing(warmup=True)
    def run(self,
            system,
            y0,
            duration,
            step_size,
            output_fs,
            # grid_labels,
            grid_values,
            noise_seed=1,
            blocksize_x=64,
            warmup_time=0.0):


        output_array = cuda.pinned_array((int(output_fs * duration),
                                          len(grid_values),
                                          len(system.saved_states)
                                          ),
                                         dtype=self.precision)
        output_array[:, :, :] = 0

        #TODO: add check here for init conditions - how to separate from real indices when it's not part of the grid?
        #Maybe tack inits onto the end of the constants array. Maybe noise too? Then we can sweep eeeeeeeverything.
        # This works I think. index inits by [-1:-num_states - 1], index noise by [-num_states - 2: -2*num_states -1].
        # I don't think it will ever clash?
        # for index, label in enumerate(grid_labels):
        #     grid_indices[index] = system.constant_indices[label]

        cp_filtercoefficients = firwin(int32(round(1 / (self.numba_precision(step_size) * self.numba_precision(output_fs)))),
                                      output_fs/2.01,
                                      window='hann',
                                      pass_zero='lowpass',
                                      fs = output_fs)
        np_filtercoefficients = asnumpy(cp_filtercoefficients).astype(self.precision)
        d_filtercoefficients = cuda.to_device(np_filtercoefficients)

        d_outputstates = cuda.to_device(output_array)
        d_saved_states = cuda.to_device(system.saved_state_indices)
        d_gridvalues = cuda.to_device(grid_values)
        d_inits = cuda.to_device(y0)


        #total threads / threads per block (or 1 if 1 is greater)
        BLOCKSPERGRID = int(max(1, np.ceil(len(grid_values) / blocksize_x)))

        #Size of shared allocation (n states per thread per block, times 2 (for sums) x 8 for float64)
        if self.numba_precision == float32:
            bytes_per_val = 4
        else:
            bytes_per_val = 8

        dynamic_sharedmem = blocksize_x * system.num_states * bytes_per_val
        # dynamic_sharedmem = 0 # Current cell immplementation requires no shared memory
        if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1":
            cuda.profile_start()
        self.integratorKernel[BLOCKSPERGRID, blocksize_x,
                              0, dynamic_sharedmem](
                                  blocksize_x,
                                  d_outputstates,
                                  d_gridvalues,
                                  d_saved_states,
                                  d_inits,
                                  step_size,
                                  duration,
                                  output_fs,
                                  d_filtercoefficients,
                                  # d_noise,
                                  # d_noisesigmas,
                                  warmup_time)

        if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1":
            cuda.profile_stop()
        cuda.synchronize()

        return np.ascontiguousarray(d_outputstates.copy_to_host().T)

    def get_psd(self, solutionsarray, fs, window='hann'):
        #Todo: make these arbitrary parameteres editable
        t_axis_length = solutionsarray.shape[2]
        num_states = solutionsarray.shape[0]
        grid_size = solutionsarray.shape[1]

        nperseg = int(t_axis_length / 4)
        noverlap = int(nperseg/2)
        nfft = nperseg*2

        mem_remaining = self.get_free_memory()


        num_segs = (t_axis_length - noverlap) // (nperseg - noverlap)
        segsize = nperseg * num_states * grid_size * 8
        #TODO: Try self.precision().itemsize when things are working.

        total_mem_for_operation = segsize * num_segs
        total_chunks = int(np.ceil(total_mem_for_operation / mem_remaining))
        chunksize = int(np.ceil(grid_size / total_chunks))

        psd_array = np.zeros((num_states,
                                  grid_size,
                                  int(nfft/2) + 1),
                                 dtype=self.precision)

        for i in range(total_chunks):

            index = chunksize * i

            mag_f, temp_psd_array = welch(asarray(solutionsarray[:,index:index + chunksize,:], order='C'),
                                       fs=fs*2*np.pi, #This is a magic adjustment for the non-dimensional system.
                                       window=window,
                                       nperseg=nperseg,
                                       nfft=nfft,
                                       detrend='linear',
                                       scaling='spectrum',
                                       axis=2)
            psd_array[:, index:index+chunksize, :] = np.abs(temp_psd_array.get())
            mag_f = mag_f.get()

        return psd_array, mag_f

    def get_fft_phase(self, solutionsarray, fs):
        t_axis_length = solutionsarray.shape[2]
        num_states = solutionsarray.shape[0]
        grid_size = solutionsarray.shape[1]

        mem_remaining = self.get_free_memory()
        total_mem_for_operation = solutionsarray.size * 16 # (will need complex128 values on the way to the phase angle)
        total_chunks = int(np.ceil(total_mem_for_operation / mem_remaining))
        chunksize = int(np.ceil(grid_size / total_chunks))

        fft_phase_array = np.zeros((num_states,
                                    grid_size,
                                    int(t_axis_length/ 2) + 1),
                                   dtype=self.precision)

        for i in range(total_chunks):

            index = chunksize * i
            fft_phase_array[:, index:index + chunksize, :] = np.angle(rfft(asarray(solutionsarray), axis=2).get())
            phase_f = rfftfreq(t_axis_length, d=1/(fs*2*np.pi)).get() # again some non-dimensional fs magic

        return fft_phase_array, phase_f

    def get_free_memory(self):
        total_mem = cuda.current_context().get_memory_info()[1] - 1024**3  # Leave 1G for misc overhead
        allocated_mem = get_default_memory_pool().used_bytes()

        return total_mem - allocated_mem


#%% Test Code
if __name__ == "__main__":
    precision = np.float32

    # #Setting up grid of params to simulate with
    a_gains = np.asarray([i * 0.01 for i in range(-500, 500)], dtype=precision)
    b_params = np.asarray([i * 0.02 for i in range(-500, 500)], dtype=precision)
    grid_params = [(a, b) for a in a_gains for b in b_params]
    grid_labels = ['omega_forcing', 'a']
    step_size = precision(0.001)
    fs = precision(1)
    duration = precision(100)
    # sys = thermal_cantilever_ax_b.diffeq_system(precision = precision)
    inits = np.asarray([1.0, 0, 1.0, 0, 1.0], dtype=precision)

    ODE = genericSolver(precision=precision)
    ODE.load_system(sys)
    solutions = ODE.run(sys,
                        inits,
                        duration,
                        step_size,
                        fs,
                        grid_labels,
                        grid_params,
                        warmup_time=precision(100.0))
    psd, f_psd = ODE.get_psd(solutions, fs)
    phase, f_phase = ODE.get_fft_phase(solutions, fs)