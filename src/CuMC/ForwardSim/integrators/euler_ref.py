# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:31:43 2024

@author: cca79
"""

# -*- coding: utf-8 -*-
import os

if __name__ == '__main__':
    os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
    os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"
    os.environ["NUMBA_OPT"] = "0"



import numpy as np
from cupy import asarray, ascontiguousarray, get_default_memory_pool
from numba import cuda, from_dtype, literally
from numba import float32, float64, int32, int64, void, int16
from numba.types import literal
from solvers.genericSolver import genericSolver
from numba.extending import as_numba_type



# global0 = 0


class Solver(genericSolver):

    def __init__(self,
                 precision = np.float32,
                 diffeq_sys = None
                 ):
        super().__init__(precision, diffeq_sys)

        self.precision = precision

    def build_kernel(self,
                     system):
        dxdtfunc = system.dxdtfunc
        calc_consts = system.calc_grid_constants
        global zero
        zero = 0
        global nstates
        nstates = system.num_states
        global nsaved
        nsaved = len(system.saved_state_indices)
        # global constants_length
        # constants_length = len(system.constants_array)


        precision = self.numba_precision
        as_numba_type.register(float, self.numba_precision)

        @cuda.jit(
            # (int32,
            # self.numba_precision[:,:,::1],
            # self.numba_precision[:,:],
            # int32[:],
            # self.numba_precision[:],
            # self.numba_precision[:],
            # self.numba_precision,
            # self.numba_precision,
            # self.numba_precision,
            # self.numba_precision[:],
            # self.numba_precision),
            max_registers=160,
            opt=True,
            lineinfo=True
            )
        def eulerkernel(xblocksize,
                        output,
                        grid_values,
                        saved_states,
                        inits,
                        step_size,
                        duration,
                        output_fs,
                        filtercoeffs,
                        warmup_time):


            #Figure out where we are on the chip
            tx = int16(cuda.threadIdx.x)
            block_index = int32(cuda.blockIdx.x)
            l_param_set = int32(xblocksize * block_index + tx)


            # Don't try and do a run that hasn't been requested.
            if l_param_set >= len(grid_values):
                return

            l_step_size = precision(step_size)
            l_ds_rate = int32(round(1 / (output_fs * l_step_size)))                   #samples per output value
            l_n_outer = int32(round((duration / l_step_size) / l_ds_rate))            #samples per output value
            l_warmup = int32(warmup_time * output_fs)

            l_sweep_params = grid_values[l_param_set]
            l_constants = cuda.local.array(shape=5, #This magic number is obtained from the length of the one-run constants function, entered here as magic until I figure out if the use is generalisable
                               dtype=precision)


            calc_consts(l_constants, l_sweep_params)

            # print(l_constants)
            l_saved_states = cuda.local.array(
                shape=(nsaved),
                dtype=int16)

            l_sums = cuda.local.array(
                shape=(nsaved),
                dtype=precision)

            for i in range(nsaved):
                l_saved_states[i] = saved_states[i]
                l_sums[i] = precision(0.0)
            # litzero = literal(zero)
            # litstates = literal(nstates)

            # Declare arrays to be kept in shared memory - very quick access.
            # dynamic_mem = cuda.shared.array(zero, dtype=precision)
            dynamic_mem = cuda.shared.array(zero, dtype=precision)
            s_state = dynamic_mem[:xblocksize*nstates] #Note: Change to shared memory was implemented incorrectly when profiling, compare this to local when optimisation is back on the agenda

            # l_state = cuda.local.array(
            #     shape=(nstates),
            #     dtype=precision)

            # vectorize local variables used in integration for convenience
            l_dxdt = cuda.local.array(
                shape=(nstates),
                dtype=precision)

            # c_constants = cuda.const.array_like(constants)

            c_filtercoefficients = cuda.const.array_like(filtercoeffs)



            #Initialise w starting states
            for i in range(nstates):
                s_state[tx*nstates+i] = inits[i]


            l_dxdt[:] = precision(0.0)
            l_t = precision(0.0)

            # from pdb import set_trace; set_trace()

            #Loop through output samples, one iteration per output
            for i in range(l_n_outer + l_warmup):

                #Loop through euler solver steps - smaller step than output for numerical accuracy reasons
                for j in range(l_ds_rate):
                    l_t += l_step_size

                    #Get current filter coefficient for the downsampling filter
                    filtercoeff = c_filtercoefficients[j]

                    # Calculate derivative at sample
                    dxdtfunc(l_dxdt,
                             s_state[tx*nstates:(tx+1)*nstates],
                             l_constants,
                             l_t)

                    #Forward-step state using euler-maruyama eq
                    #Add sum*filter coefficient to a running sum for downsampler
                    for k in range(nstates):
                        s_state[tx*nstates + k] += l_dxdt[k] * l_step_size
                    for k in range(nsaved):
                        l_sums[k] += s_state[tx*nstates + l_saved_states[k]] * filtercoeff

                #Start saving only after warmup period (to get past transient behaviour)
                if i > (l_warmup - 1):

                    for n in range(nsaved):
                        output[i-l_warmup, l_param_set, n] = l_sums[n]

                #Reset filters to zero for another run
                l_sums[:] = precision(0)


        self.integratorKernel = eulerkernel




#%% Test Code
if __name__ == "__main__":

    from cell_models.fabbri_linder_cell_globalconsts import fabbri_linder_cell, initial_values

    precision = np.float32

    # #Setting up grid of params to simulate with
    ACh_values = np.linspace(1, 100, 2, dtype=precision)
    Iso_values = np.linspace(1, 1000, 1, dtype=precision)
    grid_params = [(a, b) for a in ACh_values for b in Iso_values]
    grid_labels = ['ACh', 'Iso']
    step_size = precision(0.01)
    fs = precision(1)
    duration = precision(25)
    sys = fabbri_linder_cell(precision = precision, saved_states = ['V', 'cAMP'])
    inits = [initial_values[label] for label in sys.state_labels]

    ODE = Solver(precision=precision)
    ODE.load_system(sys)
    solutions = ODE.run(sys,
                        inits,
                        duration,
                        step_size,
                        fs,
                        # grid_labels,
                        grid_params,
                        blocksize_x = 32,
                        warmup_time=precision(20.0))
    # psd, f_psd = ODE.get_psd(solutions, fs)
    # phase, f_phase = ODE.get_fft_phase(solutions, fs)