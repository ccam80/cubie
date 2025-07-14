Currently unhoused info:
- Noise sigmas - these are used only in a Euler-maruyama loop (currently), so should belong to the integrator. This may require experimentation.s

The default constants/parameters dict layout makes it tricky to move terms between parameters and constants to optimise execution between runs with different
fixed/free parameters.

Having initial values inside the system seems like a big fudge. Why would they be in there? Perhaps they could be in there as defaults only?

Move back to cupy arrays for mem managment help - wait until current cuda version is supported to avoid a massive fartaround with reinstalling.

x = cuda.threadIdx.x
bx = cuda.blockIdx.x
if x == 0 and bx == 0:
    from pdb import set_trace;
    set_trace()

The CuNODE solvers stored the array in the order: [time, run_index, state], as this seemed to maximise cache hits and reduce 
conflicts. Each thread should proceed roughly in lockstep (not assured!), should load it's slice of states into cache, and 
write them without successive misses. 

 # def change_values(self, **kwargs):
    # Keep this as a reference for how to test if a system rebuild is required
    #     old_vals = self.system_conditions.copy()
    #     _update_dicts_from_kwargs([self.system_conditions], **kwargs)
    #
    #     #trigger a rebuild if the shape of any array has changed, or if constants have changed, as these require rebuild
    #     for array in self.system_conditions:
    #         if isinstance(self.system_conditions[array], np.ndarray):
    #             if self.system_conditions[array].shape != old_vals[array].shape:
    #                 self.is_built = False
    #
    #     if self.system_conditions['num_drivers'] != old_vals['num_drivers']:
    #         self.is_built = False
    #     if self.system_conditions['constants'] != old_vals['constants']:
    #         self.is_built = False

#TODO: tidy gitignore and project files, scrub from repo