from attrs import define, field, validators
import numpy as np

@define
class IntegratorLoopData:
    """Class to hold and return data related to the integrator loop, including compile-time constants and information
    propagated form the system."""

    precision: str = field(default='float32', validator=validators.in_(['float32', 'float64']))
    n_states: int = field(default=0, validator=validators.instance_of(int))
    n_observables: int = field(default=0, validator=validators.instance_of(int))
    n_parameters: int = field(default=0, validator=validators.instance_of(int))
    n_drivers: int = field(default=0, validator=validators.instance_of(int))
    dt_min: float = field(default=1e-6, validator=validators.instance_of(float))
    dt_max: float = field(default=1.0, validator=validators.instance_of(float))
    dt_save: float = field(default=0.1, validator=validators.instance_of(float))
    dt_summarise: float = field(default=0.1, validator=validators.instance_of(float))
    atol: float = field(default=1e-6, validator=validators.instance_of(float))
    rtol: float = field(default=1e-6, validator=validators.instance_of(float))
    save_time: bool = field(default=False, validator=validators.instance_of(bool))
    n_saved_states: int = field(default=0, validator=validators.instance_of(int))
    n_saved_observables: int = field(default=0, validator=validators.instance_of(int))
    summary_temp_memory: int = field(default=0, validator=validators.instance_of(int))
    dxdt_func: callable = field(default=None, validator=validators.instance_of(callable))
    save_state_func: callable = field(default=None, validator=validators.instance_of(callable))
    update_summary_func: callable = field(default=None, validator=validators.instance_of(callable))
    save_summary_func: callable = field(default=None, validator=validators.instance_of(callable))

    compile_settings = {'precision':           precision,
                        'n_states':            n_states,
                        'n_obs':               n_obs,
                        'n_par':               n_par,
                        'n_drivers':           n_drivers,
                        'dt_min':              dt_min,
                        'dt_max':              dt_max,
                        'dt_save':             dt_save,
                        'dt_summarise':        dt_summarise,
                        'atol':                atol,
                        'rtol':                rtol,
                        'save_time':           save_time,
                        'n_saved_states':      n_saved_states,
                        'n_saved_observables': n_saved_observables,
                        'summary_temp_memory': summary_temp_memory,
                        'dxdt_func':           dxdt_func,
                        'save_state_func':     save_state_func,
                        'update_summary_func': update_summary_func,
                        'save_summary_func':   save_summary_func,
                        }
    @property
    def n_states(self) -> int:
        """Return the number of states."""
        return self.n_states