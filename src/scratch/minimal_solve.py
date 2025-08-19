from cubie import systems
from cubie import solve_ivp
import numpy as np

precision = np.float64

try: #local root
    Pig7TVE = np.genfromtxt("./pig_7_TVE_normalised.csv", delimiter=",", dtype=precision)[:,np.newaxis]
    matlab_results = np.genfromtxt(
            "./nic_numerical_comparison/comparison_waveforms.csv",
            delimiter=",", dtype=precision)
    input_values = np.genfromtxt(
            "./nic_numerical_comparison/valid_input_sets.csv",
            delimiter=",", dtype=precision)
    valid_indices = np.genfromtxt(
            "./nic_numerical_comparison/valid_indices.csv", delimiter=",",
            dtype=int)
except:
    try: # repo root
        Pig7TVE = np.genfromtxt("./src/scratch/pig_7_TVE_normalised.csv", delimiter=",", dtype=precision)[:,np.newaxis]
        matlab_results = np.genfromtxt(
                "./src/scratch/nic_numerical_comparison/comparison_waveforms"
                ".csv",
                delimiter=",", dtype=precision)
        input_values = np.genfromtxt(
                "./src/scratch/nic_numerical_comparison/valid_input_sets.csv",
                delimiter=",", dtype=precision)
        valid_indices = np.genfromtxt(
                "./src/scratch/nic_numerical_comparison/valid_indices.csv",
                delimiter=",", dtype=int)
    except:
        try: # src root
            Pig7TVE = np.genfromtxt("./scratch/pig_7_TVE_normalised.csv", delimiter=",", dtype=precision)[:,np.newaxis]
            matlab_results = np.genfromtxt(
                    "./scratch/nic_numerical_comparison/comparison_waveforms"
                    ".csv",
                    delimiter=",", dtype=precision)
            input_values = np.genfromtxt(
                    "./scratch/nic_numerical_comparison/valid_input_sets.csv",
                    delimiter=",", dtype=precision)
            valid_indices = np.genfromtxt(
                    "./scratch/nic_numerical_comparison/valid_indices.csv",
                    delimiter=",", dtype=int)
        except:
            raise (FileNotFoundError, "Couldn't find the file.")

fs = 250
system = systems.ThreeChamberModel(precision=precision)
results = solve_ivp(system,
                  y0={'V_h': np.arange(0, 1, 0.01)},
                  parameters={'E_h': np.arange(0.4,0.6, 0.01)},
                  forcing_vectors=Pig7TVE,
                  duration=float(25*len(Pig7TVE)/fs),
                  method='euler',
                  dt_eval=1/250,
                  dt_min=1/250,
                  dt_summarise=1/(len(Pig7TVE)/fs),
                  output_types=['state', 'observables', 'mean', 'max'])