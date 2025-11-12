# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:42:35 2024

@author: cca79
"""
import os

if __name__ == "__main__":
    os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
    os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"
    os.environ["NUMBA_OPT"] = "1"

from numba import cuda, float64, int64, int32, float32, from_dtype
# from cupy import asarray as cpasarray
# from _utils import clamp_32, clamp_64
import numpy as np
import math
from numba.extending import as_numba_type

from numba import from_dtype

#Debug helper function
if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
    def selp(cond, t, f):
        if cond:
            return t
        else:
            return f
    print("selp function modified")
    cuda.selp = selp

#------------------------Compile-time constant toggles ---------------------- #
#Numba treats global variables as compile-time constants, so my hope is that
#adding global values for the conditional toggles will remove the implied
#branching when LLVM has its way with the code. From memory, this doesn't fully work,
# and values must (maybe) be declared global in the constructor function scope as well?
VW_IKs = 1
Iso_1_uM_on = 0
Iso_linear_on = 0
Iso_cas_on = 1
dynamic_Ki_Nai = 0
R231C_on = 0

# Not currently certain which form I want this dict in, so have included an inverted copy for now.
state_labels = {
    0: "V",
    1: "Nai",
    2: "Ki",
    3: "y",
    4: "m_WT",
    5: "h_WT",
    6: "m",
    7: "h",
    8: "dL",
    9: "fL",
    10: "fCa",
    11: "dT",
    12: "fT",
    13: "R_1",
    14: "O",
    15: "I",
    16: "RI",
    17: "fTC",
    18: "fTMC",
    19: "fTMM",
    20: "fCMi",
    21: "fCMs",
    22: "fCQ",
    23: "Cai",
    24: "Ca_sub",
    25: "Ca_nsr",
    26: "Ca_jsr",
    27: "r_Kur",
    28: "s_Kur",
    29: "q",
    30: "r",
    31: "paS",
    32: "paF",
    33: "piy",
    34: "n",
    35: "a",
    36: "x",
    37: "cAMP",
    38: "PLBp"
}
inverted_state_labels = {
    "V": 0,
    "Nai": 1,
    "Ki": 2,
    "y": 3,
    "m_WT": 4,
    "h_WT": 5,
    "m": 6,
    "h": 7,
    "dL": 8,
    "fL": 9,
    "fCa": 10,
    "dT": 11,
    "fT": 12,
    "R_1": 13,
    "O": 14,
    "I": 15,
    "RI": 16,
    "fTC": 17,
    "fTMC": 18,
    "fTMM": 19,
    "fCMi": 20,
    "fCMs": 21,
    "fCQ": 22,
    "Cai": 23,
    "Ca_sub": 24,
    "Ca_nsr": 25,
    "Ca_jsr": 26,
    "r_Kur": 27,
    "s_Kur": 28,
    "q": 29,
    "r": 30,
    "paS": 31,
    "paF": 32,
    "piy": 33,
    "n": 34,
    "a": 35,
    "x": 36,
    "cAMP": 37,
    "PLBp": 38
}

initial_values = {
    "V": -47.787168,
    "Nai": 5.0,
    "Ki": 140.0,
    "y": 0.009508,
    "m_WT": 0.447724,
    "h_WT": 0.003058,
    "m": 0.447724,
    "h": 0.003058,
    "dL": 0.001921,
    "fL": 0.846702,
    "fCa": 0.844449,
    "dT": 0.268909,
    "fT": 0.020484,
    "R_1": 0.9308,
    "O": 6.181512e-09,
    "I": 4.595622e-10,
    "RI": 0.069199,
    "fTC": 0.017929,
    "fTMC": 0.259947,
    "fTMM": 0.653777,
    "fCMi": 0.217311,
    "fCMs": 0.158521,
    "fCQ": 0.138975,
    "Cai": 9.15641e-06,
    "Ca_sub": 6.226104e-05,
    "Ca_nsr": 0.435148,
    "Ca_jsr": 0.409551,
    "r_Kur": 0.011845,
    "s_Kur": 0.845304,
    "q": 0.430836,
    "r": 0.014523,
    "paS": 0.283185,
    "paF": 0.011068,
    "piy": 0.709051,
    "n": 0.1162,
    "a": 0.00277,
    "x": 0.0685782388661923,
    "cAMP": 0.03288333333333339,
    "PLBp": 0.23
}

default_constants = {
    "A2": 0.07,
    "ACh": 0.0,
    "ACh_on": 0.0,
    "ATPi_max": 2.533,
    "C": 57.0,
    "CCh_cas": 0.0,
    "CM_tot": 0.045,
    "CQ_tot": 10.0,
    "Ca_intracellular_fluxes_tau_dif_Ca": 5.469e-05,
    "Ca_intracellular_fluxes_tau_tr": 0.04,
    "Cao": 1.8,
    "EC50_SK": 0.7,
    "EC50_SR": 0.45,
    "F": 96485.3415,
    "GKACh": 0.00345,
    "GKr_max": 0.00424,
    "GKs_max": 0.00065,
    "GKur_max": 0.0001539,
    "GNa_WT": 0.0223,
    "GNa_max": 0.0223,
    "GSK": 0.0,
    "Gf_K_max": 0.00268,
    "Gf_Na_max": 0.00159,
    "Gto_max": 0.0035,
    "HSR": 2.5,
    "ICaL_fCa_gate_alpha_fCa": 0.0075,
    "INaK_max": 0.08105,
    "INa_WT_ratio": 0.5,
    "Iso_slope_cas_ICaL": 0.0,
    "K1ni": 395.3,
    "K1no": 1628.0,
    "K2ni": 2.289,
    "K2no": 561.4,
    "K3ni": 26.44,
    "K3no": 4.663,
    "KATP_min": 6034.0,
    "K_05ICaL1": 0.730287,
    "K_05ICaL2": 0.66145,
    "K_05INaK": 0.719701,
    "K_05RyR": 0.682891,
    "K_05iso": 58.57114132,
    "K_AC": 0.0735,
    "K_ACCa": 2.4e-05,
    "K_ACI": 0.016,
    "K_Ca": 0.000563995,
    "K_ICaL1": 0.470657,
    "K_ICaL2": 27.526226,
    "K_INaK": 0.435692,
    "K_NaCa": 3.343,
    "K_iso": 0.007,
    "K_iso_increase": 1.0,
    "K_iso_shift": 1.0,
    "K_iso_shift_ninf": 1.0,
    "K_iso_slope_dL": 1.0,
    "K_j_SRCarel": 1.0,
    "K_j_up": 1.0,
    "K_k1": 4.64625,
    "K_up": 0.000286113,
    "Kci": 0.0207,
    "Kcni": 26.44,
    "Kco": 3.663,
    "Kif": 26.26,
    "Km_Kp": 1.4,
    "Km_Nap": 14.0,
    "Km_fCa": 0.000338,
    "Ko": 5.4,
    "L_cell": 67.0,
    "L_sub": 0.02,
    "MaxSR": 15.0,
    "Mgi": 2.5,
    "MinSR": 1.0,
    "Nao": 140.0,
    "PKAtot": 1.0,
    "PKItot": 0.3,
    "PP1": 0.00089,
    "P_CaL": 0.4578,
    "P_CaT": 0.04132,
    "P_up_basal": 5.0,
    "Qci": 0.1369,
    "Qco": 0.0,
    "Qn": 0.4315,
    "R_2": 8314.472,
    "R_cell": 3.9,
    "RyR_max": 0.02,
    "RyR_min": 0.0127,
    "T": 310.0,
    "TC_tot": 0.031,
    "TMC_tot": 0.062,
    "V_R231C": 113.8797,
    "V_WT": 102.4597,
    "V_i_part": 0.46,
    "V_jsr_part": 0.0012,
    "V_nsr_part": 0.0116,
    "delta_m": 1e-05,
    "delta_m_WT": 1e-05,
    "g_ratio": 0.3153,
    "h_inf_shift": 0.0,
    "h_inf_slope": 1.0,
    "h_tau_gain": 1.0,
    "kATP": 6142.0,
    "kATP05": 6724.0,
    "kPKA_PLB": 1.610336,
    "kPLBp": 52.25,
    "kPP1": 23850.0,
    "kPP1_PLB": 0.07457,
    "k_WT": 55.206,
    "k_dL": 4.337,
    "k_fL": 5.3,
    "kb_CM": 542.0,
    "kb_CQ": 445.0,
    "kb_TC": 446.0,
    "kb_TMC": 7.51,
    "kb_TMM": 751.0,
    "kf_CM": 1641986.0,
    "kf_CQ": 175.4,
    "kf_TC": 88800.0,
    "kf_TMC": 227700.0,
    "kf_TMM": 2277.0,
    "kiCa": 500.0,
    "kim": 5.0,
    "koCa_max": 10000.0,
    "kom": 660.0,
    "ks": 1.480410851e+08,
    "m_inf_shift": 0.0,
    "m_inf_slope": 1.0,
    "m_tau_shift": 0.0,
    "nATP": 3.36,
    "nPKA": 5.0,
    "nPLB": 1.0,
    "nRyR": 9.773,
    "n_SK": 2.2,
    "n_inf_shift": 0.0,
    "n_shift_slope": 1.0,
    "nif": 9.281,
    "niso": 0.9264,
    "offset_fT": 0.0,
    "ratio_INaL_INa": 0.0,
    "shift_fL": 0.0,
    "slope_up": 5e-05,
    "tau_y_a_shift": 0.0,
    "tau_y_b_shift": 0.0,
    "y_shift": 0.0,
    "Iso": 0.0,
    "K_05if": 17.8741 / 600.0,
    "V_dL": -16.4508532699999996,
    "V_fL": -37.4,
    "cAMPb": 20.0 / 600.0,
    "kPKA": 9000.0 / 600.0,
    "kPKA_cAMP": 284.5 / 600.0,
    "k_R231C": -34.9006,
    }

def update_calculated_constants(constants):
    # Local variable declarations
    if Iso_1_uM_on:
        Iso_increase_1 =  1.23 * constants["K_iso_increase"]
        Iso_increase_2 =  1.2 * constants["K_iso_increase"]
        Iso_shift_1 = -14.0 * constants["K_iso_shift"]
        Iso_shift_2 = 7.5 * constants["K_iso_shift"]
        Iso_shift_dL = -8.0 * constants["K_iso_shift"]
        Iso_shift_ninf = 14.568 * constants["K_iso_shift_ninf"]
        Iso_slope_dL = -27.0 * constants["K_iso_slope_dL"]

    elif Iso_linear_on:
        Iso_increase_1 = 1.0 + 0.23 * constants["K_iso_increase"]
        Iso_increase_2 = 1.0 + 0.2 * constants["K_iso_increase"]
        Iso_shift_1 = 0.0
        Iso_shift_2 = 0.0
        Iso_shift_dL = 0.0
        Iso_shift_ninf = 0.0
        Iso_slope_dL = 0.0

    else:
        Iso_increase_1 = 1.0
        Iso_increase_2 = 1.0
        Iso_shift_1 = 0.0
        Iso_shift_2 = 0.0
        Iso_shift_dL = 0.0
        Iso_shift_ninf = 0.0
        Iso_slope_dL = 0.0

    ACh_block = 0.31 * 1e-06 * constants["ACh"] / (1e-06 * constants["ACh"] + 9e-05)
    ACh_shift = (-1.0 - 9.898 * (1e-06 * constants["ACh"])**0.618 / ((1e-06 * constants["ACh"])**0.618 + 1.22422999999999998e-03)) if constants["ACh"] > 0.0 else 0.0
    Ca_intracellular_fluxes_b_up = 0.7 * constants["ACh"] / (9e-05 + constants["ACh"])
    GKr = constants["GKr_max"] * (constants["Ko"] / 5.4)**0.5
    GNa_L = constants["GNa_max"] * constants["ratio_INaL_INa"]
    RTONF = constants["R_2"] * constants["T"] / constants["F"]
    V_cell = 1e-09 * 3.141592653589793 * constants["R_cell"]**2 * constants["L_cell"]
    V_sub = 1e-09 * 2.0 * 3.141592653589793 * constants["L_sub"] * (constants["R_cell"] - constants["L_sub"] / 2.0) * constants["L_cell"]
    alpha_a = ((3.5988 - 0.025641) / (1.0 + 1.2155e-06 / (1.0 * 1e-06 * constants["ACh"] + 1e-15)**1.6951) + 0.025641) / 1000.0  # 1e-15 constant added to avoid divide by zero when ACh = 0
    k34 = constants["Nao"] / (constants["K3no"] + constants["Nao"])
    kcch = 0.0146 * (constants["CCh_cas"]**1.4402 / (51.7331**1.4402 + constants["CCh_cas"]**1.4402))
    kiso = constants["K_iso"] + 0.1181 * (constants["Iso"]**constants["niso"] / (constants["K_05iso"]**constants["niso"] + constants["Iso"]**constants["niso"] + 1e-15))  # 1e-15 constant added to avoid nan
    P_up = constants["P_up_basal"] * (1.0 - Ca_intracellular_fluxes_b_up)
    V_i = constants["V_i_part"] * V_cell - V_sub
    V_jsr = constants["V_jsr_part"] * V_cell
    V_nsr = constants["V_nsr_part"] * V_cell

    # Update constants dictionary
    constants.update({
        "Iso_increase_1": Iso_increase_1,
        "Iso_increase_2": Iso_increase_2,
        "Iso_shift_1": Iso_shift_1,
        "Iso_shift_2": Iso_shift_2,
        "Iso_shift_dL": Iso_shift_dL,
        "Iso_shift_ninf": Iso_shift_ninf,
        "Iso_slope_dL": Iso_slope_dL,
        "ACh_block": ACh_block,
        "ACh_shift": ACh_shift,
        "Ca_intracellular_fluxes_b_up": Ca_intracellular_fluxes_b_up,
        "GKr": GKr,
        "GNa_L": GNa_L,
        "RTONF": RTONF,
        "V_cell": V_cell,
        "V_sub": V_sub,
        "alpha_a": alpha_a,
        "k34": k34,
        "kcch": kcch,
        "kiso": kiso,
        "P_up": P_up,
        "V_i": V_i,
        "V_jsr": V_jsr,
        "V_nsr": V_nsr
    })

    return constants

update_calculated_constants(default_constants)



class cell_constant_class(dict):
    def set_constant(self, key, item):
        if key in self:
            self[key] = item
            update_calculated_constants(self)
        else:
            raise KeyError(f"Constant {key} not in constants dictionary")

    def get_constant(self, key):
        if key in self:
            return self[key]
        else:
            raise KeyError(f"Constant {key} not in constants dictionary")



def system_constants(constants_dict=None, default_constants=default_constants, **kwargs):
    """ Instantiate a cell_constant_class - a dict with set_constant and get_constant
    methods. The set_constant method will update any derived (calculated) constants
    so you don't need to update manually.

    This is intended for use outside of CUDA so you can play it fast and loose with
    the math you use.
    """

    constants = cell_constant_class()

    if constants_dict is None:
        constants_dict = {}

    combined_updates = {**default_constants, **constants_dict, **kwargs}

    # Note: If the same value occurs in the dict and
    # keyword args, the kwargs one will win.
    for key, item in combined_updates.items():
        constants.update(combined_updates)

    update_calculated_constants(constants)


    return constants


class fabbri_linder_cell:
    """ This class should contain all system definitions. The constants management
    scheme can be a little tricky, because the GPU stuff can't handle dictionaries.
    The constants_array will be passed to your dxdt function - you can use the indices
    given in self.constant_indices to map them out while you set up your dxdt function.

    > test_system = diffeq_system()
    > print(diffeq_system.constant_indices)

    - Place all of your system constants and their labelsin the constants_dict.
    - Update self.num_states to match the number of state variables you
    need to solve for.
    - Feel free to define any helper functions inside the __init__ function.
    These must have the cuda.jit decorator with a signature (return(arg)), like you can
    see in the example functions.
    You can call these in the dxdt function.
    - update noise_sigmas with the std dev of gaussian noise in any state if
    you're doing a "noisy" run.

    Many numpy (and other) functions won't work inside the dxdt or CUDA device
    functions. Try using the Cupy function instead if you get an error.

    The included example is a big nasty sinoatrial node cell, based on the Fabbri
    2017 model and updated by Linder to incorporate graded isoprenaline responses
    and PKA effects.

    """
    def __init__(self,
                 num_states =39, # manually set this as a literal to avoid interfering with memory allocations in cuda
                 precision=np.float32,
                 state_labels = state_labels,
                 **kwargs):
        """Set system constant values then function as a factory function to
        build CUDA device functions for use in the ODE solver kernel. No
        arguments, no returns it's all just bad coding practice in here.

        """

        self.num_states = num_states
        self.precision = precision
        self.numba_precision = from_dtype(precision)
        p = self.numba_precision
        # self.noise_sigmas = np.zeros(self.num_state, dtype=precision)

        self.constants_dict  = system_constants(kwargs)
        self.constants_array = np.asarray([constant for (label, constant) in self.constants_dict.items()], dtype=precision)
        self.constant_indices = {label: index for index, (label, constant) in enumerate(self.constants_dict.items())}

        self.state_labels = state_labels


        as_numba_type.register(float, self.numba_precision)
        @cuda.jit(
            # (self.numba_precision[:],
            #     self.numba_precision[:],
            #     self.numba_precision[:],
            #     self.numba_precision),
                  device=True,
                  inline=True,
                  lineinfo=True, #comment out if running in cudasim mode
                  debug=False,
                  opt=True)
        def dxdtfunc(out,
                     state,
                     constants,
                     t):
            """ Put your dxdt calculations in here, including any reference signal
            or other math. Ugly is good here, avoid creating local variables and
            partial calculations - a long string of multiplies and adds, referring to
            the same array, might help the compiler make it fast. Avoid low powers,
            use consecutive multiplications instead.

            For a list of supported math functions you can include, see
            :https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html"""

            #This section was modified from a myokit export to cuda, by ChatGPT,
            #until its memory of input dicts slipped, so I went to a prompt with
            # system inputs using Gemini 2.5 Pro (exp) in Google AI studio.
            #And of course, checked, because AI is an enthusiastic idiot.


            E0_m_WT = state[0] + p(41.0)
            F_PLBp = cuda.selp(state[38] > p(0.23), p(3.3931) * (state[38]**p(4.0695)) / ((p(0.2805)**p(4.0695)) + (state[38]**p(4.0695))), p(1.698) * (state[38]**p(13.5842)) / ((p(0.224)**p(13.5842)) + (state[38]**p(13.5842))))
            INa_h_WT_gate_alpha_h_WT = p(20.0) * math.exp(p(-0.125) * (state[0] + p(75.0)))
            INa_h_WT_gate_beta_h_WT = p(2000.0) / (p(320.0) * math.exp(p(-0.1) * (state[0] + p(75.0))) + p(1.0))
            INa_h_gate_alpha_h = p(20.0) * math.exp(p(-0.125) * (state[0] + p(75.0)))
            INa_h_gate_beta_h = p(2000.0) / (p(320.0) * math.exp(p(-0.1) * (state[0] + p(75.0))) + p(1.0))
            INa_m_WT_gate_beta_m_WT = p(8000.0) * math.exp(p(-0.056) * (state[0] + p(66.0)))
            ISK_x_gate_tau_x = p(1.0) / (p(0.047) * state[24] * p(1000.0) + p(1.0) / p(76.0))
            PKA = cuda.selp(state[37] * p(600.0) < p(25.87), p(-0.9483029) * math.exp(-state[37] * p(600.0) * p(6.56147900000000062e-02)) + p(9.77816459999999998e-01), p(-4.52605089999999988e-01) * math.exp(-state[37] * p(600.0) * p(3.39509399999999989e-02)) + p(9.92217140000000053e-01))
            beta_a = p(10.0) * math.exp(p(0.0133) * (state[0] + p(40.0))) / p(1000.0)
            dT_inf = p(1.0) / (p(1.0) + math.exp(-(state[0] + p(38.3)) / p(5.5)))
            fT_inf = p(1.0) / (p(1.0) + math.exp((state[0] + p(58.7)) / p(3.8)))
            h_WT_inf = p(1.0) / (p(1.0) + math.exp((state[0] + p(69.804)) / p(4.4565)))
            k2 = p(1.1) * p(237.9851) * ((state[37] * p(600.0))**p(5.101) / ((p(20.1077)**p(6.101)) + (state[37] * p(600.0))**p(6.101)))
            m_WT_inf = p(1.0) / (p(1.0) + math.exp(-(state[0] + p(42.0504)) / p(8.3106)))
            pa_infinity = p(1.0) / (p(1.0) + math.exp(-(state[0] + p(10.0144)) / p(7.6607)))
            piy_inf = p(1.0) / (p(1.0) + math.exp((state[0] + p(28.6)) / p(17.1)))
            q_inf = p(1.0) / (p(1.0) + math.exp((state[0] + p(49.0)) / p(13.0)))
            r_Kur_inf = p(1.0) / (p(1.0) + math.exp((state[0] + p(6.0)) / p(-8.6)))
            r_inf = p(1.0) / (p(1.0) + math.exp(-(state[0] - p(19.3)) / p(15.0)))
            s_Kur_inf = p(1.0) / (p(1.0) + math.exp((state[0] + p(7.5)) / p(10.0)))
            tau_dT = p(0.001) / (p(1.068) * math.exp((state[0] + p(38.3)) / p(30.0)) + p(1.068) * math.exp(-(state[0] + p(38.3)) / p(30.0))) * p(1000.0)
            tau_fL = p(0.001) * (p(44.3) + p(230.0) * math.exp(-((state[0] + p(36.0)) / p(10.0))**p(2.0))) * p(1000.0)
            tau_paF = p(1.0) / (p(30.0) * math.exp(state[0] / p(10.0)) + math.exp(-state[0] / p(12.0))) * p(1000.0)
            tau_paS = p(8.46553540000000049e-01) / (p(4.2) * math.exp(state[0] / p(17.0)) + p(0.15) * math.exp(-state[0] / p(21.6))) * p(1000.0)
            tau_piy = p(1.0) / (p(100.0) * math.exp(-state[0] / p(54.645)) + p(656.0) * math.exp(state[0] / p(106.157))) * p(1000.0)
            tau_q = p(0.001) * p(0.6) * (p(65.17) / (p(0.57) * math.exp(p(-0.08) * (state[0] + p(44.0))) + p(0.065) * math.exp(p(0.1) * (state[0] + p(45.93)))) + p(10.1)) * p(1000.0)
            tau_r = p(0.001) * p(0.66) * p(1.4) * (p(15.59) / (p(1.037) * math.exp(p(0.09) * (state[0] + p(30.61))) + p(0.369) * math.exp(p(-0.12) * (state[0] + p(23.84)))) + p(2.98)) * p(1000.0)
            tau_r_Kur = (p(0.009) / (p(1.0) + math.exp((state[0] + p(5.0)) / p(12.0))) + p(0.0005)) * p(1000.0)
            tau_s_Kur = (p(0.59) / (p(1.0) + math.exp((state[0] + p(60.0)) / p(10.0))) + p(3.05)) * p(1000.0)
            ATPi = constants[3] * (constants[99] * ((state[37] * p(100.0) / constants[146])**constants[126]) / (constants[100] + ((state[37] * p(100.0) / constants[146])**constants[126])) - constants[35]) / p(100.0)
            E0_m = state[0] + p(41.0) - constants[125]
            INa_m_WT_gate_alpha_m_WT = cuda.selp(math.fabs(E0_m_WT) < constants[94], p(2000.0), p(200.0) * E0_m_WT / -(math.exp(p(-0.1) * E0_m_WT) - p(1.0)))
            INa_m_gate_beta_m = p(8000.0) * math.exp(p(-0.056) * (state[0] + p(66.0) - constants[125]))
            _j_up_common = constants[169] / (p(1.0) + math.exp((-state[23] + constants[57]) / constants[138]))


            if not VW_IKs: # Original check was VW_IKs == 0.0f
                Iso_inc_cas_IKs = p(-0.2152) + p(0.435692) * (PKA**p(10.0808)) / ((p(0.719701)**p(10.0808)) + (PKA**p(10.0808)))
                Iso_shift_cas_IKs_n_gate = -(p(24.4) * p(1.411375) * (PKA**p(9.281)) / ((p(0.704217)**p(9.281)) + (PKA**p(9.281))) - p(18.76)) # Note: Literals used
                E_Ks = constants[162] * math.log((constants[65] + p(0.12) * constants[71]) / (state[2] + p(0.12) * state[1]))
            else:
                Iso_inc_cas_IKs = p(-0.2152) + p(0.494259) * (PKA**p(10.0808)) / ((p(0.736717)**p(10.0808)) + (PKA**p(10.0808)))
                Iso_shift_cas_IKs_n_gate = -(p(24.4) * p(1.438835) * (PKA**p(9.281)) / ((p(0.707399)**p(9.281)) + (PKA**p(9.281))) - p(18.76)) # Note: Literals used
                E_Ks = constants[162] * math.log((constants[65] + p(0.0018) * constants[71]) / (state[2] + p(0.0018) * state[1]))

            if Iso_cas_on:
                Iso_inc_cas_ICaL = p(-0.2152) + constants[45] * (PKA**p(10.0808)) / ((constants[36]**p(10.0808)) + (PKA**p(10.0808)))
                Iso_inc_cas_IKs = Iso_inc_cas_IKs
                Iso_inc_cas_INaK = p(-0.2152) + constants[47] * (PKA**p(10.0808)) / ((constants[38]**p(10.0808)) + (PKA**p(10.0808)))
                Iso_shift_cas_ICaL_dL_gate = -(constants[46] * (PKA**p(9.281)) / ((constants[37]**p(9.281)) + (PKA**p(9.281))) - p(18.76))
                Iso_shift_cas_IKs_n_gate = Iso_shift_cas_IKs_n_gate
                Iso_shift_cas_If_y_gate = constants[61] * ((state[37]**constants[133]) / ((constants[143]**constants[133]) + (state[37]**constants[133]))) - p(18.76)
                koCa = constants[120] * (constants[84] - constants[83] * (PKA**constants[129]) / ((constants[39]**constants[129]) + (PKA**constants[129])) + p(1.0))
                j_up = constants[55] * (p(0.9) * _j_up_common * F_PLBp) # Assumes F_PLBp is correct
            else:
                j_up = _j_up_common
                Iso_inc_cas_ICaL = p(0.0)
                Iso_inc_cas_IKs = p(0.0)
                Iso_inc_cas_INaK = p(0.0)
                Iso_shift_cas_ICaL_dL_gate = p(0.0)
                Iso_shift_cas_IKs_n_gate = p(0.0)
                Iso_shift_cas_If_y_gate = p(0.0)
                koCa = constants[120]


            if R231C_on: # Original check was R231C_on != 0.0f
                R231C = constants[95] * ((p(1.0) / (p(1.0) + math.exp(-(state[0] - constants[88]) / constants[149])))**p(2.0)) / (((p(1.0) - constants[0]) / (p(1.0) + math.exp(-(state[0] - constants[89]) / constants[105])) + constants[0])**p(2.0))
            else:
                R231C = p(1.0)

            delta_fCMi = constants[113] * state[23] * (p(1.0) - state[20]) - constants[108] * state[20]
            delta_fCMs = constants[113] * state[24] * (p(1.0) - state[21]) - constants[108] * state[21]
            delta_fCQ = constants[114] * state[26] * (p(1.0) - state[22]) - constants[109] * state[22]
            delta_fTC = constants[115] * state[23] * (p(1.0) - state[17]) - constants[110] * state[17]
            delta_fTMC = constants[116] * state[23] * (p(1.0) - (state[18] + state[19])) - constants[111] * state[18]
            delta_fTMM = constants[117] * constants[69] * (p(1.0) - (state[18] + state[19])) - constants[112] * state[19]
            fCa_infinity = constants[64] / (constants[64] + state[24])
            fL_inf = p(1.0) / (p(1.0) + math.exp((state[0] - constants[145] + constants[137]) / constants[107]))
            h_inf = p(1.0) / (p(1.0) + math.exp((state[0] + p(69.804) - constants[96]) / (p(4.4565) * constants[97])))
            j_Ca_dif = (state[24] - state[23]) / constants[8]
            j_SRCarel = constants[54] * constants[122] * state[14] * (state[26] - state[24])
            j_tr = (state[25] - state[26]) / constants[9]
            # Corrected k1 calculation requires intermediate step for Cai from fCMi
            _Cai_from_fCMi = constants[108] * state[20] / (constants[113] * (p(1.0) - state[20]))
            k1 = constants[56] * (constants[43] + constants[41] / (p(1.0) + math.exp((constants[44] - _Cai_from_fCMi) / constants[42])))
            k3 = constants[147] * ((state[37]**(constants[127] - p(1.0))) / ((constants[148]**constants[127]) + (state[37]**constants[127])))
            k4 = constants[102] * (PKA**constants[128]) / ((constants[101]**constants[128]) + (PKA**constants[128]))
            k43 = state[1] / (constants[33] + state[1])
            k5 = constants[103] * (constants[74] * state[38] / (constants[104] + state[38]))
            kCaSR = constants[68] - (constants[68] - constants[70]) / (p(1.0) + (constants[12] / state[26])**constants[24])


            m_inf = p(1.0) / (p(1.0) + math.exp(-(state[0] + p(42.0504) - constants[123]) / (p(8.3106) * constants[124])))
            paF_inf = pa_infinity
            paS_inf = pa_infinity
            tau_fT = (p(1.0) / (p(16.67) * math.exp(-(state[0] + p(75.0)) / p(83.3)) + p(16.67) * math.exp((state[0] + p(75.0)) / p(15.38))) + constants[135]) * p(1000.0)
            tau_h = constants[98] / (INa_h_gate_alpha_h + INa_h_gate_beta_h) * p(1000.0)
            tau_h_WT = p(1.0) / (INa_h_WT_gate_alpha_h_WT + INa_h_WT_gate_beta_h_WT) * p(1000.0)
            x_infinity = p(0.81) * (state[24] * p(1000.0))**constants[130] / ((state[24] * p(1000.0))**constants[130] + (constants[11]**constants[130]))

            E_K = constants[162] * math.log(constants[65] / state[2])
            E_Na = constants[162] * math.log(constants[71] / state[1])
            E_mh = constants[162] * math.log((constants[71] + p(0.12) * constants[65]) / (state[1] + p(0.12) * state[2]))

            if not VW_IKs: # Original check was VW_IKs == 0.0f
                if Iso_linear_on:
                    _gks_iso_factor = p(1.0) + p(0.2) * constants[50]
                elif Iso_1_uM_on:
                    _gks_iso_factor = p(1.0) + p(0.2) * constants[50]
                elif Iso_cas_on:
                    _gks_iso_factor = p(1.0) + Iso_inc_cas_IKs
                else:
                    _gks_iso_factor = p(1.0)
                GKs = constants[16] * _gks_iso_factor
            else: # VW_IKs != 0.0f
                if Iso_linear_on:
                    _gks_iso_factor = p(1.0) + p(0.25) * constants[50]
                elif Iso_1_uM_on:
                    _gks_iso_factor = p(1.25) * constants[50]
                elif Iso_cas_on:
                    _gks_iso_factor = p(1.0) + Iso_inc_cas_IKs
                else:
                    _gks_iso_factor = p(1.0)
                GKs = p(0.2) * constants[16] * _gks_iso_factor

            ICaL_fCa_gate_tau_fCa = p(0.001) * fCa_infinity / constants[25]

            #currents are linearised for the region in which the denominator is smaller than 1e-6 to avoid singularities when running at 32 bit precision
            #This matches the approach published by the authors of Chaste-codegen
            ICaT_nonsingular = p(2.0) * constants[76] * state[0] / (constants[162] * -(math.exp(p(-1.0) * state[0] * p(2.0) / constants[162]) - p(1.0))) * (state[24] - constants[10] * math.exp(p(-2.0) * state[0] / constants[162])) * state[11] * state[12]
            P_ICaT = p(2.0) * constants[76]
            G_ICaT = state[11] * state[12]
            ICaT_at_zero = P_ICaT * (p(1.0)/p(2.0)) * (state[24] - constants[10]) * G_ICaT
            ICaT_deriv_at_zero = (P_ICaT * G_ICaT / (p(2.0) * constants[162])) * (state[24] + constants[10])
            L_ICaT = ICaT_at_zero + ICaT_deriv_at_zero * state[0]
            ICaT = cuda.selp(abs(state[0]) < constants[162] * p(5.0e-7), L_ICaT, ICaT_nonsingular)

            if not VW_IKs: # Original check was VW_IKs == 0.0f
                IKs_n_gate_beta_n = p(1.0) * math.exp(-(state[0] - constants[152] - Iso_shift_cas_IKs_n_gate - p(5.0)) / p(25.0))
                n_inf = math.sqrt(p(1.0) / (p(1.0) + math.exp(-(state[0] + p(0.6383) - constants[152] - Iso_shift_cas_IKs_n_gate - constants[131]) / (p(10.7071) * constants[132]))))
                IKs_n_gate_alpha_n = p(28.0) / (p(1.0) + math.exp(-(state[0] - p(40.0) - constants[152] - Iso_shift_cas_IKs_n_gate) / p(3.0)))
            else:
                IKs_n_gate_beta_n = p(1.93) * math.exp(-state[0] / p(83.2))
                n_inf = p(1.0) / (p(1.0) + math.exp(-(state[0] + p(15.733) + constants[155] - Iso_shift_cas_IKs_n_gate - constants[131]) / (constants[132] * p(27.77))))
                n_inf_safe_denom = cuda.selp(n_inf == p(1.0), p(1e-9), p(1.0) - n_inf) # Avoid division by zero if n_inf is 1.0
                IKs_n_gate_alpha_n = n_inf / n_inf_safe_denom * IKs_n_gate_beta_n

            INa_m_gate_alpha_m = cuda.selp(math.fabs(E0_m) < constants[93], p(2000.0), p(200.0) * E0_m / -(math.exp(p(-0.1) * E0_m) - p(1.0)))
            IsiCa_nonsingular = p(2.0) * constants[75] * constants[150] * (p(1.0) - constants[157]) * (p(1.0) + Iso_inc_cas_ICaL) * (state[0] - p(0.0)) / (constants[162] * -(math.exp(p(-1.0) * (state[0] - p(0.0)) * p(2.0) / constants[162]) - p(1.0))) * (state[24] - constants[10] * math.exp(p(-2.0) * (state[0] - p(0.0)) / constants[162])) * state[8] * state[9] * state[10]
            P_IsiCa = p(2.0) * constants[75] * constants[150] * (p(1.0) - constants[157]) * (p(1.0) + Iso_inc_cas_ICaL)
            G_ICaL = state[8] * state[9] * state[10] # Common gating factor for IsiCa, IsiK, IsiNa
            IsiCa_at_zero = P_IsiCa * (p(1.0)/p(2.0)) * (state[24] - constants[10]) * G_ICaL
            IsiCa_deriv_at_zero = (P_IsiCa * G_ICaL / (p(2.0) * constants[162])) * (state[24] + constants[10])
            L_IsiCa = IsiCa_at_zero + IsiCa_deriv_at_zero * state[0]
            IsiCa = cuda.selp(abs(state[0]) < constants[162] * p(5.0e-7), L_IsiCa, IsiCa_nonsingular)


            IsiK_nonsingular = p(0.000365) * constants[75] * constants[150] * (p(1.0) - constants[157]) * (p(1.0) + Iso_inc_cas_ICaL) * (state[0] - p(0.0)) / (constants[162] * -(math.exp(p(-1.0) * (state[0] - p(0.0)) / constants[162]) - p(1.0))) * (state[2] - constants[65] * math.exp(p(-1.0) * (state[0] - p(0.0)) / constants[162])) * state[8] * state[9] * state[10]
            P_IsiK = p(0.000365) * constants[75] * constants[150] * (p(1.0) - constants[157]) * (p(1.0) + Iso_inc_cas_ICaL)
            IsiK_at_zero = P_IsiK * (p(1.0)/p(1.0)) * (state[2] - constants[65]) * G_ICaL
            IsiK_deriv_at_zero = (P_IsiK * G_ICaL / (p(2.0) * constants[162])) * (state[2] + constants[65])
            L_IsiK = IsiK_at_zero + IsiK_deriv_at_zero * state[0]
            IsiK = cuda.selp(abs(state[0]) < constants[162] * p(1e-6),L_IsiK,IsiK_nonsingular )

            IsiNa_nonsingular = p(1.85e-05) * constants[75] * constants[150] * (p(1.0) - constants[157]) * (p(1.0) + Iso_inc_cas_ICaL) * (state[0] - p(0.0)) / (constants[162] * -(math.exp(p(-1.0) * (state[0] - p(0.0)) / constants[162]) - p(1.0))) * (state[1] - constants[71] * math.exp(p(-1.0) * (state[0] - p(0.0)) / constants[162])) * state[8] * state[9] * state[10]
            P_IsiNa = p(1.85e-05) * constants[75] * constants[150] * (p(1.0) - constants[157]) * (p(1.0) + Iso_inc_cas_ICaL)
            IsiNa_at_zero = P_IsiNa * (p(1.0)/p(1.0)) * (state[1] - constants[71]) * G_ICaL
            IsiNa_deriv_at_zero = (P_IsiNa * G_ICaL / (p(2.0) * constants[162])) * (state[1] + constants[71])
            L_IsiNa = IsiNa_at_zero + IsiNa_deriv_at_zero * state[0]
            IsiNa = cuda.selp(abs(state[0]) < constants[162] * p(1e-6), L_IsiNa, IsiNa_nonsingular)

            dL_inf = p(1.0) / (p(1.0) + math.exp(-(state[0] - constants[144] - constants[154] - Iso_shift_cas_ICaL_dL_gate) / (constants[106] * (p(1.0) + constants[28]) * (p(1.0) + constants[156] / p(100.0)))))
            di = p(1.0) + state[24] / constants[58] * (p(1.0) + math.exp(-constants[78] * state[0] / constants[162]) + state[1] / constants[59]) + state[1] / constants[29] * (p(1.0) + state[1] / constants[31] * (p(1.0) + state[1] / constants[33]))
            diff_Ca_jsr = (j_tr - (j_SRCarel + constants[7] * delta_fCQ)) * p(0.001)
            diff_PLBp = (k4 - k5) / p(60000.0)
            diff_cAMP = ((constants[168] - constants[167]) * ATPi + k1 * ATPi - k2 * state[37] - k3 * state[37]) / p(60000.0)
            diff_fCMi = delta_fCMi * p(0.001)
            diff_fCMs = delta_fCMs * p(0.001)
            diff_fCQ = delta_fCQ * p(0.001)
            diff_fTC = delta_fTC * p(0.001)
            diff_fTMC = delta_fTMC * p(0.001)
            diff_fTMM = delta_fTMM * p(0.001)
            diff_x = (x_infinity - state[36]) / ISK_x_gate_tau_x
            do_ = p(1.0) + constants[10] / constants[60] * (p(1.0) + math.exp(constants[79] * state[0] / constants[162])) + constants[71] / constants[30] * (p(1.0) + constants[71] / constants[32] * (p(1.0) + constants[71] / constants[34]))
            k32 = math.exp(constants[80] * state[0] / (p(2.0) * constants[162]))
            k41 = math.exp(-constants[80] * state[0] / (p(2.0) * constants[162]))
            kiSRCa = constants[118] * kCaSR
            koSRCa = koCa / kCaSR

            tau_m_WT = p(1.0) / (INa_m_WT_gate_alpha_m_WT + INa_m_WT_gate_beta_m_WT) * p(1000.0)
            tau_y = (p(1.0) / (p(0.36) * (state[0] + p(148.8) - constants[158] - constants[153] - Iso_shift_cas_If_y_gate - constants[139]) / (math.exp(p(0.066) * (state[0] + p(148.8) - constants[158] - constants[153] - Iso_shift_cas_If_y_gate - constants[139])) - p(1.0)) + p(0.1) * (state[0] + p(87.3) - constants[158] - constants[153] - Iso_shift_cas_If_y_gate - constants[140]) / -(math.exp(p(-0.2) * (state[0] + p(87.3) - constants[158] - constants[153] - Iso_shift_cas_If_y_gate - constants[140])) - p(1.0))) - p(0.054)) * p(1000.0)

            _y_inf_cond_val = -(p(80.0) - constants[158] - constants[153] - Iso_shift_cas_If_y_gate - constants[141])
            y_inf = cuda.selp(state[0] < _y_inf_cond_val, p(0.01329) + p(0.99921) / (p(1.0) + math.exp((state[0] + p(97.134) - constants[158] - constants[153] - Iso_shift_cas_If_y_gate - constants[141]) / p(8.1752))), p(0.0002501) * math.exp(-(state[0] - constants[158] - constants[153] - Iso_shift_cas_If_y_gate - constants[141]) / p(12.861)))
            ICaL = IsiCa + IsiK + IsiNa
            _crit_val_a1 = -p(41.8) - constants[154] - Iso_shift_cas_ICaL_dL_gate
            _crit_val_a2 = -p(6.8) - constants[154] - Iso_shift_cas_ICaL_dL_gate
            _crit_val_b = -p(1.8) - constants[154] - Iso_shift_cas_ICaL_dL_gate
            a_needs_delta = (state[0] == _crit_val_a1) or (state[0] == _crit_val_a2) or (state[0] == 0) # Assuming state[0] == 0 comparison is intentional and 0 is exact
            b_needs_delta = (state[0] == _crit_val_b)
            delta = p(1e-5) # Assuming precision() was just a placeholder name for the value itself
            adVm = cuda.selp(a_needs_delta, state[0] - delta, state[0])
            bdVm = cuda.selp(b_needs_delta, state[0] - delta, state[0])

            ICaL_dL_gate_alpha_dL = p(-0.02839) * (adVm + p(41.8) - constants[154] - Iso_shift_cas_ICaL_dL_gate) / (math.exp(-(adVm + p(41.8) - constants[154] - Iso_shift_cas_ICaL_dL_gate) / p(2.5)) - p(1.0)) - p(0.0849) * (adVm + p(6.8) - constants[154] - Iso_shift_cas_ICaL_dL_gate) / (math.exp(-(adVm + p(6.8) - constants[154] - Iso_shift_cas_ICaL_dL_gate) / p(4.8)) - p(1.0))
            ICaL_dL_gate_beta_dL = p(0.01143) * (bdVm + p(1.8) - constants[154] - Iso_shift_cas_ICaL_dL_gate) / (math.exp((bdVm + p(1.8) - constants[154] - Iso_shift_cas_ICaL_dL_gate) / p(2.5)) - p(1.0))
            IKACh = cuda.selp(constants[1] > p(0.0), constants[2] * constants[14] * (state[0] - E_K) * (p(1.0) + math.exp((state[0] + p(20.0)) / p(20.0))) * state[35], p(0.0))
            IKr = constants[160] * (state[0] - E_K) * (p(0.9) * state[32] + p(0.1) * state[31]) * state[33]
            IKs = R231C * GKs * (state[0] - E_Ks) * (state[34]**p(2.0))

            IKur = constants[17] * state[27] * state[28] * (state[0] - E_K)
            INaK = (p(1.0) + Iso_inc_cas_INaK) * constants[151] * constants[26] * ((p(1.0) + (constants[62] / constants[65])**p(1.2))**p(-1.0)) * ((p(1.0) + (constants[63] / state[1])**p(1.3))**p(-1.0)) * ((p(1.0) + math.exp(-(state[0] - E_Na + p(110.0)) / p(20.0)))**p(-1.0))
            INa_ = constants[19] * (state[6]**p(3.0)) * state[7] * (state[0] - E_mh)
            INa_L = constants[161] * (state[6]**p(3.0)) * (state[0] - E_mh)
            INa_WT = constants[18] * (state[4]**p(3.0)) * state[5] * (state[0] - E_mh)
            ISK = constants[20] * (state[0] - E_K) * state[36]
            IfK = state[3] * constants[21] * (state[0] - E_K)
            IfNa = state[3] * constants[22] * (state[0] - E_Na)
            Ito = constants[23] * (state[0] - E_K) * state[29] * state[30]
            diff_I = (kiSRCa * state[24] * state[14] - constants[119] * state[15] - (constants[121] * state[15] - koSRCa * (state[24]**p(2.0)) * state[16])) * p(0.001)
            diff_O = (koSRCa * (state[24]**p(2.0)) * state[13] - constants[121] * state[14] - (kiSRCa * state[24] * state[14] - constants[119] * state[15])) * p(0.001)
            diff_RI = (constants[121] * state[15] - koSRCa * (state[24]**p(2.0)) * state[16] - (constants[119] * state[16] - kiSRCa * state[24] * state[13])) * p(0.001)
            diff_R_1 = (constants[119] * state[16] - kiSRCa * state[24] * state[13] - (koSRCa * (state[24]**p(2.0)) * state[13] - constants[121] * state[14])) * p(0.001)
            diff_fCa = (fCa_infinity - state[10]) / ICaL_fCa_gate_tau_fCa * p(0.001)

            k12 = state[24] / constants[58] * math.exp(-constants[78] * state[0] / constants[162]) / di
            k14 = state[1] / constants[29] * state[1] / constants[31] * (p(1.0) + state[1] / constants[33]) * math.exp(constants[80] * state[0] / (p(2.0) * constants[162])) / di
            k21 = constants[10] / constants[60] * math.exp(constants[79] * state[0] / constants[162]) / do_
            k23 = constants[71] / constants[30] * constants[71] / constants[32] * (p(1.0) + constants[71] / constants[34]) * math.exp(-constants[80] * state[0] / (p(2.0) * constants[162])) / do_
            tau_m = p(1.0) / (INa_m_gate_alpha_m + INa_m_gate_beta_m) * p(1000.0)
            INa = (p(1.0) - constants[27]) * (INa_ + INa_L) + constants[27] * INa_WT
            If = IfNa + IfK
            diff_Ca_nsr = (j_up - j_tr * constants[171] / constants[172]) * p(0.001)
            diff_Cai = (p(1.0) * (j_Ca_dif * constants[164] - j_up * constants[172]) / constants[170] - (constants[6] * delta_fCMi + constants[86] * delta_fTC + constants[87] * delta_fTMC)) * p(0.001)
            tau_dL = p(0.001) / (ICaL_dL_gate_alpha_dL + ICaL_dL_gate_beta_dL) * p(1000.0)
            tau_n = p(1.0) / (IKs_n_gate_alpha_n + IKs_n_gate_beta_n) * p(1000.0)
            x1 = k41 * constants[166] * (k23 + k21) + k21 * k32 * (k43 + k41)
            x2 = k32 * k43 * (k14 + k12) + k41 * k12 * (constants[166] + k32)
            x3 = k14 * k43 * (k23 + k21) + k12 * k23 * (k43 + k41)
            x4 = k23 * constants[166] * (k14 + k12) + k14 * k21 * (constants[166] + k32)
            INaCa = constants[48] * (x2 * k21 - x1 * k12) / (x1 + x2 + x3 + x4)
            Itot = If + IKr + IKs + Ito + INaK + INaCa + INa + ICaL + ICaT + IKACh + IKur + ISK
            diff_Ca_sub = (j_SRCarel * constants[171] / constants[164] - ((IsiCa + ICaT - p(2.0) * INaCa) / (p(2.0) * constants[13] * constants[164]) + j_Ca_dif + constants[6] * delta_fCMs)) * p(0.001)
            if dynamic_Ki_Nai:
                diff_Nai = p(-1.0) * (INa + IfNa + IsiNa + p(3.0) * INaK + p(3.0) * INaCa) / (p(1.0) * (constants[170] + constants[164]) * constants[13]) * p(0.001)
                diff_Ki = p(-1.0) * (IKur + Ito + IKr + IKs + IfK + IsiK + ISK - p(2.0) * INaK) / (p(1.0) * (constants[170] + constants[164]) * constants[13]) * p(0.001)
            else:
                diff_Nai = p(0.0)
                diff_Ki = p(0.0)
            Iion = Itot * p(1000.0) / constants[4]

            out[11] = (dT_inf - state[11]) / tau_dT
            out[33] = (piy_inf - state[33]) / tau_piy
            out[29] = (q_inf - state[29]) / tau_q
            out[30] = (r_inf - state[30]) / tau_r
            out[27] = (r_Kur_inf - state[27]) / tau_r_Kur
            out[28] = (s_Kur_inf - state[28]) / tau_s_Kur
            out[35] = constants[165] * (1.0 - state[35]) - beta_a * state[35]
            out[9] = (fL_inf - state[9]) / tau_fL
            out[12] = (fT_inf - state[12]) / tau_fT
            out[7] = (h_inf - state[7]) / tau_h
            out[5] = (h_WT_inf - state[5]) / tau_h_WT
            out[32] = (paF_inf - state[32]) / tau_paF
            out[31] = (paS_inf - state[31]) / tau_paS
            out[26] = diff_Ca_jsr
            out[38] = diff_PLBp
            out[37] = diff_cAMP
            out[20] = diff_fCMi
            out[21] = diff_fCMs
            out[22] = diff_fCQ
            out[17] = diff_fTC
            out[18] = diff_fTMC
            out[19] = diff_fTMM
            out[4] = (m_WT_inf - state[4]) / tau_m_WT
            out[36] = diff_x
            out[3] = (y_inf - state[3]) / tau_y
            out[15] = diff_I
            out[14] = diff_O
            out[16] = diff_RI
            out[13] = diff_R_1
            out[10] = diff_fCa
            out[6] = (m_inf - state[6]) / tau_m
            out[25] = diff_Ca_nsr
            out[23] = diff_Cai
            out[2] = diff_Ki
            out[8] = (dL_inf - state[8]) / tau_dL
            out[34] = (n_inf - state[34]) / tau_n
            out[24] = diff_Ca_sub
            out[1] = diff_Nai
            out[0] = -Iion

        self.dxdtfunc = dxdtfunc


    def update_constants(self, updates_dict=None, **kwargs):
        if updates_dict is None:
            updates_dict = {}

        combined_updates = {**updates_dict, **kwargs}

        # Note: If the same value occurs in the dict and
        # keyword args, the kwargs one will win.
        for key, item in combined_updates.items():
            self.constants_dict.set_constant(key, self.precision(item))

        self.constants_array = np.asarray([constant for (label, constant) in self.constants_dict.items()], dtype=self.precision)


# ******************************* TEST CODE ******************************** #
if __name__ == '__main__':
    precision = np.float64
    sys = fabbri_linder_cell(precision = precision)
    numba_precision = sys.numba_precision

    dxdt = sys.dxdtfunc
    NUM_STATES= 39
    NUM_CONSTANTS = 173
    constants = cuda.to_device(sys.constants_array)
    inits = cuda.to_device([initial_values[state_labels[i]] for i in range(NUM_STATES)])

    @cuda.jit((sys.numba_precision[:],
               sys.numba_precision[:],
               sys.numba_precision[:]
               ),
              opt=True,
              debug=False)
    def testkernel(out, constants, inits):

        l_state = cuda.local.array(shape=NUM_STATES, dtype=numba_precision)
        l_constants = cuda.const.array_like(constants)
        for i in range(len(inits)):
            l_state[i] = inits[i]

        t = precision(1.0)
        dxdt(out,
            l_state,
            l_constants,
            t)

    out = cuda.pinned_array_like(np.zeros(sys.num_states))
    d_out = cuda.to_device(out)
    print("Testing to see if your dxdt function compiles using CUDA...")
    testkernel[1,1](d_out, constants, inits)
    cuda.synchronize()
    out = d_out.copy_to_host()
    print(out)