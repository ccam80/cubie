# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:42:35 2024

@author: cca79
"""
import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"

from numba import cuda, float64, int64, int32, float32, from_dtype
from numpy import asarray
# from _utils import clamp_32, clamp_64
import numpy as np
from math import cos
from numba import from_dtype


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
    "Iso_1_uM": 0.0,
    "Iso_linear": 0.0,
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
    "R231C_on": 0.0,
    "R_2": 8314.472,
    "R_cell": 3.9,
    "RyR_max": 0.02,
    "RyR_min": 0.0127,
    "T": 310.0,
    "TC_tot": 0.031,
    "TMC_tot": 0.062,
    "VW_IKs": 0.0,
    "V_R231C": 113.8797,
    "V_WT": 102.4597,
    "V_i_part": 0.46,
    "V_jsr_part": 0.0012,
    "V_nsr_part": 0.0116,
    "delta_m": 1e-05,
    "delta_m_WT": 1e-05,
    "dynamic_Ki_Nai": 0.0,
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
    "Iso_cas": -1.0,
    "K_05if": 17.8741 / 600.0,
    "V_dL": -16.4508532699999996,
    "V_fL": -37.4,
    "cAMPb": 20.0 / 600.0,
    "kPKA": 9000.0 / 600.0,
    "kPKA_cAMP": 284.5 / 600.0,
    "k_R231C": -34.9006,
    "Iso_increase_1": 1.0, #remove if only for iso_1_um or iso_linear
    "Iso_increase_2": 1.0, #remove if only for iso_1_um or iso_linear
    "Iso_shift_1": 0.0,    #remove if only for iso_1_um or iso_linear
    "Iso_shift_2": 0.0,    #remove if only for iso_1_um or iso_linear
    "Iso_shift_dL": 0.0,   #remove if only for iso_1_um or iso_linear
    "Iso_shift_ninf": 0.0, #remove if only for iso_1_um or iso_linear
    "Iso_slope_dL": 0.0,   #remove if only for iso_1_um or iso_linear
    }

def update_calculated_constants(constants):
    # Local variable declarations
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
    kiso = constants["K_iso"] + 0.1181 * (constants["Iso_cas"]**constants["niso"] / (constants["K_05iso"]**constants["niso"] + constants["Iso_cas"]**constants["niso"] + 1e-15))  # 1e-15 constant added to avoid nan
    P_up = constants["P_up_basal"] * (1.0 - Ca_intracellular_fluxes_b_up)
    V_i = constants["V_i_part"] * V_cell - V_sub
    V_jsr = constants["V_jsr_part"] * V_cell
    V_nsr = constants["V_nsr_part"] * V_cell

    # Update constants dictionary
    constants.update({
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


class diffeq_system:
    """ This class should contain all system definitions. The constants management
    scheme can be a little tricky, because the GPU stuff can't handle dictionaries.
    The constants_array will be passed to your dxdt function - you can use the indices
    given in self.constant_indices to map them out while you set up your dxdt function.

    > test_system = diffeq_system()
    > print(diffeq_system.constant_indices)

    - Place all of your system constants and their labelsin the constants_dict.
    - Update self.num_state to match the number of state variables you
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
                 num_state =3,
                 num_algebraics=5,
                 precision=np.float64,
                 state_labels = state_labels,
                 **kwargs):
        """Set system constant values then function as a factory function to
        build CUDA device functions for use in the ODE solver kernel. No
        arguments, no returns it's all just bad coding practice in here.

        Everything except for the constants_array and constant_indices generators
        and dxdt assignment at the end is an example, you will need to overwrite"""

        self.num_state = num_state
        self.precision = precision
        self.numba_precision = from_dtype(precision)
        # self.noise_sigmas = np.zeros(self.num_state, dtype=precision)

        self.constants_dict  = system_constants(kwargs)
        self.constants_array = asarray([constant for (label, constant) in self.constants_dict.items()], dtype=precision)
        self.constant_indices = {label: index for index, (label, constant) in enumerate(self.constants_dict.items())}

        self.state_labels = state_labels



        @cuda.jit((self.numba_precision[:],
                   self.numba_precision[:],
                   self.numba_precision[:],
                   self.numba_precision[:],
                   self.numba_precision),
                  device=True,
                  inline=True,)
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
            import math

            E0_m_WT = state[0] + 41.0
            F_PLBp = (3.3931 * (PLBp**4.0695) / ((0.2805**4.0695) + (PLBp**4.0695))
                      if PLBp > 0.23
                      else 1.698 * (PLBp**13.5842) / ((0.224**13.5842) + (PLBp**13.5842))) #selp candidate - level set equation
            INa_h_WT_gate_alpha_h_WT = 20.0 * math.exp(-0.125 * (state[0] + 75.0))
            INa_h_WT_gate_beta_h_WT = 2000.0 / (320.0 * math.exp(-0.1 * (state[0] + 75.0)) + 1.0)
            INa_h_gate_alpha_h = 20.0 * math.exp(-0.125 * (state[0] + 75.0))
            INa_h_gate_beta_h = 2000.0 / (320.0 * math.exp(-0.1 * (state[0] + 75.0)) + 1.0)
            INa_m_WT_gate_beta_m_WT = 8000.0 * math.exp(-0.056 * (state[0] + 66.0))
            ISK_x_gate_tau_x = 1.0 / (0.047 * state[24] * 1000.0 + 1.0 / 76.0)
            PKA = (-0.9483029 * math.exp(-state[37] * 600.0 * 0.06561479) + 0.97781646
                   if cAMP * 600.0 < 25.87
                   else -0.45260509 * math.exp(-state[37] * 600.0 * 0.03395094) + 0.99221714) #selp candidate - level set equation
            beta_a = 10.0 * math.exp(0.0133 * (state[0] + 40.0)) / 1000.0
            dT_inf = 1.0 / (1.0 + math.exp(-(state[0] + 38.3) / 5.5))
            fT_inf = 1.0 / (1.0 + math.exp((state[0] + 58.7) / 3.8))
            h_WT_inf = 1.0 / (1.0 + math.exp((state[0] + 69.804) / 4.4565))
            k2 = 1.1 * 237.9851 * ((state[37] * 600.0)**5.101 /
                 ((20.1077**6.101) + (state[37] * 600.0)**6.101))
            m_WT_inf = 1.0 / (1.0 + math.exp(-(state[0] + 42.0504) / 8.3106))
            pa_infinity = 1.0 / (1.0 + math.exp(-(state[0] + 10.0144) / 7.6607))
            piy_inf = 1.0 / (1.0 + math.exp((state[0] + 28.6) / 17.1))
            q_inf = 1.0 / (1.0 + math.exp((state[0] + 49.0) / 13.0))
            r_Kur_inf = 1.0 / (1.0 + math.exp((state[0] + 6.0) / -8.6))
            r_inf = 1.0 / (1.0 + math.exp(-(state[0] - 19.3) / 15.0))
            s_Kur_inf = 1.0 / (1.0 + math.exp((state[0] + 7.5) / 10.0))
            tau_dT = 0.001 / (1.068 * math.exp((state[0] + 38.3) / 30.0) +
                              1.068 * math.exp(-(state[0] + 38.3) / 30.0)) * 1000.0
            tau_fL = 0.001 * (44.3 + 230.0 * math.exp(-((state[0] + 36.0) / 10.0)**2)) * 1000.0
            tau_paF = 1.0 / (30.0 * math.exp(state[0] / 10.0) + math.exp(-state[0] / 12.0)) * 1000.0
            tau_paS = 0.84655354 / (4.2 * math.exp(state[0] / 17.0) +
                                    0.15 * math.exp(-state[0] / 21.6)) * 1000.0
            tau_piy = (1.0 / (100.0 * math.exp(-state[0] / 54.645) +
                       656.0 * math.exp(state[0] / 106.157))) * 1000.0
            tau_q = (0.001 * 0.6 * (65.17 / (0.57 * math.exp(-0.08 * (state[0] + 44.0)) +
                                             0.065 * math.exp(0.1 * (state[0] + 45.93))) + 10.1)
                     * 1000.0)
            tau_r = (0.001 * 0.66 * 1.4 * (15.59 / (1.037 * math.exp(0.09 * (state[0] + 30.61)) +
                                                  0.369 * math.exp(-0.12 * (state[0] + 23.84))) + 2.98)
                     * 1000.0)
            tau_r_Kur = (0.009 / (1.0 + math.exp((state[0] + 5.0) / 12.0)) + 0.0005) * 1000.0
            tau_s_Kur = (0.59 / (1.0 + math.exp((state[0] + 60.0) / 10.0)) + 3.05) * 1000.0
            ATPi = (constants[3] * (constants[104] * ((state[37] * 100.0) / constants[151])**constants[131] /
                   (constants[105] + ((state[37] * 100.0) / constants[151])**constants[131])
                   - constants[37]) / 100.0)
            E0_m = state[0] + 41.0 - constants[130]
            INa_m_WT_gate_alpha_m_WT = (2000.0 if abs(E0_m_WT) < delta_m_WT
                                        else 200.0 * E0_m_WT / -(math.exp(-0.1 * E0_m_WT) - 1.0)) #selp candidate
            INa_m_gate_alpha_m = (2000.0 if abs(E0_m) < delta_m
                                  else 200.0 * E0_m / -(math.exp(-0.1 * E0_m) - 1.0)) #selp candidate
            INa_m_gate_beta_m = 8000.0 * math.exp(-0.056 * (state[0] + 66.0 - constants[130]))
            Iso_inc_cas_ICaL = (-0.2152 + constants[47] * (PKA**10.0808) /
                                ((constants[38]**10.0808) + (PKA**10.0808)))

            Iso_inc_cas_IKs = (
                (-0.2152 +
                 (0.435692 if VW_IKs == 0.0 else 0.494259) *
                 (PKA**10.0808) / ((0.719701 if VW_IKs == 0.0 else 0.736717)**10.0808 + (PKA**10.0808))
                )            )
            Iso_inc_cas_INaK = (
                -0.2152 + constants[49] * (PKA**10.0808) /
                ((constants[40]**10.0808) + (PKA**10.0808))
            )
            Iso_shift_cas_ICaL_dL_gate = (
                -(constants[48] * (PKA**9.281) /
                  ((constants[39]**9.281) + (PKA**9.281)) - 18.76)
            )
            Iso_shift_cas_IKs_n_gate = (
                (-(24.4 * (1.411375 if VW_IKs == 0.0 else 1.438835) *
                   (PKA**9.281) / ((0.704217 if VW_IKs == 0.0 else 0.707399)**9.281 + (PKA**9.281))
                   - 18.76))
            )
            Iso_shift_cas_If_y_gate = (
                constants[63] * ((state[37])**constants[138] /
                ((constants[148]**constants[138]) + (state[37]**constants[138]))) - 18.76
            )
            R231C = (g_ratio * (1.0 / (1.0 + math.exp(-(state[0] - constants[92]) / constants[154])))**2 /
                     ((1.0 - state[0] * 0.0) / (1.0 + math.exp(-(state[0] - constants[93]) /
                      constants[110])) + state[0] * 0.0)**2
                     if R231C_on != 0.0 else 1.0)
            delta_fCMi = (constants[118] * state[23] * (1.0 - state[20]) -
                          constants[113] * state[20])
            delta_fCMs = (constants[118] * state[24] * (1.0 - state[21]) -
                          constants[113] * state[21])
            delta_fCQ = (constants[119] * state[26] * (1.0 - state[22]) -
                         constants[114] * state[22])
            delta_fTC = (constants[120] * state[23] * (1.0 - state[17]) -
                         constants[115] * state[17])
            delta_fTMC = (constants[121] * state[23] * (1.0 - (state[18] + state[19])) -
                          constants[116] * state[18])
            delta_fTMM = (constants[122] * constants[71] * (1.0 - (state[18] + state[19])) -
                          constants[117] * state[19])
            fCa_infinity = constants[66] / (constants[66] + state[24])
            fL_inf = 1.0 / (1.0 + math.exp((state[0] - constants[150] + constants[142]) / constants[112]))
            h_inf = 1.0 / (1.0 + math.exp((state[0] + 69.804 - constants[101]) /
                                         (4.4565 * constants[102])))
            j_Ca_dif = (state[24] - state[23]) / constants[8]
            j_SRCarel = constants[56] * constants[127] * state[14] * (state[26] - state[24])
            j_tr = (state[25] - state[26]) / constants[9]
            k1 = (constants[58] * (constants[45] + constants[43] / (1.0 + math.exp(
                  (constants[46] - constants[113]*(state[20]/(constants[118]*(1.0 - state[20])))) /
                  constants[44]))))
            k3 = (constants[152] * (state[37]**(constants[132] - 1.0) /
                  ((constants[153]**constants[132]) + (state[37]**constants[132]))))
            k4 = (constants[107] * (PKA**constants[133]) /
                  ((constants[106]**constants[133]) + (PKA**constants[133])))
            k43 = state[1] / (constants[35] + state[1])
            k5 = (constants[108] * (constants[76] * state[38] /
                  (constants[109] + state[38])))
            kCaSR = (constants[70] - (constants[70] - constants[72]) /
                     (1.0 + (constants[12] / state[26])**constants[24]))
            koCa = ((constants[125] * (constants[87] - constants[86] * (PKA**constants[134]) /
                     ((constants[41]**constants[134]) + (PKA**constants[134])) + 1.0))
                   )
            m_inf = (1.0 / (1.0 + math.exp(-(state[0] + 42.0504 - constants[128]) /
                                           (8.3106 * constants[129]))))
            paF_inf = pa_infinity
            paS_inf = pa_infinity
            tau_fT = ((1.0 / (16.67 * math.exp(-(state[0] + 75.0) / 83.3) +
                       16.67 * math.exp((state[0] + 75.0) / 15.38)) + constants[140]) * 1000.0)
            tau_h = (constants[103] / (INa_h_gate_alpha_h + INa_h_gate_beta_h) * 1000.0)
            tau_h_WT = 1.0 / (INa_h_WT_gate_alpha_h_WT + INa_h_WT_gate_beta_h_WT) * 1000.0
            x_infinity = (0.81 * (state[24] * 1000.0)**constants[135] /
                          ((state[24] * 1000.0)**constants[135] + constants[11]**constants[135]))
            E_K = constants[167] * math.log(constants[67] / state[2])
            E_Ks = (constants[167] * math.log((constants[67] + 0.12 * constants[73]) /
                     (state[2] + 0.12 * state[1])) if VW_IKs == 0.0 else
                    constants[167] * math.log((constants[67] + 0.0018 * constants[73]) /
                     (state[2] + 0.0018 * state[1])))
            E_Na = constants[167] * math.log(constants[73] / state[1])
            E_mh = constants[167] * math.log((constants[73] + 0.12 * constants[67]) /
                                             (state[1] + 0.12 * state[2]))
            GKs = (
                constants[16] * (1.0 + Iso_inc_cas_IKs)
                if VW_IKs == 0.0 else
                0.2 * constants[16] * (1.0 + Iso_inc_cas_IKs)
            )
            ICaL_fCa_gate_tau_fCa = 0.001 * fCa_infinity / constants[25]
            ICaT = (2.0 * constants[78] * state[0] / (constants[167] * -
                    (math.exp(-1.0 * state[0] * 2.0 / constants[167]) - 1.0)) *
                    (state[24] - constants[10] * math.exp(-2.0 * state[0] / constants[167])) *
                    state[11] * state[12])
            IKs_n_gate_beta_n = ((1.0 * math.exp(-(state[0] - constants[157] -
                                   Iso_shift_cas_IKs_n_gate - 5.0) / 25.0))
                                 if VW_IKs == 0.0 else
                                 (1.93 * math.exp(-state[0] / 83.2)))
            INa_m_gate_alpha_m = INa_m_gate_alpha_m  # already assigned above
            IsiCa = (2.0 * constants[77] * constants[155] * (1.0 - constants[162]) *
                     (1.0 + Iso_inc_cas_ICaL) * (state[0] - 0.0) /
                     (constants[167] * -(math.exp(-1.0 * (state[0] - 0.0) * 2.0 / constants[167]) - 1.0)) *
                     (state[24] - constants[10] * math.exp(-2.0 * (state[0] - 0.0) / constants[167])) *
                     state[8] * state[9] * state[10])
            IsiK = (0.000365 * constants[77] * constants[155] * (1.0 - constants[162]) *
                    (1.0 + Iso_inc_cas_ICaL) * (state[0] - 0.0) /
                    (constants[167] * -(math.exp(-1.0 * (state[0] - 0.0) / constants[167]) - 1.0)) *
                    (state[2] - constants[67] * math.exp(-1.0 * (state[0] - 0.0) / constants[167])) *
                    state[8] * state[9] * state[10])
            IsiNa = (1.85e-05 * constants[77] * constants[155] * (1.0 - constants[162]) *
                     (1.0 + Iso_inc_cas_ICaL) * (state[0] - 0.0) /
                     (constants[167] * -(math.exp(-1.0 * (state[0] - 0.0) / constants[167]) - 1.0)) *
                     (state[1] - constants[73] * math.exp(-1.0 * (state[0] - 0.0) / constants[167])) *
                     state[8] * state[9] * state[10])
            adVm = ( -41.80001 - constants[159] - Iso_shift_cas_ICaL_dL_gate
                     if state[0] == -41.8 - constants[159] - Iso_shift_cas_ICaL_dL_gate
                     else (0.0 if state[0] == 0.0 else
                           (-6.80001 - constants[159] - Iso_shift_cas_ICaL_dL_gate
                            if state[0] == -6.8 - constants[159] - Iso_shift_cas_ICaL_dL_gate
                            else state[0])))
            bdVm = (-1.80001 - constants[159] - Iso_shift_cas_ICaL_dL_gate
                    if state[0] == -1.8 - constants[159] - Iso_shift_cas_ICaL_dL_gate
                    else state[0])
            dL_inf = 1.0 / (1.0 + math.exp(-(state[0] - constants[149] - constants[159] -
                                             Iso_shift_cas_ICaL_dL_gate) /
                                           (constants[111] * (1.0 + constants[30]) *
                                            (1.0 + constants[161] / 100.0))))
            di = (1.0 + state[24] / constants[60] * (1.0 + math.exp(-constants[80]*state[0] /
                 constants[167]) + state[1] / constants[61]) + state[1] / constants[31] *
                 (1.0 + state[1] / constants[33] * (1.0 + state[1] / constants[35])))
            diff_Ca_jsr = (j_tr - (j_SRCarel + constants[7] * delta_fCQ)) * 0.001
            diff_PLBp = (k4 - k5) / 60000.0
            diff_cAMP = ((constants[173] - constants[172]) * ATPi + k1 * ATPi -
                         k2 * state[37] - k3 * state[37]) / 60000.0
            diff_fCMi = delta_fCMi * 0.001
            diff_fCMs = delta_fCMs * 0.001
            diff_fCQ = delta_fCQ * 0.001
            diff_fTC = delta_fTC * 0.001
            diff_fTMC = delta_fTMC * 0.001
            diff_fTMM = delta_fTMM * 0.001
            diff_x = (x_infinity - state[36]) / ISK_x_gate_tau_x
            do_ = (1.0 + constants[10] / constants[62] * (1.0 + math.exp(constants[81]*state[0] /
                  constants[167])) + constants[73] / constants[32] * (1.0 + constants[73] /
                  constants[34] * (1.0 + constants[73] / constants[36])))
            k32 = math.exp(constants[82] * state[0] / (2.0 * constants[167]))
            k41 = math.exp(-constants[82] * state[0] / (2.0 * constants[167]))
            kiSRCa = constants[123] * kCaSR
            koSRCa = koCa / kCaSR
            n_inf = ((1.0 / (1.0 + math.exp(-(state[0] + 0.6383 - constants[157] -
                                             Iso_shift_cas_IKs_n_gate - constants[136]) /
                                            (10.7071 * constants[137]))))
                     if VW_IKs == 0.0 else
                     (1.0 / (1.0 + math.exp(-(state[0] + 15.733 + constants[160] -
                                              Iso_shift_cas_IKs_n_gate - constants[136]) /
                                            (constants[137]*27.77)))))
            tau_m_WT = 1.0 / (INa_m_WT_gate_alpha_m_WT + INa_m_WT_gate_beta_m_WT) * 1000.0
            tau_y = (1.0 / (0.36 * (state[0] + 148.8 - constants[163] -
                     constants[158] - Iso_shift_cas_If_y_gate - constants[144]) /
                     (math.exp(0.066 * (state[0] + 148.8 - constants[163] -
                     constants[158] - Iso_shift_cas_If_y_gate - constants[144])) - 1.0)
                     + 0.1 * (state[0] + 87.3 - constants[163] -
                     constants[158] - Iso_shift_cas_If_y_gate - constants[145]) /
                     -(math.exp(-0.2*(state[0] + 87.3 - constants[163] -
                     constants[158] - Iso_shift_cas_If_y_gate - constants[145])) - 1.0))
                     - 0.054) * 1000.0
            y_inf = ((0.01329 + 0.99921 /
                      (1.0 + math.exp((state[0] + 97.134 - constants[163] -
                                       constants[158] - Iso_shift_cas_If_y_gate -
                                       constants[146]) / 8.1752)))
                     if state[0] < -(80.0 - constants[163] - constants[158] -
                                     Iso_shift_cas_If_y_gate - constants[146])
                     else 0.0002501 * math.exp(-(state[0] - constants[163] -
                            constants[158] - Iso_shift_cas_If_y_gate - constants[146]) / 12.861))
            ICaL = IsiCa + IsiK + IsiNa
            ICaL_dL_gate_alpha_dL = (-0.02839 * (adVm + 41.8 - constants[159] -
                                      Iso_shift_cas_ICaL_dL_gate) /
                                     (math.exp(-(adVm + 41.8 - constants[159] -
                                      Iso_shift_cas_ICaL_dL_gate)/2.5) - 1.0)
                                     - 0.0849*(adVm + 6.8 - constants[159] -
                                     Iso_shift_cas_ICaL_dL_gate) /
                                     (math.exp(-(adVm + 6.8 - constants[159] -
                                      Iso_shift_cas_ICaL_dL_gate)/4.8) - 1.0))
            ICaL_dL_gate_beta_dL = (0.01143*(bdVm + 1.8 - constants[159] -
                                             Iso_shift_cas_ICaL_dL_gate) /
                                    (math.exp((bdVm + 1.8 - constants[159] -
                                       Iso_shift_cas_ICaL_dL_gate)/2.5)-1.0))
            IKACh = ((constants[2] * constants[14] * (state[0] - E_K) *
                      (1.0 + math.exp((state[0] + 20.0)/20.0)) * state[35])
                     if ACh > 0.0 else 0.0)
            IKr = constants[165] * (state[0] - E_K) * (0.9*state[32] + 0.1*state[31]) * state[33]
            IKs = R231C * GKs * (state[0] - E_Ks) * (state[34]**2)
            IKs_n_gate_alpha_n = ((28.0 / (1.0 + math.exp(-(state[0] - 40.0 - constants[157] -
                                       Iso_shift_cas_IKs_n_gate)/3.0))) if VW_IKs == 0.0
                                  else (n_inf/(1.0 - n_inf)*IKs_n_gate_beta_n))
            IKur = constants[17] * state[27] * state[28] * (state[0] - E_K)
            INaK = ((1.0 + Iso_inc_cas_INaK) * constants[156] * constants[26] *
                    (1.0 + (constants[64]/constants[67])**1.2)**-1.0 *
                    (1.0 + (constants[65]/state[1])**1.3)**-1.0 *
                    (1.0 + math.exp(-(state[0] - E_Na + 110.0)/20.0))**-1.0)
            INa_ = constants[19]*(state[6]**3)*state[7]*(state[0]-E_mh)
            INa_L = constants[166]*(state[6]**3)*(state[0]-E_mh)
            INa_WT = constants[18]*(state[4]**3)*state[5]*(state[0]-E_mh)
            ISK = constants[20]*(state[0]-E_K)*state[36]
            IfK = state[33]*constants[21]*(state[0]-E_K)  # Actually, "y" is state[3] or [33]? The code uses "y" for state[3]? The original uses y for [3]?
            # But line 297 says IfK = y * Gf_K_max * (V-E_K). Actually "y" is state[3] in labels.
            # We'll correct that to state[3]:
            IfK = state[3]*constants[21]*(state[0]-E_K)
            IfNa = state[3]*constants[22]*(state[0]-E_Na)
            Ito = constants[23]*(state[0]-E_K)*state[29]*state[30]
            diff_I = ((kiSRCa*state[24]*state[14] - constants[124]*state[15])
                      - (constants[126]*state[15] - koSRCa*(state[24]**2)*state[16]))*0.001
            diff_O = ((koSRCa*(state[24]**2)*state[13] - constants[126]*state[14])
                      - (kiSRCa*state[24]*state[14] - constants[124]*state[15]))*0.001
            diff_RI = ((constants[126]*state[15] - koSRCa*(state[24]**2)*state[16])
                       - (constants[124]*state[16] - kiSRCa*state[24]*state[13]))*0.001
            diff_R_1 = ((constants[124]*state[16] - kiSRCa*state[24]*state[13])
                        - (koSRCa*(state[24]**2)*state[13] - constants[126]*state[14]))*0.001
            diff_fCa = (fCa_infinity - state[10]) / ICaL_fCa_gate_tau_fCa * 0.001
            j_up = ((constants[57]*(0.9*constants[174]*F_PLBp /
                     (1.0 + math.exp((-state[23] + constants[59])/constants[143])))))
            k12 = (state[24]/constants[60]*math.exp(-constants[80]*state[0]/constants[167])/di)
            k14 = (state[1]/constants[31]*state[1]/constants[33]*(1.0+state[1]/constants[35])*
                   math.exp(constants[82]*state[0]/(2.0*constants[167]))/di)
            k21 = (constants[10]/constants[62]*math.exp(constants[81]*state[0]/constants[167]) /
                   do_)
            k23 = (constants[73]/constants[32]*constants[73]/constants[34]*
                   (1.0 + constants[73]/constants[36])*
                   math.exp(-constants[82]*state[0]/(2.0*constants[167]))/do_)
            tau_m = 1.0 / (INa_m_gate_alpha_m + INa_m_gate_beta_m) * 1000.0
            INa = ((1.0 - constants[27])*(INa_ + INa_L) + constants[27]*INa_WT)
            If_ = IfNa + IfK
            diff_Ca_nsr = (j_up - j_tr*constants[176]/constants[177]) * 0.001
            diff_Cai = ((1.0*(j_Ca_dif*constants[169] - j_up*constants[177]) /
                         constants[175]) - (constants[6]*delta_fCMi +
                         constants[89]*delta_fTC + constants[90]*delta_fTMC))*0.001
            diff_Ki = ((-1.0*(IKur + Ito + IKr + IKs + IfK + IsiK + ISK
                       - 2.0*INaK)/(1.0*(constants[175]+constants[169])*constants[13]))
                       * 0.001 if dynamic_Ki_Nai == 1.0 else 0.0)
            tau_dL = 0.001 / (ICaL_dL_gate_alpha_dL + ICaL_dL_gate_beta_dL) * 1000.0
            tau_n = 1.0 / (IKs_n_gate_alpha_n + IKs_n_gate_beta_n) * 1000.0
            x1 = (k41 * constants[171] * (k23 + k21) + k21 * k32 * (k43 + k41))
            x2 = (k32 * k43 * (k14 + k12) + k41 * k12 * (constants[171] + k32))
            x3 = (k14 * k43 * (k23 + k21) + k12 * k23 * (k43 + k41))
            x4 = (k23 * constants[171] * (k14 + k12) + k14 * k21 * (constants[171] + k32))
            INaCa = constants[50]*(x2*k21 - x1*k12)/(x1 + x2 + x3 + x4)
            Itot = If_ + IKr + IKs + Ito + INaK + INaCa + INa + ICaL + ICaT + IKACh + IKur + ISK
            diff_Ca_sub = ((j_SRCarel*constants[176]/constants[169]) -
                           ((IsiCa + ICaT - 2.0*INaCa)/(2.0*constants[13]*constants[169])
                            + j_Ca_dif + constants[6]*delta_fCMs))*0.001
            diff_Nai = ((-1.0*(INa + IfNa + IsiNa + 3.0*INaK + 3.0*INaCa) /
                         (1.0*(constants[175]+constants[169])*constants[13])) * 0.001
                        if dynamic_Ki_Nai == 1.0 else 0.0)
            Iion = Itot * 1000.0 / constants[4]

            out[4] = (m_WT_inf - state[4]) / tau_m_WT
            out[38] = diff_PLBp
            out[37] = diff_cAMP
            out[20] = diff_fCMi
            out[21] = diff_fCMs
            out[22] = diff_fCQ
            out[17] = diff_fTC
            out[18] = diff_fTMC
            out[19] = diff_fTMM
            out[36] = diff_x
            out[23] = diff_Cai
            out[25] = diff_Ca_nsr
            out[26] = diff_Ca_jsr
            out[24] = diff_Ca_sub
            out[1] = diff_Nai
            out[2] = diff_Ki
            out[3] = (y_inf - state[3]) / tau_y
            out[8] = (dL_inf - state[8]) / tau_dL
            out[33] = (piy_inf - state[33]) / tau_piy
            out[29] = (q_inf - state[29]) / tau_q
            out[30] = (r_inf - state[30]) / tau_r
            out[27] = (r_Kur_inf - state[27]) / tau_r_Kur
            out[28] = (s_Kur_inf - state[28]) / tau_s_Kur
            out[35] = (constants[170]*(1.0 - state[35]) - beta_a*state[35])
            out[9] = (fL_inf - state[9]) / tau_fL
            out[12] = (fT_inf - state[12]) / tau_fT
            out[7] = (h_inf - state[7]) / tau_h
            out[5] = (h_WT_inf - state[5]) / tau_h_WT
            out[32] = (paF_inf - state[32]) / tau_paF
            out[31] = (paS_inf - state[31]) / tau_paS
            out[11] = (dT_inf - state[11]) / tau_dT  # original code had Real D_dT at state[11]
            out[10] = diff_fCa
            out[14] = diff_O
            out[15] = diff_I
            out[16] = diff_RI
            out[13] = diff_R_1
            out[6] = (m_inf - state[6]) / tau_m
            out[34] = (n_inf - state[34]) / tau_n
            out[8] = (dT_inf - state[8]) / tau_dT  # repeated index 8, as in original code
            out[37] = diff_cAMP  # repeated index 37, already assigned
            out[0] = -Iion - I_diff


        self.dxdtfunc = dxdtfunc


    def update_constants(self, updates_dict=None, **kwargs):
        if updates_dict is None:
            updates_dict = {}

        combined_updates = {**updates_dict, **kwargs}

        # Note: If the same value occurs in the dict and
        # keyword args, the kwargs one will win.
        for key, item in combined_updates.items():
            self.constants_dict.set_constant(key, self.precision(item))

        self.constants_array = asarray([constant for (label, constant) in self.constants_dict.items()], dtype=self.precision)


#******************************* TEST CODE ******************************** #
# if __name__ == '__main__':
    # sys = diffeq_system()
    # dxdt = sys.dxdtfunc

    # @cuda.jit()
    # def testkernel(out):
    #     # precision = np.float32
    #     # numba_precision = float32
    #     l_dxdt = cuda.local.array(shape=NUM_state, dtype=numba_precision)
    #     l_state = cuda.local.array(shape=NUM_state, dtype=numba_precision)
    #     l_constants = cuda.local.array(shape=NUM_CONSTANTS, dtype=numba_precision)
    #     l_state[:] = precision(1.0)
    #     l_constants[:] = precision(1.0)

    #     t = precision(1.0)
    #     dxdt(l_dxdt,
    #         l_state,
    #         l_constants,
    #         t)

    #     out = l_dxdt


    #     NUM_state = 5
    #     NUM_CONSTANTS = 14
    #     outtest = np.zeros(NUM_state, dtype=np.float4)
    #     out = cuda.to_device(outtest)
    #     print("Testing to see if your dxdt function compiles using CUDA...")
    #     testkernel[1,1](out)
    #     cuda.synchronize()
    #     out.copy_to_host(outtest)
    #     print(outtest)