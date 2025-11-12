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
VW_IKs = 0
Iso_1_uM_on = 0
Iso_linear_on = 0
Iso_cas_on = 1
dynamic_Ki_Nai = 0
R231C_on = 0
ACh_on = 1.0


state_labels = {
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
    "cAMP": 19.73/600,
    "PLBp": 0.23
}

# default_constants = {
#     "A2": 0.07,
#     "ACh": 0.0,
#     "ACh_on": 0.0,
#     "ATPi_max": 2.533,
#     "C": 57.0,
#     "CCh_cas": 0.0,
#     "CM_tot": 0.045,
#     "CQ_tot": 10.0,
#     "Ca_intracellular_fluxes_tau_dif_Ca": 5.469e-05,
#     "Ca_intracellular_fluxes_tau_tr": 0.04,
#     "Cao": 1.8,
#     "EC50_SK": 0.7,
#     "EC50_SR": 0.45,
#     "F": 96485.3415,
#     "GKACh": 0.00345,
#     "GKr_max": 0.00424,
#     "GKs_max": 0.00065,
#     "GKur_max": 0.0001539,
#     "GNa_WT": 0.0223,
#     "GNa_max": 0.0223,
#     "GSK": 0.0,
#     "Gf_K_max": 0.00268,
#     "Gf_Na_max": 0.00159,
#     "Gto_max": 0.0035,
#     "HSR": 2.5,
#     "ICaL_fCa_gate_alpha_fCa": 0.0075,
#     "INaK_max": 0.08105,
#     "INa_WT_ratio": 0.5,
#     "Iso_slope_cas_ICaL": 0.0,
#     "K1ni": 395.3,
#     "K1no": 1628.0,
#     "K2ni": 2.289,
#     "K2no": 561.4,
#     "K3ni": 26.44,
#     "K3no": 4.663,
#     "KATP_min": 6034.0,
#     "K_05ICaL1": 0.730287,
#     "K_05ICaL2": 0.66145,
#     "K_05INaK": 0.719701,
#     "K_05RyR": 0.682891,
#     "K_05iso": 58.57114132,
#     "K_AC": 0.0735,
#     "K_ACCa": 2.4e-05,
#     "K_ACI": 0.016,
#     "K_Ca": 0.000563995,
#     "K_ICaL1": 0.470657,
#     "K_ICaL2": 27.526226,
#     "K_INaK": 0.435692,
#     "K_NaCa": 3.343,
#     "K_iso": 0.007,
#     "K_iso_increase": 1.0,
#     "K_iso_shift": 1.0,
#     "K_iso_shift_ninf": 1.0,
#     "K_iso_slope_dL": 1.0,
#     "K_j_SRCarel": 1.0,
#     "K_j_up": 1.0,
#     "K_k1": 4.64625,
#     "K_up": 0.000286113,
#     "Kci": 0.0207,
#     "Kcni": 26.44,
#     "Kco": 3.663,
#     "Kif": 26.26,
#     "Km_Kp": 1.4,
#     "Km_Nap": 14.0,
#     "Km_fCa": 0.000338,
#     "Ko": 5.4,
#     "L_cell": 67.0,
#     "L_sub": 0.02,
#     "MaxSR": 15.0,
#     "Mgi": 2.5,
#     "MinSR": 1.0,
#     "Nao": 140.0,
#     "PKAtot": 1.0,
#     "PKItot": 0.3,
#     "PP1": 0.00089,
#     "P_CaL": 0.4578,
#     "P_CaT": 0.04132,
#     "P_up_basal": 5.0,
#     "Qci": 0.1369,
#     "Qco": 0.0,
#     "Qn": 0.4315,
#     "R_2": 8314.472,
#     "R_cell": 3.9,
#     "RyR_max": 0.02,
#     "RyR_min": 0.0127,
#     "T": 310.0,
#     "TC_tot": 0.031,
#     "TMC_tot": 0.062,
#     "V_R231C": 113.8797,
#     "V_WT": 102.4597,
#     "V_i_part": 0.46,
#     "V_jsr_part": 0.0012,
#     "V_nsr_part": 0.0116,
#     "delta_m": 1e-05,
#     "delta_m_WT": 1e-05,
#     "g_ratio": 0.3153,
#     "h_inf_shift": 0.0,
#     "h_inf_slope": 1.0,
#     "h_tau_gain": 1.0,
#     "kATP": 6142.0,
#     "kATP05": 6724.0,
#     "kPKA_PLB": 1.610336,
#     "kPLBp": 52.25,
#     "kPP1": 23850.0,
#     "kPP1_PLB": 0.07457,
#     "k_WT": 55.206,
#     "k_dL": 4.337,
#     "k_fL": 5.3,
#     "kb_CM": 542.0,
#     "kb_CQ": 445.0,
#     "kb_TC": 446.0,
#     "kb_TMC": 7.51,
#     "kb_TMM": 751.0,
#     "kf_CM": 1641986.0,
#     "kf_CQ": 175.4,
#     "kf_TC": 88800.0,
#     "kf_TMC": 227700.0,
#     "kf_TMM": 2277.0,
#     "kiCa": 500.0,
#     "kim": 5.0,
#     "koCa_max": 10000.0,
#     "kom": 660.0,
#     "ks": 1.480410851e+08,
#     "m_inf_shift": 0.0,
#     "m_inf_slope": 1.0,
#     "m_tau_shift": 0.0,
#     "nATP": 3.36,
#     "nPKA": 5.0,
#     "nPLB": 1.0,
#     "nRyR": 9.773,
#     "n_SK": 2.2,
#     "n_inf_shift": 0.0,
#     "n_shift_slope": 1.0,
#     "nif": 9.281,
#     "niso": 0.9264,
#     "offset_fT": 0.0,
#     "ratio_INaL_INa": 0.0,
#     "shift_fL": 0.0,
#     "slope_up": 5e-05,
#     "tau_y_a_shift": 0.0,
#     "tau_y_b_shift": 0.0,
#     "y_shift": 0.0,
#     "Iso": 0.0,
#     "K_05if": 17.8741 / 600.0,
#     "V_dL": -16.4508532699999996,
#     "V_fL": -37.4,
#     "cAMPb": 20.0 / 600.0,
#     "kPKA": 9000.0 / 600.0,
#     "kPKA_cAMP": 284.5 / 600.0,
#     "k_R231C": -34.9006,
#     }
A2 = 0.07
# ACh = 0.0
ATPi_max = 2.533
C = 57.0
CCh_cas = 0.0
CM_tot = 0.045
CQ_tot = 10.0
Ca_intracellular_fluxes_tau_dif_Ca = 5.469e-05
Ca_intracellular_fluxes_tau_tr = 0.04
Cao = 1.8
EC50_SK = 0.7
EC50_SR = 0.45
F = 96485.3415
GKACh = 0.00345
GKr_max = 0.00424
GKs_max = 0.00065
GKur_max = 0.0001539
GNa_WT = 0.0223
GNa_max = 0.0223
GSK = 0.0
Gf_K_max = 0.00268
Gf_Na_max = 0.00159
Gto_max = 0.0035
HSR = 2.5
ICaL_fCa_gate_alpha_fCa = 0.0075
INaK_max = 0.08105
INa_WT_ratio = 0.5
Iso_slope_cas_ICaL = 0.0
K1ni = 395.3
K1no = 1628.0
K2ni = 2.289
K2no = 561.4
K3ni = 26.44
K3no = 4.663
KATP_min = 6034.0
K_05ICaL1 = 0.730287
K_05ICaL2 = 0.66145
K_05INaK = 0.719701
K_05RyR = 0.682891
K_05iso = 58.57114132
K_AC = 0.0735
K_ACCa = 2.4e-05
K_ACI = 0.016
K_Ca = 0.000563995
K_ICaL1 = 0.470657
K_ICaL2 = 27.526226
K_INaK = 0.435692
K_NaCa = 3.343
K_iso = 0.007
K_iso_increase = 1.0
K_iso_shift = 1.0
K_iso_shift_ninf = 1.0
K_iso_slope_dL = 1.0
K_j_SRCarel = 1.0
K_j_up = 1.0
K_k1 = 4.64625
K_up = 0.000286113
Kci = 0.0207
Kcni = 26.44
Kco = 3.663
Kif = 26.26
Km_Kp = 1.4
Km_Nap = 14.0
Km_fCa = 0.000338
Ko = 5.4
L_cell = 67.0
L_sub = 0.02
MaxSR = 15.0
Mgi = 2.5
MinSR = 1.0
Nao = 140.0
PKAtot = 1.0
PKItot = 0.3
PP1 = 0.00089
P_CaL = 0.4578
P_CaT = 0.04132
P_up_basal = 5.0
Qci = 0.1369
Qco = 0.0
Qn = 0.4315
R_2 = 8314.472
R_cell = 3.9
RyR_max = 0.02
RyR_min = 0.0127
T = 310.0
TC_tot = 0.031
TMC_tot = 0.062
V_R231C = 113.8797
V_WT = 102.4597
V_i_part = 0.46
V_jsr_part = 0.0012
V_nsr_part = 0.0116
delta_m = 1e-05
delta_m_WT = 1e-05
g_ratio = 0.3153
h_inf_shift = 0.0
h_inf_slope = 1.0
h_tau_gain = 1.0
kATP = 6142.0
kATP05 = 6724.0
kPKA_PLB = 1.610336
kPLBp = 52.25
kPP1 = 23850.0
kPP1_PLB = 0.07457
k_WT = 55.206
k_dL = 4.337
k_fL = 5.3
kb_CM = 542.0
kb_CQ = 445.0
kb_TC = 446.0
kb_TMC = 7.51
kb_TMM = 751.0
kf_CM = 1641986.0
kf_CQ = 175.4
kf_TC = 88800.0
kf_TMC = 227700.0
kf_TMM = 2277.0
kiCa = 500.0
kim = 5.0
koCa_max = 10000.0
kom = 660.0
ks = 1.480410851e+08
m_inf_shift = 0.0
m_inf_slope = 1.0
m_tau_shift = 0.0
nATP = 3.36
nPKA = 5.0
nPLB = 1.0
nRyR = 9.773
n_SK = 2.2
n_inf_shift = 0.0
n_shift_slope = 1.0
nif = 9.281
niso = 0.9264
offset_fT = 0.0
ratio_INaL_INa = 0.0
shift_fL = 0.0
slope_up = 5e-05
tau_y_a_shift = 0.0
tau_y_b_shift = 0.0
y_shift = 0.0
# Iso = 0.0
K_05if = 0.029790166666666668
V_dL = -16.4508532699999996
V_fL = -37.4
cAMPb = 0.03333333333333333
kPKA = 15.0
kPKA_cAMP = 0.4741666666666667
k_R231C = -34.9006

if Iso_1_uM_on:
    Iso_increase_1 =  1.23 * K_iso_increase
    Iso_increase_2 =  1.2 * K_iso_increase
    Iso_shift_1 = -14.0 * K_iso_shift
    Iso_shift_2 = 7.5 * K_iso_shift
    Iso_shift_dL = -8.0 * K_iso_shift
    Iso_shift_ninf = 14.568 * K_iso_shift_ninf
    Iso_slope_dL = -27.0 * K_iso_slope_dL

elif Iso_linear_on:
    Iso_increase_1 = 1.0 + 0.23 * K_iso_increase
    Iso_increase_2 = 1.0 + 0.2 * K_iso_increase
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


GKr = GKr_max * (Ko / 5.4)**0.5
GNa_L = GNa_max * ratio_INaL_INa
RTONF = R_2 * T / F
V_cell = 1e-09 * 3.141592653589793 * R_cell**2 * L_cell
V_sub = 1e-09 * 2.0 * 3.141592653589793 * L_sub * (R_cell - L_sub / 2.0) * L_cell

k34 = Nao / (K3no + Nao)
kcch = 0.0146 * (CCh_cas**1.4402 / (51.7331**1.4402 + CCh_cas**1.4402))
V_i = V_i_part * V_cell - V_sub
V_jsr = V_jsr_part * V_cell
V_nsr = V_nsr_part * V_cell





class cell_constant_class(dict):
    def set_constant(self, key, item):
        if key in self:
            self[key] = item
        else:
            raise KeyError(f"Constant {key} not in constants dictionary")

    def get_constant(self, key):
        if key in self:
            return self[key]
        else:
            raise KeyError(f"Constant {key} not in constants dictionary")



# def system_constants(constants_dict=None, default_constants=default_constants, **kwargs):
#     """ Instantiate a cell_constant_class - a dict with set_constant and get_constant
#     methods. The set_constant method will update any derived (calculated) constants
#     so you don't need to update manually.

#     This is intended for use outside of CUDA so you can play it fast and loose with
#     the math you use.
#     """

#     constants = cell_constant_class()

#     if constants_dict is None:
#         constants_dict = {}

#     combined_updates = {**default_constants, **constants_dict, **kwargs}

#     # Note: If the same value occurs in the dict and
#     # keyword args, the kwargs one will win.
#     for key, item in combined_updates.items():
#         constants.update(combined_updates)


#     return constants


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
                 saved_states = None,
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

        #constants dict and list removed in favour of globals to help out compiler
        # self.constants_dict  = system_constants(kwargs)
        # self.constants_array = np.asarray([constant for (label, constant) in self.constants_dict.items()], dtype=precision)
        # self.constant_indices = {label: index for index, (label, constant) in enumerate(self.constants_dict.items())}

        self.state_labels = state_labels
        if saved_states is not None:
            self.set_saved_states(saved_states)


        as_numba_type.register(float, self.numba_precision)
        @cuda.jit(
            # (self.numba_precision[:],
            #     self.numba_precision[:],
            #     self.numba_precision[:],
            #     self.numba_precision),
                  device=True,
                  inline=True,
                  # fastmath=True,
                  # lineinfo=True, #comment out if running in cudasim mode
                  debug=False,
                  opt=True)
        def dxdtfunc(out,
                     state,
                     l_constants,
                     t):
            """ Put your dxdt calculations in here, including any reference signal
            or other math. Ugly is good here, avoid creating local variables and
            partial calculations - a long string of multiplies and adds, referring to
            the same array, might help the compiler make it fast. Avoid low powers,
            use consecutive multiplications instead.

            For a list of supported math functions you can include, see
            :https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html

            Fastmath exp, log, sqrt will allow inaccuracy -
            exp: 2 + floor(abs(1.173 * x)) ULP vs 2 ULP non-fast
            log: if x[0.5,2] then 2^-21.41, otherwise, 3 ULP vs 1 ULP
            sqrt is IEEE compliant
            div seems IEEE compliant
            also we cast denormals to zero, but these are 2^-126, which seems welll smaller than
            tolerance.
            """

            #If recording multiple variables, align them here (i.e. make them adjacent) and also change in the "state labels" dict
            V = state[0]
            Nai = state[1]
            Ki = state[2]
            y = state[3]
            m_WT = state[4]
            h_WT = state[5]
            m = state[6]
            h = state[7]
            dL = state[8]
            fL = state[9]
            fCa = state[10]
            dT = state[11]
            fT = state[12]
            R_1 = state[13]
            O = state[14]
            I = state[15]
            RI = state[16]
            fTC = state[17]
            fTMC = state[18]
            fTMM = state[19]
            fCMi = state[20]
            fCMs = state[21]
            fCQ = state[22]
            Cai = state[23]
            Ca_sub = state[24]
            Ca_nsr = state[25]
            Ca_jsr = state[26]
            r_Kur = state[27]
            s_Kur = state[28]
            q = state[29]
            r = state[30]
            paS = state[31]
            paF = state[32]
            piy = state[33]
            n = state[34]
            a = state[35]
            x = state[36]
            cAMP = state[37]
            PLBp = state[38]


            alpha_a = l_constants[0]
            ACh_block = l_constants[1]
            ACh_shift = l_constants[2]
            kiso = l_constants[3]
            P_up = l_constants[4]


            E0_m_WT = V + p(41.0)
            F_PLBp = cuda.selp(PLBp > p(0.23), p(3.3931) * (PLBp**p(4.0695)) / ((p(0.2805)**p(4.0695)) + (PLBp**p(4.0695))), p(1.698) * (PLBp**p(13.5842)) / ((p(0.224)**p(13.5842)) + (PLBp**p(13.5842))))
            INa_h_WT_gate_alpha_h_WT = p(20.0) * math.exp(p(-0.125) * (V + p(75.0)))
            INa_h_WT_gate_beta_h_WT = p(2000.0) / (p(320.0) * math.exp(p(-0.1) * (V + p(75.0))) + p(1.0))
            INa_h_gate_alpha_h = p(20.0) * math.exp(p(-0.125) * (V + p(75.0)))
            INa_h_gate_beta_h = p(2000.0) / (p(320.0) * math.exp(p(-0.1) * (V + p(75.0))) + p(1.0))
            INa_m_WT_gate_beta_m_WT = p(8000.0) * math.exp(p(-0.056) * (V + p(66.0)))
            ISK_x_gate_tau_x = p(1.0) / (p(0.047) * Ca_sub * p(1000.0) + p(1.0) / p(76.0))
            PKA = cuda.selp(cAMP * p(600.0) < p(25.87), p(-0.9483029) * math.exp(-cAMP * p(600.0) * p(6.56147900000000062e-02)) + p(9.77816459999999998e-01), p(-4.52605089999999988e-01) * math.exp(-cAMP * p(600.0) * p(3.39509399999999989e-02)) + p(9.92217140000000053e-01))
            beta_a = p(10.0) * math.exp(p(0.0133) * (V + p(40.0))) / p(1000.0)
            dT_inf = p(1.0) / (p(1.0) + math.exp(-(V + p(38.3)) / p(5.5)))
            fT_inf = p(1.0) / (p(1.0) + math.exp((V + p(58.7)) / p(3.8)))
            h_WT_inf = p(1.0) / (p(1.0) + math.exp((V + p(69.804)) / p(4.4565)))
            k2 = p(1.1) * p(237.9851) * ((cAMP * p(600.0))**p(5.101) / ((p(20.1077)**p(6.101)) + (cAMP * p(600.0))**p(6.101)))
            m_WT_inf = p(1.0) / (p(1.0) + math.exp(-(V + p(42.0504)) / p(8.3106)))
            pa_infinity = p(1.0) / (p(1.0) + math.exp(-(V + p(10.0144)) / p(7.6607)))
            piy_inf = p(1.0) / (p(1.0) + math.exp((V + p(28.6)) / p(17.1)))
            q_inf = p(1.0) / (p(1.0) + math.exp((V + p(49.0)) / p(13.0)))
            r_Kur_inf = p(1.0) / (p(1.0) + math.exp((V + p(6.0)) / p(-8.6)))
            r_inf = p(1.0) / (p(1.0) + math.exp(-(V - p(19.3)) / p(15.0)))
            s_Kur_inf = p(1.0) / (p(1.0) + math.exp((V + p(7.5)) / p(10.0)))
            tau_dT = p(0.001) / (p(1.068) * math.exp((V + p(38.3)) / p(30.0)) + p(1.068) * math.exp(-(V + p(38.3)) / p(30.0))) * p(1000.0)
            tau_fL = p(0.001) * (p(44.3) + p(230.0) * math.exp(-((V + p(36.0)) / p(10.0))*((V + p(36.0)) / p(10.0)))) * p(1000.0)
            tau_paF = p(1.0) / (p(30.0) * math.exp(V / p(10.0)) + math.exp(-V / p(12.0))) * p(1000.0)
            tau_paS = p(8.46553540000000049e-01) / (p(4.2) * math.exp(V / p(17.0)) + p(0.15) * math.exp(-V / p(21.6))) * p(1000.0)
            tau_piy = p(1.0) / (p(100.0) * math.exp(-V / p(54.645)) + p(656.0) * math.exp(V / p(106.157))) * p(1000.0)
            tau_q = p(0.001) * p(0.6) * (p(65.17) / (p(0.57) * math.exp(p(-0.08) * (V + p(44.0))) + p(0.065) * math.exp(p(0.1) * (V + p(45.93)))) + p(10.1)) * p(1000.0)
            tau_r = p(0.001) * p(0.66) * p(1.4) * (p(15.59) / (p(1.037) * math.exp(p(0.09) * (V + p(30.61))) + p(0.369) * math.exp(p(-0.12) * (V + p(23.84)))) + p(2.98)) * p(1000.0)
            tau_r_Kur = (p(0.009) / (p(1.0) + math.exp((V + p(5.0)) / p(12.0))) + p(0.0005)) * p(1000.0)
            tau_s_Kur = (p(0.59) / (p(1.0) + math.exp((V + p(60.0)) / p(10.0))) + p(3.05)) * p(1000.0)
            ATPi = p(ATPi_max) * (p(kATP) * ((cAMP * p(100.0) / p(cAMPb))**p(nATP)) / (p(kATP05) + ((cAMP * p(100.0) / p(cAMPb))**p(nATP))) - p(KATP_min)) / p(100.0)
            E0_m = V + p(41.0) - p(m_tau_shift)
            INa_m_WT_gate_alpha_m_WT = cuda.selp(abs(E0_m_WT) < p(delta_m_WT), p(2000.0), p(200.0) * E0_m_WT / -(math.exp(p(-0.1) * E0_m_WT) - p(1.0)))
            INa_m_gate_beta_m = p(8000.0) * math.exp(p(-0.056) * (V + p(66.0) - p(m_tau_shift)))
            _j_up_common = p(P_up) / (p(1.0) + math.exp((-Cai + p(K_up)) / p(slope_up)))

            if not VW_IKs:
                Iso_inc_cas_IKs = p(-0.2152) + p(0.435692) * (PKA**p(10.0808)) / ((p(0.719701)**p(10.0808)) + (PKA**p(10.0808)))
                Iso_shift_cas_IKs_n_gate = -(p(24.4) * p(1.411375) * (PKA**p(9.281)) / ((p(0.704217)**p(9.281)) + (PKA**p(9.281))) - p(18.76)) # Note: Literals used
                E_Ks = p(RTONF) * math.log((p(Ko) + p(0.12) * p(Nao)) / (Ki + p(0.12) * Nai))
            else:
                Iso_inc_cas_IKs = p(-0.2152) + p(0.494259) * (PKA**p(10.0808)) / ((p(0.736717)**p(10.0808)) + (PKA**p(10.0808)))
                Iso_shift_cas_IKs_n_gate = -(p(24.4) * p(1.438835) * (PKA**p(9.281)) / ((p(0.707399)**p(9.281)) + (PKA**p(9.281))) - p(18.76)) # Note: Literals used
                E_Ks = p(RTONF) * math.log((p(Ko) + p(0.0018) * p(Nao)) / (Ki + p(0.0018) * Nai))

            if Iso_cas_on:
                Iso_inc_cas_ICaL = p(-0.2152) + p(K_ICaL1) * (PKA**p(10.0808)) / ((p(K_05ICaL1)**p(10.0808)) + (PKA**p(10.0808)))
                Iso_inc_cas_IKs = Iso_inc_cas_IKs # Assigned above based on VW_IKs
                Iso_inc_cas_INaK = p(-0.2152) + p(K_INaK) * (PKA**p(10.0808)) / ((p(K_05INaK)**p(10.0808)) + (PKA**p(10.0808)))
                Iso_shift_cas_ICaL_dL_gate = -(p(K_ICaL2) * (PKA**p(9.281)) / ((p(K_05ICaL2)**p(9.281)) + (PKA**p(9.281))) - p(18.76))
                Iso_shift_cas_IKs_n_gate = Iso_shift_cas_IKs_n_gate # Assigned above based on VW_IKs
                Iso_shift_cas_If_y_gate = p(Kif) * ((cAMP**p(nif)) / ((p(K_05if)**p(nif)) + (cAMP**p(nif)))) - p(18.76)
                koCa = p(koCa_max) * (p(RyR_min) - p(RyR_max) * (PKA**p(nRyR)) / ((p(K_05RyR)**p(nRyR)) + (PKA**p(nRyR))) + p(1.0))
                j_up = p(K_j_up) * (p(0.9) * _j_up_common * F_PLBp) # Assumes F_PLBp is correct
            else:
                j_up = _j_up_common
                Iso_inc_cas_ICaL = p(0.0)
                Iso_inc_cas_IKs = p(0.0)
                Iso_inc_cas_INaK = p(0.0)
                Iso_shift_cas_ICaL_dL_gate = p(0.0)
                Iso_shift_cas_IKs_n_gate = p(0.0)
                Iso_shift_cas_If_y_gate = p(0.0)
                koCa = p(koCa_max)


            if R231C_on: # Original check was R231C_on != 0.0f
                R231C = p(g_ratio) * ((p(1.0) / (p(1.0) + math.exp(-(V - p(V_R231C)) / p(k_R231C))))**p(2.0)) / (((p(1.0) - p(A2)) / (p(1.0) + math.exp(-(V - p(V_WT)) / p(k_WT))) + p(A2))**p(2.0))
            else:
                R231C = p(1.0)

            delta_fCMi = p(kf_CM) * Cai * (p(1.0) - fCMi) - p(kb_CM) * fCMi
            delta_fCMs = p(kf_CM) * Ca_sub * (p(1.0) - fCMs) - p(kb_CM) * fCMs
            delta_fCQ = p(kf_CQ) * Ca_jsr * (p(1.0) - fCQ) - p(kb_CQ) * fCQ
            delta_fTC = p(kf_TC) * Cai * (p(1.0) - fTC) - p(kb_TC) * fTC
            delta_fTMC = p(kf_TMC) * Cai * (p(1.0) - (fTMC + fTMM)) - p(kb_TMC) * fTMC
            delta_fTMM = p(kf_TMM) * p(Mgi) * (p(1.0) - (fTMC + fTMM)) - p(kb_TMM) * fTMM
            fCa_infinity = p(Km_fCa) / (p(Km_fCa) + Ca_sub)
            fL_inf = p(1.0) / (p(1.0) + math.exp((V - p(V_fL) + p(shift_fL)) / p(k_fL)))
            h_inf = p(1.0) / (p(1.0) + math.exp((V + p(69.804) - p(h_inf_shift)) / (p(4.4565) * p(h_inf_slope))))
            j_Ca_dif = (Ca_sub - Cai) / p(Ca_intracellular_fluxes_tau_dif_Ca)
            j_SRCarel = p(K_j_SRCarel) * p(ks) * O * (Ca_jsr - Ca_sub)
            j_tr = (Ca_nsr - Ca_jsr) / p(Ca_intracellular_fluxes_tau_tr)
            _Cai_from_fCMi = p(kb_CM) * fCMi / (p(kf_CM) * (p(1.0) - fCMi))
            k1 = p(K_k1) * (p(K_ACI) + p(K_AC) / (p(1.0) + math.exp((p(K_Ca) - _Cai_from_fCMi) / p(K_ACCa))))
            k3 = p(kPKA) * ((cAMP**(p(nPKA) - p(1.0))) / ((p(kPKA_cAMP)**p(nPKA)) + (cAMP**p(nPKA))))
            k4 = p(kPLBp) * (PKA**p(nPLB)) / ((p(kPKA_PLB)**p(nPLB)) + (PKA**p(nPLB)))
            k43 = Nai / (p(K3ni) + Nai)
            k5 = p(kPP1) * (p(PP1) * PLBp / (p(kPP1_PLB) + PLBp))
            kCaSR = p(MaxSR) - (p(MaxSR) - p(MinSR)) / (p(1.0) + (p(EC50_SR) / Ca_jsr)**p(HSR))


            m_inf = p(1.0) / (p(1.0) + math.exp(-(V + p(42.0504) - p(m_inf_shift)) / (p(8.3106) * p(m_inf_slope))))
            paF_inf = pa_infinity
            paS_inf = pa_infinity
            tau_fT = (p(1.0) / (p(16.67) * math.exp(-(V + p(75.0)) / p(83.3)) + p(16.67) * math.exp((V + p(75.0)) / p(15.38))) + p(offset_fT)) * p(1000.0)
            tau_h = p(h_tau_gain) / (INa_h_gate_alpha_h + INa_h_gate_beta_h) * p(1000.0)
            tau_h_WT = p(1.0) / (INa_h_WT_gate_alpha_h_WT + INa_h_WT_gate_beta_h_WT) * p(1000.0)
            x_infinity = p(0.81) * (Ca_sub * p(1000.0))**p(n_SK) / ((Ca_sub * p(1000.0))**p(n_SK) + (p(EC50_SK)**p(n_SK)))

            E_K = p(RTONF) * math.log(p(Ko) / Ki)
            E_Na = p(RTONF) * math.log(p(Nao) / Nai)
            E_mh = p(RTONF) * math.log((p(Nao) + p(0.12) * p(Ko)) / (Nai + p(0.12) * Ki))

            if not VW_IKs:
                if Iso_linear_on:
                    _gks_iso_factor = p(1.0) + p(0.2) * p(K_iso_increase)
                elif Iso_1_uM_on:
                    _gks_iso_factor = p(1.0) + p(0.2) * p(K_iso_increase)
                elif Iso_cas_on:
                    _gks_iso_factor = p(1.0) + Iso_inc_cas_IKs
                else:
                    _gks_iso_factor = p(1.0)
                GKs = p(GKs_max) * _gks_iso_factor
            else:
                if Iso_linear_on:
                    _gks_iso_factor = p(1.0) + p(0.25) * p(K_iso_increase)
                elif Iso_1_uM_on:
                    _gks_iso_factor = p(1.25) * p(K_iso_increase)
                elif Iso_cas_on:
                    _gks_iso_factor = p(1.0) + Iso_inc_cas_IKs
                else:
                    _gks_iso_factor = p(1.0)
                GKs = p(0.2) * p(GKs_max) * _gks_iso_factor

            ICaL_fCa_gate_tau_fCa = p(0.001) * fCa_infinity / p(ICaL_fCa_gate_alpha_fCa)

            #currents are linearised for the region in which the denominator is smaller than 1e-6 to avoid singularities when running at 32 bit precision
            #This matches the approach published by the authors of Chaste-codegen
            ICaT_nonsingular = p(2.0) * p(P_CaT) * V / (p(RTONF) * -(math.exp(p(-1.0) * V * p(2.0) / p(RTONF)) - p(1.0))) * (Ca_sub - p(Cao) * math.exp(p(-2.0) * V / p(RTONF))) * dT * fT
            P_ICaT = p(2.0) * p(P_CaT)
            G_ICaT = dT * fT
            ICaT_at_zero = P_ICaT * (p(1.0)/p(2.0)) * (Ca_sub - p(Cao)) * G_ICaT
            ICaT_deriv_at_zero = (P_ICaT * G_ICaT / (p(2.0) * p(RTONF))) * (Ca_sub + p(Cao))
            L_ICaT = ICaT_at_zero + ICaT_deriv_at_zero * V
            ICaT = cuda.selp(abs(V) < p(RTONF) * p(5.0e-7), L_ICaT, ICaT_nonsingular)

            if not VW_IKs: # Original check was VW_IKs == 0.0f
                IKs_n_gate_beta_n = p(1.0) * math.exp(-(V - p(Iso_shift_1) - Iso_shift_cas_IKs_n_gate - p(5.0)) / p(25.0))
                n_inf = math.sqrt(p(1.0) / (p(1.0) + math.exp(-(V + p(0.6383) - p(Iso_shift_1) - Iso_shift_cas_IKs_n_gate - p(n_inf_shift)) / (p(10.7071) * p(n_shift_slope)))))
                IKs_n_gate_alpha_n = p(28.0) / (p(1.0) + math.exp(-(V - p(40.0) - p(Iso_shift_1) - Iso_shift_cas_IKs_n_gate) / p(3.0)))
            else:
                IKs_n_gate_beta_n = p(1.93) * math.exp(-V / p(83.2))
                n_inf = p(1.0) / (p(1.0) + math.exp(-(V + p(15.733) + p(Iso_shift_ninf) - Iso_shift_cas_IKs_n_gate - p(n_inf_shift)) / (p(n_shift_slope) * p(27.77))))
                n_inf_safe_denom = cuda.selp(n_inf == p(1.0), p(1e-9), p(1.0) - n_inf) # Avoid division by zero if n_inf is 1.0
                IKs_n_gate_alpha_n = n_inf / n_inf_safe_denom * IKs_n_gate_beta_n

            INa_m_gate_alpha_m = cuda.selp(abs(E0_m) < p(delta_m), p(2000.0), p(200.0) * E0_m / -(math.exp(p(-0.1) * E0_m) - p(1.0)))
            IsiCa_nonsingular = p(2.0) * p(P_CaL) * p(Iso_increase_1) * (p(1.0) - p(ACh_block)) * (p(1.0) + Iso_inc_cas_ICaL) * (V - p(0.0)) / (p(RTONF) * -(math.exp(p(-1.0) * (V - p(0.0)) * p(2.0) / p(RTONF)) - p(1.0))) * (Ca_sub - p(Cao) * math.exp(p(-2.0) * (V - p(0.0)) / p(RTONF))) * dL * fL * fCa
            P_IsiCa = p(2.0) * p(P_CaL) * p(Iso_increase_1) * (p(1.0) - p(ACh_block)) * (p(1.0) + Iso_inc_cas_ICaL)
            G_ICaL = dL * fL * fCa # Common gating factor for IsiCa, IsiK, IsiNa
            IsiCa_at_zero = P_IsiCa * (p(1.0)/p(2.0)) * (Ca_sub - p(Cao)) * G_ICaL
            IsiCa_deriv_at_zero = (P_IsiCa * G_ICaL / (p(2.0) * p(RTONF))) * (Ca_sub + p(Cao))
            L_IsiCa = IsiCa_at_zero + IsiCa_deriv_at_zero * V
            IsiCa = cuda.selp(abs(V) < p(RTONF) * p(5.0e-7), L_IsiCa, IsiCa_nonsingular)


            IsiK_nonsingular = p(0.000365) * p(P_CaL) * p(Iso_increase_1) * (p(1.0) - p(ACh_block)) * (p(1.0) + Iso_inc_cas_ICaL) * (V - p(0.0)) / (p(RTONF) * -(math.exp(p(-1.0) * (V - p(0.0)) / p(RTONF)) - p(1.0))) * (Ki - p(Ko) * math.exp(p(-1.0) * (V - p(0.0)) / p(RTONF))) * dL * fL * fCa
            P_IsiK = p(0.000365) * p(P_CaL) * p(Iso_increase_1) * (p(1.0) - p(ACh_block)) * (p(1.0) + Iso_inc_cas_ICaL)
            IsiK_at_zero = P_IsiK * (p(1.0)/p(1.0)) * (Ki - p(Ko)) * G_ICaL
            IsiK_deriv_at_zero = (P_IsiK * G_ICaL / (p(2.0) * p(RTONF))) * (Ki + p(Ko))
            L_IsiK = IsiK_at_zero + IsiK_deriv_at_zero * V
            IsiK = cuda.selp(abs(V) < p(RTONF) * p(1e-6),L_IsiK,IsiK_nonsingular )

            IsiNa_nonsingular = p(1.85e-05) * p(P_CaL) * p(Iso_increase_1) * (p(1.0) - p(ACh_block)) * (p(1.0) + Iso_inc_cas_ICaL) * (V - p(0.0)) / (p(RTONF) * -(math.exp(p(-1.0) * (V - p(0.0)) / p(RTONF)) - p(1.0))) * (Nai - p(Nao) * math.exp(p(-1.0) * (V - p(0.0)) / p(RTONF))) * dL * fL * fCa
            P_IsiNa = p(1.85e-05) * p(P_CaL) * p(Iso_increase_1) * (p(1.0) - p(ACh_block)) * (p(1.0) + Iso_inc_cas_ICaL)
            IsiNa_at_zero = P_IsiNa * (p(1.0)/p(1.0)) * (Nai - p(Nao)) * G_ICaL
            IsiNa_deriv_at_zero = (P_IsiNa * G_ICaL / (p(2.0) * p(RTONF))) * (Nai + p(Nao))
            L_IsiNa = IsiNa_at_zero + IsiNa_deriv_at_zero * V
            IsiNa = cuda.selp(abs(V) < p(RTONF) * p(1e-6), L_IsiNa, IsiNa_nonsingular)

            dL_inf = p(1.0) / (p(1.0) + math.exp(-(V - p(V_dL) - p(Iso_shift_dL) - Iso_shift_cas_ICaL_dL_gate) / (p(k_dL) * (p(1.0) + p(Iso_slope_cas_ICaL)) * (p(1.0) + p(Iso_slope_dL) / p(100.0)))))
            di = p(1.0) + Ca_sub / p(Kci) * (p(1.0) + math.exp(-p(Qci) * V / p(RTONF)) + Nai / p(Kcni)) + Nai / p(K1ni) * (p(1.0) + Nai / p(K2ni) * (p(1.0) + Nai / p(K3ni)))
            diff_Ca_jsr = (j_tr - (j_SRCarel + p(CQ_tot) * delta_fCQ)) * p(0.001)
            diff_PLBp = (k4 - k5) / p(60000.0)
            diff_cAMP = ((p(kiso) - p(kcch)) * ATPi + k1 * ATPi - k2 * cAMP - k3 * cAMP) / p(60000.0)
            diff_fCMi = delta_fCMi * p(0.001)
            diff_fCMs = delta_fCMs * p(0.001)
            diff_fCQ = delta_fCQ * p(0.001)
            diff_fTC = delta_fTC * p(0.001)
            diff_fTMC = delta_fTMC * p(0.001)
            diff_fTMM = delta_fTMM * p(0.001)
            diff_x = (x_infinity - x) / ISK_x_gate_tau_x
            do_ = p(1.0) + p(Cao) / p(Kco) * (p(1.0) + math.exp(p(Qco) * V / p(RTONF))) + p(Nao) / p(K1no) * (p(1.0) + p(Nao) / p(K2no) * (p(1.0) + p(Nao) / p(K3no)))
            k32 = math.exp(p(Qn) * V / (p(2.0) * p(RTONF)))
            k41 = math.exp(-p(Qn) * V / (p(2.0) * p(RTONF)))
            kiSRCa = p(kiCa) * kCaSR
            koSRCa = koCa / kCaSR

            tau_m_WT = p(1.0) / (INa_m_WT_gate_alpha_m_WT + INa_m_WT_gate_beta_m_WT) * p(1000.0)
            tau_y = (p(1.0) / (p(0.36) * (V + p(148.8) - p(ACh_shift) - p(Iso_shift_2) - Iso_shift_cas_If_y_gate - p(tau_y_a_shift)) / (math.exp(p(0.066) * (V + p(148.8) - p(ACh_shift) - p(Iso_shift_2) - Iso_shift_cas_If_y_gate - p(tau_y_a_shift))) - p(1.0)) + p(0.1) * (V + p(87.3) - p(ACh_shift) - p(Iso_shift_2) - Iso_shift_cas_If_y_gate - p(tau_y_b_shift)) / -(math.exp(p(-0.2) * (V + p(87.3) - p(ACh_shift) - p(Iso_shift_2) - Iso_shift_cas_If_y_gate - p(tau_y_b_shift))) - p(1.0))) - p(0.054)) * p(1000.0)

            _y_inf_cond_val = -(p(80.0) - p(ACh_shift) - p(Iso_shift_2) - Iso_shift_cas_If_y_gate - p(y_shift))
            y_inf = cuda.selp(V < _y_inf_cond_val, p(0.01329) + p(0.99921) / (p(1.0) + math.exp((V + p(97.134) - p(ACh_shift) - p(Iso_shift_2) - Iso_shift_cas_If_y_gate - p(y_shift)) / p(8.1752))), p(0.0002501) * math.exp(-(V - p(ACh_shift) - p(Iso_shift_2) - Iso_shift_cas_If_y_gate - p(y_shift)) / p(12.861)))
            ICaL = IsiCa + IsiK + IsiNa
            _crit_val_a1 = -p(41.8) - p(Iso_shift_dL) - Iso_shift_cas_ICaL_dL_gate
            _crit_val_a2 = -p(6.8) - p(Iso_shift_dL) - Iso_shift_cas_ICaL_dL_gate
            _crit_val_b = -p(1.8) - p(Iso_shift_dL) - Iso_shift_cas_ICaL_dL_gate
            #TODO: Either linearise or add nonzero offset, currenty you're doing both in different places
            a_needs_delta = (V == _crit_val_a1) or (V == _crit_val_a2) or (abs(V) < 1e-7)
            b_needs_delta = (V == _crit_val_b)
            delta = p(1e-9)
            adVm = cuda.selp(a_needs_delta, V - delta, V)
            bdVm = cuda.selp(b_needs_delta, V - delta, V)

            ICaL_dL_gate_alpha_dL = p(-0.02839) * (adVm + p(41.8) - p(Iso_shift_dL) - Iso_shift_cas_ICaL_dL_gate) / (math.exp(-(adVm + p(41.8) - p(Iso_shift_dL) - Iso_shift_cas_ICaL_dL_gate) / p(2.5)) - p(1.0)) - p(0.0849) * (adVm + p(6.8) - p(Iso_shift_dL) - Iso_shift_cas_ICaL_dL_gate) / (math.exp(-(adVm + p(6.8) - p(Iso_shift_dL) - Iso_shift_cas_ICaL_dL_gate) / p(4.8)) - p(1.0))
            ICaL_dL_gate_beta_dL = p(0.01143) * (bdVm + p(1.8) - p(Iso_shift_dL) - Iso_shift_cas_ICaL_dL_gate) / (math.exp((bdVm + p(1.8) - p(Iso_shift_dL) - Iso_shift_cas_ICaL_dL_gate) / p(2.5)) - p(1.0))
            IKACh = cuda.selp(p(ACh_block) > p(0.0), p(ACh_on) * p(GKACh) * (V - E_K) * (p(1.0) + math.exp((V + p(20.0)) / p(20.0))) * a, p(0.0))
            IKr = p(GKr) * (V - E_K) * (p(0.9) * paF + p(0.1) * paS) * piy
            IKs = R231C * GKs * (V - E_Ks) * (n*n)

            IKur = p(GKur_max) * r_Kur * s_Kur * (V - E_K)
            INaK = (p(1.0) + Iso_inc_cas_INaK) * p(Iso_increase_2) * p(INaK_max) * ((p(1.0) + (p(Km_Kp) / p(Ko))**p(1.2))**p(-1.0)) * ((p(1.0) + (p(Km_Nap) / Nai)**p(1.3))**p(-1.0)) * ((p(1.0) + math.exp(-(V - E_Na + p(110.0)) / p(20.0)))**p(-1.0))
            INa_ = p(GNa_max) * (m*m*m) * h * (V - E_mh)
            INa_L = p(GNa_L) * (m*m*m) * (V - E_mh)
            INa_WT = p(GNa_WT) * (m_WT*m_WT*m_WT) * h_WT * (V - E_mh)
            ISK = p(GSK) * (V - E_K) * x
            IfK = y * p(Gf_K_max) * (V - E_K)
            IfNa = y * p(Gf_Na_max) * (V - E_Na)
            Ito = p(Gto_max) * (V - E_K) * q * r
            diff_I = (kiSRCa * Cai * O - p(kim) * I - (p(kom) * I - koSRCa * (Cai*Cai) * RI)) * p(0.001)
            diff_O = (koSRCa * (Cai*Cai) * R_1 - p(kom) * O - (kiSRCa * Cai * O - p(kim) * I)) * p(0.001)
            diff_RI = (p(kom) * I - koSRCa * (Cai*Cai) * RI - (p(kim) * RI - kiSRCa * Cai * R_1)) * p(0.001)
            diff_R_1 = (p(kim) * RI - kiSRCa * Cai * R_1 - (koSRCa * (Cai*Cai) * R_1 - p(kom) * O)) * p(0.001)
            diff_fCa = (fCa_infinity - fCa) / ICaL_fCa_gate_tau_fCa * p(0.001)

            k12 = Ca_sub / p(Kci) * math.exp(-p(Qci) * V / p(RTONF)) / di
            k14 = Nai / p(K1ni) * Nai / p(K2ni) * (p(1.0) + Nai / p(K3ni)) * math.exp(p(Qn) * V / (p(2.0) * p(RTONF))) / di
            k21 = p(Cao) / p(Kco) * math.exp(p(Qco) * V / p(RTONF)) / do_
            k23 = p(Nao) / p(K1no) * p(Nao) / p(K2no) * (p(1.0) + p(Nao) / p(K3no)) * math.exp(-p(Qn) * V / (p(2.0) * p(RTONF))) / do_
            tau_m = p(1.0) / (INa_m_gate_alpha_m + INa_m_gate_beta_m) * p(1000.0)
            INa = (p(1.0) - p(INa_WT_ratio)) * (INa_ + INa_L) + p(INa_WT_ratio) * INa_WT
            If = IfNa + IfK
            diff_Ca_nsr = (j_up - j_tr * p(V_jsr) / p(V_nsr)) * p(0.001)
            diff_Cai = (p(1.0) * (j_Ca_dif * p(V_sub) - j_up * p(V_nsr)) / p(V_i) - (p(CM_tot) * delta_fCMi + p(TC_tot) * delta_fTC + p(TMC_tot) * delta_fTMC)) * p(0.001)
            tau_dL = p(0.001) / (ICaL_dL_gate_alpha_dL + ICaL_dL_gate_beta_dL) * p(1000.0)
            tau_n = p(1.0) / (IKs_n_gate_alpha_n + IKs_n_gate_beta_n) * p(1000.0)
            x1 = k41 * p(k34) * (k23 + k21) + k21 * k32 * (k43 + k41)
            x2 = k32 * k43 * (k14 + k12) + k41 * k12 * (p(k34) + k32)
            x3 = k14 * k43 * (k23 + k21) + k12 * k23 * (k43 + k41)
            x4 = k23 * p(k34) * (k14 + k12) + k14 * k21 * (p(k34) + k32)
            INaCa = p(K_NaCa) * (x2 * k21 - x1 * k12) / (x1 + x2 + x3 + x4)
            Itot = If + IKr + IKs + Ito + INaK + INaCa + INa + ICaL + ICaT + IKACh + IKur + ISK
            diff_Ca_sub = (j_SRCarel * p(V_jsr) / p(V_sub) - ((IsiCa + ICaT - p(2.0) * INaCa) / (p(2.0) * p(F) * p(V_sub)) + j_Ca_dif + p(CM_tot) * delta_fCMs)) * p(0.001)
            if dynamic_Ki_Nai:
                diff_Nai = p(-1.0) * (INa + IfNa + IsiNa + p(3.0) * INaK + p(3.0) * INaCa) / (p(1.0) * (p(V_i) + p(V_sub)) * p(F)) * p(0.001)
                diff_Ki = p(-1.0) * (IKur + Ito + IKr + IKs + IfK + IsiK + ISK - p(2.0) * INaK) / (p(1.0) * (p(V_i) + p(V_sub)) * p(F)) * p(0.001)
            else:
                diff_Nai = p(0.0)
                diff_Ki = p(0.0)
            Iion = Itot * p(1000.0) / p(C)

            out[11] = (dT_inf - dT) / tau_dT
            out[33] = (piy_inf - piy) / tau_piy
            out[29] = (q_inf - q) / tau_q
            out[30] = (r_inf - r) / tau_r
            out[27] = (r_Kur_inf - r_Kur) / tau_r_Kur
            out[28] = (s_Kur_inf - s_Kur) / tau_s_Kur
            out[35] = p(alpha_a) * (p(1.0) - a) - beta_a * a
            out[9] = (fL_inf - fL) / tau_fL
            out[12] = (fT_inf - fT) / tau_fT
            out[7] = (h_inf - h) / tau_h
            out[5] = (h_WT_inf - h_WT) / tau_h_WT
            out[32] = (paF_inf - paF) / tau_paF
            out[31] = (paS_inf - paS) / tau_paS
            out[26] = diff_Ca_jsr
            out[38] = diff_PLBp
            out[37] = diff_cAMP
            out[20] = diff_fCMi
            out[21] = diff_fCMs
            out[22] = diff_fCQ
            out[17] = diff_fTC
            out[18] = diff_fTMC
            out[19] = diff_fTMM
            out[4] = (m_WT_inf - m_WT) / tau_m_WT
            out[36] = diff_x
            out[3] = (y_inf - y) / tau_y
            out[15] = diff_I
            out[14] = diff_O
            out[16] = diff_RI
            out[13] = diff_R_1
            out[10] = diff_fCa
            out[6] = (m_inf - m) / tau_m
            out[25] = diff_Ca_nsr
            out[23] = diff_Cai
            out[2] = diff_Ki
            out[8] = (dL_inf - dL) / tau_dL
            out[34] = (n_inf - n) / tau_n
            out[24] = diff_Ca_sub
            out[1] = diff_Nai
            out[0] = -Iion

        @cuda.jit(device=True,
                  opt=True,
                  inline=True,
                  # lineinfo=True
                  )
        def calculate_grid_dependent_constants(l_constants,
                                               grid_values):
            # There are a number of "run once" constants that are missed by the compiler due to this section being included as a function call. It might pay to make the non-Ach/iso dependent ones into shared memory elements.
            ACh = grid_values[0]
            Iso = grid_values[1]

            alpha_a = ((p(3.5988) - p(0.025641)) / (p(1.0) + p(1.2155e-06) / (p(1.0) * p(1e-06) * ACh + p(1e-15))**p(1.6951)) + p(0.025641)) / p(1000.0)  # p(1e-15) constant added to avoid divide by zero when ACh = 0
            ACh_block = p(0.31) * p(1e-06) * ACh / (p(1e-06) * ACh + p(9e-05))
            ACh_shift = cuda.selp(ACh > p(0.0), (p(-1.0) - p(9.898) * (p(1e-06) * ACh)**p(0.618) / ((p(1e-06) * ACh)**p(0.618) + p(1.22422999999999998e-03))), p(0.0))
            kiso = K_iso + p(0.1181) * (Iso**niso / (K_05iso**niso + Iso**niso + p(1e-15)))  # p(1e-15) constant added to avoid nan
            Ca_intracellular_fluxes_b_up = p(0.7) * ACh / (p(9e-05) + ACh)
            P_up = P_up_basal * (p(1.0) - Ca_intracellular_fluxes_b_up)

            l_constants[0] = alpha_a
            l_constants[1] = ACh_block
            l_constants[2] = ACh_shift
            l_constants[3] = kiso
            l_constants[4] = P_up


        self.calc_grid_constants = calculate_grid_dependent_constants
        self.dxdtfunc = dxdtfunc

    def set_saved_states(self, states):
        self.saved_states = states
        self.saved_state_indices = [self.state_labels[label] for label in states]

    # def update_constants(self, updates_dict=None, **kwargs):
    #     if updates_dict is None:
    #         updates_dict = {}

    #     combined_updates = {**updates_dict, **kwargs}

    #     # Note: If the same value occurs in the dict and
    #     # keyword args, the kwargs one will win.
    #     for key, item in combined_updates.items():
    #         self.constants_dict.set_constant(key, self.precision(item))

    #     self.constants_array = np.asarray([constant for (label, constant) in self.constants_dict.items()], dtype=self.precision)


# ******************************* TEST CODE ******************************** #
if __name__ == '__main__':
    precision = np.float32
    sys = fabbri_linder_cell(precision = precision)
    numba_precision = sys.numba_precision

    dxdt = sys.dxdtfunc
    NUM_STATES= 39
    # NUM_CONSTANTS = 173
    # constants = cuda.to_device(sys.constants_array)
    inits = cuda.to_device([initial_values[label] for label in state_labels])
    sweep_params = cuda.to_device(np.asarray([1.0, 1.0], dtype=precision))

    @cuda.jit((sys.numba_precision[:],
               sys.numba_precision[:],
               sys.numba_precision[:]
               ),
              opt=True,
              debug=False)
    def testkernel(out,
                   params,
                   inits):

        l_state = cuda.local.array(shape=NUM_STATES, dtype=numba_precision)
        c_params = cuda.const.array_like(params)

        for i in range(len(inits)):
            l_state[i] = inits[i]

        t = precision(1.0)
        dxdt(out,
            l_state,
            c_params,
            t)

    out = cuda.pinned_array_like(np.zeros(sys.num_states))
    d_out = cuda.to_device(out)
    print("Testing to see if your dxdt function compiles using CUDA...")
    testkernel[1,1](d_out,
                    sweep_params,
                    inits)
    cuda.synchronize()
    out = d_out.copy_to_host()
    print(out)