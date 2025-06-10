# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 19:24:23 2024

@author: cca79
"""
from numba import cuda, float64, int32, from_dtype, float32
from time import time
import numpy as np
from functools import wraps
from numba.cuda.random import xoroshiro128p_normal_float64,xoroshiro128p_normal_float32, xoroshiro128p_dtype
xoro_type = from_dtype(xoroshiro128p_dtype)


def timing(f, nruns=3):
    @wraps(f)
    def wrap(*args, **kw):
        ts = np.zeros(nruns)
        te = np.zeros(nruns)
        for  i in range(nruns):
            ts[i] = time()
            result = f(*args, **kw)
            te[i] = time()
        durations = te-ts
        print('func:%r took: \n %2.6f sec avg \n %2.6f max \n %2.6f min \n over %d runs' % \
          (f.__name__, np.mean(durations), np.amax(durations), np.amin(durations), nruns))
        return result
    return wrap

@cuda.jit(float64(float64,
                  float64),
          device=True,
          inline=True)
def clamp_64(value,
         clip_value):
    if value <= clip_value and value >= -clip_value:
        return value
    elif value > clip_value:
        return clip_value
    else:
        return -clip_value

@cuda.jit(float32(float32,
                  float32),
          device=True,
          inline=True)
def clamp_32(value,
         clip_value):
    if value <= clip_value and value >= -clip_value:
        return value
    elif value > clip_value:
        return clip_value
    else:
        return -clip_value

@cuda.jit((float64[:],
            float64[:],
            int32,
            xoro_type[:]
            ),
          device=True,
          inline=True)
def get_noise_64(noise_array,
              sigmas,
              idx,
              RNG):

    for i in range(len(noise_array)):
        if sigmas[i] != 0.0:
            noise_array[i] = xoroshiro128p_normal_float64(RNG, idx) * sigmas[i]

@cuda.jit((float32[:],
            float32[:],
            int32,
            xoro_type[:]
            ),
          device=True,
          inline=True)

def get_noise_32(noise_array,
              sigmas,
              idx,
              RNG):

    for i in range(len(noise_array)):
        if sigmas[i] != 0.0:
            noise_array[i] = xoroshiro128p_normal_float32(RNG, idx) * sigmas[i]


def round_sf(num, sf):
    if num == 0.0:
        return 0.0
    else:
        return round(num, sf - 1 - int(np.floor(np.log10(abs(num)))))

def round_list_sf(list, sf):
    return [round_sf(num, sf) for num in list]

def get_readonly_view(array):
    view = array.view()
    view.flags.writeable = False
    return view