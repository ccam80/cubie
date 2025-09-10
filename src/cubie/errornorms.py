from numba import cuda, from_dtype

def static_weighted_l2_norm_factory(precision, n, weights):
    """Generates a function to compute weighted L2 norm with static weights"""

    precision = from_dtype(precision)
    for i in range(n):
        #Normalise error by weight ** 2
        weights[i] = precision(1 / (weights[i]*weights[i]))

    @cuda.jit(device=True, inline=True)
    def norm_static_weighted_l2_squared(i, in1, in2):
        acc = precision(0.0)
        for j in range(n):
            val = in1[i, j] - in2[i, j]
            acc += val * val * weights[j]
        return acc

def dynamic_weighted_l2_norm_factory(precision, n):
    """Generates a function to compute weighted L2 norm with dynamic weights"""
    precision = from_dtype(precision)

    @cuda.jit(device=True, inline=True)
    def norm_static_weighted_l2_squared(i, in1, in2, weights):
        acc = precision(0.0)
        for j in range(n):
            # Normalise error by weight ** 2
            weight = precision(1 / (weights[i] * weights[i]))
            val = in1[i, j] - in2[i, j]
            acc += val * val * weight
        return acc
def l2_squared_norm_factory(precision, n):
    precision = from_dtype(precision)

    @cuda.jit(device=True, inline=True)
    def norm_l2_squared(i, in1, in2):
        acc = precision(0.0)
        for j in range(n):
            val = in1[i, j] - in2[i, j]
            acc += val * val
        return acc
