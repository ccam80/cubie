from numba import cuda, from_dtype

def get_norm_factory(norm_type):
    known_norms = ["l2", "weighted_l2", "weighted_l2_static"]
    if norm_type == "l2":
        return l2_squared_norm_factory
    elif norm_type == "weighted_l2":
        return dynamic_weighted_l2_norm_factory
    elif norm_type == "weighted_l2_static":
        return static_weighted_l2_norm_factory
    elif norm_type == "hairer":
        return hairer_squared_factory
    else:
        raise ValueError(f"Unknown norm type: {norm_type}. Options are: "
                         f"{known_norms}.")


def hairer_squared_factory(precision, n):
    precision = from_dtype(precision)
    n_squared = n * n
    # step sizes and norms can be approximate - fastmath is fine
    @cuda.jit(device=True, inline=True, fastmath=True)
    def hairer_squared(arr):
        acc = precision(0.0)
        for j in range(n):
            acc += arr[j] * arr[j]
        return acc / n_squared
    return hairer_squared


def static_weighted_l2_norm_factory(precision, n, weights):
    """Generates a function to compute weighted L2 norm with static weights"""

    precision = from_dtype(precision)
    for i in range(n):
        #Normalise error by weight ** 2
        weights[i] = precision(1 / (weights[i]*weights[i]))

    # step sizes and norms can be approximate - fastmath is fine
    @cuda.jit(device=True, inline=True)
    def static_weighted_l2_squared(arr):
        acc = precision(0.0)
        for j in range(n):
            acc += arr[j] * arr[j] * weights[j]
        return acc
    return static_weighted_l2_squared

def dynamic_weighted_l2_norm_factory(precision, n):
    """Generates a function to compute weighted L2 norm with dynamic weights"""
    precision = from_dtype(precision)

    # step sizes and norms can be approximate - fastmath is fine
    @cuda.jit(device=True, inline=True)
    def static_weighted_l2_squared(arr, weights):
        acc = precision(0.0)
        for j in range(n):
            # Normalise error by weight ** 2
            weight = precision(1 / (weights[j] * weights[j]))
            acc += arr[j] * arr[j] * weight
        return acc
    return static_weighted_l2_squared

def l2_squared_norm_factory(precision, n):
    precision = from_dtype(precision)

    # step sizes and norms can be approximate - fastmath is fine
    @cuda.jit(device=True, inline=True)
    def norm_l2_squared(arr):
        acc = precision(0.0)
        for j in range(n):
            acc += arr[j] * arr[j]
        return acc

    return norm_l2_squared
