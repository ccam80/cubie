# Improvement: Currently forcing vectors are passed as an array, whereas a device function would be more flexible.
#  The option of either would be more flexible again.
# TODO [$6873084f7cdbf00008a72cfe]: CUDA-wide edits:
#  - Add a thread count to integrator kernels, so that multi-threaded loops are possible (e.g. one thread per dxdt
#  evaluation)
#  - Add a stream argument propogated down through whatever layers require it.
