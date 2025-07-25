# Improvement: Currently forcing vectors are passed as an array, whereas a device function would be more flexible.
#  The option of either would be more flexible again.
# TODO [$6873084f7cdbf00008a72cfe]: CUDA-wide edits:
#  - Add a stream argument propogated down through whatever layers require it.

# TODO: "In other words: if your dict has a fixed and known set of keys, it is an object, not a hash. So if you never
## iterate over the keys of a dict, you should use a proper class." Take guidance from attrs - make fixed-field items
# like compile_settings into classes