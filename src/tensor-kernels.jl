## Tensor Kernels

module TensorKernels

using Compat: allequal
using LinearAlgebra: mul!
export create_mttkrp_buffer, mttkrp, mttkrp!, khatrirao, khatrirao!

include("tensor-kernels/khatrirao.jl")
include("tensor-kernels/mttkrp.jl")

end
