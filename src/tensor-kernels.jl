## Tensor Kernels

module TensorKernels

using Compat: allequal
using LinearAlgebra: mul!
export create_mttkrp_buffer, mttkrp, mttkrp!, mttkrps, mttkrps!, khatrirao, khatrirao!

include("tensor-kernels/khatrirao.jl")
include("tensor-kernels/mttkrp.jl")
include("tensor-kernels/mttkrps.jl")

end
