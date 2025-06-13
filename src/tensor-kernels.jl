## Tensor Kernels

"""
Tensor kernels for Generalized CP Decomposition.
"""
module TensorKernels

using Compat: allequal
using LinearAlgebra: mul!
using SparseArrays: AbstractSparseMatrix
export create_mttkrp_buffer, mttkrp, mttkrp!, mttkrps, mttkrps!, khatrirao, khatrirao!, checksym, sparse_mttkrp!

include("tensor-kernels/khatrirao.jl")
include("tensor-kernels/mttkrp.jl")
include("tensor-kernels/mttkrps.jl")
include("tensor-kernels/checksym.jl")
include("tensor-kernels/sparse_mttkrp.jl")

end
