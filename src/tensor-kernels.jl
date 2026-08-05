## Tensor Kernels

"""
Tensor kernels for Generalized CP Decomposition.
"""
module TensorKernels

using Compat: allequal
using LinearAlgebra: mul!, rmul!
using SparseArrays: AbstractSparseMatrix, sparse
using SparseArrayKit: SparseArray, nonzero_length, nonzero_keys, nonzero_values
using Base.Cartesian: @nloops, @ntuple
using StaticArrays: MVector
#using SparseTensors: AbstractSparseTensor, numstored, storedindices, storedvalues
export create_mttkrp_buffer, mttkrp, mttkrp!, mttkrps, mttkrps!, khatrirao, khatrirao!, checksym, sparse_mttkrp!, sparse_mttkrps!, 
    symmetric_mttkrp!, create_symmetric_mttkrp_buffer, symmetric_kr!, symmetric_self_kr!, symmetric_kr_unweighted!, symmetric_self_kr_unweighted!
    
include("tensor-kernels/khatrirao.jl")
include("tensor-kernels/mttkrp.jl")
include("tensor-kernels/mttkrps.jl")
include("tensor-kernels/checksym.jl")

end
