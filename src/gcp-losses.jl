## Loss function types

"""
Loss functions for Generalized CP Decomposition.
"""
module GCPLosses

using ..GCPDecompositions
using ..TensorKernels: mttkrps!, mttkrp, mttkrp!, sparse_mttkrp!, sparse_mttkrps!, checksym, khatrirao, symmetric_mttkrp!, symmetric_kr!
using IntervalSets: Interval
using LinearAlgebra: mul!, rmul!, Diagonal, norm
using SparseArrayKit: SparseArray, nonzero_keys, nonzero_values
using StaticArrays: MVector
using Base.Cartesian: @nloops, @ntuple, @ncall
using Combinatorics: with_replacement_combinations
import ForwardDiff

# Abstract type

"""
    AbstractLoss

Abstract type for GCP loss functions ``f(x,m)``,
where ``x`` is the data entry and ``m`` is the model entry.

Concrete types `ConcreteLoss <: AbstractLoss` should implement:

  - `value(loss::ConcreteLoss, x, m)` that computes the value of the loss function ``f(x,m)``
  - `deriv(loss::ConcreteLoss, x, m)` that computes the value of the partial derivative ``\\partial_m f(x,m)`` with respect to ``m``
  - `domain(loss::ConcreteLoss)` that returns an `Interval` from IntervalSets.jl defining the domain for ``m``
"""
abstract type AbstractLoss end

"""
    value(loss, x, m)

Compute the value of the (entrywise) loss function `loss`
for data entry `x` and model entry `m`.
"""
function value end

"""
    deriv(loss, x, m)

Compute the derivative of the (entrywise) loss function `loss`
at the model entry `m` for the data entry `x`.
"""
function deriv end

"""
    domain(loss)

Return the domain of the (entrywise) loss function `loss`.
"""
function domain end

# Objective function and gradients

"""
    objective(M::CPD, X::AbstractArray, loss)

Compute the GCP objective function for the model tensor `M`, data tensor `X`,
and loss function `loss`.
"""
function objective(M::CPD{T,N}, X::Array{TX,N}, loss) where {T,TX,N}
    return sum(value(loss, X[I], M[I]) for I in CartesianIndices(X) if !ismissing(X[I]))
end

"""
    objective(M::SymCPD, X::AbstractArray, loss)

Compute the symmetric GCP objective function for the symmetric model tensor `M`, data tensor `X`,
and loss function `loss`.
"""
function objective(M::SymCPD{T,N}, X::Array{TX,N}, loss, γ) where {T,TX,N}
    return sum(value(loss, X[I], M[I]) for I in CartesianIndices(X) if !ismissing(X[I])) + γ * sum(sum((norm(M.U[k][:, r])^2 - 1)^2 for r in 1:ncomps(M)) for k in 1:ngroups(M))
end

"""
    grad_U!(GU, M::CPD, X::AbstractArray, loss)

Compute the GCP gradient with respect to the factor matrices `U = (U[1],...,U[N])`
for the model tensor `M`, data tensor `X`, and loss function `loss`, and store
the result in `GU = (GU[1],...,GU[N])`.
"""
function grad_U!(
    GU::NTuple{N,TGU},
    M::CPD{T,N},
    X::Array{TX,N},
    loss,
) where {T,TX,N,TGU<:AbstractMatrix{T}}
    Y = [
        ismissing(X[I]) ? zero(nonmissingtype(eltype(X))) : deriv(loss, X[I], M[I]) for
        I in CartesianIndices(X)
    ]
    mttkrps!(GU, Y, M.U)
    for k in 1:N
        rmul!(GU[k], Diagonal(M.λ))
    end
    return GU
end

"""
    grad_U_λ!(GU_λ, M::SymCPD, X::AbstractArray, loss, sym_data, γ)

Compute the SymGCP gradient with respect to the factor matrices `U = (U[1],...,U[N])` and the 
weights `λ` for the model tensor `M`, data tensor `X`, and loss function `loss`, and store
the result in `GU_λ = (GU[1],...,GU[K], Gλ)`. Simplify gradients for symmetry of model tensor matching 
symmetry of data tensor if sym_data is true. γ controls the strength of the (column-norm - 1) regularization.
"""
function grad_U_λ!(
    GU_λ::Tuple,
    M::SymCPD{T,N,K},
    X::Array{TX,N},
    loss,
    sym_data,
    γ,
) where {T,TX,N,K}

    missing_or_deriv(x, m) = ismissing(x) ? zero(nonmissingtype(typeof(x))) : deriv(loss, x, m)
    Y = Array(convertCPD(M))
    Y .= missing_or_deriv.(X, Y)

    # Weights gradient
    GU_λ[K+1] .= khatrirao([M.U[k] for k in reverse(M.S)]...)' * vec(Y)

    # Factor matrix gradients
    for j in 1:K
        if sym_data
            mttkrp!(GU_λ[j], Y, tuple([M.U[k] for k in M.S]...), findall(M.S .== j)[1])
            rmul!(GU_λ[j], count(M.S .== j))
        else
            for (index, mode) in enumerate(findall(M.S .== j))
                if index == 1  # Overwrite
                    mttkrp!(GU_λ[j], Y, tuple([M.U[k] for k in M.S]...), mode)
                else  # Add in-place
                    added_factor = similar(GU_λ[j])
                    mttkrp!(added_factor, Y, tuple([M.U[k] for k in M.S]...), mode)
                    GU_λ[j] .= GU_λ[j] + added_factor
                end
            end
        end
        rmul!(GU_λ[j], Diagonal(M.λ))
        GU_λ[j] .+= mapslices(x -> 4γ * (norm(x)^2 - 1) * x, M.U[j]; dims=1)
    end

    return GU_λ
end

"""
    grad_U_λ_symmetric!(GU_λ, M::SymCPD, X::AbstractArray, idx_map_mats::NTuple, loss, γ)

Compute the SymGCP gradient with respect to the factor matrices `U = (U[1],...,U[N])` and the 
weights `λ` for the model tensor `M`, data tensor `X`, and loss function `loss`, exploiting symmetry
for more efficient computation. Stores the result in `GU_λ = (GU[1],...,GU[K], Gλ)`. 
idx_map_mats should contain a matrix for each symmetric cell which maps to reduced linear indices. 
Use form_reduced_linear_mapping_matrix in symcpd.jl for construction of these matrices.
Note that this function assumes that X and M have matching symmetry. 
γ controls the strength of the (column-norm - 1) regularization.
"""
function grad_U_λ_symmetric!(
    GU_λ::Tuple,
    M::SymCPD{T,N,K},
    X::Array{TX,N},
    idx_map_mats::NTuple{K, AbstractMatrix},
    loss,
    γ,
) where {T,TX,N,K}

    # Weights gradient
    vec_size = prod(k -> prod(i -> size(M.U[k],1)+i-1, 1:count(M.S .== k))÷factorial(count(M.S .== k)), unique(M.S))
    Y_vec = similar(X, vec_size)
    fill_reduced_Y_vec_version_1b!(Y_vec, X, M, loss, Val(N), Val(ncomps(M)))
    # fill_reduced_Y_vec_version_2c!(Y_vec, X, M, loss)
    kr_tilde = similar(M.U[1], vec_size, ncomps(M))
    flip_group_ordering(k) = ngroups(M) - k + 1
    GU_λ[K+1] .= symmetric_kr!(kr_tilde, reverse(flip_group_ordering.(M.S)), reverse(M.U)...)' * Y_vec

    # Factor matrix gradients
    for j in 1:K
        mode = findall(M.S .== j)[1]
        S_reduced = M.S[setdiff(1:N,mode)]
        # Form reduced matricization, splitting out some special cases
        if count(M.S .== j) == 1 && mode == 1
            Y_mat = reshape(Y_vec, size(X,mode), :)
        else
            mat_size = prod(k -> prod(i -> size(M.U[k],1)+i-1, 1:count(S_reduced .== k))÷factorial(count(S_reduced .== k)), unique(S_reduced))
            Y_mat = similar(X, size(X, mode), mat_size)
            if count(M.S .== j) == 1 && mode == N
                vec_idx = 1
                for row in 1:size(X, mode)
                    Y_mat[row,:] = Y_vec[vec_idx:vec_idx+mat_size-1]
                    vec_idx += mat_size
                end
            else
                fill_reduced_Y_mode_n!(Y_mat, Y_vec, idx_map_mats[j])
            end
        end
        symmetric_mttkrp!(GU_λ[j], Y_mat, M.U, M.S, mode)
        rmul!(GU_λ[j], Diagonal(M.λ))
        GU_λ[j] .+= mapslices(x -> 4γ * (norm(x)^2 - 1) * x, M.U[j]; dims=1)
    end

    return GU_λ
end

struct SymmetricIndices{N}
    sz::Int
end
function Base.iterate(iter::SymmetricIndices{N}) where {N}
    out = CartesianIndex(ntuple(i -> 1, N))
    return out, out
end
function Base.iterate(iter::SymmetricIndices{N}, index::CartesianIndex{N}) where {N}
    j = findfirst(!=(iter.sz), Tuple(index))
    j === nothing && return nothing
    out = CartesianIndex(ntuple(k -> k > j ? index[k] : k == j ? index[j] + 1 : index[j]+1, N))
    return out, out
end
Base.length(iter::SymmetricIndices{N}) where {N} = binomial(iter.sz+N-1, N)
Base.eltype(iter::SymmetricIndices{N}) where {N} = CartesianIndex{N}

"""
    fill_reduced_Y_vec_version_1a!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N}) where {N}

Forms reduced vectorization of derivative tensor Y where duplicate entries due to symmetry are removed,
using @nloops macro.
"""
@generated function fill_reduced_Y_vec_version_1a!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N}) where {N}
    set_idx = [:(tensor_idx[$k] = $(Symbol("i_$k"))) for k in 1:N]
    quote
        tensor_idx = zeros(MVector{$N, Int})
        T = eltype(M.U[1])
        vec_idx = 1
        @inbounds @nloops $N i k -> (k == $N ? 1 : M.S[k+1] == M.S[k] ? i_{k+1} : 1):size(M.U[M.S[k]], 1) begin
            $(set_idx...)
            x = X[tensor_idx...]
            Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, M[tensor_idx...])
            vec_idx += 1
        end
    end
end
@generated function fill_reduced_Y_vec_version_1a_directM!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N}, ::Val{R}) where {N,R}
    set_idx = [:(tensor_idx[$k] = $(Symbol("i_$k"))) for k in 1:N]
    Us_exprs = [:(M.U[M.S[$l]]) for l in 1:N]
    quote
        tensor_idx = zeros(MVector{$N, Int})
        T = eltype(M.U[1])
        Us = tuple($(Us_exprs...)) 
        vec_idx = 1
        @inbounds @nloops $N i k -> (k == $N ? 1 : M.S[k+1] == M.S[k] ? i_{k+1} : 1):size(M.U[M.S[k]], 1) begin
            $(set_idx...)
            x = X[tensor_idx...]
            m = zero(T)
            for j in 1:$R
                p = M.λ[j]
                for l in 1:$N
                    p *= Us[l][tensor_idx[l], j]
                end
                m += p
            end
            Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, m)
            vec_idx += 1
        end
    end
end
@generated function fill_reduced_Y_vec_version_1a_directM_rank1fullsym!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N}) where {N}
    set_idx = [:(tensor_idx[$k] = $(Symbol("i_$k"))) for k in 1:N]
    quote
        tensor_idx = zeros(MVector{$N, Int})
        T = eltype(M.U[1])
        vec_idx = 1
        @inbounds @nloops $N i k -> (k == $N ? 1 : M.S[k+1] == M.S[k] ? i_{k+1} : 1):size(M.U[M.S[k]], 1) begin
            $(set_idx...)
            x = X[tensor_idx...]
            m = M.λ[1] * Base.Cartesian.@ncall $N (*) k -> M.U[1][i_k,1]
            Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, m)
            vec_idx += 1
        end
    end
end
@generated function fill_reduced_Y_vec_version_1a_test_iter!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N}) where {N}
    set_idx = [:(tensor_idx[$k] = $(Symbol("i_$k"))) for k in 1:N]
    quote
        tensor_idx = zeros(MVector{$N, Int})
        vec_idx = 1
        @nloops $N i k -> (k == $N ? 1 : M.S[k+1] == M.S[k] ? i_{k+1} : 1):size(M.U[M.S[k]], 1) begin
            $(set_idx...)
            @inbounds Y_vec[vec_idx] = tensor_idx[1]
            vec_idx += 1
        end
    end
end

"""
    fill_reduced_Y_vec_version_1b!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N}, ::Val{R}) where {N,R}

Forms reduced vectorization of derivative tensor Y where duplicate entries due to symmetry are removed,
using @nloops macro, computing partial products for all modes except the first for efficiency.
"""
@generated function fill_reduced_Y_vec_version_1b!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N}, ::Val{R}) where {N,R}
    set_idx = [:(tensor_idx[$(k+1)] = $(Symbol("i_$k"))) for k in 0:N-1]
    set_partial_m = map(1:R) do j
        terms = [:(M.U[M.S[$k]][$(Symbol("i_$(k-1)")), $j]) for k in 2:N]
        :(partial_m[$j] = M.λ[$j] * *( $(terms...) ))
    end
    quote
        tensor_idx = zeros(MVector{$N, Int})
        T = eltype(M.U[1])
        partial_m = zeros(MVector{$R, T})
        mode1_factors = M.U[M.S[1]]
        vec_idx = 1
        @inbounds @nloops $(N-1) i k -> (k == $(N-1) ? 1 : M.S[k+2] == M.S[k+1] ? i_{k+1} : 1):size(M.U[M.S[k+1]], 1) begin
            $(set_partial_m...) 
            for i_0 in (M.S[2] == M.S[1] ? i_1 : 1):size(M.U[M.S[1]], 1)
                $(set_idx...)
                x = X[tensor_idx...]
                m = zero(T)
                for r in 1:$R
                    m += mode1_factors[i_0,r] * partial_m[r]
                end 
                Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, m)
                vec_idx += 1
            end
        end
    end
end

"""
    fill_reduced_Y_vec_version_1c!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N}, ::Val{R}) where {N,R}

Forms reduced vectorization of derivative tensor Y where duplicate entries due to symmetry are removed,
using @nloops macro, computing partial products for efficiency.
"""
@generated function fill_reduced_Y_vec_version_1c!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N}, ::Val{R}) where {N,R}
    set_idx = vcat(
        [:(tensor_idx[1] = i_mode1)],
        [:(tensor_idx[2] = i_mode2)],
        [:(tensor_idx[$(k+2)] = $(Symbol("i_$k"))) for k in 1:N-2]
    )
    set_partial_m_mode2 = map(1:R) do j
        terms = [:(M.U[M.S[$k]][$(Symbol("i_$(k-2)")), $j]) for k in 3:N]
        :(partial_m_mode2[$j] = M.λ[$j] * *( $(terms...) ))
    end
    set_partial_m_mode1 = map(1:R) do j
        :(partial_m_mode1[$j] = partial_m_mode2[$j] * M.U[M.S[2]][i_mode2, $j])
    end
    quote
        tensor_idx = zeros(MVector{$N, Int})
        T = eltype(M.U[1])
        partial_m_mode1 = zeros(MVector{$R, T})
        partial_m_mode2 = zeros(MVector{$R, T})
        mode1_factors = M.U[M.S[1]]
        vec_idx = 1
        @inbounds @nloops $(N-2) i k -> (k == $(N-2) ? 1 : M.S[k+3] == M.S[k+2] ? i_{k+1} : 1):size(M.U[M.S[k+2]], 1) begin
            $(set_partial_m_mode2...) 
            for i_mode2 in (M.S[3] == M.S[2] ? i_1 : 1):size(M.U[M.S[2]], 1)
                $(set_partial_m_mode1...) 
                for i_mode1 in (M.S[2] == M.S[1] ? i_mode2 : 1):size(M.U[M.S[1]], 1)
                    $(set_idx...)
                    x = X[tensor_idx...]
                    m = zero(T)
                    for r in 1:$R
                        m += mode1_factors[i_mode1,r] * partial_m_mode1[r]
                    end 
                    Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, m)
                    vec_idx += 1
                end
            end
        end
    end
end

"""
    fill_reduced_Y_vec_version_2a!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss) where {T,N,K}

Forms reduced vectorization of derivative tensor Y where duplicate entries due to symmetry are removed,
using symmetric indices iterator.
"""
function fill_reduced_Y_vec_version_2a!(Y_vec::AbstractVector, X::Array{T,N}, M::SymCPD{T,N,K}, loss) where {T,N,K}
    sym_block_sizes = ntuple(k -> count(==(k), M.S), Val(K))
    inds = Iterators.product(
        ntuple(k -> SymmetricIndices{sym_block_sizes[k]}(size(M.U[k], 1)), Val(K))...
    )
    _fill_reduced_Y_vec_version_2a!(Y_vec, X, M, loss, inds)
end
function _fill_reduced_Y_vec_version_2a!(Y_vec, X::Array{T,N}, M, loss, inds) where {T,N}
    for (vec_idx, tensor_idx) in enumerate(inds)
        I = CartesianIndex(tensor_idx...)
        x = X[I]
        m = M[I]
        @inbounds Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, m)
    end
end

function fill_reduced_Y_vec_version_2a_directM!(Y_vec::AbstractVector, X::Array{T,N}, M::SymCPD{T,N,K}, loss) where {T,N,K}
    sym_block_sizes = ntuple(k -> count(==(k), M.S), Val(K))
    inds = Iterators.product(
        ntuple(k -> SymmetricIndices{sym_block_sizes[k]}(size(M.U[k], 1)), Val(K))...
    )
    _fill_reduced_Y_vec_version_2a_directM!(Y_vec, X, M, loss, inds, Val(ncomps(M)))
end
function _fill_reduced_Y_vec_version_2a_directM!(Y_vec, X::Array{T,N}, M, loss, inds, ::Val{R}) where {T,N,R}
    Us = ntuple(l -> M.U[M.S[l]], N) 
    @inbounds for (vec_idx, tensor_idx) in enumerate(inds)
        I = CartesianIndex(tensor_idx...)
        x = X[I]
        m = zero(T)
        for j in 1:R
            p = one(T)
            for l in 1:N
                p *= Us[l][I[l],j]
            end
            p *= M.λ[j]
            m += p
        end
        Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, m)
    end
end
function fill_reduced_Y_vec_version_2a_directM_rank1fullsym!(Y_vec::AbstractVector, X::Array{T,N}, M::SymCPD{T,N,K}, loss) where {T,N,K}
    sym_block_sizes = ntuple(k -> count(==(k), M.S), Val(K))
    inds = Iterators.product(
        ntuple(k -> SymmetricIndices{sym_block_sizes[k]}(size(M.U[k], 1)), Val(K))...
    )
    _fill_reduced_Y_vec_version_2a_directM_rank1fullsym!(Y_vec, X, M, loss, inds)
end
function _fill_reduced_Y_vec_version_2a_directM_rank1fullsym!(Y_vec, X::Array{T,N}, M, loss, inds) where {T,N}
    @inbounds λ1 = M.λ[1]
    @inbounds U = M.U[1]
    @inbounds for (vec_idx, tensor_idx) in enumerate(inds)
        I = CartesianIndex(tensor_idx...)
        x = X[I]
        m = one(T)
        for l in 1:N
            m *= U[I[l],1]
        end
        m *= λ1
        Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, m)
    end
end

# function fill_reduced_Y_vec_version_2a_test_iter!(Y_vec::AbstractVector, X::Array{T,N}, M::SymCPD{T,N,K}, loss) where {T,N,K}
#     sym_block_sizes = ntuple(k -> count(==(k), M.S), Val(K))
#     inds = Iterators.product(
#         ntuple(k -> SymmetricIndices{sym_block_sizes[k]}(size(M.U[k], 1)), Val(K))...
#     )
#     _fill_reduced_Y_vec_version_2a_test_iter!(Y_vec, X, M, loss, inds)
# end
# function fill_reduced_Y_vec_version_2a_test_iter!(Y_vec, X::Array{T,N}, M, loss, inds) where {T,N}
#     for (vec_idx, tensor_idx) in enumerate(inds)
#         I = CartesianIndex(tensor_idx...)
#         @inbounds Y_vec[vec_idx] = I[1]
#     end
# end
# function _fill_reduced_Y_vec_version_2a_test_iter!(Y_vec, X::Array{T,N}, M, loss, inds) where {T,N}
#     for (vec_idx, tensor_idx) in enumerate(inds)
#         I = CartesianIndex(tensor_idx...)
#         @inbounds Y_vec[vec_idx] = I[1]
#     end
# end
# function fill_reduced_Y_vec_version_2a_test_iter_array!(Y_vec, X::Array{T,N}, M, loss, inds_array) where {T,N}
#     for (vec_idx, I) in enumerate(inds_array)
#         @inbounds Y_vec[vec_idx] = I[1]
#     end
# end
# function collect_symmetric_inds(inds)
#     inds_array = Vector{CartesianIndex{3}}(undef, length(inds))
#     for (idx, ind) in enumerate(inds)
#         inds_array[idx] = CartesianIndex(ind...)
#     end
#     return inds_array
# end


"""
    fill_reduced_Y_vec_version_2b!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss) where {T,N,K}

Forms reduced vectorization of derivative tensor Y where duplicate entries due to symmetry are removed,
using symmetric indices iterator and computing partial products for all but the first mode, using control
flow in the iterator.
"""
function fill_reduced_Y_vec_version_2b!(Y_vec::AbstractVector, X::Array{T,N}, M::SymCPD{T,N,K}, loss) where {T,N,K}
    sym_block_sizes = ntuple(k -> count(==(k), M.S), Val(K))
    inds = Iterators.product(
        ntuple(k -> SymmetricIndices{sym_block_sizes[k]}(size(M.U[k], 1)), Val(K))...
    )
    _fill_reduced_Y_vec_version_2b!(Y_vec, X, M, loss, inds, Val(ncomps(M)))
end
function _fill_reduced_Y_vec_version_2b!(Y_vec, X::Array{T,N}, M, loss, inds, ::Val{R}) where {T,N,R}    
    Us = ntuple(k -> M.U[M.S[k]], Val(N))
    partial_m = zeros(MVector{R, eltype(Us[1])})
    tail_prev = ntuple(_ -> 0, Val(N-1))
    @inbounds for (vec_idx, tensor_idx) in enumerate(inds)
        I = CartesianIndex(tensor_idx...)

        # Update partial product if index other than first one changes
        tail = ntuple(i -> I[i+1], Val(N-1))
        if tail != tail_prev
            for j in 1:R
                p = M.λ[j]
                for l in 2:N
                    p *= Us[l][I[l],j]
                end
                partial_m[j] = p
            end
            tail_prev = tail
        end

        m = zero(eltype(Us[1]))
        for j in 1:R
            m += Us[1][I[1],j] * partial_m[j]
        end 

        x = X[I]
        Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, m)
    end
end

"""
    fill_reduced_Y_vec_version_2c!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss) where {T,N,K}

Forms reduced vectorization of derivative tensor Y where duplicate entries due to symmetry are removed,
using symmetric indices iterator and computing partial products for all but the first mode, by splitting
out separate inner loop for first mode.
"""
function fill_reduced_Y_vec_version_2c!(Y_vec::AbstractVector, X::Array{T,N}, M::SymCPD{T,N,K}, loss) where {T,N,K}
    sym_block_sizes = ntuple(k -> count(==(k), M.S), Val(K))
    # Split out indices for mode 1
    if sym_block_sizes[1] == 1
        inds_minus_mode1 = Iterators.product(
            ntuple(k -> SymmetricIndices{sym_block_sizes[k+1]}(size(M.U[k+1], 1)), Val(K-1))...
    )
    else
        inds_minus_mode1 = Iterators.product(
            ntuple(k -> k == 1 
                ? SymmetricIndices{sym_block_sizes[k]-1}(size(M.U[k], 1))
                : SymmetricIndices{sym_block_sizes[k]}(size(M.U[k], 1)), Val(K))...
        )
    end
    _fill_reduced_Y_vec_version_2c!(Y_vec, X, M, loss, inds_minus_mode1, Val(ncomps(M)))
end
function _fill_reduced_Y_vec_version_2c!(Y_vec, X::Array{T,N}, M, loss, inds_minus_mode1, ::Val{R}) where {T,N,R}
    Us = ntuple(k -> M.U[M.S[k]], Val(N))
    partial_m = zeros(MVector{R, eltype(Us[1])})
    mode_1_singleton = M.S[1] != M.S[2]
    mode_1_size = size(X,1) 
    vec_idx = 1
    @inbounds for tensor_idx in inds_minus_mode1
        # Update partial product
        for j in 1:R
            p = M.λ[j]
            for k in 2:N
                p *= Us[k][CartesianIndex(tensor_idx...)[k-1],j]
            end
            partial_m[j] = p
        end
        inner_loop_start = mode_1_singleton ? 1 : CartesianIndex(tensor_idx...)[1]
        for i1 in inner_loop_start:mode_1_size
            I = CartesianIndex(i1, tensor_idx...)
            m = zero(eltype(Us[1]))
            for j in 1:R
                m += Us[1][i1,j] * partial_m[j]
            end 
            x = X[I]
            Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, m)
            vec_idx += 1
        end
    end
end

"""
    fill_reduced_Y_mode_n!(Y_mat::AbstractMatrix, n::Integer, X::Array, M::SymCPD, loss, ::Val{N})

Forms reduced mode-n matricization of derivative tensor Y where duplicate columns due to symmetry are removed,
by copying values from Y_vec using mapping in idx_map_mat.
"""
function fill_reduced_Y_mode_n!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, idx_map_mat::AbstractMatrix)
    @inbounds for I in eachindex(Y_mat, idx_map_mat)
        Y_mat[I] = Y_vec[idx_map_mat[I]]
    end
    return Y_mat
end

"""
    fill_reduced_Y_mode_n!(Y_mat::AbstractMatrix, n::Integer, X::Array, M::SymCPD, loss, ::Val{N})

Forms reduced mode-n matricization of derivative tensor Y where duplicate columns due to symmetry are removed,
using data tensor X, model tensor M, and loss function loss.
"""
@generated function fill_reduced_Y_mode_n!(Y_mat::AbstractMatrix, n::Integer, X::Array, M::SymCPD, loss, ::Val{N}) where {N}
    set_idx = [:(idx[col_inds_pos[$k]] = $(Symbol("i_$k"))) for k in 1:N-1]
    quote
        num_rows = size(X,n)
        col_inds_pos = setdiff(1:$N,n)
        S_reduced = M.S[col_inds_pos]
        idx = zeros(MVector{$N, Int})
        col = 1
        @nloops $(N-1) i k -> (k == $(N-1) ? 1 : S_reduced[k+1] == S_reduced[k] ? i_{k+1} : 1):size(M.U[S_reduced[k]], 1) begin
            $(set_idx...)
            for row in 1:num_rows
                idx[n] = row
                x = X[idx...]
                Y_mat[row,col] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, M[idx...])
            end
            col += 1
        end
    end
end

"""
    fill_reduced_Y_mode_n!(Y_mat::AbstractMatrix, n::Integer, X::Array, M::SymCPD, loss, ::Val{N})

Forms reduced mode-n matricization of derivative tensor Y where duplicate columns due to symmetry are removed,
using data tensor X, model tensor M, dense model tensor M_array and loss function loss. Accesses values of M_array
rather than M.
"""
@generated function fill_reduced_Y_mode_n!(Y_mat::AbstractMatrix, n::Integer, X::Array, M::SymCPD, M_array::Array, loss, ::Val{N}) where {N}
    set_idx = [:(idx[col_inds_pos[$k]] = $(Symbol("i_$k"))) for k in 1:N-1]
    quote
        num_rows = size(X,n)
        col_inds_pos = setdiff(1:$N,n)
        S_reduced = M.S[col_inds_pos]
        idx = zeros(MVector{$N, Int})
        col = 1
        @nloops $(N-1) i k -> (k == $(N-1) ? 1 : S_reduced[k+1] == S_reduced[k] ? i_{k+1} : 1):size(M.U[S_reduced[k]], 1) begin
            $(set_idx...)
            for row in 1:num_rows
                idx[n] = row
                I = CartesianIndex(Tuple(idx))
                x = X[I]
                Y_mat[row,col] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, M_array[I])
            end
            col += 1
        end
    end
end

"""
    fill_reduced_Y_mode1_mat_from_vec_order3!(Y_mode1_mat::AbstractMatrix, Y_vec::AbstractVector)
Fill in values of Y_mode1_mat from values in Y_vec with specialized algorithm for fully symmetric order 3 tensors.
"""
function fill_reduced_Y_mode1_mat_from_vec_order3_fullsym!(Y_mode1_mat::AbstractMatrix, Y_vec::AbstractVector)
    sz = size(Y_mode1_mat,1)
    start_col = 1
    start_row = 1
    vec_idx = 1
    for block in 1:sz
        block_size = sz-block+1
        # Fill in unique, then duplciate entries below first row in symmetric sub-matrices
        for (col_idx, col) in enumerate(start_col:start_col+block_size)
            for (row_idx, row) in enumerate(start_row+col_idx-1:sz)
                Y_mode1_mat[row,col] = Y_vec[vec_idx]
                if row_idx != 1 && col_idx != 1 && col_idx != sz  # Skip top row which is handled later
                    Y_mode1_mat[start_row+col_idx-1, col+row_idx-1] = Y_vec[vec_idx]
                end
                vec_idx += 1
            end
        end
        start_col += block_size
        start_row = block + 1
    end

    # Fill in remaining rows above sub-matrices
    row = 1
    start_col = 2
    num_cols = prod(i -> sz+i-1, 1:2)÷2
    vec_idx = 2
    for block in 1:sz
        block_size = sz-block+1
        for col in start_col:num_cols
            Y_mode1_mat[row, col] = Y_vec[vec_idx]
            vec_idx += 1
        end
        row += 1
        start_col += block_size
        vec_idx += 1 # Skip entry already filled in
    end
end

insert_row(row, t::Tuple{}) = (row,)
insert_row(row, t::Tuple) = 
    row >= t[1] ? (row, t...) : (t[1], insert_row(row, t[2:end])...)
"""
    fill_reduced_Y_mode_n_from_vec_fullsym_orderN!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, n, M::SymCPD, ::Val{N}) where {N}

    Fill out values of Y_mat from values of Y_vec, using specialized algorithm for fully symmetric tensors.
"""
@generated function fill_reduced_Y_mode_n_from_vec_fullsym_orderN!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, n, M::SymCPD, ::Val{N}) where {N}
    quote
        sz = size(Y_mat, 1)
        col_inds_pos = setdiff(1:$N,n)
        S_reduced = M.S[col_inds_pos]
        vec_idx = 1
        col = 1
        col_starts = Array{Int}(undef, ntuple(Returns(sz), $(N-2)))
        @nloops $(N-2) i k -> (k == $(N-2) ? 1 : S_reduced[k+1] == S_reduced[k] ? i_{k+1} : 1):size(M.U[S_reduced[k]], 1) begin
            @ncall $(N-2) setindex! col_starts col i
            for j in $(Symbol("i_1")):sz
                for row in 1:sz
                    if row >= j 
                        Y_mat[row,col] = Y_vec[vec_idx]
                        vec_idx += 1
                    else
                        # Sort indices lexicographically
                        idx = insert_row(row, (j, (@ntuple $(N-2) i)...))
                        # Copy from corresponding filled out entry
                        target_col = col_starts[idx[3:end]...] + idx[2] - idx[3] # idx[2] - idx[3] is how many times j has been incremented since col_starts[idx[3:end]...]
                        Y_mat[row, col] = Y_mat[idx[1], target_col]
                    end
                end
                col += 1
            end
        end
    end
end

"""
    stochastic_grad_U_λ!(GU_λ, M::SymCPD, X::AbstractArray, loss, B)

Compute the SymGCP gradient with respect to the factor matrices `U = (U[1],...,U[N])` and the 
weights `λ` for the model tensor `M`, elements of the data tensor `X` with indices given by B, and loss function `loss`, and store
the result in `GU_λ = (GU[1],...,GU[K], Gλ)`. Simplify gradients for symmetry of model tensor matching 
symmetry of data tensor if sym_data is true. γ controls the strength of the (column-norm - 1) regularization.
    p - number of nonzero elements in batch
    q - number of zero elements in batch
"""
function stochastic_grad_U_λ!(
    GU_λ::Tuple,
    M::SymCPD{T,N,K},
    X::Array{TX,N},
    loss,
    sym_data,
    γ,
    B,
    sampling_strategy;
    p=1,
    q=1
) where {T,TX,N,K}
    
    η = count(!iszero, X)
    ζ = length(X) - η
    ω = length(X)

    # Initialize sparse subsampled derivative tensor
    inds = unique(B)
    Y = SparseArray{T,N}(Dict([(idx, zero(T)) for idx in inds]), size(X))

    # Compute bias-corrected derivatives
    for (i,idx) in enumerate(B)
        if sampling_strategy == "uniform"
            Y[idx] += (ω / length(B)) * deriv(loss, X[idx], M[idx])
        elseif sampling_strategy == "stratified"
            # First p entries of B are nonzeros, remaining q entries are zeros
            if i <= p
                Y[idx] += (η / p) * deriv(loss, X[idx], M[idx])
            else
                Y[idx] += (ζ / q) * deriv(loss, X[idx], M[idx])
            end
        elseif sampling_strategy == "semi-stratified"
            # First p entries of B are nonzeros, remaining q entries are possible zeros
            if i <= p
                Y[idx] += (η / p) * (deriv(loss, X[idx], M[idx]) - deriv(loss, zero(T), M[idx]))
            else
                Y[idx] += (ω / q) * deriv(loss, zero(T), M[idx])
            end
        else
            error(
                "The only supported sampling strategies are uniform and stratified",
            )
        end
    end

    # Factor matrix gradients
    Us = tuple([M.U[k] for k in M.S]...)

    # Compute mttkrp for each mode
    mode_GUs = similar.(Us)
    sparse_mttkrps!(mode_GUs, Y, Us)

    for j in 1:K
        if sym_data
            first_n = findall(M.S .== j)[1]
            GU_λ[j] .= mode_GUs[first_n]
            rmul!(GU_λ[j], count(M.S .== j))
        else
            for (index, mode) in enumerate(findall(M.S .== j))
                if index == 1  # Overwrite
                    GU_λ[j] .= mode_GUs[mode]
                else  # Add in-place
                    GU_λ[j] .+= mode_GUs[mode]
                end
            end
        end
        rmul!(GU_λ[j], Diagonal(M.λ))
        if !iszero(γ)
            GU_λ[j] .+= mapslices(x -> 4γ * (norm(x)^2 - 1) * x, M.U[j]; dims=1)
        end
    end

    # Weights gradient
    inds, vals = nonzero_keys(Y), nonzero_values(Y)
	Uh = reduce(.*, Us[k][getindex.(inds, k), :] for k in eachindex(Us))
    mul!(GU_λ[K+1], Uh', collect(vals))

    return GU_λ
end

# Statistically motivated losses

"""
    LeastSquares()

Loss corresponding to conventional CP decomposition.
Corresponds to a statistical assumption of Gaussian data `X`
with mean given by the low-rank model tensor `M`.

  - **Distribution:** ``x_i \\sim \\mathcal{N}(\\mu_i, \\sigma)``
  - **Link function:** ``m_i = \\mu_i``
  - **Loss function:** ``f(x,m) = (x-m)^2``
  - **Domain:** ``m \\in \\mathbb{R}``
"""
struct LeastSquares <: AbstractLoss end
value(::LeastSquares, x, m) = (x - m)^2
deriv(::LeastSquares, x, m) = 2 * (m - x)
domain(::LeastSquares) = Interval(-Inf, +Inf)

"""
    NonnegativeLeastSquares()

Loss corresponding to nonnegative CP decomposition.
Corresponds to a statistical assumption of Gaussian data `X`
with nonnegative mean given by the low-rank model tensor `M`.

  - **Distribution:** ``x_i \\sim \\mathcal{N}(\\mu_i, \\sigma)``
  - **Link function:** ``m_i = \\mu_i``
  - **Loss function:** ``f(x,m) = (x-m)^2``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct NonnegativeLeastSquares <: AbstractLoss end
value(::NonnegativeLeastSquares, x, m) = (x - m)^2
deriv(::NonnegativeLeastSquares, x, m) = 2 * (m - x)
domain(::NonnegativeLeastSquares) = Interval(0.0, Inf)

"""
    Poisson(eps::Real = 1e-10)

Loss corresponding to a statistical assumption of Poisson data `X`
with rate given by the low-rank model tensor `M`.

  - **Distribution:** ``x_i \\sim \\operatorname{Poisson}(\\lambda_i)``
  - **Link function:** ``m_i = \\lambda_i``
  - **Loss function:** ``f(x,m) = m - x \\log(m + \\epsilon)``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct Poisson{T<:Real} <: AbstractLoss
    eps::T
    Poisson{T}(eps::T) where {T<:Real} =
        eps >= zero(eps) ? new(eps) :
        throw(DomainError(eps, "Poisson loss requires nonnegative `eps`"))
end
Poisson(eps::T = 1e-10) where {T<:Real} = Poisson{T}(eps)
value(loss::Poisson, x, m) = m - x * log(m + loss.eps)
deriv(loss::Poisson, x, m) = one(m) - x / (m + loss.eps)
domain(::Poisson) = Interval(0.0, +Inf)

"""
    PoissonLog()

Loss corresponding to a statistical assumption of Poisson data `X`
with log-rate given by the low-rank model tensor `M`.

  - **Distribution:** ``x_i \\sim \\operatorname{Poisson}(\\lambda_i)``
  - **Link function:** ``m_i = \\log \\lambda_i``
  - **Loss function:** ``f(x,m) = e^m - x m``
  - **Domain:** ``m \\in \\mathbb{R}``
"""
struct PoissonLog <: AbstractLoss end
value(::PoissonLog, x, m) = exp(m) - x * m
deriv(::PoissonLog, x, m) = exp(m) - x
domain(::PoissonLog) = Interval(-Inf, +Inf)

"""
    Gamma(eps::Real = 1e-10)

Loss corresponding to a statistical assumption of Gamma-distributed data `X`
with scale given by the low-rank model tensor `M`.

- **Distribution:** ``x_i \\sim \\operatorname{Gamma}(k, \\sigma_i)``
- **Link function:** ``m_i = k \\sigma_i``
- **Loss function:** ``f(x,m) = \\frac{x}{m + \\epsilon} + \\log(m + \\epsilon)``
- **Domain:** ``m \\in [0, \\infty)``
"""
struct Gamma{T<:Real} <: AbstractLoss
    eps::T
    Gamma{T}(eps::T) where {T<:Real} =
        eps >= zero(eps) ? new(eps) :
        throw(DomainError(eps, "Gamma loss requires nonnegative `eps`"))
end
Gamma(eps::T = 1e-10) where {T<:Real} = Gamma{T}(eps)
value(loss::Gamma, x, m) = x / (m + loss.eps) + log(m + loss.eps)
deriv(loss::Gamma, x, m) = -x / (m + loss.eps)^2 + inv(m + loss.eps)
domain(::Gamma) = Interval(0.0, +Inf)

"""
    Rayleigh(eps::Real = 1e-10)

Loss corresponding to the statistical assumption of Rayleigh data `X`
with sacle given by the low-rank model tensor `M`

  - **Distribution:** ``x_i \\sim \\operatorname{Rayleigh}(\\theta_i)``
  - **Link function:** ``m_i = \\sqrt{\\frac{\\pi}{2}\\theta_i}``
  - **Loss function:** ``f(x, m) = 2\\log(m + \\epsilon) + \\frac{\\pi}{4}(\\frac{x}{m + \\epsilon})^2``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct Rayleigh{T<:Real} <: AbstractLoss
    eps::T
    Rayleigh{T}(eps::T) where {T<:Real} =
        eps >= zero(eps) ? new(eps) :
        throw(DomainError(eps, "Rayleigh loss requires nonnegative `eps`"))
end
Rayleigh(eps::T = 1e-10) where {T<:Real} = Rayleigh{T}(eps)
value(loss::Rayleigh, x, m) = 2 * log(m + loss.eps) + (pi / 4) * ((x / (m + loss.eps))^2)
deriv(loss::Rayleigh, x, m) = 2 / (m + loss.eps) - (pi / 2) * (x^2 / (m + loss.eps)^3)
domain(::Rayleigh) = Interval(0.0, +Inf)

"""
    BernoulliOdds(eps::Real = 1e-10)

Loss corresponding to the statistical assumption of Bernouli data `X`
with odds-sucess rate given by the low-rank model tensor `M`

  - **Distribution:** ``x_i \\sim \\operatorname{Bernouli}(\\rho_i)``
  - **Link function:** ``m_i = \\frac{\\rho_i}{1 - \\rho_i}``
  - **Loss function:** ``f(x, m) = \\log(m + 1) - x\\log(m + \\epsilon)``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct BernoulliOdds{T<:Real} <: AbstractLoss
    eps::T
    BernoulliOdds{T}(eps::T) where {T<:Real} =
        eps >= zero(eps) ? new(eps) :
        throw(DomainError(eps, "BernoulliOdds requires nonnegative `eps`"))
end
BernoulliOdds(eps::T = 1e-10) where {T<:Real} = BernoulliOdds{T}(eps)
value(loss::BernoulliOdds, x, m) = log(m + 1) - x * log(m + loss.eps)
deriv(loss::BernoulliOdds, x, m) = 1 / (m + 1) - (x / (m + loss.eps))
domain(::BernoulliOdds) = Interval(0.0, +Inf)

"""
    BernoulliLogit(eps::Real = 1e-10)

Loss corresponding to the statistical assumption of Bernouli data `X`
with log odds-success rate given by the low-rank model tensor `M`

  - **Distribution:** ``x_i \\sim \\operatorname{Bernouli}(\\rho_i)``
  - **Link function:** ``m_i = \\log(\\frac{\\rho_i}{1 - \\rho_i})``
  - **Loss function:** ``f(x, m) = \\log(1 + e^m) - xm``
  - **Domain:** ``m \\in \\mathbb{R}``
"""
struct BernoulliLogit{T<:Real} <: AbstractLoss
    eps::T
    BernoulliLogit{T}(eps::T) where {T<:Real} =
        eps >= zero(eps) ? new(eps) :
        throw(DomainError(eps, "BernoulliLogitsLoss requires nonnegative `eps`"))
end
BernoulliLogit(eps::T = 1e-10) where {T<:Real} = BernoulliLogit{T}(eps)
value(::BernoulliLogit, x, m) = log(1 + exp(m)) - x * m
deriv(::BernoulliLogit, x, m) = exp(m) / (1 + exp(m)) - x
domain(::BernoulliLogit) = Interval(-Inf, +Inf)

"""
    NegativeBinomialOdds(r::Integer, eps::Real = 1e-10)

Loss corresponding to the statistical assumption of Negative Binomial
data `X` with log odds failure rate given by the low-rank model tensor `M`

  - **Distribution:** ``x_i \\sim \\operatorname{NegativeBinomial}(r, \\rho_i) ``
  - **Link function:** ``m = \\frac{\\rho}{1 - \\rho}``
  - **Loss function:** ``f(x, m) = (r + x) \\log(1 + m) - x\\log(m + \\epsilon) ``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct NegativeBinomialOdds{S<:Integer,T<:Real} <: AbstractLoss
    r::S
    eps::T
    function NegativeBinomialOdds{S,T}(r::S, eps::T) where {S<:Integer,T<:Real}
        eps >= zero(eps) ||
            throw(DomainError(eps, "NegativeBinomialOdds requires nonnegative `eps`"))
        r >= zero(r) ||
            throw(DomainError(r, "NegativeBinomialOdds requires nonnegative `r`"))
        return new(r, eps)
    end
end
NegativeBinomialOdds(r::S, eps::T = 1e-10) where {S<:Integer,T<:Real} =
    NegativeBinomialOdds{S,T}(r, eps)
value(loss::NegativeBinomialOdds, x, m) = (loss.r + x) * log(1 + m) - x * log(m + loss.eps)
deriv(loss::NegativeBinomialOdds, x, m) = (loss.r + x) / (1 + m) - x / (m + loss.eps)
domain(::NegativeBinomialOdds) = Interval(0.0, +Inf)

"""
    Huber(Δ::Real)

  Huber Loss for given Δ

  - **Loss function:** ``f(x, m) = (x - m)^2 if \\abs(x - m)\\leq\\Delta, 2\\Delta\\abs(x - m) - \\Delta^2 otherwise``
  - **Domain:** ``m \\in \\mathbb{R}``
"""
struct Huber{T<:Real} <: AbstractLoss
    Δ::T
    Huber{T}(Δ::T) where {T<:Real} =
        Δ >= zero(Δ) ? new(Δ) : throw(DomainError(Δ, "Huber requires nonnegative `Δ`"))
end
Huber(Δ::T) where {T<:Real} = Huber{T}(Δ)
value(loss::Huber, x, m) =
    abs(x - m) <= loss.Δ ? (x - m)^2 : 2 * loss.Δ * abs(x - m) - loss.Δ^2
deriv(loss::Huber, x, m) =
    abs(x - m) <= loss.Δ ? -2 * (x - m) : -2 * sign(x - m) * loss.Δ * x
domain(::Huber) = Interval(-Inf, +Inf)

"""
    BetaDivergence(β::Real, eps::Real)

    BetaDivergence Loss for given β

  - **Loss function:** ``f(x, m; β) = \\frac{1}{\\beta}m^{\\beta} - \\frac{1}{\\beta - 1}xm^{\\beta - 1}
                          if \\beta \\in \\mathbb{R}  \\{0, 1\\},
                            m - x\\log(m) if \\beta = 1,
                            \\frac{x}{m} + \\log(m) if \\beta = 0``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct BetaDivergence{S<:Real,T<:Real} <: AbstractLoss
    β::T
    eps::T
    BetaDivergence{S,T}(β::S, eps::T) where {S<:Real,T<:Real} =
        eps >= zero(eps) ? new(β, eps) :
        throw(DomainError(eps, "BetaDivergence requires nonnegative `eps`"))
end
BetaDivergence(β::S, eps::T = 1e-10) where {S<:Real,T<:Real} = BetaDivergence{S,T}(β, eps)
function value(loss::BetaDivergence, x, m)
    if loss.β == 0
        return x / (m + loss.eps) + log(m + loss.eps)
    elseif loss.β == 1
        return m - x * log(m + loss.eps)
    else
        return 1 / loss.β * m^loss.β - 1 / (loss.β - 1) * x * m^(loss.β - 1)
    end
end
function deriv(loss::BetaDivergence, x, m)
    if loss.β == 0
        return -x / (m + loss.eps)^2 + 1 / (m + loss.eps)
    elseif loss.β == 1
        return 1 - x / (m + loss.eps)
    else
        return m^(loss.β - 1) - x * m^(loss.β - 2)
    end
end
domain(::BetaDivergence) = Interval(0.0, +Inf)

# User-defined loss
"""
    UserDefined

Type for user-defined loss functions ``f(x,m)``,
where ``x`` is the data entry and ``m`` is the model entry.

Contains three fields:

 1. `func::Function`   : function that evaluates the loss function ``f(x,m)``
 2. `deriv::Function`  : function that evaluates the partial derivative ``\\partial_m f(x,m)`` with respect to ``m``
 3. `domain::Interval` : `Interval` from IntervalSets.jl defining the domain for ``m``

The constructor is `UserDefined(func; deriv, domain)`.
If not provided,

  - `deriv` is automatically computed from `func` using forward-mode automatic differentiation
  - `domain` gets a default value of `Interval(-Inf, +Inf)`
"""
struct UserDefined <: AbstractLoss
    func::Function
    deriv::Function
    domain::Interval
    function UserDefined(
        func::Function;
        deriv::Function = (x, m) -> ForwardDiff.derivative(m -> func(x, m), m),
        domain::Interval = Interval(-Inf, Inf),
    )
        hasmethod(func, Tuple{Real,Real}) ||
            error("`func` must accept two inputs `(x::Real, m::Real)`")
        hasmethod(deriv, Tuple{Real,Real}) ||
            error("`deriv` must accept two inputs `(x::Real, m::Real)`")
        return new(func, deriv, domain)
    end
end
value(loss::UserDefined, x, m) = loss.func(x, m)
deriv(loss::UserDefined, x, m) = loss.deriv(x, m)
domain(loss::UserDefined) = loss.domain

end


function form_sym(X, M, loss)
    sz = size(X,1)
    Y_mode1_mat = similar(X, sz, (sz*(sz+1))÷2)
    # What loop order for cache efficiency?
    for row in 1:sz
        col = 1
        for i in 1:sz
            for j in i:sz
                if i == j
                    Y_mode1_mat[row,col] = ismissing(X[row,i,j]) ? zero(nonmissingtype(eltype(X))) : deriv(loss, X[row,i,j], M[row,i,j])
                else
                    Y_mode1_mat[row,col] = ismissing(X[row,i,j]) ? zero(nonmissingtype(eltype(X))) : 2 * deriv(loss, X[row,i,j], M[row,i,j])
                end
                col += 1
            end
        end
    end
    return Y_mode1_mat
end

function form_nonsym(X, M, loss)
    Y = [
        ismissing(X[I]) ? zero(nonmissingtype(eltype(X))) : deriv(loss, X[I], M[I]) for
        I in CartesianIndices(X)
    ]
    return Y
end

