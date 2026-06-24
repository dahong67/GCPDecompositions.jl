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
    grad_U_λ!(GU, M::SymCPD, X::AbstractArray, loss, sym_data, γ)

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

    if !sym_data
        missing_or_deriv(x, m) = ismissing(x) ? zero(nonmissingtype(typeof(x))) : deriv(loss, x, m)
        Y = Array(convertCPD(M))
        Y .= missing_or_deriv.(X, Y)
    end

    # Weights gradient
    if sym_data
        vec_size = prod(k -> prod(i -> size(M.U[k],1)+i-1, 1:count(M.S .== k))÷factorial(count(M.S .== k)), unique(M.S))
        Y_vec = similar(X, vec_size)
        fill_reduced_Y_vec!(Y_vec, X, M, loss, Val(N))
        kr_tilde = similar(M.U[1], vec_size, ncomps(M))
        flip_group_ordering(k) = ngroups(M) - k + 1
        GU_λ[K+1] .= symmetric_kr!(kr_tilde, reverse(flip_group_ordering.(M.S)), reverse(M.U)...)' * Y_vec
    else
        GU_λ[K+1] .= khatrirao([M.U[k] for k in reverse(M.S)]...)' * vec(Y)
    end

    # Factor matrix gradients
    for j in 1:K
        if sym_data
            mode = findall(M.S .== j)[1]
            S_reduced = M.S[setdiff(1:N,mode)]
            # Form reduced matricization, splitting out some special cases
            if count(M.S .== j) == 1 && mode == 1
                Y_mat = reshape(Y_vec, size(X,mode), :)
            elseif maximum(M.S) >= N-1 && count(M.S .== j) > 1
                Y_mat = reshape(Y, size(X,mode), :)
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
                    fill_reduced_Y_mode_n!(Y_mat, mode, X, M, loss, Val(N))
                end
            end
            symmetric_mttkrp!(GU_λ[j], Y_mat, M.U, M.S, mode)
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
    fill_reduced_Y_vec!(Y_vec, X, M, loss, Val(N))
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

"""
    fill_reduced_Y_mode_n!(Y_mat::AbstractMatrix, n::Integer, X::Array, M::SymCPD, loss, ::Val{N})

Forms reduced mode-n matricization of derivative tensor Y where duplicate columns due to symmetry are removed.
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
# function fill_reduced_Y_mode_n_fullsym!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, mapping_mat::AbstractMatrix)
#     @inbounds for I in eachindex(Y_mat, mapping_mat)
#         Y_mat[I] = Y_vec[mapping_mat[I]]
#     end
#     return Y_mat
# end
function fill_reduced_Y_mode_n!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, idx_map_mat::AbstractMatrix)
    @inbounds for I in eachindex(Y_mat, idx_map_mat)
        Y_mat[I] = Y_vec[idx_map_mat[I]]
    end
    return Y_mat
end
# @generated function fill_reduced_Y_mode_n_multi_inds!(Y_mat::AbstractMatrix, n::Integer, X::Array, M::SymCPD, loss, ::Val{N}) where {N}
#     set_idx = [:(idx[col_inds_pos[$k]] = $(Symbol("i_$k"))) for k in 1:N-1]
#     quote
#         num_rows = size(X,n)
#         col_inds_pos = setdiff(1:$N,n)
#         S_reduced = M.S[col_inds_pos]
#         idx = zeros(MVector{$N, Int})
#         col = 1
#         @nloops $(N-1) i k -> (k == $(N-1) ? 1 : S_reduced[k+1] == S_reduced[k] ? i_{k+1} : 1):size(M.U[S_reduced[k]], 1) begin
#             $(set_idx...)
#             for row in 1:num_rows
#                 idx[n] = row
#                 Y_mat[row,col] = Tuple(idx)
#             end
#             col += 1
#         end
#     end
# end
# @generated function fill_reduced_Y_mode_n_vec_inds!(Y_mat::AbstractMatrix, n::Integer, X::Array, M::SymCPD, loss, ::Val{N}) where {N}
#     set_idx = [:(idx[col_inds_pos[$k]] = $(Symbol("i_$k"))) for k in 1:N-1]
#     quote
#         num_rows = size(X,n)
#         col_inds_pos = setdiff(1:$N,n)
#         S_reduced = M.S[col_inds_pos]
#         idx = zeros(MVector{$N, Int})
#         col = 1
#         @nloops $(N-1) i k -> (k == $(N-1) ? 1 : S_reduced[k+1] == S_reduced[k] ? i_{k+1} : 1):size(M.U[S_reduced[k]], 1) begin
#             for row in 1:num_rows
#                 $(set_idx...)
#                 idx[n] = row
#                 sort!(idx, rev=true)
#                 Y_mat[row,col] = lin_reduced(idx,num_rows)
#             end
#             col += 1
#         end
#     end
# end

# function lin_reduced(I::NTuple{2, Int}, n::Int)
#     i1, i2 = I
#     lin_idx = 1
#     # Add i2 offset
#     lin_idx += (i2-1)*(n+1) - binomial(i2, 2)
#     # Add i1 offset
#     lin_idx += i1 - i2
#     return lin_idx
# end

# function lin_reduced(I::NTuple{3, Int}, n::Int)
#     i1, i2, i3 = I
#     lin_idx = 1
#     # Add i3 offset
#     for j in 1:(i3-1)
#         lin_idx += binomial(n-j+2, 2)
#     end
#     # Add i2 offset
#     k2 = i2 - i3
#     lin_idx += (n-i3+2)*(k2) - binomial(k2+1, 2)
#     # Add i1 offset
#     lin_idx += i1 - i2
#     return lin_idx
# end

# function lin_reduced(I::NTuple{4,Int}, n::Int)
#     i1, i2, i3, i4 = I
#     lin_idx = 1
#     # Add i1 offset
#     lin_idx += i1 - i2
#     # Add i2 offset
#     lin_idx += (n-i3+2)*(i2-i3) - binomial(i2-i3+1, 2)
#     # Add i3 offset
#     for j in 1:(i3-i4)
#         lin_idx += binomial(n - i4 - j + 3, 2)
#     end
#     # Add i4 offset
#     for j in 1:(i4-1)
#         lin_idx += binomial(n - j + 3, 3)
#     end
#     return lin_idx
# end

# function lin_reduced(I::NTuple{M,Int}, n::Int) where M
#     return binomial(n + M - 1, M) - sum(t -> binomial(n - I[t] + t - 1, t), 1:M)
# end
# function lin_reduced(I::MVector{M,Int}, n::Int) where M
#     return binomial(n + M - 1, M) - sum(t -> binomial(n - I[t] + t - 1, t), 1:M)
# end

# # Fill by traversing Y_vec
# @generated function fill_reduced_Y_mode_n_from_vec_fullsym!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, n, M::SymCPD, ::Val{N}) where {N}
#     set_idx = [:(tensor_idx[$k] = $(Symbol("i_$k"))) for k in 1:N]
#     quote
#         # col_indices_array = make_col_indices_array(Y_mat, M, 1, Val($N));
#         tensor_idx = zeros(MVector{$N, Int})
#         permuted_tensor_idx = zeros(MVector{$(N-1), Int})
#         unique_vals = zeros(MVector{$N, Int})
#         sz = size(M.U[1], 1)
#         vec_idx = 1
#         @nloops $N i k -> (k == $N ? 1 : M.S[k+1] == M.S[k] ? i_{k+1} : 1):size(M.U[M.S[k]], 1) begin
#             $(set_idx...)
#             y = Y_vec[vec_idx]
#             # Iterate over all permutations of the indices where the last N-1 indices are non-decreasing
#             # This is equivalent to swapping i_1 with all unique elements in i_2, ..., i_N,
#             # and keeping order of remaining elements the same. 

#             # Find unique values
#             n_unique = 0
#             for val in tensor_idx
#                 if val ∉ @view(unique_vals[1:n_unique])
#                     n_unique += 1
#                     unique_vals[n_unique] = val
#                 end
#             end

#             # Iterate over unique values
#             for i in 1:n_unique
#                 row = unique_vals[i]
#                 # Find location
#                 loc = 0
#                 for j in 1:$N
#                     if tensor_idx[j] == row
#                         loc = j
#                         break
#                     end     
#                 end
#                 # Copy to permuted
#                 idx = 1
#                 for j in 1:$N
#                     if j != loc
#                         permuted_tensor_idx[idx] = tensor_idx[j]
#                         idx += 1
#                     end
#                 end
                
#                 # col = col_indices_array[permuted_tensor_idx...]
#                 col = lin_reduced(permuted_tensor_idx, sz)
#                 Y_mat[row, col] = y
#             end

#             vec_idx += 1
#         end
#     end
# end

# @generated function fill_reduced_Y_mode_n_from_vec_fullsym_precomputed_inds!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, n, M::SymCPD, col_indices_array, ::Val{N}) where {N}
#     set_idx = [:(tensor_idx[$k] = $(Symbol("i_$k"))) for k in 1:N]
#     quote
#         tensor_idx = zeros(MVector{$N, Int})
#         permuted_tensor_idx = zeros(MVector{$(N-1), Int})
#         unique_vals = zeros(MVector{$N, Int})
#         vec_idx = 1
#         @nloops $N i k -> (k == $N ? 1 : M.S[k+1] == M.S[k] ? i_{k+1} : 1):size(M.U[M.S[k]], 1) begin
#             $(set_idx...)
#             y = Y_vec[vec_idx]
#             # Iterate over all permutations of the indices where the last N-1 indices are non-decreasing
#             # This is equivalent to swapping i_1 with all unique elements in i_2, ..., i_N,
#             # and keeping order of remaining elements the same. 

#             # Find unique values
#             n_unique = 0
#             for val in tensor_idx
#                 if val ∉ @view(unique_vals[1:n_unique])
#                     n_unique += 1
#                     unique_vals[n_unique] = val
#                 end
#             end

#             # Iterate over unique values
#             for i in 1:n_unique
#                 row = unique_vals[i]
#                 # Find location
#                 loc = 0
#                 for j in 1:$N
#                     if tensor_idx[j] == row
#                         loc = j
#                         break
#                     end     
#                 end
#                 # Copy to permuted
#                 idx = 1
#                 for j in 1:$N
#                     if j != loc
#                         permuted_tensor_idx[idx] = tensor_idx[j]
#                         idx += 1
#                     end
#                 end
                
#                 col = col_indices_array[permuted_tensor_idx...]
#                 Y_mat[row, col] = y
#             end



#             # for t in unique(tensor_idx)
#             #     loc = findfirst(==(t), tensor_idx)
#             #     permuted_tensor_idx = tensor_idx[1:end .!= loc]
#             #     col = col_indices_array[permuted_tensor_idx...]
#             #     Y_mat[t, col] = y
#             # end
#             vec_idx += 1
#         end
#     end
# end

# @generated function make_col_indices_array(Y_mat, M, mode, ::Val{N}) where {N}
#     quote
#         col_inds_pos = setdiff(1:$N,mode)
#         S_reduced = M.S[col_inds_pos]
#         mode_size = size(Y_mat, 1)
#         col_indices_array = Array{Int}(undef, ntuple(Returns(mode_size), Val($(N-1))))
#         col = 1
#         @nloops $(N-1) i k -> (k == $(N-1) ? 1 : S_reduced[k+1] == S_reduced[k] ? i_{k+1} : 1):size(M.U[S_reduced[k]], 1) begin
#             col_indices_array[(@ntuple $(N-1) i)...] = col
#             col += 1
#         end
#         return col_indices_array
#     end
# end


# @generated function fill_reduced_Y_vec!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N}) where {N}
#     set_idx = [:(tensor_idx[$k] = $(Symbol("i_$k"))) for k in 1:N]
#     quote
#         tensor_idx = zeros(MVector{$N, Int})
#         vec_idx = 1
#         @nloops $N i k -> (k == $N ? 1 : M.S[k+1] == M.S[k] ? i_{k+1} : 1):size(M.U[M.S[k]], 1) begin
#             $(set_idx...)
#             x = X[tensor_idx...]
#             Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, M[tensor_idx...])
#             vec_idx += 1
#         end
#     end
# end




insert_row(row, t::Tuple{}) = (row,)
insert_row(row, t::Tuple) = 
    row >= t[1] ? (row, t...) : (t[1], insert_row(row, t[2:end])...)

# For mode-n MTTKRP where mode n is in a singleton cell
# @generated function fill_reduced_Y_mode_n_from_vec_singleton_cell!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, ::Val{n}, M::SymCPD, ::Val{N}) where {n,N}
#     num_parent_loops = N-n  
#     num_child_loops = n-1
#     quote
#         mode_sz = size(Y_mat, $n)
#         full_inds = 1:$N
#         parent_col_inds_pos = full_inds[$n+1:end]
#         child_col_inds_pos = full_inds[1:$n-1]
#         S_parent = M.S[parent_col_inds_pos]
#         S_child = M.S[child_col_inds_pos]
#         # col_inds_pos = setdiff(1:$N,n)
#         # S_reduced = M.S[col_inds_pos]
#         vec_idx = 1
#         col = 1
#         row_start_col = 1
#         @nloops $(num_parent_loops) i k -> (k == $(num_parent_loops) ? 1 : S_parent[k+1] == S_parent[k] ? i_{k+1} : 1):size(M.U[S_parent[k]], 1) begin
#             for row in 1:mode_sz
#                 col = row_start_col
#                 @nloops $(num_child_loops) j k -> (k == $(num_child_loops) ? 1 : S_child[k+1] == S_child[k] ? j_{k+1} : 1):size(M.U[S_child[k]], 1) begin
#                     Y_mat[row,col] = Y_vec[vec_idx]
#                     vec_idx += 1
#                     col += 1
#                 end
#             end
#             row_start_col = col
#         end
#     end
# end

@generated function fill_reduced_Y_mode_n_from_vec_singleton_cell!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, n, M::SymCPD, ::Val{N}) where {N}
    quote
        mode_sz = size(Y_mat, n)
        col_inds_pos = setdiff(1:$N,n)
        S_reduced = M.S[col_inds_pos]
        vec_idx = 1
        col = 1
        @nloops $(N-1) i k -> (k == $(N-1) ? 1 : S_reduced[k+1] == S_reduced[k] ? i_{k+1} : 1):size(M.U[S_reduced[k]], 1) begin 
            for row in 1:mode_sz
                Y_mat[row,col] = Y_vec[vec_idx]
                vec_idx += 1
            end
            col += 1
        end
    end
end

# For mode-n MTTKRP where mode n is in a cell with 2 modes
@generated function fill_reduced_Y_mode_n_from_vec_doubleton_cell!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, n, M::SymCPD, ::Val{N}) where {N}
    quote
        mode_size = size(Y_mat, 1)
        col_inds_pos = setdiff(1:$N,[n,n+1])
        S_reduced = M.S[col_inds_pos]
        vec_idx = 1
        col = 1
        @nloops $(N-2) i k -> (k == $(N-2) ? 1 : S_reduced[k+1] == S_reduced[k] ? i_{k+1} : 1):size(M.U[S_reduced[k]], 1) begin
            col_start = col
            for j in 1:mode_size
                for row in 1:mode_size
                    if row >= j 
                        Y_mat[row,col] = Y_vec[vec_idx]
                        vec_idx += 1
                    else
                        # Copy from corresponding filled out entry
                        target_col = col_start + row - 1
                        Y_mat[row,col] = Y_mat[j, target_col]
                    end
                end
                col += 1
            end
        end
    end
end

# @generated function fill_reduced_Y_mode_n_from_vec!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, n, M::SymCPD, ::Val{N}) where {N}
#     quote
#         mode_size = size(Y_mat, 1)
#         col_inds_pos = setdiff(1:$N,[n,n+1])
#         S_reduced = M.S[col_inds_pos]
#         vec_idx = 1
#         col = 1
#         num_modes_cell = count(M.S .== M.S[n])
#         col_starts = Array{Int}(undef, ntuple(k -> size(M.U[S_reduced[k]], 1), Val($(N-2))))
#         @nloops $(N-2) i k -> (k == $(N-2) ? 1 : S_reduced[k+1] == S_reduced[k] ? i_{k+1} : 1):size(M.U[S_reduced[k]], 1) begin
#             @ncall $(N-2) setindex! col_starts col i
#             for j in (@ntuple $(N-2) i)[n]:mode_size
#                 for row in 1:mode_size
#                     if row >= j 
#                         Y_mat[row,col] = Y_vec[vec_idx]
#                         vec_idx += 1
#                     else
#                         # Sort indices in cell lexicographically
#                         full_inds = @ntuple $(N-2) i
#                         cell_inds = ntuple(t -> full_inds[n+t-1], num_modes_cell - 2)
#                         idx = insert_row(row, (j, cell_inds...))
#                         # idx = insert_row(row, (j, (@ntuple $(N-2) i)...), ?)
#                         # idx should only have num_modes_cell elements
#                         # Copy from corresponding filled out entry
#                         target_col = col_starts[idx[3:end]...] + idx[2] - idx[3]  
#                         Y_mat[row, col] = Y_mat[idx[1], target_col]
#                     end
#                 end
#                 col += 1
#             end
#         end
#     end
# end

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

function fill_reduced_Y_mode_n_from_vec_fullsym_order3!(Y_mat::AbstractMatrix, Y_vec::AbstractVector)
    sz = size(Y_mat, 1)
    vec_idx = 1
    col = 1
    col_starts = Vector{Int}(undef, sz)

    for i3 in 1:sz
        col_starts[i3] = col
        for i2 in i3:sz
            for row in 1:sz
                if row >= i2
                    Y_mat[row, col] = Y_vec[vec_idx]
                    vec_idx += 1
                else
                    # Sort (row, i2, i3) in descneding/lexicographic order. Know i2 >= i3, so just insert row
                    idx1, idx2, idx3 = row >= i2 ? (row, i2, i3) :
                                                row >= i3 ? (i2, row, i3) :
                                                            (i2, i3, row)
                    # Copy from corresponding entry already filled out
                    target_col = col_starts[idx3] + (idx2 - idx3)
                    Y_mat[row, col] = Y_mat[idx1, target_col]
                end
            end
            col += 1
        end
    end
end

function fill_reduced_Y_mode_n_from_vec_fullsym_order4!(Y_mat::AbstractMatrix, Y_vec::AbstractVector)
    sz = size(Y_mat, 1)
    vec_idx = 1
    col = 1
    col_starts = Matrix{Int}(undef, sz, sz) # Row indexes i3, col indexes i4

    for i4 in 1:sz
        for i3 in i4:sz
            col_starts[i3, i4] = col
            for i2 in i3:sz
                for row in 1:sz
                    if row >= i2
                        Y_mat[row, col] = Y_vec[vec_idx]
                        vec_idx += 1
                    else
                        # Sort (row, i2, i3, i4) in descneding/lexicographic order. Know i2 >= i3 >= i4, so just insert row
                        idx1, idx2, idx3, idx4 = row >= i2 ? (row, i2, i3, i4) :
                                                 row >= i3 ? (i2, row, i3, i4) :
                                                 row >= i4 ? (i2, i3, row, i4) :
                                                             (i2, i3, i4, row)
                        # Copy from corresponding entry already filled out
                        target_col = col_starts[idx3, idx4] + (idx2 - idx3)
                        Y_mat[row, col] = Y_mat[idx1, target_col]
                    end
                end
                col += 1
            end
        end
    end
end


# @generated function fill_reduced_Y_mode_n_from_vec_fullsym_orderN!(Y_mat::AbstractMatrix, Y_vec::AbstractVector,  ::Val{N}) where {N}
#     quote
#         sz = size(Y_mat, 1)
#         vec_idx = 1
#         col = 1
#         col_starts = Array{Int}(undef, ntuple(_ -> sz, $N-2))
#         @nloops $(N-2) i k -> (k == $(N-2) ? 1 : S_reduced[k+1] == S_reduced[k] ? i_{k+1} : 1):size(M.U[S_reduced[k]], 1) begin
            
#             for j in 
#             for row in 1:num_rows

#             end
#             col += 1
#         end
#     end
# end



# function sort_lexicographic!(idx::MVector{N,Int}, S) where N
#     start_idx = 1
#     for cell in unique(S)
#         end_idx = start_idx + count(S .== cell) - 1
#         sort!(@view(idx[start_idx:end_idx]), rev=true)
#         start_idx = end_idx + 1
#     end
# end

# @generated function fill_reduced_Y_mode_n_from_vec_fullsym!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, n::Integer, M::SymCPD, ::Val{N}) where {N}
#     set_idx = [:(tensor_idx[col_inds_pos[$k]] = $(Symbol("i_$k"))) for k in 1:N-1]
#     quote
#         num_rows = size(Y_mat, 1)
#         col_inds_pos = setdiff(1:$N,n)
#         S_reduced = M.S[col_inds_pos]
#         tensor_idx = ones(MVector{$N+1, Int})  # Extra index at end for use when computing vec idx
#         col = 1
#         vec_idx = 1
#         @nloops $(N-1) i k -> (k == $(N-1) ? 1 : S_reduced[k+1] == S_reduced[k] ? i_{k+1} : 1):size(M.U[S_reduced[k]], 1) begin
#             # Compute contribution of columns to vec_idx sum
#             $(set_idx...)
#             partial_sum_col = get_vec_idx_minus_one(tensor_idx, num_rows, n)   
#             # println(tensor_idx, "  ", partial_sum_col) 
#             for row in 1:num_rows
#                 tensor_idx[n] = row
#                 if row >= $(Symbol("i_1"))
#                     Y_mat[row,col] = Y_vec[vec_idx]
#                     vec_idx += 1
#                 else
#                     # Get lexicographically forward permutation
#                     sort!(@view(tensor_idx[2:$N]), rev=true)
#                     # sort_lexicographic!(tensor_idx, M.S)
#                     # Copy from corresponding entry in vec
#                     # Y_mat[row,col] = Y_vec[get_vec_idx(tensor_idx, num_rows)]
#                     search_idx = partial_sum_col + binomial(num_rows - tensor_idx[2] + 1, 1) - binomial(num_rows - tensor_idx[1] + 1, 1)
#                     Y_mat[row,col] = Y_vec[search_idx]
#                 end
#             end
#             col += 1
#         end
#     end
# end



# M = N + 1
# function get_vec_idx(idx::MVector{M, Int}, size) where M
#     rank = 1
#     for m in 1:M-1
#         for w in idx[m+1]:idx[m]-1
#             binom_arg1 = size-w+m-1
#             binom_arg2 = m-1
#             if binom_arg1 >= binom_arg2
#                 rank += binomial(size-w+m-1,m-1)
#             end
#         end
#     end
#     return rank
# end

# function get_vec_idx_minus_one(idx::MVector{M,Int}, size, skip_m) where M
#     rank = 1
#     for m in 1:M-1
#         if m != skip_m
#             rank += binomial(size - idx[m+1] + m, m) - binomial(size - idx[m] + m, m)
#         end
#     end
#     return rank
# end

# function get_vec_idx(idx::MVector{M,Int}, size) where M
#     rank = 1
#     for m in 1:M-1
#         rank += binomial(size - idx[m+1] + m, m) - binomial(size - idx[m] + m, m)
#     end
#     return rank
# end



# @generated function fill_reduced_Y_mode_n_from_vec_unique!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, n::Integer, X::Array, M::SymCPD, ::Val{N}) where {N}
#     quote
#         num_rows = size(X,n)
#         col_inds_pos = setdiff(1:$N,n)
#         S_reduced = M.S[col_inds_pos]
#         vec_idx = 1
#         start_col = 1
#         for row in 1:num_rows
#             col = start_col
#             @nloops $(N-1) i k -> (k == $(N-1) ? 1 : S_reduced[k+1] == S_reduced[k] ? i_{k+1} : 1):size(M.U[S_reduced[k]], 1) begin
#                 Y_mat[row,col] = Y_vec[vec_idx]
#                 col += 1
#             end
#             start_col += size(X,n)-row+1
#         end
#     end
# end

# @generated function fill_reduced_Y_mode_n_from_vec_one_mode_cell!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, n::Integer, X::Array, M::SymCPD, ::Val{N}) where {N}
#     quote
#         num_rows = size(X,n)
#         col_inds_pos = setdiff(1:$N,n)
#         S_reduced = M.S[col_inds_pos]
#         col = 1
#         vec_idx = 1
#         @nloops $(N-1) i k -> (k == $(N-1) ? 1 : S_reduced[k+1] == S_reduced[k] ? i_{k+1} : 1):size(M.U[S_reduced[k]], 1) begin
#             for row in 1:num_rows
#                 Y_mat[row,col] = Y_vec[vec_idx]
#                 vec_idx += 1
#             end
#             col += 1
#         end
#     end
# end


"""
    fill_reduced_Y_vec!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N})

Forms reduced vectorization of derivative tensor Y where duplicate entries due to symmetry are removed.
"""
@generated function fill_reduced_Y_vec!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N}) where {N}
    set_idx = [:(tensor_idx[$k] = $(Symbol("i_$k"))) for k in 1:N]
    quote
        tensor_idx = zeros(MVector{$N, Int})
        vec_idx = 1
        @nloops $N i k -> (k == $N ? 1 : M.S[k+1] == M.S[k] ? i_{k+1} : 1):size(M.U[M.S[k]], 1) begin
            $(set_idx...)
            x = X[tensor_idx...]
            Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, M[tensor_idx...])
            vec_idx += 1
        end
    end
end
@generated function fill_reduced_Y_vec!(Y_vec::AbstractVector, X::Array, M::SymCPD, M_array::Array, loss, ::Val{N}) where {N}
    set_idx = [:(tensor_idx[$k] = $(Symbol("i_$k"))) for k in 1:N]
    quote
        tensor_idx = zeros(MVector{$N, Int})
        vec_idx = 1
        @nloops $N i k -> (k == $N ? 1 : M.S[k+1] == M.S[k] ? i_{k+1} : 1):size(M.U[M.S[k]], 1) begin
            $(set_idx...)
            x = X[tensor_idx...]
            Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, M_array[tensor_idx...])
            vec_idx += 1
        end
    end
end
# @generated function fill_reduced_Y_vec_multi_inds!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N}) where {N}
#     set_idx = [:(tensor_idx[$k] = $(Symbol("i_$k"))) for k in 1:N]
#     quote
#         tensor_idx = zeros(MVector{$N, Int})
#         vec_idx = 1
#         @nloops $N i k -> (k == $N ? 1 : M.S[k+1] == M.S[k] ? i_{k+1} : 1):size(M.U[M.S[k]], 1) begin
#             $(set_idx...)
#             x = X[tensor_idx...]
#             Y_vec[vec_idx] = Tuple(tensor_idx)
#             vec_idx += 1
#         end
#     end
# end

# function fill_reduced_Y_vec_combin!(Y_vec::AbstractVector, X::Array{T,N}, M::SymCPD, loss) where {T,N}
#     vec_idx = 1
#     mode_size = size(X, 1)
#     a = 0
#     for multi_idx_vec in with_replacement_combinations(1:mode_size, N)
#         # multi_idx = CartesianIndex(ntuple(i -> multi_idx_vec[i], Val(N)))
#         # x = X[multi_idx]
#         # Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, M[multi_idx])
#         # vec_idx += 1
#         a += 1
#     end
# end


function fill_reduced_Y_mode1_mat_from_vec_order3!(Y_mode1_mat::AbstractMatrix, Y_vec::AbstractVector)
    sz = size(Y_mode1_mat, 1)
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
    num_cols = prod(i -> sz+i-1, 1:2)÷factorial(2)
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

# function fill_reduced_Y_mode1_mat_from_vec_order4!(Y_mode1_mat::AbstractMatrix, Y_vec::AbstractVector)
#     sz = size(Y_mode1_mat, 1)
#     start_col = 1
#     vec_idx = 1
#     # Fill in unique, then duplciate entries below first row in symmetric sub-matrices
#     for outer_block in 1:sz
#         start_row = outer_block
#         inner_block_initial_sz = sz-(outer_block-1)
#         num_inner_blocks = sz - outer_block + 1
#         for inner_block in 1:num_inner_blocks
#             inner_block_sz = inner_block_initial_sz - inner_block + 1
#             for (col_idx, col) in enumerate(start_col:start_col+inner_block_sz)
#                 for (row_idx, row) in enumerate(start_row+col_idx-1:sz)
#                     Y_mode1_mat[row,col] = Y_vec[vec_idx]
#                     if row_idx != 1 && col_idx != 1 && col_idx != inner_block_sz  # Skip top row which is handled later
#                         Y_mode1_mat[start_row+col_idx-1, col+row_idx-1] = Y_vec[vec_idx]
#                     end
#                     vec_idx += 1
#                 end
#             end
#             start_col += inner_block_sz
#             start_row += 1
#         end
#     end
#     # Fill in remaining rows above sub-matrices
#     for outer_block in 1:sz
#         num_inner_blocks = sz - outer_block + 1
#         for inner_block in 1:num_inner_blocks
#             row = 1
#             start_col = 2
#             num_cols = num_cols = prod(i -> sz+i-1, 1:N-1)÷factorial(N-1)
#             vec_idx = 2
#         end
#     end
# end


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

