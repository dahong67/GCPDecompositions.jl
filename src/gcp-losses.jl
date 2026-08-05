## Loss function types

"""
Loss functions for Generalized CP Decomposition.
"""
module GCPLosses

using ..GCPDecompositions
using ..TensorKernels: mttkrps!, mttkrp, mttkrp!, sparse_mttkrp!, sparse_mttkrps!, checksym, khatrirao
using ..TensorKernels: symmetric_mttkrp!, symmetric_kr!, symmetric_kr_unweighted!
using IntervalSets: Interval
using LinearAlgebra: mul!, rmul!, Diagonal, norm, dot
using SparseArrayKit: SparseArray, nonzero_keys, nonzero_values
using StaticArrays: MVector, SVector
using Base.Cartesian: @nloops, @ntuple, @ncall, @nexprs, @nref
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

# """
#     grad_U_λ_symmetric!(GU_λ, M::SymCPD, X::AbstractArray, idx_map_mats::NTuple, loss, γ)

# Compute the SymGCP gradient with respect to the factor matrices `U = (U[1],...,U[N])` and the 
# weights `λ` for the model tensor `M`, data tensor `X`, and loss function `loss`, exploiting symmetry
# for more efficient computation. Stores the result in `GU_λ = (GU[1],...,GU[K], Gλ)`. 
# idx_map_mats should contain a matrix for each symmetric cell which maps to reduced linear indices. 
# Use form_reduced_linear_mapping_matrix in symcpd.jl for construction of these matrices.
# Note that this function assumes that X and M have matching symmetry. 
# γ controls the strength of the (column-norm - 1) regularization.
# """
# function grad_U_λ_symmetric!(
#     GU_λ::Tuple,
#     M::SymCPD{T,N,K},
#     X::Array{TX,N},
#     idx_map_mats::NTuple{K, AbstractMatrix},
#     loss,
#     γ,
# ) where {T,TX,N,K}

#     # Weights gradient
#     vec_size = prod(k -> prod(i -> size(M.U[k],1)+i-1, 1:count(M.S .== k))÷factorial(count(M.S .== k)), unique(M.S))
#     Y_vec = similar(X, vec_size)
#     fill_reduced_Y_vec_version_1b!(Y_vec, X, M, loss, Val(N), Val(ncomps(M)))
#     # fill_reduced_Y_vec_version_2c!(Y_vec, X, M, loss)
#     kr_tilde = similar(M.U[1], vec_size, ncomps(M))
#     flip_group_ordering(k) = ngroups(M) - k + 1
#     GU_λ[K+1] .= symmetric_kr!(kr_tilde, reverse(flip_group_ordering.(M.S)), reverse(M.U)...)' * Y_vec

#     # Factor matrix gradients
#     for j in 1:K
#         mode = findall(M.S .== j)[1]
#         S_reduced = M.S[setdiff(1:N,mode)]
#         # Form reduced matricization, splitting out some special cases
#         if count(M.S .== j) == 1 && mode == 1
#             Y_mat = reshape(Y_vec, size(X,mode), :)
#         else
#             mat_size = prod(k -> prod(i -> size(M.U[k],1)+i-1, 1:count(S_reduced .== k))÷factorial(count(S_reduced .== k)), unique(S_reduced))
#             Y_mat = similar(X, size(X, mode), mat_size)
#             if count(M.S .== j) == 1 && mode == N
#                 vec_idx = 1
#                 for row in 1:size(X, mode)
#                     Y_mat[row,:] = Y_vec[vec_idx:vec_idx+mat_size-1]
#                     vec_idx += mat_size
#                 end
#             else
#                 fill_reduced_Y_mode_n!(Y_mat, Y_vec, idx_map_mats[j])
#             end
#         end
#         symmetric_mttkrp!(GU_λ[j], Y_mat, M.U, M.S, mode)
#         rmul!(GU_λ[j], Diagonal(M.λ))
#         GU_λ[j] .+= mapslices(x -> 4γ * (norm(x)^2 - 1) * x, M.U[j]; dims=1)
#     end

#     return GU_λ
# end

# """
#     grad_U_λ_symmetric_ttsv!(GU_λ, M::SymCPD, X::AbstractArray, idx_map_mats::NTuple, loss, γ)

# Compute the SymGCP gradient with respect to the factor matrices `U = (U[1],...,U[N])` and the 
# weights `λ` for the model tensor `M`, data tensor `X`, and loss function `loss`, exploiting symmetry
# for more efficient computation, by computing TTSVs. Stores the result in `GU_λ = (GU[1],...,GU[K], Gλ)`. 
# idx_map_mats should contain a matrix for each symmetric cell which maps to reduced linear indices. 
# Note that this function assumes that X and M have matching symmetry. 
# γ controls the strength of the (column-norm - 1) regularization.
# Currently only for full symmetry.
# """
# function grad_U_λ_symmetric_ttsv!(
#     GU_λ::Tuple,
#     M::SymCPD{T,N,K},
#     X::Array{TX,N},
#     loss,
#     γ,
# ) where {T,TX,N,K}

#     R = ncomps(M)

#     # Collect unique entries of derivative tensor
#     vec_size = prod(k -> prod(i -> size(M.U[k],1)+i-1, 1:count(M.S .== k))÷factorial(count(M.S .== k)), unique(M.S))
#     Y_vec = similar(X, vec_size)
#     fill_reduced_Y_vec_version_1b!(Y_vec, X, M, loss, Val(N), Val(R))

#     # Compute TTSVs for weights gradient
#     kr_tilde = similar(M.U[1], vec_size, R)
#     flip_group_ordering(k) = ngroups(M) - k + 1
#     GU_λ[K+1] .= symmetric_kr!(kr_tilde, reverse(flip_group_ordering.(M.S)), reverse(M.U)...)' * Y_vec

#     GU_λ[K+1] .= columnwise_ttsv_all_modes(Y_vec, M.U[1], Val(N), Val(R))
#     # for j in 1:R
#     #     GU_λ[K+1][j] .= ttsv_all_modes(Y_vec, M.U[1][:,j], Val(N))
#     # end

#     # for j in 1:ncomps(M)
#     #     GU[K+1][j] = GCPLosses.ttsv_all_modes(Y_vec, M.U[1][:,j], Val(3))
#     # end




#     # Factor matrix gradients
#     for j in 1:K
#         mode = findall(M.S .== j)[1]
#         S_reduced = M.S[setdiff(1:N,mode)]
#         # Form reduced matricization, splitting out some special cases
#         if count(M.S .== j) == 1 && mode == 1
#             Y_mat = reshape(Y_vec, size(X,mode), :)
#         else
#             mat_size = prod(k -> prod(i -> size(M.U[k],1)+i-1, 1:count(S_reduced .== k))÷factorial(count(S_reduced .== k)), unique(S_reduced))
#             Y_mat = similar(X, size(X, mode), mat_size)
#             if count(M.S .== j) == 1 && mode == N
#                 vec_idx = 1
#                 for row in 1:size(X, mode)
#                     Y_mat[row,:] = Y_vec[vec_idx:vec_idx+mat_size-1]
#                     vec_idx += mat_size
#                 end
#             else
#                 fill_reduced_Y_mode_n!(Y_mat, Y_vec, idx_map_mats[j])
#             end
#         end
#         symmetric_mttkrp!(GU_λ[j], Y_mat, M.U, M.S, mode)
#         rmul!(GU_λ[j], Diagonal(M.λ))
#         GU_λ[j] .+= mapslices(x -> 4γ * (norm(x)^2 - 1) * x, M.U[j]; dims=1)
#     end

#     return GU_λ
# end

"""
    collect_multinomial_coefficients(S::NTuple{N,Int}, cell_sizes::NTuple{K,Int}, ::Val{N}) where {N,K}

Collect the multinomial coefficient (number of repeated entries) for each
unique value in the tensor with symmetry pattern given by S and mode sizes for each cell
given by cell_sizes.
"""
@generated function collect_multinomial_coefficients(
    S::NTuple{N,Int}, 
    cell_sizes::NTuple{K,Int}, 
    ::Val{N}
) where {N,K}
    quote
        coef_dtype = Float64
        sz = 1
        num = 1.0
        for k in 1:$K
            sz *= binomial(cell_sizes[k] + count(S .== k) - 1, count(S .== k))
            num *= factorial(count(S .== k))
        end
        coefs = Vector{coef_dtype}(undef, sz)
        vec_idx = 1
        @nloops $N i k -> (k == $N ? 1 : S[k] == S[k+1] ? i_{k+1} : 1):cell_sizes[S[k]] begin
            if $K == 1 && $N == 2
                α = i_1 == i_2 ? 1.0 : 2.0
            elseif $K == 1 && $N == 3
                α = i_1 == i_2 ? (i_2 == i_3 ? 1.0 : 3.0) : (i_2 == i_3 ? 3.0 : 6.0)
            elseif $K == 1 && $N == 4
                n_equal = (i_1==i_2) + (i_1==i_3) + (i_1==i_4) + (i_2==i_3) + (i_2==i_4) + (i_3==i_4)
                α = n_equal == 0 ? 24.0 : n_equal == 1 ? 12.0 : n_equal == 2 ? 6.0 : n_equal == 3 ? 4.0 : 1.0
            else
                idx = @ntuple $N i
                denom = 1.0
                num_modes = 1
                for m in 2:$N
                    if S[m] == S[m-1] && idx[m] == idx[m-1]
                        num_modes += 1
                    else
                        denom *= factorial(num_modes)
                        num_modes = 1
                    end
                end
                denom *= factorial(num_modes)
                α = num / denom
            end
            coefs[vec_idx] = α
            vec_idx += 1
        end
        return coefs
    end
end

"""
    fill_reduced_Y_vec!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N}, ::Val{R}) where {N,R}

Forms reduced vectorization of derivative tensor Y with only structually unique entries.
Computes partial products at loop level 2 for efficiency. May be able to get further improvements
    by saving partial products at all loop levels 2 through N.
"""
@generated function fill_reduced_Y_vec!(Y_vec::AbstractVector, X::Array, M::SymCPD, loss, ::Val{N}, ::Val{R}) where {N,R}
    set_partial_m = map(1:R) do j
        terms = [:(M.U[M.S[$k]][$(Symbol("i_$(k)")), $j]) for k in 2:N]
        :(partial_prod[$j] = M.λ[$j] * *( $(terms...) ))
    end
    pre_body = Expr(:block, set_partial_m...)

    quote
        S = M.S
        T = eltype(M.U[1])
        partial_prod = zeros(MVector{$R, T})
        mode1_factors = M.U[S[1]]
        vec_idx = 1
        @inbounds @nloops(
            $N,
            i,
            k -> (k == $N ? 1 : S[k] == S[k+1] ? i_{k+1} : 1):size(M.U[S[k]], 1),
            d -> d == 2 ? $pre_body : nothing,
            begin
                x = @nref $N X i
                m = zero(T)
                for col in 1:$R
                    m = muladd(mode1_factors[i_1, col], partial_prod[col], m)
                end 
                Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, m)
                vec_idx += 1
            end
        )
    end
end

"""
    fill_reduced_Y_mode_n!(Y_mat::AbstractMatrix, n::Integer, X::Array, M::SymCPD, loss, ::Val{N})

Fill mode-n matricization of derivative tensor Y where duplicate columns due to symmetry are removed,
    by copying values from Y_vec using mapping in idx_map_mat.
"""
function fill_reduced_Y_mode_n!(Y_mat::AbstractMatrix, Y_vec::AbstractVector, idx_map_mat::AbstractMatrix)
    @inbounds for I in eachindex(Y_mat, idx_map_mat)
        Y_mat[I] = Y_vec[idx_map_mat[I]]
    end
    return Y_mat
end

"""
    weight_grad_krp_mult!(
        GU_λ::NTuple{V, AbstractArray}, 
        Y_vec::AbstractVector, 
        Y_vec_scaled_buffer::AbstractVector,
        kr_buffer::AbstractMatrix, 
        M::SymCPD{T,N,K}, 
        multinomial_coefs::AbstractVector
    ) where {V,T,N,K}
    
Compute the gradient for the weights by forming the reduced Khatri-Rao product and then multiplying
    by the vector of unique values of the derivative tensor, writing the result into GU_λ[K+1].
multinomial_coefs should be a vector of the same length as Y_vec and Y_vec_scaled_buffer, with the 
    multinomial coefficient for each corresponding entry of Y_vec (note this is different from factor_grad_krp_mult!,
    where multinomial_coefs should be an NTuple).
Uses Y_vec_scaled_buffer for Y_vec scaled by the multinomial coefficients, and
    kr_buffer for the reduced Khatri-Rao product.
"""
function weight_grad_krp_mult!(
    GU_λ::NTuple{V, AbstractArray}, 
    Y_vec::AbstractVector, 
    Y_vec_scaled_buffer::AbstractVector,
    kr_buffer::AbstractMatrix, 
    M::SymCPD{T,N,K}, 
    multinomial_coefs::AbstractVector
) where {V,T,N,K}
    flip_group_ordering(j) = K - j + 1
    @inbounds for idx in eachindex(Y_vec, Y_vec_scaled_buffer, multinomial_coefs)
        Y_vec_scaled_buffer[idx] = Y_vec[idx] * multinomial_coefs[idx]
    end
    symmetric_kr_unweighted!(kr_buffer, reverse(flip_group_ordering.(M.S)), reverse(M.U)...)
    mul!(GU_λ[K+1], kr_buffer', Y_vec_scaled_buffer)
    return GU_λ[K+1]
end

"""
    factor_grad_krp_mult!(
        GU_λ::NTuple{V, AbstractArray}, 
        Y_vec::AbstractVector, 
        Y_mat_buffer::AbstractMatrix, 
        kr_buffer::AbstractMatrix, 
        M::SymCPD{T,N,K}, 
        idx_map_mat::AbstractMatrix, 
        multinomial_coefs::NTuple{L, AbstractVector}, 
        cell::Int
    ) where {V,T,N,K}

Compute the gradient for the factors for the given cell by forming the reduced Khatri-Rao product and then 
    multiplying by the reduced matricziation formed from the vector of unique values of the derivative tensor,
    writing the result into GU_λ[cell].
idx_map_mat should be matrix for maps each element of the given cell's 
    reduced matricization to its reduced linear index.
multinomial_coefs should be a NTuple of length L with coefficients for each cell of symmetric modes, where
    the coefs for the given cell assume one mode is held out, and the cells are in reverse order of M.
    Should have L == K-1 if there is one mode in the given cell, or L == K otherwise.
Uses Y_mat_buffer as a buffer for the reduced matricization of the derivative tensor, and
    kr_buffer as a buffer for the reduced Khatri-Rao product.
"""
function factor_grad_krp_mult!(
    GU_λ::NTuple{V, AbstractArray}, 
    Y_vec::AbstractVector, 
    Y_mat_buffer::AbstractMatrix, 
    kr_buffer::AbstractMatrix, 
    M::SymCPD{T,N,K}, 
    idx_map_mat::AbstractMatrix, 
    multinomial_coefs::NTuple{L, AbstractVector}, 
    cell::Int
) where {V,T,N,K,L}
    (cell >= 1 && cell <= K) || 
        throw(ArgumentError("`cell` must satisfy 1 <= cell <= ngroups(M)"))
    (L == K && count(==(cell), M.S) > 1) || (L == K-1 && count(==(cell), M.S) == 1) ||
        throw(ArgumentError("`multinomial_coefs`` should be an NTuple with a vector of coefficients for each cell included in the Khatri-Rao product"))

    mode = findfirst(==(cell), M.S)
    fill_reduced_Y_mode_n!(Y_mat_buffer, Y_vec, idx_map_mat)
    symmetric_mttkrp!(GU_λ[cell], Y_mat_buffer, M.U, M.S, mode, multinomial_coefs, kr_buffer)
    rmul!(GU_λ[cell], Diagonal(M.λ))
    return GU_λ[cell]
end

"""
    weight_grad_ttv!(
        GU_λ::NTuple{V, AbstractArray}, 
        Y_vec::AbstractVector, 
        M::SymCPD{T,N,K}, 
        multinomial_coefs::Vector{TC}
    ) where {V,T,N,K,TC}

Compute the gradients for the weights vector using the TTV in all modes,
    writing the result into GU_λ[K+1].
"""
function weight_grad_ttv!(
    GU_λ::NTuple{V, AbstractArray},
    Y_vec::AbstractVector, 
    M::SymCPD{T,N,K}, 
    multinomial_coefs::Vector{TC}
) where {V,T,N,K,TC}
    fill!(GU_λ[K+1], zero(T))
    columnwise_ttv_all_modes!(GU_λ[K+1], Y_vec, M.U, multinomial_coefs, Val(M.S), Val(N), Val(ncomps(M)))
    return GU_λ[K+1]
end

"""
    columnwise_ttv_all_modes!(
        result::Vector{TX},
        y::Vector{TY}, 
        Xs::NTuple{K,AbstractMatrix{TX}}, 
        multinomial_coefs::Vector{TC}, 
        ::Val{S}, ::Val{N}, ::Val{R}
    ) where {TY,K,TX,TC,S,N,R}

Compute the TTV in all modes (i.e., the weights gradient) given the reduced vectorization of 
    the derivative tensor y and factor matrices in Xs, using the coefficients in 
    multinomial_coefs, for general symmetry given by S, order N, and rank R.
"""
@generated function columnwise_ttv_all_modes!(
    result::Vector{TX},
    y::Vector{TY}, 
    Xs::NTuple{K,AbstractMatrix{TX}}, 
    multinomial_coefs::Vector{TC}, 
    ::Val{S}, ::Val{N}, ::Val{R}
) where {TY,K,TX,TC,S,N,R}

    # Define functions for different symbols in expressions
    Iv(d) = Symbol(:i_, d);  Partial_Prod(d) = Symbol(:partial_prod_, d)
    XM(d) = Symbol(:Xt_, S[d])        # transposed factor at loop position d

    pre_exprs  = [Any[] for _ in 1:N]
    post_exprs = [Any[] for _ in 1:N]

    # Add pre-expression at each level for partial products
    for d in 2:N
        push!(pre_exprs[d], :(for col in 1:$R
            $(Partial_Prod(d))[col] = $(d == N ? :($(XM(d))[col, $(Iv(d))]) :
                              :($(XM(d))[col, $(Iv(d))] * $(Partial_Prod(d+1))[col]))
        end))
    end
    
    # Add pre/post expressions for zeroing/flushing the accumulator
    push!(pre_exprs[2],  :(fill!(acc, zero($TX))))
    push!(post_exprs[2], :(for col in 1:$R
                               result[col] = muladd(acc[col], partial_prod_2[col], result[col])
                           end))

    chain(exprs) = Expr(:->, :d, foldr((k, rest) -> isempty(exprs[k]) ? rest :
                     Expr(:if, :(d == $k), Expr(:block, exprs[k]...), rest),
                     1:N; init = :nothing))
    pre, post = chain(pre_exprs), chain(post_exprs)

    alloc = Expr(:block, (:($(Partial_Prod(d)) = zeros(MVector{$R, $TX})) for d in 2:N)...)
    
    quote
        Sv = $(S)
        acc = zeros(MVector{$R, TX})
        @nexprs $K k -> (Xt_k = permutedims(Xs[k]))
        Xfirst = $(XM(1))
        $alloc
        vec_idx = 1
        @inbounds @nloops(
            $N,
            i,
            k -> (k == $N ? 1 : Sv[k] == Sv[k+1] ? i_{k+1} : 1):size(Xs[Sv[k]], 1),
            $pre,
            $post,
            begin
                tensor_term = multinomial_coefs[vec_idx] * y[vec_idx]
                for col in 1:$R
                    acc[col] = muladd(tensor_term, Xfirst[col, i_1], acc[col])
                end
                vec_idx += 1
            end
        )
        return result
    end
end

"""
    factor_grad_ttv!(GU_λ::NTuple{V, AbstractArray},
        Y_vec::AbstractVector, 
        M::SymCPD{T,N,K}, 
        multinomial_coefs::Vector{TC},
        cell::Int
    ) where {V,T,N,K,TC}

Compute the gradients for the factor matrix for the given cell using the TTV in all modes except one,
    writing the result into GU_λ[cell]. Note that the multiplication by number of modes in cell is handled
    by columnwise_ttv_all_modes_except_one!.
"""
function factor_grad_ttv!(
    GU_λ::NTuple{V, AbstractArray},
    Y_vec::AbstractVector, 
    M::SymCPD{T,N,K}, 
    multinomial_coefs::Vector{TC},
    cell::Int
) where {V,T,N,K,TC}
    (1 <= cell <= K) || 
        throw(ArgumentError("`cell` must satisfy 1 <= cell <= ngroups(M)"))

    GU_T = zeros(T, size(GU_λ[cell], 2), size(GU_λ[cell], 1))
    columnwise_ttv_all_modes_except_one!(GU_T, Y_vec, M.U, multinomial_coefs, Val(M.S), Val(cell), Val(N), Val(ncomps(M)))
    GU_λ[cell] .= permutedims(GU_T)
    rmul!(GU_λ[cell], Diagonal(M.λ))
    return GU_λ[cell]
end

"""
    columnwise_ttv_all_modes_except_one!(
        result::AbstractMatrix,
        y::Vector{TY}, 
        Xs::NTuple{K,AbstractMatrix{TX}}, 
        multinomial_coefs::Vector{TC}, 
        ::Val{S}, ::Val{c}, ::Val{N}, ::Val{R}
    ) where {TY,K,TX,TC,S,c,N,R}

Compute the TTSV in all modes except the first one corresponding to cell c, i.e., 
    the gradient for the factors for cell c minus scaling by diag(λ), 
    given the reduced vectorization of the derivative tensor y 
    and the factor matrices in Xs, with pre-computed multinomial coefficients.
result should be dimensions R x n, where R is the rank and n is the mode size for cell c.
This function currently doesa permutedims on the factor matrices in Xs, but
    it would probably be better if there was an option to store the factor matrices transposed from the start.
"""
@generated function columnwise_ttv_all_modes_except_one!(
    result::AbstractMatrix,
    y::Vector{TY}, 
    Xs::NTuple{K,AbstractMatrix{TX}}, 
    multinomial_coefs::Vector{TC}, 
    ::Val{S}, ::Val{c}, ::Val{N}, ::Val{R}
) where {TY,K,TX,TC,S,c,N,R}

    num_modes_cell = count(==(c), S)
    mode = findfirst(==(c), S)
    last_mode_cell = mode + num_modes_cell - 1
    acc_modes = [j + mode - 1 for j in 1:num_modes_cell if j + mode - 1 >= 2]

    need_pref = last_mode_cell >= 3  # Flag whether we can save multiplies by computing prefix products
    x_needed  = sort!(unique!([collect(2:last_mode_cell-1); collect(mode+1:N)])) # Included X in prefix/suffix products, excluding mode 1
    peel = (mode == 1 && num_modes_cell >= 2)  # Flag whether to peel first i_1 iteration
    
    # Define functions for different symbols in expressions
    Iv(d)=Symbol(:i_,d);  Pv(d)=Symbol(:p_,d);   Av(d)=Symbol(:acc_,d)
    Xv(d)=Symbol(:x_,d);  SUF(d)=Symbol(:suf_,d); SP(d)=Symbol(:sp_,d)
    XM(d)=Symbol(:X_, S[d])  # transposed factor matrix at loop position d, defined in quote
    col_assign_loop(lhs, rhs) = :(for col in 1:$R; $lhs = $rhs; end)

    pre_exprs  = [Any[] for _ in 1:N]
    post_exprs = [Any[] for _ in 1:N]
    
    # Add pre- and post-expressions for each level
    for d in 1:N
        # Load row i_d of matrix X[S[d]] into x_d (pre)
        d in x_needed && push!(pre_exprs[d],
            col_assign_loop(:($(Xv(d))[col]), :($(XM(d))[col, $(Iv(d))])))

        # Compute suffix product x_d * ... * x_N (pre) if it will be used
        if d > mode
            push!(pre_exprs[d], col_assign_loop(:($(SUF(d))[col]),
                d == N ? :($(Xv(d))[col]) : :($(Xv(d))[col] * $(SUF(d+1))[col])))
        end

        # Compute coefficient p_d, the multiplicity of i_d among i_j for j in d:last_mode_cell (pre)
        if d >= max(mode, 2) && d <= last_mode_cell
            push!(pre_exprs[d], d == last_mode_cell ? :($(Pv(d)) = 1) :
                :($(Pv(d)) = ifelse($(Iv(d)) == $(Iv(d+1)), $(Pv(d+1)) + 1, 1)))
        end

        # For level d = 2, compute i_1 loop-invariant product sp_D = p_D * prod_{k=2...N, k != D} x_k,
        # i.e., the leave-one-out-product excluding x_1, but including the coefficient p_D,
        # for D in {2,..,last_mode_cell} if D >= mode (pre).
        # Also compute prefix product x_2 * ... * x_{last_mode_cell - 1} (pre).
        if d == 2
            need_pref && push!(pre_exprs[2], :(fill!(pref, one($TX))))
            for D in 2:last_mode_cell
                if D >= mode
                    fac = Any[Pv(D)]
                    D > 2 && push!(fac, :(pref[col]))
                    D < N && push!(fac, :($(SUF(D+1))[col]))
                    push!(pre_exprs[2], col_assign_loop(:($(SP(D))[col]),
                        length(fac) == 1 ? only(fac) : Expr(:call, :*, fac...)))
                end
                D < last_mode_cell && push!(pre_exprs[2],
                    col_assign_loop(:(pref[col]), :(pref[col] * $(Xv(D))[col])))
            end
        end

        # Zero (pre) and flush (post) accumulators
        if d in acc_modes
            !(peel && d == 2) && push!(pre_exprs[d], :(fill!($(Av(d)), zero($TX)))) # If peel && d == 2 we directly overwrite acc_2
            push!(post_exprs[d], :(for col in 1:$R
                                    result[col, $(Iv(d))] += $(Av(d))[col]
                                end))
        end
    end

    # Peel off first iteration of i_1 loop (i.e., i_1 = i_2) when mode == 1 and num_modes_cell >= 2 to reduce branching
    # and coefficient computation logic (pre).
    if peel
        push!(pre_exprs[2], :(yw = multinomial_coefs[vec_idx] * y[vec_idx]))
        push!(pre_exprs[2], :(tt = (p_2 + 1) * yw))
        push!(pre_exprs[2], col_assign_loop(:(acc_2[col]), :(tt * suf_2[col])))
        if last_mode_cell >= 3
            push!(pre_exprs[2], col_assign_loop(:(yx1[col]), :(yw * x_2[col])))
            for D in 3:last_mode_cell
                push!(pre_exprs[2], Expr(:if, :($(Iv(D)) != $(Iv(D-1))),
                    col_assign_loop(:($(Av(D))[col]),
                        :(muladd(yx1[col], $(SP(D))[col], $(Av(D))[col])))))
            end
        end
        push!(pre_exprs[2], :(vec_idx += 1))
    end

    chain(exprs) = Expr(:->, :d, foldr((k, rest) -> isempty(exprs[k]) ? rest :
                     Expr(:if, :(d == $k), Expr(:block, exprs[k]...), rest),
                     1:N; init = :nothing))
    pre, post = chain(pre_exprs), chain(post_exprs)

    # Expressions for allocations
    alloc = Expr(:block)
    for d in x_needed;  push!(alloc.args, :($(Xv(d))  = zeros(MVector{$R, $TX}))) end
    for d in mode+1:N;  push!(alloc.args, :($(SUF(d)) = zeros(MVector{$R, $TX}))) end
    for D in acc_modes; push!(alloc.args, :($(Av(D))  = zeros(MVector{$R, $TX}))) end
    for D in acc_modes; push!(alloc.args, :($(SP(D))  = zeros(MVector{$R, $TX}))) end
    need_pref           && push!(alloc.args, :(pref = zeros(MVector{$R, $TX})))
    !isempty(acc_modes) && push!(alloc.args, :(yx1  = zeros(MVector{$R, $TX})))

    # Expressions for loop body
    # Compute value of p_1*yw if mode != 1
    tt1_def = peel                ? :(tt1 = yw) :
              mode != 1           ? :nothing :
              num_modes_cell == 1 ? :(tt1 = yw) :
                                    :(tt1 = ifelse(i_1 == i_2, p_2 + 1, 1) * yw)

    yx1_def = isempty(acc_modes)  ? :nothing :
              :(for col in 1:$R
                    yx1[col] = yw * Xfirst[col, i_1]
                end)

    quote
        Sv = $(S)
        @nexprs $K k -> (X_k = permutedims(Xs[k]))
        Xfirst = $(Symbol(:X_, S[1]))
        $alloc
        vec_idx = 1
        @inbounds @nloops(
            $N, 
            i, 
            k -> (k == $N ? 1 :
                    k == 1 ? $(peel ? :(i_2 + 1) : (S[1] == S[2] ? :(i_2) : 1)) :
                    Sv[k] == Sv[k+1] ? i_{k+1} : 1):size(Xs[Sv[k]], 1),
            $pre, 
            $post, 
            begin   # Body expr
                yw = multinomial_coefs[vec_idx] * y[vec_idx]
                $tt1_def
                $yx1_def
                @nexprs $num_modes_cell j -> begin
                    if (j == 1 ? true : i_{j+$mode-1} != i_{j+$mode-2})
                        if j > $(2 - mode)
                            for col in 1:$R
                                acc_{j+$mode-1}[col] = muladd(yx1[col], sp_{j+$mode-1}[col], acc_{j+$mode-1}[col])
                            end
                        else
                            for col in 1:$R
                                result[col, i_1] = muladd(tt1, suf_2[col], result[col, i_1])
                            end
                        end
                    end
                end
                vec_idx += 1
            end
        )
        return result
    end
end

"""
    make_krp_row_mappings(
        M::SymCPD{T,N,K}, 
        vec_size::Int, 
        ::Val{S}, ::Val{c}
    ) where {T,N,K,S,c}

Make vectors of mappings to row in KRP matrix, to use in KRP-TTV
    gradient for factors for cell c.
vec_size should be length of the reduced vectorized derivative tensor.
S should be symmetry pattern for M (i.e, M.S).
"""
@generated function make_krp_row_mappings(
    M::SymCPD{T,N,K}, 
    vec_size::Int, 
    ::Val{S}, ::Val{c}
) where {T,N,K,S,c}
    num_modes_cell = count(==(c), S)
    mode = findfirst(==(c), S)
    S_reduced = (S[1:mode-1]..., S[mode+1:end]...)
    ns_expr = Expr(:tuple, (:(size(M.U[$mode], 1)) for mode in S_reduced)...)
    assigns = map(1:num_modes_cell) do j
        l = j + mode - 1
        idxs = Expr(:tuple, (Symbol(:i_, k) for k in 1:N if k != l)...)
        :(row_mappings[$l][vec_idx] = lin_reduced_general_sym($idxs, $(S_reduced), ns))
    end
    quote
        Sv = $(S)
        ns = $ns_expr
        row_mappings = Base.Cartesian.@ntuple $N k -> zeros(Int64, vec_size)
        vec_idx = 1
        @nloops(
            $N,
            i,
            k -> (k == $N ? 1 : Sv[k] == Sv[k+1] ? i_{k+1} : 1):size(M, k),
            begin
                $(assigns...)
                vec_idx += 1
            end
        )
        return row_mappings
    end
end
function lin_reduced_general_sym(I::NTuple{N, Int}, S::NTuple{N, Int}, sizes::NTuple{N, Int}) where {N}
    idx = 1
    cell_offset = 1
    for k in unique(S)
        cell_modes = findall(S .== k)
        num_modes = length(cell_modes)
        mode_size = sizes[k]
        total_idxs_cell = binomial(mode_size + num_modes - 1, num_modes)
        idx += cell_offset * (total_idxs_cell - sum(t -> binomial(mode_size - I[cell_modes[t]] + t - 1, t), 1:num_modes) - 1)
        cell_offset *= total_idxs_cell
    end
    return idx
end

"""
    factor_grad_krp_ttv!(
        GU_λ::NTuple{V, AbstractArray},  
        Y_vec::AbstractVector,
        kr_buffer::AbstractMatrix, 
        M::SymCPD{T,N,K},  
        krp_row_mappings::NTuple{N, Vector{Int}}, 
        multinomial_coefs::AbstractVector, 
        cell::Int
    ) where {V,T,N,K}

Compute the gradient for the factor matrix for the given cell by forming the Khatri-Rao product matrix 
    and using that matrix for factor terms of the TTV in all modes except one, writing the result to GU_λ[cell].
Uses kr_buffer to store the reduced Khatri-Rao product matrix.
"""
function factor_grad_krp_ttv!(
    GU_λ::NTuple{V, AbstractArray},  
    Y_vec::AbstractVector,
    kr_buffer::AbstractMatrix, 
    M::SymCPD{T,N,K},  
    krp_row_mappings::NTuple{N, Vector{Int}}, 
    multinomial_coefs::AbstractVector, 
    cell::Int
) where {V,T,N,K}
    (cell >= 1 && cell <= K) || 
        throw(ArgumentError("`cell` must satisfy 1 <= cell <= ngroups(M)"))

    mode = findfirst(==(cell), M.S)
    r = ncomps(M)
    S_reduced = M.S[setdiff(1:N,mode)]
    flip_group_ordering(k) = ngroups(M) - k + 1
    symmetric_kr_unweighted!(kr_buffer, reverse(flip_group_ordering.(S_reduced)), reverse(M.U)...)
    mode_sizes = ntuple(i -> size(M, i), N)
    GU_T = zeros(eltype(GU_λ[cell]), size(GU_λ[cell], 2), size(GU_λ[cell], 1))
    # Calling permutedims on the kr_bufer is not ideal, but doesn't add too much time
    GCPLosses.ttv_all_modes_except_one_krp!(GU_T, Y_vec, permutedims(kr_buffer), multinomial_coefs, krp_row_mappings, mode_sizes, Val(M.S), Val(cell), Val(N), Val(r))
    GU_λ[cell] .= permutedims(GU_T)
    rmul!(GU_λ[cell], Diagonal(M.λ))
    return GU_λ[cell]
end

"""
    ttv_all_modes_except_one_krp!(
        result::AbstractMatrix,
        y::Vector{T}, 
        Ktilde_T::Matrix{TK}, 
        multinomial_coefs::Vector{TC}, 
        row_mappings::NTuple{N, Vector{TM}}, 
        mode_sizes::NTuple{N, Int},
        ::Val{S},
        ::Val{c}, 
        ::Val{N}, 
        ::Val{R}
    ) where {T,TK,TM<:Integer,TC,N,S,c,R}

Compute the TTV in all modes except the first mode for cell c, i.e., the factors gradient
    for cell c minus scaling by diag(λ), 
    and write to result, which should be size R x n, where n is the mode size for cell c.
"""
@generated function ttv_all_modes_except_one_krp!(
    result::AbstractMatrix,
    y::Vector{T}, 
    Ktilde_T::Matrix{TK}, 
    multinomial_coefs::Vector{TC}, 
    row_mappings::NTuple{N, Vector{TM}}, 
    mode_sizes::NTuple{N, Int},
    ::Val{S},
    ::Val{c}, 
    ::Val{N}, 
    ::Val{R}
) where {T,TK,TM<:Integer,TC,N,S,c,R}

    num_modes_cell = count(==(c), S)
    mode = findfirst(==(c), S)
    last_mode = mode + num_modes_cell - 1
    acc_modes = [j + mode - 1 for j in 1:num_modes_cell if j + mode - 1 >= 2]
    peel = (mode == 1 && num_modes_cell >= 2)
    fire_free = peel ? 2 : 1
    save_kr1 = mode == 1

    Iv(d) = Symbol(:i_, d);  Pv(d) = Symbol(:p_, d); Av(d) = Symbol(:acc_, d)

    pre_lv  = [Any[] for _ in 1:N]
    post_lv = [Any[] for _ in 1:N]

    # Add expressions to zero accumulators (pre), and flush accumulators to result (post)
    for D in acc_modes
        !(peel && D == 2) && push!(pre_lv[D], :(fill!($(Av(D)), zero(TK))))
        push!(post_lv[D], :(for col in 1:$R
                                result[col, $(Iv(D))] += $(Av(D))[col]
                            end))
    end

    # Add expressions to compute p_d, the multiplicity of value i_d among positions d:last_mode
    for d in last_mode:-1:max(mode, 2)
        push!(pre_lv[d], d == last_mode ? :($(Pv(d)) = 1) :
              :($(Pv(d)) = ifelse($(Iv(d)) == $(Iv(d+1)), $(Pv(d+1)) + 1, 1)))
    end

    # Define expression for coefficient p_1 
    p1_def = peel                ? :(p_1 = 1) :
             mode != 1           ? :nothing :
             num_modes_cell == 1 ? :(p_1 = 1) :
                                   :(p_1 = ifelse(i_1 == i_2, p_2 + 1, 1))

    # Add expression to save KR row for (i_2, ..., i_N), invariant across i_1 loop
    if save_kr1
        push!(pre_lv[2], quote
            kr1 = rm_1[vec_idx]
            for col in 1:$R
                kr_row_save[col] = Ktilde_T[col, kr1]
            end
        end)
    end

    # Peel first iteration i_1 = i_2 of i_1 loop
    if peel
        push!(pre_lv[2], :(yw = multinomial_coefs[vec_idx] * y[vec_idx]))
        push!(pre_lv[2], :(tensor_term = (p_2 + 1) * yw))
        push!(pre_lv[2], :(for col in 1:$R
                               acc_2[col] = tensor_term * kr_row_save[col]
                           end))
        for D in 3:last_mode
            push!(pre_lv[2], Expr(:if, :($(Iv(D)) != $(Iv(D-1))), quote
                tt = $(Pv(D)) * yw
                kr_row = $(Symbol(:rm_, D))[vec_idx]
                for col in 1:$R
                    $(Av(D))[col] = muladd(tt, Ktilde_T[col, kr_row], $(Av(D))[col])
                end
            end))
        end
        push!(pre_lv[2], :(vec_idx += 1))
    end

    chain(lv) = Expr(:->, :d, foldr((k, rest) -> isempty(lv[k]) ? rest :
                     Expr(:if, :(d == $k), Expr(:block, lv[k]...), rest),
                     1:N; init = :nothing))
    pre, post = chain(pre_lv), chain(post_lv)

    acc_alloc = Expr(:block, (:($(Av(D)) = zeros(MVector{$R, TK})) for D in acc_modes)...)
    kr_save_alloc  = save_kr1 ? :(kr_row_save = zeros(MVector{$R, TK})) : :(nothing)

    # If mode = 1, uses saved KR row
    pos1_arm = save_kr1 ?
        :(for col in 1:$R
              result[col, v] = muladd(tensor_term, kr_row_save[col], result[col, v])
          end) :
        quote
            kr_row = rm_1[vec_idx]
            for col in 1:$R
                result[col, v] = muladd(tensor_term, Ktilde_T[col, kr_row], result[col, v])
            end
        end

    quote
        Sv = $(S)
        @nexprs $N j -> (rm_j = row_mappings[j])
        $kr_save_alloc
        $acc_alloc
        vec_idx = 1
        @inbounds @nloops(
            $N,
            i,
            k -> (k == $N ? 1 :
                    k == 1 ? $(peel ? :(i_2 + 1) : (S[1] == S[2] ? :(i_2) : 1)) :
                    Sv[k] == Sv[k+1] ? i_{k+1} : 1):mode_sizes[k],
            $pre,
            $post,
            begin   # Body expr
                yw = multinomial_coefs[vec_idx] * y[vec_idx]
                idx = @ntuple $N i
                $p1_def
                @nexprs $num_modes_cell j -> begin
                    if (j <= $fire_free ? true : idx[j+$mode-1] != idx[j+$mode-2])
                        v = idx[j+$mode-1]
                        tensor_term =  p_{j+$mode-1} * yw
                        if j > $(2 - mode)
                            kr_row = rm_{j+$mode-1}[vec_idx]
                            for col in 1:$R
                                acc_{j+$mode-1}[col] = muladd(tensor_term, Ktilde_T[col, kr_row], acc_{j+$mode-1}[col])
                            end
                        else
                            $pos1_arm
                        end
                    end
                end
                vec_idx += 1
            end
        )
        return result
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

