## Tensor Kernel: khatrirao

"""
    khatrirao(A1, A2, ...)

Compute the Khatri-Rao product (i.e., the column-wise Kronecker product)
of the matrices `A1`, `A2`, etc.
"""
function khatrirao(A::Vararg{AbstractMatrix})
    I, r = _checked_khatrirao_dims(A...)
    return khatrirao!(similar(A[1], prod(I), r), A...)
end

"""
    khatrirao!(K, A1, A2, ...)

Compute the Khatri-Rao product (i.e., the column-wise Kronecker product)
of the matrices `A1`, `A2`, etc. and store the result in `K`.
"""
function khatrirao!(K::AbstractMatrix, A::Vararg{AbstractMatrix,N}) where {N}
    I, r = _checked_khatrirao_dims(A...)

    # Check output dimensions
    Base.require_one_based_indexing(K)
    size(K) == (prod(I), r) || throw(
        DimensionMismatch(
            "Output `K` must have size equal to `(prod(size.(A,1)), size(A[1],2))",
        ),
    )

    # Compute recursively, using a good order for intermediate multiplications
    if N == 1        # base case: N = 1
        K .= A[1]
    elseif N == 2    # base case: N = 2
        reshape(K, I[2], I[1], r) .= reshape(A[2], :, 1, r) .* reshape(A[1], 1, :, r)
    else             # recursion: N > 2
        n = argmin(n -> I[n] * I[n+1], 1:N-1)
        khatrirao!(K, A[1:n-1]..., khatrirao(A[n], A[n+1]), A[n+2:end]...)
    end

    return K
end

"""
    _checked_khatrirao_dims(A1, A2, ...)

Check that `A1`, `A2`, etc. have compatible dimensions for the Khatri-Rao product.
If so, return a tuple of the number of rows and the shared number of columns.
If not, throw an error.
"""
function _checked_khatrirao_dims(A::Vararg{AbstractMatrix})
    Base.require_one_based_indexing(A...)
    allequal(size.(A, 2)) || throw(
        DimensionMismatch(
            "Matrices in a Khatri-Rao product must have the same number of columns.",
        ),
    )
    return size.(A, 1), size(A[1], 2)
end

"""
    symmetric_kr!(Ktilde, S_reduced, A1, A2, ...)

Compute only the unique rows of the Khatri-Rao product of A1, A2, ..., AK,
with symmetry given by S_reduced, rescaling each row with the number of times it is duplicated,
and store the result in Ktilde. S_reduced has L groups of symmetric modes, and should include all
modes except mode n when computing the Khatri-Rao product for the mode-n MTTKRP.
"""
function symmetric_kr!(Ktilde::AbstractMatrix, S_reduced::NTuple{N}, A::Vararg{AbstractMatrix,L}) where {N, L}
    R = size(Ktilde, 2)
    if L == 1
        # return symmetric_self_kr!(Ktilde, A[1], Val(N))
        return symmetric_self_kr_partial_products_v2!(Ktilde, A[1], Val(N), Val(R))
        # return return symmetric_self_kr_iter_partial_products_v2!(Ktilde, A[1], Val(N), Val(R))
    else
        r = size(A[1], 2)
        groups = unique(S_reduced)
        intermediate_buffers = [similar(A[i], prod(k -> size(A[i],1)+k-1, 1:count(S_reduced .== i))÷factorial(count(S_reduced .== i)), r) for i in groups]
        for (buffer_idx, sym_group) in enumerate(groups)
            num_repeat = count(S_reduced .== sym_group)
            if num_repeat == 1
                intermediate_buffers[buffer_idx] .= A[sym_group]
            else
                # symmetric_self_kr!(intermediate_buffers[buffer_idx], A[sym_group], Val(count(S_reduced .== sym_group)))
                symmetric_self_kr_partial_products_v2!(intermediate_buffers[buffer_idx], A[sym_group], Val(count(S_reduced .== sym_group)), Val(R))
                # symmetric_self_kr_iter_partial_products_v2!(intermediate_buffers[buffer_idx], A[sym_group], Val(count(S_reduced .== sym_group)), Val(R))
            end
        end
        return khatrirao!(Ktilde, intermediate_buffers...)
    end
end

"""
    symmetric_self_kr!(Ktilde, A, N)

Compute only the unique rows of the Khatri-Rao product of A with itself 
N times, rescaling each row with the number of times it is duplicated,
store the result in Ktilde.
Implements more efficient direct computation of scaling αs for 
up to 4th order case, falls back on general form for higher orders.
"""
@generated function symmetric_self_kr!(Ktilde::AbstractMatrix, A::AbstractMatrix, ::Val{N}) where {N}
    quote
        n, r = size(A)
        if $N > 4
            α_num = factorial($N)
        end
        @inbounds for col in 1:r
            row = 1
            @nloops $N i k -> (k == N ? 1 : i_{k+1}):n begin
                if $N == 2
                    α = i_1 == i_2 ? 1 : 2
                elseif $N == 3
                    α = i_1 == i_2 ? (i_2 == i_3 ? 1 : 3) : (i_2 == i_3 ? 3 : 6)
                elseif $N == 4
                    n_equal = (i_1==i_2) + (i_1==i_3) + (i_1==i_4) + (i_2==i_3) + (i_2==i_4) + (i_3==i_4)
                    α = n_equal == 0 ? 24 : n_equal == 1 ? 12 : n_equal == 2 ? 6 : n_equal == 3 ? 4 : 1
                else
                    indices = @ntuple $N i 
                    α_denom = 1
                    n_equal = 1
                    for i in 2:$N
                        if indices[i] == indices[i-1]
                            n_equal += 1
                        else
                            α_denom *= factorial(n_equal)
                            n_equal = 1
                        end
                    end
                    α_denom *= factorial(n_equal)
                    α = α_num / α_denom
                end
                Ktilde[row,col] = α * prod(@ntuple $N j -> A[i_j, col])
                row += 1
            end
        end
        return Ktilde
    end
end
# function symmetric_self_kr_iter!(Ktilde::AbstractMatrix, A::AbstractMatrix, ::Val{N}, ::Val{R}) where {N,R}
#     n = size(A, 1)
#     if N > 4
#         α_num = factorial(N)
#     end
#     inds = GCPLosses.SymmetricIndices{N}(n)
#     @inbounds for col in 1:R
#         for (row,I) in enumerate(inds)
#             if N == 2
#                 α = I[1] == I[2] ? 1 : 2
#             elseif N == 3
#                 α = I[1] == I[2] ? (I[2] == I[3] ? 1 : 3) : (I[2] == I[3] ? 3 : 6)
#             elseif N == 4
#                 n_equal = (I[1]==I[2]) + (I[1]==I[3]) + (I[1]==I[4]) + (I[2]==I[3]) + (I[2]==I[4]) + (I[3]==I[4])
#                 α = n_equal == 0 ? 24 : n_equal == 1 ? 12 : n_equal == 2 ? 6 : n_equal == 3 ? 4 : 1
#             else 
#                 α_denom = 1
#                 n_equal = 1
#                 for i in 2:N
#                     if I[i] == I[i-1]
#                         n_equal += 1
#                     else
#                         α_denom *= factorial(n_equal)
#                         n_equal = 1
#                     end
#                 end
#                 α_denom *= factorial(n_equal)
#                 α = α_num / α_denom
#             end
#             Ktilde[row,col] = α * prod(j -> A[I[j], col], 1:N)
#         end
#     end
#     return Ktilde
# end
@generated function symmetric_self_kr_partial_products_v1!(Ktilde::AbstractMatrix{T}, A::AbstractMatrix, ::Val{N}, ::Val{R}) where {T,N,R}
    terms = [:(A[$(Symbol("i_$(k-1)")), col]) for k in 2:N]
    set_partial_product = :(partial_product = *( $(terms...) ))
    quote
        n = size(A,1)
        partial_product = zero(T)
        if $N > 4
            α_num = factorial($N)
        end
        @inbounds for col in 1:$R
            row = 1
            @nloops $(N-1) i k -> (k == $(N-1) ? 1 : i_{k+1}):n begin
                $set_partial_product
                for i_0 in i_1:n
                    if $N == 2
                        α = i_0 == i_1 ? 1 : 2
                    elseif $N == 3
                        α = i_0 == i_1 ? (i_1 == i_2 ? 1 : 3) : (i_1 == i_2 ? 3 : 6)
                    elseif $N == 4
                        n_equal = (i_0==i_1) + (i_0==i_2) + (i_0==i_3) + (i_1==i_2) + (i_1==i_3) + (i_2==i_3)
                        α = n_equal == 0 ? 24 : n_equal == 1 ? 12 : n_equal == 2 ? 6 : n_equal == 3 ? 4 : 1
                    else
                        indices = (i_0, (@ntuple $(N-1) i)...)
                        α_denom = 1
                        n_equal = 1
                        for i in 2:$N
                            if indices[i] == indices[i-1]
                                n_equal += 1
                            else
                                α_denom *= factorial(n_equal)
                                n_equal = 1
                            end
                        end
                        α_denom *= factorial(n_equal)
                        α = α_num / α_denom
                    end
                    Ktilde[row,col] = α * A[i_0,col] * partial_product
                    row += 1
                end
            end
        end
        return Ktilde
    end
end
# function symmetric_self_kr_iter_partial_products_v1!(Ktilde::AbstractMatrix{T}, A::AbstractMatrix, ::Val{N}, ::Val{R}) where {T,N,R}
#     n = size(A, 1)
#     if N > 4
#         α_num = factorial(N)
#     end
#     inds_minus_mode1 = GCPLosses.SymmetricIndices{N-1}(n)
#     partial_product = zero(T)
#     @inbounds for col in 1:R
#         row = 1
#         for I_minus_mode1 in inds_minus_mode1
#             partial_product = prod(j -> A[I_minus_mode1[j], col], 1:N-1)
#             for i1 in I_minus_mode1[1]:n
#                 I = CartesianIndex(i1, I_minus_mode1)
#                 if N == 2
#                     α = I[1] == I[2] ? 1 : 2
#                 elseif N == 3
#                     α = I[1] == I[2] ? (I[2] == I[3] ? 1 : 3) : (I[2] == I[3] ? 3 : 6)
#                 elseif N == 4
#                     n_equal = (I[1]==I[2]) + (I[1]==I[3]) + (I[1]==I[4]) + (I[2]==I[3]) + (I[2]==I[4]) + (I[3]==I[4])
#                     α = n_equal == 0 ? 24 : n_equal == 1 ? 12 : n_equal == 2 ? 6 : n_equal == 3 ? 4 : 1
#                 else 
#                     α_denom = 1
#                     n_equal = 1
#                     for i in 2:N
#                         if I[i] == I[i-1]
#                             n_equal += 1
#                         else
#                             α_denom *= factorial(n_equal)
#                             n_equal = 1
#                         end
#                     end
#                     α_denom *= factorial(n_equal)
#                     α = α_num / α_denom
#                 end
#                 Ktilde[row,col] = α * A[I[1], col] * partial_product
#                 row += 1
#                 # Ktilde[row,col] = α * prod(j -> A[I[j], col], 1:N)
#             end
#         end
#     end
#     return Ktilde
# end
@generated function symmetric_self_kr_partial_products_v2!(Ktilde::AbstractMatrix{T}, A::AbstractMatrix, ::Val{N}, ::Val{R}) where {T,N,R}
    terms = [:(A[$(Symbol("i_$(k-1)")), col]) for k in 2:N]
    set_partial_product = map(1:R) do j
        terms = [:(A[$(Symbol("i_$(k-1)")), $j]) for k in 2:N]
        :(partial_product[$j] = *( $(terms...) ))
    end
    quote
        n = size(A,1)
        partial_product = zeros(MVector{$R, T})
        if $N > 4
            α_num = factorial($N)
        end
        row = 1
        @inbounds @nloops $(N-1) i k -> (k == $(N-1) ? 1 : i_{k+1}):n begin
            $(set_partial_product...)
            for i_0 in i_1:n
                if $N == 2
                    α = i_0 == i_1 ? 1 : 2
                elseif $N == 3
                    α = i_0 == i_1 ? (i_1 == i_2 ? 1 : 3) : (i_1 == i_2 ? 3 : 6)
                elseif $N == 4
                    n_equal = (i_0==i_1) + (i_0==i_2) + (i_0==i_3) + (i_1==i_2) + (i_1==i_3) + (i_2==i_3)
                    α = n_equal == 0 ? 24 : n_equal == 1 ? 12 : n_equal == 2 ? 6 : n_equal == 3 ? 4 : 1
                else
                    indices = (i_0, (@ntuple $(N-1) i)...)
                    α_denom = 1
                    n_equal = 1
                    for i in 2:$N
                        if indices[i] == indices[i-1]
                            n_equal += 1
                        else
                            α_denom *= factorial(n_equal)
                            n_equal = 1
                        end
                    end
                    α_denom *= factorial(n_equal)
                    α = α_num / α_denom
                end
                for j in 1:$R
                    Ktilde[row,j] = α * A[i_0,j] * partial_product[j]
                end 
                row += 1
            end
        end
        return Ktilde
    end
end
# function symmetric_self_kr_iter_partial_products_v2!(Ktilde::AbstractMatrix{T}, A::AbstractMatrix, ::Val{N}, ::Val{R}) where {T,N,R}
#     n = size(A, 1)
#     if N > 4
#         α_num = factorial(N)
#     end
#     inds_minus_mode1 = GCPLosses.SymmetricIndices{N-1}(n)
#     partial_product = zeros(MVector{R, T})
#     row = 1
#     @inbounds for I_minus_mode1 in inds_minus_mode1
#         for j in 1:R
#             partial_product[j] = prod(k -> A[I_minus_mode1[k], j], 1:N-1)
#         end
#         for i1 in I_minus_mode1[1]:n
#             I = CartesianIndex(i1, I_minus_mode1)
#             if N == 2
#                 α = I[1] == I[2] ? 1 : 2
#             elseif N == 3
#                 α = I[1] == I[2] ? (I[2] == I[3] ? 1 : 3) : (I[2] == I[3] ? 3 : 6)
#             elseif N == 4
#                 n_equal = (I[1]==I[2]) + (I[1]==I[3]) + (I[1]==I[4]) + (I[2]==I[3]) + (I[2]==I[4]) + (I[3]==I[4])
#                 α = n_equal == 0 ? 24 : n_equal == 1 ? 12 : n_equal == 2 ? 6 : n_equal == 3 ? 4 : 1
#             else 
#                 α_denom = 1
#                 n_equal = 1
#                 for i in 2:N
#                     if I[i] == I[i-1]
#                         n_equal += 1
#                     else
#                         α_denom *= factorial(n_equal)
#                         n_equal = 1
#                     end
#                 end
#                 α_denom *= factorial(n_equal)
#                 α = α_num / α_denom
#             end
#             for j in 1:R
#                 Ktilde[row,j] = α * A[i1,j] * partial_product[j]
#             end 
#             # Ktilde[row,col] = α * A[I[1], col] * partial_product
#             row += 1
#             # Ktilde[row,col] = α * prod(j -> A[I[j], col], 1:N)
#         end
#     end
#     return Ktilde
# end