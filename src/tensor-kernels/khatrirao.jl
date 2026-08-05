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
function symmetric_kr!(Ktilde::AbstractMatrix, S_reduced::NTuple{N}, multinomial_coefs::NTuple{L}, A::Vararg{AbstractMatrix,L}) where {N, L}
    R = size(Ktilde, 2)
    if L == 1
        return symmetric_self_kr!(Ktilde, A[1], multinomial_coefs[1], Val(N), Val(R))
    else
        r = size(A[1], 2)
        groups = unique(S_reduced)
        intermediate_buffers = [similar(A[i], prod(k -> size(A[i],1)+k-1, 1:count(S_reduced .== i))÷factorial(count(S_reduced .== i)), r) for i in groups]
        for (buffer_idx, sym_group) in enumerate(groups)
            num_repeat = count(S_reduced .== sym_group)
            if num_repeat == 1
                intermediate_buffers[buffer_idx] .= A[sym_group]
            else
                symmetric_self_kr!(intermediate_buffers[buffer_idx], A[sym_group], multinomial_coefs[buffer_idx], Val(num_repeat), Val(R))
            end
        end
        return khatrirao!(Ktilde, intermediate_buffers...)
    end
end
@generated function symmetric_self_kr!(Ktilde::AbstractMatrix{T}, A::AbstractMatrix, multionmial_coefs::Vector, ::Val{N}, ::Val{R}) where {T,N,R}
    quote
        n = size(A, 1)
        @inbounds for col in 1:$R
            row = 1
            @nloops(
                $N,
                i,
                k -> (k == $N ? 1 : i_{k+1}):n,
                d -> d == 1 ? nothing : d == $N ? a_d = A[i_d, col] : a_d = A[i_d, col] * a_{d+1},
                begin
                    α = multionmial_coefs[row]
                    Ktilde[row,col] = α * A[i_1,col] * a_2
                    row += 1
                end
            )
        end
        return Ktilde
    end
end

"""
    symmetric_kr_unweighted!(Ktilde_T, S_reduced, A1, A2, ...)

Compute only the unique rows of the Khatri-Rao product of A1, A2, ..., AK,
with symmetry given by S_reduced, without rescaling each row with the number of times it is duplicated,
and store the result in Ktilde_T. S_reduced has L groups of symmetric modes, and should include all
modes except mode n when computing the Khatri-Rao product for the mode-n MTTKRP.
Used for KRP-TTSV function for factor gradients.
"""
function symmetric_kr_unweighted!(Ktilde::AbstractMatrix, S_reduced::NTuple{N}, A::Vararg{AbstractMatrix,L}) where {N, L}
    R = size(Ktilde, 2)
    if L == 1
        return symmetric_self_kr_unweighted!(Ktilde, A[1], Val(N), Val(R))
    else
        r = size(A[1], 2)
        groups = unique(S_reduced)
        intermediate_buffers = [similar(A[i], prod(k -> size(A[i],1)+k-1, 1:count(S_reduced .== i))÷factorial(count(S_reduced .== i)), r) for i in groups]
        for (buffer_idx, sym_group) in enumerate(groups)
            num_repeat = count(S_reduced .== sym_group)
            if num_repeat == 1
                intermediate_buffers[buffer_idx] .= A[sym_group]
            else
                symmetric_self_kr_unweighted!(intermediate_buffers[buffer_idx], A[sym_group], Val(count(S_reduced .== sym_group)), Val(R))
            end
        end
        return khatrirao!(Ktilde, intermediate_buffers...)
    end
end
@generated function symmetric_self_kr_unweighted!(Ktilde::AbstractMatrix{T}, A::AbstractMatrix, ::Val{N}, ::Val{R}) where {T,N,R}
    quote
        n = size(A, 1)
        @inbounds for col in 1:$R
            row = 1
            @nloops(
                $N,
                i,
                k -> (k == $N ? 1 : i_{k+1}):n,
                d -> d == 1 ? nothing : 
                d == $N ? 
                    a_d = A[i_d, col] 
                : 
                    a_d = A[i_d, col] * a_{d+1}
                ,
                begin
                    Ktilde[row,col] = A[i_1,col] * a_2
                    row += 1
                end
            )
        end
        return Ktilde
    end
end