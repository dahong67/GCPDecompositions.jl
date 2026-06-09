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
    symmetric_kr!(Ktilde, S, A1, A2, ...)

Compute only the unique rows of the Khatri-Rao product of A1, A2, ..., AK,
with symmetry given by S, rescaling each row with the number of times it is duplicated,
and store the result in Ktilde. S has L groups of symmetric modes, and should not include
mode n when computing the Khatri-Rao product for the mode-n MTTKRP.
"""
function symmetric_kr!(Ktilde::AbstractMatrix, S::NTuple{N}, A::Vararg{AbstractMatrix,L}) where {N, L}
    if L == 1
        return symmetric_self_kr!(Ktilde, A[1], Val(N))
    else
        r = size(A[1], 2)
        groups = unique(S)
        intermediate_buffers = [similar(A[i], prod(k -> size(A[i],1)+k-1, 1:count(S .== i))÷factorial(count(S .== i)), r) for i in groups]
        for (buffer_idx, sym_group) in enumerate(groups)
            num_repeat = count(S .== sym_group)
            if num_repeat == 1
                intermediate_buffers[buffer_idx] .= A[sym_group]
            else
                symmetric_self_kr!(intermediate_buffers[buffer_idx], A[sym_group], Val(count(S .== sym_group)))
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
        for col in 1:r
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