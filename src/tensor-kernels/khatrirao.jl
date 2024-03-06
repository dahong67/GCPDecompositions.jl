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
