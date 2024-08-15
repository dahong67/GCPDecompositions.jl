## CP decomposition type

@testitem "constructors" begin
    using OffsetArrays

    @testset "T=$T, K=$K" for T in [Float64, Float16], K in 0:2
        λfull = T[1, 100, 10000]
        U1full, U2full, U3full = T[1 2 3; 4 5 6], T[-1 0 1], T[1 2 3; 4 5 6; 7 8 9]
        λ = λfull[1:K]
        U1, U2, U3 = U1full[:, 1:K], U2full[:, 1:K], U3full[:, 1:K]

        # Check type for various orders
        @test CPD{T,0,Vector{T},Matrix{T}}(λ, ()) isa CPD{T,0,Vector{T},Matrix{T}}
        @test CPD(λ, (U1,)) isa CPD{T,1,Vector{T},Matrix{T}}
        @test CPD(λ, (U1, U2)) isa CPD{T,2,Vector{T},Matrix{T}}
        @test CPD(λ, (U1, U2, U3)) isa CPD{T,3,Vector{T},Matrix{T}}

        # Check requirement of one-based indexing
        O1, O2 = OffsetArray(U1, 0:1, 0:K-1), OffsetArray(U2, 0:0, 0:K-1)
        @test_throws ArgumentError CPD(λ, (O1, O2))

        # Check dimension matching (for number of components)
        @test_throws DimensionMismatch CPD(λfull, (U1, U2, U3))
        @test_throws DimensionMismatch CPD(λ, (U1full, U2, U3))
        @test_throws DimensionMismatch CPD(λ, (U1, U2full, U3))
        @test_throws DimensionMismatch CPD(λ, (U1, U2, U3full))
    end
end

@testitem "ncomponents" begin
    λ = [1, 100, 10000]
    U1, U2, U3 = [1 2 3; 4 5 6], [-1 0 1], [1 2 3; 4 5 6; 7 8 9]

    @test ncomponents(CPD(λ, (U1,))) ==
          ncomponents(CPD(λ, (U1, U2))) ==
          ncomponents(CPD(λ, (U1, U2, U3))) ==
          3
    @test ncomponents(CPD(λ[1:2], (U1[:, 1:2],))) ==
          ncomponents(CPD(λ[1:2], (U1[:, 1:2], U2[:, 1:2]))) ==
          ncomponents(CPD(λ[1:2], (U1[:, 1:2], U2[:, 1:2], U3[:, 1:2]))) ==
          2
    @test ncomponents(CPD(λ[1:1], (U1[:, 1:1],))) ==
          ncomponents(CPD(λ[1:1], (U1[:, 1:1], U2[:, 1:1]))) ==
          ncomponents(CPD(λ[1:1], (U1[:, 1:1], U2[:, 1:1], U3[:, 1:1]))) ==
          1
    @test ncomponents(CPD(λ[1:0], (U1[:, 1:0],))) ==
          ncomponents(CPD(λ[1:0], (U1[:, 1:0], U2[:, 1:0]))) ==
          ncomponents(CPD(λ[1:0], (U1[:, 1:0], U2[:, 1:0], U3[:, 1:0]))) ==
          0
end

@testitem "ndims" begin
    λ = [1, 100, 10000]
    U1, U2, U3 = [1 2 3; 4 5 6], [-1 0 1], [1 2 3; 4 5 6; 7 8 9]

    @test ndims(CPD{Int,0,Vector{Int},Matrix{Int}}(λ, ())) == 0
    @test ndims(CPD(λ, (U1,))) == 1
    @test ndims(CPD(λ, (U1, U2))) == 2
    @test ndims(CPD(λ, (U1, U2, U3))) == 3
end

@testitem "size" begin
    λ = [1, 100, 10000]
    U1, U2, U3 = [1 2 3; 4 5 6], [-1 0 1], [1 2 3; 4 5 6; 7 8 9]

    @test size(CPD(λ, (U1,))) == (size(U1, 1),)
    @test size(CPD(λ, (U1, U2))) == (size(U1, 1), size(U2, 1))
    @test size(CPD(λ, (U1, U2, U3))) == (size(U1, 1), size(U2, 1), size(U3, 1))

    M = CPD(λ, (U1, U2, U3))
    @test size(M, 1) == 2
    @test size(M, 2) == 1
    @test size(M, 3) == 3
    @test size(M, 4) == 1
end

@testitem "show / summary" begin
    M = CPD(rand.(2), rand.((3, 4, 5), 2))
    Mstring = sprint((t, s) -> show(t, "text/plain", s), M)
    λstring = sprint((t, s) -> show(t, "text/plain", s), M.λ)
    Ustrings = sprint.((t, s) -> show(t, "text/plain", s), M.U)
    @test Mstring == string(
        "$(summary(M))\nλ weights:\n$λstring",
        ["\nU[$k] factor matrix:\n$Ustring" for (k, Ustring) in enumerate(Ustrings)]...,
    )
end

@testitem "getindex" begin
    @testset "K=$K" for K in 0:2
        T = Float64
        λfull = T[1, 100, 10000]
        U1full, U2full, U3full = T[1 2 3; 4 5 6], T[-1 0 1], T[1 2 3; 4 5 6; 7 8 9]
        λ = λfull[1:K]
        U1, U2, U3 = U1full[:, 1:K], U2full[:, 1:K], U3full[:, 1:K]

        M = CPD(λ, (U1, U2, U3))
        for i1 in axes(U1, 1), i2 in axes(U2, 1), i3 in axes(U3, 1)
            Mi = sum(λ .* U1[i1, :] .* U2[i2, :] .* U3[i3, :])
            @test Mi == M[i1, i2, i3]
            @test Mi == M[CartesianIndex((i1, i2, i3))]
        end
        @test_throws BoundsError M[size(U1, 1)+1, 1, 1]
        @test_throws BoundsError M[1, size(U2, 1)+1, 1]
        @test_throws BoundsError M[1, 1, size(U3, 1)+1]

        M = CPD(λ, (U1, U2))
        for i1 in axes(U1, 1), i2 in axes(U2, 1)
            Mi = sum(λ .* U1[i1, :] .* U2[i2, :])
            @test Mi == M[i1, i2]
            @test Mi == M[CartesianIndex((i1, i2))]
        end
        @test_throws BoundsError M[size(U1, 1)+1, 1]
        @test_throws BoundsError M[1, size(U2, 1)+1]

        M = CPD(λ, (U1,))
        for i1 in axes(U1, 1)
            Mi = sum(λ .* U1[i1, :])
            @test Mi == M[i1]
            @test Mi == M[CartesianIndex((i1,))]
        end
        @test_throws BoundsError M[size(U1, 1)+1]
    end
end

@testitem "norm" begin
    using LinearAlgebra

    @testset "K=$K" for K in 0:2
        T = Float64
        λfull = T[1, 100, 10000]
        U1full, U2full, U3full = T[1 2 3; 4 5 6], T[-1 0 1], T[1 2 3; 4 5 6; 7 8 9]
        λ = λfull[1:K]
        U1, U2, U3 = U1full[:, 1:K], U2full[:, 1:K], U3full[:, 1:K]

        M = CPD(λ, (U1, U2, U3))
        @test norm(M) ==
              norm(M, 2) ==
              sqrt(sum(abs2, M[I] for I in CartesianIndices(size(M))))
        @test norm(M, 1) == sum(abs, M[I] for I in CartesianIndices(size(M)))
        @test norm(M, 3) ==
              (sum(m -> abs(m)^3, M[I] for I in CartesianIndices(size(M))))^(1 / 3)

        M = CPD(λ, (U1, U2))
        @test norm(M) ==
              norm(M, 2) ==
              sqrt(sum(abs2, M[I] for I in CartesianIndices(size(M))))
        @test norm(M, 1) == sum(abs, M[I] for I in CartesianIndices(size(M)))
        @test norm(M, 3) ==
              (sum(m -> abs(m)^3, M[I] for I in CartesianIndices(size(M))))^(1 / 3)

        M = CPD(λ, (U1,))
        @test norm(M) ==
              norm(M, 2) ==
              sqrt(sum(abs2, M[I] for I in CartesianIndices(size(M))))
        @test norm(M, 1) == sum(abs, M[I] for I in CartesianIndices(size(M)))
        @test norm(M, 3) ==
              (sum(m -> abs(m)^3, M[I] for I in CartesianIndices(size(M))))^(1 / 3)
    end
end

@testset "Array" begin
    @testset "N=$N, K=$K" for N in 1:3, K in 0:2
        T = Float64
        λfull = T[1, 100, 10000]
        U1full, U2full, U3full = T[1 2 3; 4 5 6], T[-1 0 1], T[1 2 3; 4 5 6; 7 8 9]
        λ = λfull[1:K]
        U = (U1full[:, 1:K], U2full[:, 1:K], U3full[:, 1:K])[1:N]
        M = CPD(λ, U)

        X = CPD_to_Array(M, λfull)
        @test all(I -> M[I] == X[I], CartesianIndices(X))
    end
end