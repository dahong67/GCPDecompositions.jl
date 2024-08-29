## CPD component type

@testitem "constructors" begin
    using OffsetArrays

    @testset "T=$T" for T in [Float64, Float16]
        λ = T(100)
        u1, u2, u3 = T[1, 4], T[-1], T[2, 5, 8]

        # Check type for various orders
        @test CPDComp{T,0,Vector{T}}(λ, ()) isa CPDComp{T,0,Vector{T}}
        @test CPDComp(λ, (u1,)) isa CPDComp{T,1,Vector{T}}
        @test CPDComp(λ, (u1, u2)) isa CPDComp{T,2,Vector{T}}
        @test CPDComp(λ, (u1, u2, u3)) isa CPDComp{T,3,Vector{T}}

        # Check requirement of one-based indexing
        O1, O2 = OffsetArray(u1, 0:1), OffsetArray(u2, 0:0)
        @test_throws ArgumentError CPDComp(λ, (O1, O2))
    end
end

@testitem "ndims" begin
    λ = 100
    u1, u2, u3 = [1, 4], [-1], [2, 5, 8]

    @test ndims(CPDComp{Int,0,Vector{Int}}(λ, ())) == 0
    @test ndims(CPDComp(λ, (u1,))) == 1
    @test ndims(CPDComp(λ, (u1, u2))) == 2
    @test ndims(CPDComp(λ, (u1, u2, u3))) == 3
end

@testitem "size" begin
    λ = 100
    u1, u2, u3 = [1, 4], [-1], [2, 5, 8]

    @test size(CPDComp(λ, (u1,))) == (length(u1),)
    @test size(CPDComp(λ, (u1, u2))) == (length(u1), length(u2))
    @test size(CPDComp(λ, (u1, u2, u3))) == (length(u1), length(u2), length(u3))

    M = CPDComp(λ, (u1, u2, u3))
    @test size(M, 1) == 2
    @test size(M, 2) == 1
    @test size(M, 3) == 3
    @test size(M, 4) == 1
end

@testitem "show / summary" begin
    M = CPDComp(rand(), rand.((3, 4, 5)))
    Mstring = sprint((t, s) -> show(t, "text/plain", s), M)
    λstring = sprint((t, s) -> show(t, "text/plain", s), M.λ)
    ustrings = sprint.((t, s) -> show(t, "text/plain", s), M.u)
    @test Mstring == string(
        "$(summary(M))\nλ weight:\n$λstring",
        ["\nu[$k] factor vector:\n$ustring" for (k, ustring) in enumerate(ustrings)]...,
    )
end

@testitem "getindex" begin
    T = Float64
    λ = T(100)
    u1, u2, u3 = T[1, 4], T[-1], T[2, 5, 8]

    M = CPDComp(λ, (u1, u2, u3))
    for i1 in axes(u1, 1), i2 in axes(u2, 1), i3 in axes(u3, 1)
        Mi = λ * u1[i1] * u2[i2] * u3[i3]
        @test Mi == M[i1, i2, i3]
        @test Mi == M[CartesianIndex((i1, i2, i3))]
    end
    @test_throws BoundsError M[length(u1)+1, 1, 1]
    @test_throws BoundsError M[1, length(u2)+1, 1]
    @test_throws BoundsError M[1, 1, length(u3)+1]

    M = CPDComp(λ, (u1, u2))
    for i1 in axes(u1, 1), i2 in axes(u2, 1)
        Mi = λ * u1[i1] * u2[i2]
        @test Mi == M[i1, i2]
        @test Mi == M[CartesianIndex((i1, i2))]
    end
    @test_throws BoundsError M[length(u1)+1, 1]
    @test_throws BoundsError M[1, length(u2)+1]

    M = CPDComp(λ, (u1,))
    for i1 in axes(u1, 1)
        Mi = λ * u1[i1]
        @test Mi == M[i1]
        @test Mi == M[CartesianIndex((i1,))]
    end
    @test_throws BoundsError M[length(u1)+1]
end

@testitem "Array" begin
    @testset "N=$N" for N in 1:3
        T = Float64
        λ = T(100)
        u1, u2, u3 = T[1, 4], T[-1], T[2, 5, 8]
        M = CPDComp(λ, (u1, u2, u3))

        X = Array(M)
        @test all(I -> M[I] == X[I], CartesianIndices(X))
    end
end

@testitem "norm" begin
    using LinearAlgebra

    T = Float64
    λ = T(100)
    u1, u2, u3 = T[1, 4], T[-1], T[2, 5, 8]

    M = CPDComp(λ, (u1, u2, u3))
    @test norm(M) == norm(M, 2) == sqrt(sum(abs2, M[I] for I in CartesianIndices(size(M))))
    @test norm(M, 1) == sum(abs, M[I] for I in CartesianIndices(size(M)))
    @test norm(M, 3) ==
          (sum(m -> abs(m)^3, M[I] for I in CartesianIndices(size(M))))^(1 / 3)

    M = CPDComp(λ, (u1, u2))
    @test norm(M) == norm(M, 2) == sqrt(sum(abs2, M[I] for I in CartesianIndices(size(M))))
    @test norm(M, 1) == sum(abs, M[I] for I in CartesianIndices(size(M)))
    @test norm(M, 3) ==
          (sum(m -> abs(m)^3, M[I] for I in CartesianIndices(size(M))))^(1 / 3)

    M = CPDComp(λ, (u1,))
    @test norm(M) == norm(M, 2) == sqrt(sum(abs2, M[I] for I in CartesianIndices(size(M))))
    @test norm(M, 1) == sum(abs, M[I] for I in CartesianIndices(size(M)))
    @test norm(M, 3) ==
          (sum(m -> abs(m)^3, M[I] for I in CartesianIndices(size(M))))^(1 / 3)
end
