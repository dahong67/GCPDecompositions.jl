## Types

@testset "constructors" begin
    @testset "T=$T, K=$K" for T in [Float64, Float16], K in 0:2
        λfull = T[1, 100, 10000]
        U1full, U2full, U3full = T[1 2 3; 4 5 6], T[-1 0 1], T[1 2 3; 4 5 6; 7 8 9]
        λ = λfull[1:K]
        U1, U2, U3 = U1full[:,1:K], U2full[:,1:K], U3full[:,1:K]

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

@testset "ncomponents" begin
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

@testset "ndims" begin
    λ = [1, 100, 10000]
    U1, U2, U3 = [1 2 3; 4 5 6], [-1 0 1], [1 2 3; 4 5 6; 7 8 9]

    @test ndims(CPD{Int,0,Vector{Int},Matrix{Int}}(λ, ())) == 0
    @test ndims(CPD(λ, (U1,))) == 1
    @test ndims(CPD(λ, (U1, U2))) == 2
    @test ndims(CPD(λ, (U1, U2, U3))) == 3
end

@testset "size" begin
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

@testset "show / summary" begin
    M = CPD(rand.(2), rand.((3, 4, 5), 2))
    Mstring = sprint((t, s) -> show(t, "text/plain", s), M)
    λstring = sprint((t, s) -> show(t, "text/plain", s), M.λ)
    Ustrings = sprint.((t, s) -> show(t, "text/plain", s), M.U)
    @test Mstring == string(
        "$(summary(M))\nλ weights:\n$λstring",
        ["\nU[$i] factor matrix:\n$Ustring" for (i, Ustring) in enumerate(Ustrings)]...,
    )
end
