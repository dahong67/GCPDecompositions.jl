## SymCP decomposition type

@testitem "constructors" begin
    using OffsetArrays

    @testset "T=$T, K=$K" for T in [Float64, Float16], K in 0:2
        λfull = T[1, 100, 10000]
        U1full, U2full, U3full = T[1 2 3; 4 5 6], T[-1 0 1], T[1 2 3; 4 5 6; 7 8 9]
        λ = λfull[1:K]
        U1, U2, U3 = U1full[:, 1:K], U2full[:, 1:K], U3full[:, 1:K]

        # Check type for various orders
        @test SymCPD{T,0,0,Vector{T},Matrix{T}}(λ, (), ()) isa
              SymCPD{T,0,0,Vector{T},Matrix{T}}
        @test SymCPD(λ, (U1,), (1,)) isa SymCPD{T,1,1,Vector{T},Matrix{T}}
        @test SymCPD(λ, (U1, U2), (1, 2)) isa SymCPD{T,2,2,Vector{T},Matrix{T}}
        @test SymCPD(λ, (U1, U2, U3), (1, 2, 3)) isa SymCPD{T,3,3,Vector{T},Matrix{T}}

        # Check requirement of one-based indexing
        O1, O2 = OffsetArray(U1, 0:1, 0:K-1), OffsetArray(U2, 0:0, 0:K-1)
        @test_throws ArgumentError SymCPD(λ, (O1, O2), (1, 2))

        # Check dimension matching (for number of components)
        @test_throws DimensionMismatch SymCPD(λfull, (U1, U2, U3), (1, 2, 3))
        @test_throws DimensionMismatch SymCPD(λ, (U1full, U2, U3), (1, 2, 3))
        @test_throws DimensionMismatch SymCPD(λ, (U1, U2full, U3), (1, 2, 3))
        @test_throws DimensionMismatch SymCPD(λ, (U1, U2, U3full), (1, 2, 3))

        # Check different symmetric cases
        @test SymCPD(λ, (U1,), (1, 1, 1)) isa SymCPD{T,3,1,Vector{T},Matrix{T}}
        @test SymCPD(λ, (U1, U2), (1, 2, 1)) isa SymCPD{T,3,2,Vector{T},Matrix{T}}
        @test SymCPD(λ, (U1, U2), (1, 1, 2)) isa SymCPD{T,3,2,Vector{T},Matrix{T}}
        @test SymCPD(λ, (U1, U2), (2, 1, 1)) isa SymCPD{T,3,2,Vector{T},Matrix{T}}

        # Check dimension matching for S
        @test_throws DimensionMismatch SymCPD(λ, (U1, U2), (1, 2, 3))
        @test_throws DimensionMismatch SymCPD(λ, (U1, U2), (2, 2, 2))
        @test_throws DimensionMismatch SymCPD(λ, (U1, U2), (1, 2, 3))
    end
end

@testitem "ncomps" begin
    λ = [1, 100, 10000]
    U1, U2, U3 = [1 2 3; 4 5 6], [-1 0 1], [1 2 3; 4 5 6; 7 8 9]

    @test ncomps(SymCPD(λ, (U1,), (1,))) ==
          ncomps(SymCPD(λ, (U1, U2), (1, 2))) ==
          ncomps(SymCPD(λ, (U1, U2, U3), (1, 2, 3, 3))) ==
          ncomps(SymCPD(λ, (U1, U2, U3), (2, 1, 1, 2, 3))) ==
          3
    @test ncomps(SymCPD(λ[1:2], (U1[:, 1:2],), (1,))) ==
          ncomps(SymCPD(λ[1:2], (U1[:, 1:2], U2[:, 1:2]), (1, 2))) ==
          ncomps(SymCPD(λ[1:2], (U1[:, 1:2], U2[:, 1:2], U3[:, 1:2]), (1, 2, 3))) ==
          ncomps(SymCPD(λ[1:2], (U1[:, 1:2], U2[:, 1:2], U3[:, 1:2]), (3, 1, 1, 2, 3)))
    2
    @test ncomps(SymCPD(λ[1:1], (U1[:, 1:1],), (1,))) ==
          ncomps(SymCPD(λ[1:1], (U1[:, 1:1], U2[:, 1:1]), (1, 2))) ==
          ncomps(SymCPD(λ[1:1], (U1[:, 1:1], U2[:, 1:1], U3[:, 1:1]), (1, 2, 3))) ==
          ncomps(SymCPD(λ[1:1], (U1[:, 1:1], U2[:, 1:1], U3[:, 1:1]), (3, 1, 1, 2, 3, 1)))
    1
    @test ncomps(SymCPD(λ[1:0], (U1[:, 1:0],), (1,))) ==
          ncomps(SymCPD(λ[1:0], (U1[:, 1:0], U2[:, 1:0]), (1, 2))) ==
          ncomps(SymCPD(λ[1:0], (U1[:, 1:0], U2[:, 1:0], U3[:, 1:0]), (1, 2, 3))) ==
          ncomps(
              SymCPD(λ[1:0], (U1[:, 1:0], U2[:, 1:0], U3[:, 1:0]), (1, 2, 3, 2, 2, 2)),
          ) ==
          0
end

@testitem "ndims" begin
    λ = [1, 100, 10000]
    U1, U2, U3 = [1 2 3; 4 5 6], [-1 0 1], [1 2 3; 4 5 6; 7 8 9]

    @test ndims(SymCPD{Int,0,0,Vector{Int},Matrix{Int}}(λ, (), ())) == 0
    @test ndims(SymCPD(λ, (U1,), (1,))) == 1
    @test ndims(SymCPD(λ, (U1, U2), (1, 2))) == 2
    @test ndims(SymCPD(λ, (U1, U2, U3), (1, 2, 3))) == 3
    @test ndims(SymCPD(λ, (U1, U2), (1, 2, 2))) == 3
    @test ndims(SymCPD(λ, (U1,), (1, 1, 1))) == 3
end

@testitem "size" begin
    λ = [1, 100, 10000]
    U1, U2, U3 = [1 2 3; 4 5 6], [-1 0 1], [1 2 3; 4 5 6; 7 8 9]

    @test size(SymCPD(λ, (U1,), (1,))) == (size(U1, 1),)
    @test size(SymCPD(λ, (U1, U2), (1, 2))) == (size(U1, 1), size(U2, 1))
    @test size(SymCPD(λ, (U1, U2, U3), (1, 2, 3))) ==
          (size(U1, 1), size(U2, 1), size(U3, 1))

    M = SymCPD(λ, (U1, U2, U3), (1, 2, 3, 3))
    @test size(M, 1) == 2
    @test size(M, 2) == 1
    @test size(M, 3) == 3
    @test size(M, 4) == 3
    @test size(M, 5) == 1
end

@testitem "getindex" begin
    @testset "K=$K" for K in 0:2
        T = Float64
        λfull = T[1, 100, 10000]
        U1full, U2full, U3full = T[1 2 3; 4 5 6], T[-1 0 1], T[1 2 3; 4 5 6; 7 8 9]
        λ = λfull[1:K]
        U1, U2, U3 = U1full[:, 1:K], U2full[:, 1:K], U3full[:, 1:K]

        M = SymCPD(λ, (U1, U2, U3), (1, 2, 3))
        for i1 in axes(U1, 1), i2 in axes(U2, 1), i3 in axes(U3, 1)
            Mi = sum(λ .* U1[i1, :] .* U2[i2, :] .* U3[i3, :])
            @test Mi == M[i1, i2, i3]
            @test Mi == M[CartesianIndex((i1, i2, i3))]
        end
        @test_throws BoundsError M[size(U1, 1)+1, 1, 1]
        @test_throws BoundsError M[1, size(U2, 1)+1, 1]
        @test_throws BoundsError M[1, 1, size(U3, 1)+1]

        M = SymCPD(λ, (U1, U2), (1, 2))
        for i1 in axes(U1, 1), i2 in axes(U2, 1)
            Mi = sum(λ .* U1[i1, :] .* U2[i2, :])
            @test Mi == M[i1, i2]
            @test Mi == M[CartesianIndex((i1, i2))]
        end
        @test_throws BoundsError M[size(U1, 1)+1, 1]
        @test_throws BoundsError M[1, size(U2, 1)+1]

        M = SymCPD(λ, (U1,), (1,))
        for i1 in axes(U1, 1)
            Mi = sum(λ .* U1[i1, :])
            @test Mi == M[i1]
            @test Mi == M[CartesianIndex((i1,))]
        end
        @test_throws BoundsError M[size(U1, 1)+1]

        # Symmetric cases
        M = SymCPD(λ, (U3,), (1, 1, 1))
        for i1 in axes(U3, 1), i2 in axes(U3, 1), i3 in axes(U3, 1)
            Mi = sum(λ .* U3[i1, :] .* U3[i2, :] .* U3[i3, :])
            @test Mi == M[i1, i2, i3]
            @test Mi == M[CartesianIndex((i1, i2, i3))]
        end
        @test_throws BoundsError M[size(U3, 1)+1, 1, 1]
        @test_throws BoundsError M[1, size(U3, 1)+1, 1]
        @test_throws BoundsError M[1, 1, size(U3, 1)+1]

        M = SymCPD(λ, (U2, U3), (1, 2, 1))
        for i1 in axes(U2, 1), i2 in axes(U3, 1), i3 in axes(U2, 1)
            Mi = sum(λ .* U2[i1, :] .* U3[i2, :] .* U2[i3, :])
            @test Mi == M[i1, i2, i3]
            @test Mi == M[CartesianIndex((i1, i2, i3))]
        end
        @test_throws BoundsError M[size(U2, 1)+1, 1, 1]
        @test_throws BoundsError M[1, size(U3, 1)+1, 1]
        @test_throws BoundsError M[1, 1, size(U2, 1)+1]
    end
end

@testitem "Array" begin
    @testset "N=$N, K=$K" for N in 1:3, K in 1:3
        T = Float64
        λfull = T[1, 100, 10000]
        U1full, U2full, U3full = T[1 2 3; 4 5 6], T[-1 0 1], T[1 2 3; 4 5 6; 7 8 9]
        λ = λfull[1:K]
        U = (U1full[:, 1:K], U2full[:, 1:K], U3full[:, 1:K])[1:N]
        M = SymCPD(λ, U, (1, 2, 3)[1:N])

        X = Array(M)
        @test all(I -> M[I] == X[I], CartesianIndices(X))
    end

    # Symmetric cases
    T = Float64
    λ = T[1, 100, 10000]
    U1, U2 = T[1 2 3; 4 5 6], T[1 2 3; 4 5 6; 7 8 9]

    M = SymCPD(λ, (U1,), (1, 1, 1))
    X = Array(M)
    @test all(I -> M[I] == X[I], CartesianIndices(X))

    M = SymCPD(λ, (U1, U2), (1, 2, 1))
    X = Array(M)
    @test all(I -> M[I] == X[I], CartesianIndices(X))
end

@testitem "norm" begin
    using LinearAlgebra

    @testset "K=$K" for K in 0:2
        T = Float64
        λfull = T[1, 100, 10000]
        U1full, U2full, U3full = T[1 2 3; 4 5 6], T[-1 0 1], T[1 2 3; 4 5 6; 7 8 9]
        λ = λfull[1:K]
        U1, U2, U3 = U1full[:, 1:K], U2full[:, 1:K], U3full[:, 1:K]

        M = SymCPD(λ, (U1, U2, U3), (1, 2, 3))
        @test norm(M) ==
              norm(M, 2) ==
              sqrt(sum(abs2, M[I] for I in CartesianIndices(size(M))))
        @test norm(M, 1) == sum(abs, M[I] for I in CartesianIndices(size(M)))
        @test norm(M, 3) ==
              (sum(m -> abs(m)^3, M[I] for I in CartesianIndices(size(M))))^(1 / 3)

        M = SymCPD(λ, (U1, U2), (1, 2))
        @test norm(M) ==
              norm(M, 2) ==
              sqrt(sum(abs2, M[I] for I in CartesianIndices(size(M))))
        @test norm(M, 1) == sum(abs, M[I] for I in CartesianIndices(size(M)))
        @test norm(M, 3) ==
              (sum(m -> abs(m)^3, M[I] for I in CartesianIndices(size(M))))^(1 / 3)

        M = SymCPD(λ, (U1,), (1,))
        @test norm(M) ==
              norm(M, 2) ==
              sqrt(sum(abs2, M[I] for I in CartesianIndices(size(M))))
        @test norm(M, 1) == sum(abs, M[I] for I in CartesianIndices(size(M)))
        @test norm(M, 3) ==
              (sum(m -> abs(m)^3, M[I] for I in CartesianIndices(size(M))))^(1 / 3)

        # Symmetric Cases
        M = SymCPD(λ, (U1,), (1, 1, 1))
        @test norm(M) ==
              norm(M, 2) ==
              sqrt(sum(abs2, M[I] for I in CartesianIndices(size(M))))
        @test norm(M, 1) == sum(abs, M[I] for I in CartesianIndices(size(M)))
        @test norm(M, 3) ==
              (sum(m -> abs(m)^3, M[I] for I in CartesianIndices(size(M))))^(1 / 3)

        M = SymCPD(λ, (U1, U2), (1, 2, 1))
        @test norm(M) ==
              norm(M, 2) ==
              sqrt(sum(abs2, M[I] for I in CartesianIndices(size(M))))
        @test norm(M, 1) == sum(abs, M[I] for I in CartesianIndices(size(M)))
        @test norm(M, 3) ==
              (sum(m -> abs(m)^3, M[I] for I in CartesianIndices(size(M))))^(1 / 3)
    end
end

@testitem "normalizecomps-symmetric" begin
    using LinearAlgebra
    
    @testset "3way,r=$r" for r in [1,3]
        sz = 10
        M_orig = SymCPD(ones(r), (rand(sz,r),), (1,1,1))
        M_norm = deepcopy(M_orig)
        X = Array(M_orig)
        normalizecomps!(M_norm)
        @test isapprox(norm.(eachcol(M_norm.U[1])), ones(r))
        @test maximum(I -> abs(M_orig[I] - M_norm[I]), CartesianIndices(X)) <= 1e-5
    end 
    @testset "3way,r=$r,partially symmetric" for r in [1,3]
        sz1 = 10
        sz2 = 20
        M_orig = SymCPD(ones(r), (rand(sz1,r),rand(sz2,r)), (1,1,2))
        M_norm = deepcopy(M_orig)
        X = Array(M_orig)
        normalizecomps!(M_norm)
        @test isapprox(norm.(eachcol(M_norm.U[1])), ones(r))
        @test maximum(I -> abs(M_orig[I] - M_norm[I]), CartesianIndices(X)) <= 1e-5
    end 
    @testset "4way,r=$r" for r in [1,3]
        sz = 10
        M_orig = SymCPD(ones(r), (rand(sz,r),), (1,1,1,1))
        M_norm = normalizecomps(M_orig)
        X = Array(M_orig)
        @test isapprox(norm.(eachcol(M_norm.U[1])), ones(r))
        @test maximum(I -> abs(M_orig[I] - M_norm[I]), CartesianIndices(X)) <= 1e-5
    end 
    @testset "3way,r=$r,dist-U" for r in [1,3]
        sz = 10
        M_orig = SymCPD(ones(r)*2, (rand(sz,r),), (1,1,1))
        M_norm = deepcopy(M_orig)
        X = Array(M_orig)
        normalizecomps!(M_norm, distribute_to=1)
        @test isapprox(M_norm.λ, ones(r))
        @test maximum(I -> abs(M_orig[I] - M_norm[I]), CartesianIndices(X)) <= 1e-5
    end 
end

@testitem "permutecomps-symmetric" begin
    using Combinatorics

    @testset "K=$K" for K in 1:3
        T = Float64
        λfull = T[1, 100, 10000]
        Ufull = T[1 2 3; 4 5 6]
        λ = λfull[1:K]
        U = (Ufull[:, 1:K], )
        M = SymCPD(λ, U, (1,1,1))

        @testset "perm=$perm" for perm in permutations(1:K)
            Mback = deepcopy(M)
            Mperm = permutecomps(M, Tuple(perm))

            # Check for mutation
            @test M.λ == Mback.λ
            @test M.U == Mback.U

            # Check weights and factors
            @test Mperm.λ == M.λ[perm]
            @test all(k -> Mperm.U[k] == M.U[k][:, perm], 1:ngroups(Mperm))

            # Check in-place version
            permutecomps!(M, Tuple(perm))
            @test M.λ == Mperm.λ
            @test M.U == Mperm.U
        end
    end
end

@testitem "convertCPD" begin
    λ = [1, 100, 10000]
    U1, U2 = [1 2 3; 4 5 6], [1 2 3; 4 5 6; 7 8 9]

    M_symcpd = SymCPD(λ, (U1, U2), (1, 2, 1))
    M_cpd = convertCPD(M_symcpd)

    @test ncomps(M_symcpd) == ncomps(M_cpd)
    @test ndims(M_symcpd) == ndims(M_cpd)
    @test size(M_symcpd) == size(M_cpd)

    for i in Base.OneTo(size(M_symcpd)[1])
        for j in Base.OneTo(size(M_symcpd)[2])
            for k in Base.OneTo(size(M_symcpd)[3])
                @test M_symcpd[i, j, k] == M_cpd[i, j, k]
            end
        end
    end
end

@testitem "CPDtoSymCPD" begin
    λ = [1.0, 100, 10000]
    U1, U2, U3 = randn(10, 3), randn(20, 3), randn(30, 3)

    M_cpd = CPD(λ, (U1, U2, U3))
    M_symcpd = SymCPD(M_cpd)

    @test ncomps(M_symcpd) == ncomps(M_cpd)
    @test ndims(M_symcpd) == ndims(M_cpd)
    @test size(M_symcpd) == size(M_cpd)

    for i in Base.OneTo(size(M_symcpd)[1])
        for j in Base.OneTo(size(M_symcpd)[2])
            for k in Base.OneTo(size(M_symcpd)[3])
                @test M_symcpd[i, j, k] == M_cpd[i, j, k]
            end
        end
    end
end