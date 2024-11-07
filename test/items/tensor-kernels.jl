## Tensor Kernels

@testitem "mttkrp" begin
    using Random
    using GCPDecompositions.TensorKernels

    @testset "size=$sz, rank=$r" for sz in [(10, 30, 40)], r in [5]
        Random.seed!(0)
        X = randn(sz)
        U = randn.(sz, r)
        N = length(sz)

        for n in 1:N
            Xn = reshape(permutedims(X, [n; setdiff(1:N, n)]), size(X, n), :)
            Zn = reduce(
                hcat,
                [reduce(kron, [U[i][:, j] for i in reverse(setdiff(1:N, n))]) for j in 1:r],
            )
            @test mttkrp(X, U, n) ≈ Xn * Zn
        end
    end
end

@testitem "khatrirao" begin
    using Random
    using GCPDecompositions.TensorKernels

    @testset "size=$sz, rank=$r" for sz in [(10,), (10, 20), (10, 30, 40)], r in [5]
        Random.seed!(0)
        U = randn.(sz, r)
        Zn = reduce(hcat, [reduce(kron, [Ui[:, j] for Ui in U]) for j in 1:r])
        @test khatrirao(U...) ≈ Zn
    end
end

@testitem "mttkrps" begin
    using Random
    using GCPDecompositions.TensorKernels

    @testset "size=$sz, rank=$r" for sz in [(10, 30), (10, 30, 40)], r in [5]
        Random.seed!(0)
        X = randn(sz)
        U = randn.(sz, r)
        N = length(sz)

        G = map(1:N) do n
            Xn = reshape(permutedims(X, [n; setdiff(1:N, n)]), size(X, n), :)
            Zn = reduce(
                hcat,
                [reduce(kron, [U[i][:, j] for i in reverse(setdiff(1:N, n))]) for j in 1:r],
            )
            return Xn * Zn
        end
        @test all(mttkrps(X, U) .≈ G)
    end
end

@testitem "checksym" begin
    using GCPDecompositions.TensorKernels

    U1 = randn(10, 5)
    U2 = randn(20, 5)
    U3 = randn(30, 5)

    # Nonsymmetric case
    X = zeros(10, 20, 30)
    for i1 in axes(U1, 1), i2 in axes(U2, 1), i3 in axes(U3, 1)
        X[i1, i2, i3] = sum(U1[i1, :] .* U2[i2, :] .* U3[i3, :])
    end

    @test_throws DimensionMismatch checksym(X, (1, 2))
    @test_throws DimensionMismatch checksym(X, (1, 2, 3, 4))
    @test_throws DimensionMismatch checksym(X, (2, 3, 4))
    @test_throws DimensionMismatch checksym(X, (1, 2, 4))

    @test checksym(X, (1, 2, 3)) == true
    @test checksym(X, (1, 1, 1)) == false
    @test checksym(X, (1, 2, 2)) == false
    @test checksym(X, (2, 1, 2)) == false

    # Fully symmetric
    X = zeros(10, 10, 10)
    for i1 in axes(U1, 1), i2 in axes(U1, 1), i3 in axes(U1, 1)
        X[i1, i2, i3] = sum(U1[i1, :] .* U1[i2, :] .* U1[i3, :])
    end

    @test checksym(X, (1, 1, 1)) == true
    @test checksym(X, (1, 2, 3)) == true
    @test checksym(X, (1, 2, 2)) == true
    @test checksym(X, (2, 2, 1)) == true

    # Partially symmetric
    X = zeros(10, 20, 10)
    for i1 in axes(U1, 1), i2 in axes(U2, 1), i3 in axes(U1, 1)
        X[i1, i2, i3] = sum(U1[i1, :] .* U2[i2, :] .* U1[i3, :])
    end

    @test checksym(X, (1, 2, 1)) == true
    @test checksym(X, (2, 1, 2)) == true
    @test checksym(X, (1, 2, 3)) == true
    @test checksym(X, (1, 1, 1)) == false
    @test checksym(X, (1, 1, 2)) == false
    @test checksym(X, (1, 2, 2)) == false

    # 4-way partially symmetric case
    X = zeros(10, 10, 20, 10)
    for i1 in axes(U1, 1), i2 in axes(U1, 1), i3 in axes(U2, 1), i4 in axes(U1, 1)
        X[i1, i2, i3, i4] = sum(U1[i1, :] .* U1[i2, :] .* U2[i3, :] .* U1[i4, :])
    end

    @test checksym(X, (1, 1, 2, 1)) == true
    @test checksym(X, (2, 2, 1, 2)) == true
    @test checksym(X, (1, 2, 3, 4)) == true
    @test checksym(X, (1, 1, 1, 1)) == false
    @test checksym(X, (1, 1, 2, 2)) == false
    @test checksym(X, (2, 2, 1, 1)) == false
    @test checksym(X, (1, 2, 1, 1)) == false
end