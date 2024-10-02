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
