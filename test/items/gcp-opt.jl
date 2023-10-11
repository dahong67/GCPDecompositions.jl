## GCP decomposition - full optimization

@testitem "LeastSquaresLoss" begin
    using Random

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh = gcp(X, r, LeastSquaresLoss())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Xm = convert(Array{Union{Missing,eltype(X)}}, X)
        Xm[1, 1, 1] = missing
        Mm = gcp(Xm, r, LeastSquaresLoss())
        @test maximum(I -> abs(Mm[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Mh = gcp(X, r) # test default (least-squares) loss
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "PoissonLoss" begin
    using Random
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
        Random.seed!(0)
        M = CPD(fill(10.0, r), rand.(sz, r))
        X = [rand(Poisson(M[I])) for I in CartesianIndices(size(M))]

        # Compute reference
        Random.seed!(0)
        Mr = GCPDecompositions._gcp(
            X,
            r,
            (x, m) -> m - x * log(m + 1e-10),
            (x, m) -> 1 - x / (m + 1e-10),
            0.0,
            (;),
        )

        # Test
        Random.seed!(0)
        Mh = gcp(X, r, PoissonLoss())
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "UserDefinedLoss" begin
    using Random, Distributions, IntervalSets

    @testset "Least Squares" begin
        @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
            Random.seed!(0)
            M = CPD(ones(r), randn.(sz, r))
            X = [M[I] for I in CartesianIndices(size(M))]

            # Compute reference
            Random.seed!(0)
            Mr = GCPDecompositions._gcp(
                X,
                r,
                (x, m) -> (x - m)^2,
                (x, m) -> 2 * (m - x),
                -Inf,
                (;),
            )

            # Test
            Random.seed!(0)
            Mh = gcp(X, r, UserDefinedLoss((x, m) -> (x - m)^2))
            @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
        end
    end

    @testset "Poisson" begin
        @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
            Random.seed!(0)
            M = CPD(fill(10.0, r), rand.(sz, r))
            X = [rand(Poisson(M[I])) for I in CartesianIndices(size(M))]

            # Compute reference
            Random.seed!(0)
            Mr = GCPDecompositions._gcp(
                X,
                r,
                (x, m) -> m - x * log(m + 1e-10),
                (x, m) -> 1 - x / (m + 1e-10),
                0.0,
                (;),
            )

            # Test
            Random.seed!(0)
            Mh = gcp(
                X,
                r,
                UserDefinedLoss((x, m) -> m - x * log(m + 1e-10); domain = 0.0 .. Inf),
            )
            @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
        end
    end
end
