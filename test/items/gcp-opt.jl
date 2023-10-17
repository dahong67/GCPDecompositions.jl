## GCP decomposition - full optimization

@testitem "unsupported constraints" begin
    using Random, IntervalSets

    sz = (15, 20, 25)
    r = 2
    Random.seed!(0)
    M = CPD(ones(r), rand.(sz, r))
    X = [M[I] for I in CartesianIndices(size(M))]

    # Exercise `default_constraints`
    @test_throws ErrorException gcp(
        X,
        r,
        UserDefinedLoss((x, m) -> (x - m)^2; domain = Interval(1, Inf)),
    )

    # Exercise `_gcp`
    @test_throws ErrorException gcp(
        X,
        r,
        LeastSquaresLoss();
        constraints = (GCPConstraints.LowerBound(1),),
    )
    @test_throws ErrorException gcp(X, r, PoissonLoss(); constraints = ())
    @test_throws ErrorException gcp(
        X,
        r,
        UserDefinedLoss((x, m) -> (x - m)^2; domain = Interval(1, Inf));
        constraints = (GCPConstraints.LowerBound(1),),
    )
end

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

@testitem "NonnegativeLeastSquaresLoss" begin
    using Random

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh = gcp(X, r, NonnegativeLeastSquaresLoss())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Xm = convert(Array{Union{Missing,eltype(X)}}, X)
        Xm[1, 1, 1] = missing
        Mm = gcp(Xm, r, NonnegativeLeastSquaresLoss())
        @test maximum(I -> abs(Mm[I] - X[I]), CartesianIndices(X)) <= 1e-5
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

@testitem "PoissonLogLoss" begin
    using Random
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), randn.(sz, r))
        X = [rand(Poisson(exp(M[I]))) for I in CartesianIndices(size(M))]

        # Compute reference
        Random.seed!(0)
        Mr = GCPDecompositions._gcp(
            X,
            r,
            (x, m) -> exp(m) - x * m,
            (x, m) -> exp(m) - x,
            -Inf,
            (;),
        )

        # Test
        Random.seed!(0)
        Mh = gcp(X, r, PoissonLogLoss())
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "GammaLoss" begin
    using Random
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        k = 1.5
        X = [rand(Gamma(k, M[I]/k)) for I in CartesianIndices(size(M))]

        # Compute reference
        Random.seed!(0)
        Mr = GCPDecompositions._gcp(
            X,
            r,
            (x, m) -> log(m + 1e-10) + x / (m + 1e-10),
            (x, m) -> -1 * (x / (m + 1e-10)^2) + (1 / (m + 1e-10)),
            0.0,
            (;),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r, GammaLoss())
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "RayleighLoss" begin
    using Random
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [rand(Rayleigh(M[I]/(sqrt(pi/2)))) for I in CartesianIndices(size(M))]

        # Compute reference
        Random.seed!(0)
        Mr = GCPDecompositions._gcp(
            X,
            r,
            (x, m) -> 2*log(m + 1e-10) + (pi / 4) * ((x/(m + 1e-10))^2),
            (x, m) -> 2/(m + 1e-10) - (pi / 2) * (x^2 / (m + 1e-10)^3),
            0.0,
            (;),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r, RayleighLoss())
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "BernoulliOddsLoss" begin
    using Random
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [rand(Bernoulli(M[I]/(M[I] + 1))) for I in CartesianIndices(size(M))]

        # Compute reference
        Random.seed!(0)
        Mr = GCPDecompositions._gcp(
            X,
            r,
            (x, m) -> log(m + 1) - x * log(m + 1e-10),
            (x, m) -> 1 / (m + 1) - (x / (m + 1e-10)),
            0.0,
            (;),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r, BernoulliOddsLoss())
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "BernoulliLogitsLoss" begin
    using Random
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [rand(Bernoulli(exp(M[I])/(exp(M[I]) + 1))) for I in CartesianIndices(size(M))]

        # Compute reference
        Random.seed!(0)
        Mr = GCPDecompositions._gcp(
            X,
            r,
            (x, m) -> log(1 + exp(m)) - x * m,
            (x, m) -> exp(m) / (1 + exp(m)) - x,
            -Inf,
            (;),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r, BernoulliLogitLoss())
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "NegativeBinomialOddsLoss" begin
    using Random
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        num_failures = 5
        X = [rand(NegativeBinomial(num_failures, M[I]/(M[I] + 1))) for I in CartesianIndices(size(M))]
  
        # Compute reference
        Random.seed!(0)
        Mr = GCPDecompositions._gcp(
            X,
            r,
            (x, m) -> (num_failures + x) * log(1 + m) - x * log(m + 1e-10),
            (x, m) -> (num_failures + x) / (1 + m) - x / (m + 1e-10),
            0.0,
            (;),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r, NegativeBinomialOddsLoss(num_failures))
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
