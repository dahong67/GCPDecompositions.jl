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
        r;
        loss = GCPLosses.UserDefined((x, m) -> (x - m)^2; domain = Interval(1, Inf)),
    )

    # Exercise `_gcp`
    @test_throws ErrorException gcp(
        X,
        r;
        loss = GCPLosses.LeastSquares(),
        constraints = (GCPConstraints.LowerBound(1),),
    )
    @test_throws ErrorException gcp(X, r; loss = GCPLosses.Poisson(), constraints = ())
    @test_throws ErrorException gcp(
        X,
        r;
        loss = GCPLosses.UserDefined((x, m) -> (x - m)^2; domain = Interval(1, Inf)),
        constraints = (GCPConstraints.LowerBound(1),),
    )
end

@testitem "LeastSquares" begin
    using Random

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh = gcp(X, r; loss = GCPLosses.LeastSquares())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Xm = convert(Array{Union{Missing,eltype(X)}}, X)
        Xm[1, 1, 1] = missing
        Mm = gcp(Xm, r; loss = GCPLosses.LeastSquares())
        @test maximum(I -> abs(Mm[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Mh = gcp(X, r) # test default (least-squares) loss
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end

    # 4-way tensor to exercise recursive part of the Khatri-Rao code
    @testset "size(X)=$sz, rank(X)=$r" for sz in [(50, 40, 30, 2)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh = gcp(X, r; loss = GCPLosses.LeastSquares())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Xm = convert(Array{Union{Missing,eltype(X)}}, X)
        Xm[1, 1, 1, 1] = missing
        Mm = gcp(Xm, r; loss = GCPLosses.LeastSquares())
        @test maximum(I -> abs(Mm[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Mh = gcp(X, r) # test default (least-squares) loss
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end

    # 5 way tensor to exercise else case in FastALS
    @testset "size(X)=$sz, rank(X)=$r" for sz in [(10, 15, 20, 25, 30), (30, 25, 5, 5, 5)],
        r in [2]

        r = 2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh = gcp(X, r; loss = GCPLosses.LeastSquares())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Xm = convert(Array{Union{Missing,eltype(X)}}, X)
        Xm[1, 1, 1, 1, 1] = missing
        Mm = gcp(Xm, r; loss = GCPLosses.LeastSquares())
        @test maximum(I -> abs(Mm[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Mh = gcp(X, r) # test default (least-squares) loss
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end

    # Test old ALS method
    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25)], r in [2]
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh = gcp(X, r; loss = GCPLosses.LeastSquares(), algorithm = GCPAlgorithms.ALS())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "NonnegativeLeastSquares" begin
    using Random

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh = gcp(X, r; loss = GCPLosses.NonnegativeLeastSquares())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Xm = convert(Array{Union{Missing,eltype(X)}}, X)
        Xm[1, 1, 1] = missing
        Mm = gcp(Xm, r; loss = GCPLosses.NonnegativeLeastSquares())
        @test maximum(I -> abs(Mm[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "Poisson" begin
    using Random, IntervalSets
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
        Random.seed!(0)
        M = CPD(fill(10.0, r), rand.(sz, r))
        X = [rand(Poisson(M[I])) for I in CartesianIndices(size(M))]

        # Compute reference
        Random.seed!(0)
        Mr = gcp(
            X,
            r;
            loss = GCPLosses.UserDefined(
                (x, m) -> m - x * log(m + 1e-10);
                deriv = (x, m) -> 1 - x / (m + 1e-10),
                domain = Interval(0.0, +Inf),
            ),
            constraints = (GCPConstraints.LowerBound(0.0),),
            algorithm = GCPAlgorithms.LBFGSB(),
        )

        # Test
        Random.seed!(0)
        Mh = gcp(X, r; loss = GCPLosses.Poisson())
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "PoissonLog" begin
    using Random, IntervalSets
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), randn.(sz, r))
        X = [rand(Poisson(exp(M[I]))) for I in CartesianIndices(size(M))]

        # Compute reference
        Random.seed!(0)
        Mr = gcp(
            X,
            r;
            loss = GCPLosses.UserDefined(
                (x, m) -> exp(m) - x * m;
                deriv = (x, m) -> exp(m) - x,
                domain = Interval(-Inf, +Inf),
            ),
            constraints = (),
            algorithm = GCPAlgorithms.LBFGSB(),
        )

        # Test
        Random.seed!(0)
        Mh = gcp(X, r; loss = GCPLosses.PoissonLog())
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "Gamma" begin
    using Random, IntervalSets
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        k = 1.5
        X = [rand(Gamma(k, M[I] / k)) for I in CartesianIndices(size(M))]

        # Compute reference
        Random.seed!(0)
        Mr = gcp(
            X,
            r;
            loss = GCPLosses.UserDefined(
                (x, m) -> log(m + 1e-10) + x / (m + 1e-10);
                deriv = (x, m) -> -1 * (x / (m + 1e-10)^2) + (1 / (m + 1e-10)),
                domain = Interval(0.0, +Inf),
            ),
            constraints = (GCPConstraints.LowerBound(0.0),),
            algorithm = GCPAlgorithms.LBFGSB(),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r; loss = GCPLosses.Gamma())
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "Rayleigh" begin
    using Random, IntervalSets
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [rand(Rayleigh(M[I] / (sqrt(pi / 2)))) for I in CartesianIndices(size(M))]

        # Compute reference
        Random.seed!(0)
        Mr = gcp(
            X,
            r;
            loss = GCPLosses.UserDefined(
                (x, m) -> 2 * log(m + 1e-10) + (pi / 4) * ((x / (m + 1e-10))^2);
                deriv = (x, m) -> 2 / (m + 1e-10) - (pi / 2) * (x^2 / (m + 1e-10)^3),
                domain = Interval(0.0, +Inf),
            ),
            constraints = (GCPConstraints.LowerBound(0.0),),
            algorithm = GCPAlgorithms.LBFGSB(),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r; loss = GCPLosses.Rayleigh())
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "BernoulliOdds" begin
    using Random, IntervalSets
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [rand(Bernoulli(M[I] / (M[I] + 1))) for I in CartesianIndices(size(M))]

        # Compute reference
        Random.seed!(0)
        Mr = gcp(
            X,
            r;
            loss = GCPLosses.UserDefined(
                (x, m) -> log(m + 1) - x * log(m + 1e-10);
                deriv = (x, m) -> 1 / (m + 1) - (x / (m + 1e-10)),
                domain = Interval(0.0, +Inf),
            ),
            constraints = (GCPConstraints.LowerBound(0.0),),
            algorithm = GCPAlgorithms.LBFGSB(),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r; loss = GCPLosses.BernoulliOdds())
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "BernoulliLogitsLoss" begin
    using Random, IntervalSets
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [
            rand(Bernoulli(exp(M[I]) / (exp(M[I]) + 1))) for I in CartesianIndices(size(M))
        ]

        # Compute reference
        Random.seed!(0)
        Mr = gcp(
            X,
            r;
            loss = GCPLosses.UserDefined(
                (x, m) -> log(1 + exp(m)) - x * m;
                deriv = (x, m) -> exp(m) / (1 + exp(m)) - x,
                domain = Interval(-Inf, +Inf),
            ),
            constraints = (),
            algorithm = GCPAlgorithms.LBFGSB(),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r; loss = GCPLosses.BernoulliLogit())
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "NegativeBinomialOdds" begin
    using Random, IntervalSets
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        num_failures = 5
        X = [
            rand(NegativeBinomial(num_failures, M[I] / (M[I] + 1))) for
            I in CartesianIndices(size(M))
        ]

        # Compute reference
        Random.seed!(0)
        Mr = gcp(
            X,
            r;
            loss = GCPLosses.UserDefined(
                (x, m) -> (num_failures + x) * log(1 + m) - x * log(m + 1e-10);
                deriv = (x, m) -> (num_failures + x) / (1 + m) - x / (m + 1e-10),
                domain = Interval(0.0, +Inf),
            ),
            constraints = (GCPConstraints.LowerBound(0.0),),
            algorithm = GCPAlgorithms.LBFGSB(),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r; loss = GCPLosses.NegativeBinomialOdds(num_failures))
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "Huber" begin
    using Random, IntervalSets
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]

        # Compute reference
        Δ = 1
        Random.seed!(0)
        Mr = gcp(
            X,
            r;
            loss = GCPLosses.UserDefined(
                (x, m) -> abs(x - m) <= Δ ? (x - m)^2 : 2 * Δ * abs(x - m) - Δ^2;
                deriv = (x, m) ->
                    abs(x - m) <= Δ ? -2 * (x - m) : -2 * sign(x - m) * Δ * x,
                domain = Interval(-Inf, +Inf),
            ),
            constraints = (),
            algorithm = GCPAlgorithms.LBFGSB(),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r; loss = GCPLosses.Huber(Δ))
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "BetaDivergence" begin
    using Random, IntervalSets
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r, β" for sz in [(15, 20, 25), (50, 40, 30)],
        r in 1:2,
        β in [0, 0.5, 1]

        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        # May want to consider other distributions depending on value of β
        X = [rand(Poisson(M[I])) for I in CartesianIndices(size(M))]

        function beta_value(β, x, m)
            if β == 0
                return x / (m + 1e-10) + log(m + 1e-10)
            elseif β == 1
                return m - x * log(m + 1e-10)
            else
                return 1 / β * m^β - 1 / (β - 1) * x * m^(β - 1)
            end
        end
        function beta_deriv(β, x, m)
            if β == 0
                return -x / (m + 1e-10)^2 + 1 / (m + 1e-10)
            elseif β == 1
                return 1 - x / (m + 1e-10)
            else
                return m^(β - 1) - x * m^(β - 2)
            end
        end

        # Compute reference
        Random.seed!(0)
        Mr = gcp(
            X,
            r;
            loss = GCPLosses.UserDefined(
                (x, m) -> beta_value(β, x, m);
                deriv = (x, m) -> beta_deriv(β, x, m),
                domain = Interval(0.0, +Inf),
            ),
            constraints = (GCPConstraints.LowerBound(0.0),),
            algorithm = GCPAlgorithms.LBFGSB(),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r; loss = GCPLosses.BetaDivergence(β))
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "UserDefined" begin
    using Random, Distributions, IntervalSets

    @testset "Least Squares" begin
        @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
            Random.seed!(0)
            M = CPD(ones(r), randn.(sz, r))
            X = [M[I] for I in CartesianIndices(size(M))]

            # Compute reference
            Random.seed!(0)
            Mr = gcp(
                X,
                r;
                loss = GCPLosses.UserDefined(
                    (x, m) -> (x - m)^2;
                    deriv = (x, m) -> 2 * (m - x),
                    domain = Interval(-Inf, +Inf),
                ),
                constraints = (),
                algorithm = GCPAlgorithms.LBFGSB(),
            )

            # Test
            Random.seed!(0)
            Mh = gcp(X, r; loss = GCPLosses.UserDefined((x, m) -> (x - m)^2))
            @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
        end
    end

    @testset "Poisson" begin
        @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
            Random.seed!(0)
            M = CPD(fill(10.0, r), rand.(sz, r))
            X = [rand(Poisson(M[I])) for I in CartesianIndices(size(M))]

            # Compute reference
            Random.seed!(0)
            Mr = gcp(
                X,
                r;
                loss = GCPLosses.UserDefined(
                    (x, m) -> m - x * log(m + 1e-10);
                    deriv = (x, m) -> 1 - x / (m + 1e-10),
                    domain = Interval(0.0, +Inf),
                ),
                constraints = (GCPConstraints.LowerBound(0.0),),
                algorithm = GCPAlgorithms.LBFGSB(),
            )

            # Test
            Random.seed!(0)
            Mh = gcp(
                X,
                r;
                loss = GCPLosses.UserDefined(
                    (x, m) -> m - x * log(m + 1e-10);
                    domain = 0.0 .. Inf,
                ),
            )
            @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
        end
    end
end
