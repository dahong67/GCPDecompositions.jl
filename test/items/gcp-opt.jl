## GCP decomposition - full optimization

@testitem "unsupported inputs" begin
    using Random, IntervalSets

    sz = (15, 20, 25)
    r = 2
    Random.seed!(0)
    M = CPD(ones(r), rand.(sz, r))
    X = [M[I] for I in CartesianIndices(size(M))]

    # Exercise `default_gcp_constraints`
    @test_throws ErrorException gcp(
        X,
        r;
        loss = UserDefinedLoss((x, m) -> (x - m)^2; domain = Interval(1, Inf)),
    )

    # Exercise `_gcp!`
    @test_throws ErrorException gcp(
        X,
        r;
        loss = LeastSquaresLoss(),
        constraints = (LowerBoundConstraint(1),),
    )
    @test_throws ErrorException gcp(X, r; loss = PoissonLoss(), constraints = ())
    @test_throws ErrorException gcp(
        X,
        r;
        loss = UserDefinedLoss((x, m) -> (x - m)^2; domain = Interval(1, Inf)),
        constraints = (LowerBoundConstraint(1),),
    )

    # Exercise `_gcp!` for Adam
    @test_throws ErrorException gcp(
        X,
        r;
        loss = LeastSquaresLoss(),
        constraints = (LowerBoundConstraint(1),),
        algorithm = Adam(; fsampler = UniformSampler(10), gsampler = UniformSampler(10)),
    )
    @test_throws ErrorException gcp(
        X,
        r;
        loss = PoissonLoss(),
        constraints = (),
        algorithm = Adam(; fsampler = UniformSampler(10), gsampler = UniformSampler(10)),
    )
    @test_throws ErrorException gcp(
        X,
        r;
        loss = UserDefinedLoss((x, m) -> (x - m)^2; domain = Interval(1, Inf)),
        constraints = (LowerBoundConstraint(1),),
        algorithm = Adam(; fsampler = UniformSampler(10), gsampler = UniformSampler(10)),
    )

    # Exercise check in `gcp` for supported inputs to algorithm
    @test_throws ErrorException gcp(
        X,
        r;
        constraints = (LowerBoundConstraint(0),),
        algorithm = ALS(),
    )
end

@testitem "default_gcp_init" begin
    X = randn(2, 3, 4)
    M = default_gcp_init(X, 2, LeastSquaresLoss(), (), ALS())
    @test M isa CPD
end

@testitem "LeastSquares" begin
    using Random

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh = gcp(X, r; loss = LeastSquaresLoss())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Xm = convert(Array{Union{Missing,eltype(X)}}, X)
        Xm[1, 1, 1] = missing
        Mm = gcp(Xm, r; loss = LeastSquaresLoss())
        @test maximum(I -> abs(Mm[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Mh = gcp(X, r) # test default (least-squares) loss
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end

    # 4-way tensor to exercise recursive part of the Khatri-Rao code
    @testset "size(X)=$sz, rank(X)=$r" for sz in [(50, 40, 30, 2)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh = gcp(X, r; loss = LeastSquaresLoss())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Xm = convert(Array{Union{Missing,eltype(X)}}, X)
        Xm[1, 1, 1, 1] = missing
        Mm = gcp(Xm, r; loss = LeastSquaresLoss())
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
        Mh = gcp(X, r; loss = LeastSquaresLoss())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Xm = convert(Array{Union{Missing,eltype(X)}}, X)
        Xm[1, 1, 1, 1, 1] = missing
        Mm = gcp(Xm, r; loss = LeastSquaresLoss())
        @test maximum(I -> abs(Mm[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Mh = gcp(X, r) # test default (least-squares) loss
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end

    # Test old ALS method
    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25)], r in [2]
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh = gcp(X, r; loss = LeastSquaresLoss(), algorithm = ALS())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "NonnegativeLeastSquares" begin
    using Random

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh = gcp(X, r; loss = NonnegativeLeastSquaresLoss())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Xm = convert(Array{Union{Missing,eltype(X)}}, X)
        Xm[1, 1, 1] = missing
        Mm = gcp(Xm, r; loss = NonnegativeLeastSquaresLoss())
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
            loss = UserDefinedLoss(
                (x, m) -> m - x * log(m + 1e-10);
                deriv = (x, m) -> 1 - x / (m + 1e-10),
                domain = Interval(0.0, +Inf),
            ),
            constraints = (LowerBoundConstraint(0.0),),
            algorithm = LBFGSB(),
        )

        # Test
        Random.seed!(0)
        Mh = gcp(X, r; loss = PoissonLoss())
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
            loss = UserDefinedLoss(
                (x, m) -> exp(m) - x * m;
                deriv = (x, m) -> exp(m) - x,
                domain = Interval(-Inf, +Inf),
            ),
            constraints = (),
            algorithm = LBFGSB(),
        )

        # Test
        Random.seed!(0)
        Mh = gcp(X, r; loss = PoissonLogLoss())
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
            loss = UserDefinedLoss(
                (x, m) -> log(m + 1e-10) + x / (m + 1e-10);
                deriv = (x, m) -> -1 * (x / (m + 1e-10)^2) + (1 / (m + 1e-10)),
                domain = Interval(0.0, +Inf),
            ),
            constraints = (LowerBoundConstraint(0.0),),
            algorithm = LBFGSB(),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r; loss = GammaLoss())
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
            loss = UserDefinedLoss(
                (x, m) -> 2 * log(m + 1e-10) + (pi / 4) * ((x / (m + 1e-10))^2);
                deriv = (x, m) -> 2 / (m + 1e-10) - (pi / 2) * (x^2 / (m + 1e-10)^3),
                domain = Interval(0.0, +Inf),
            ),
            constraints = (LowerBoundConstraint(0.0),),
            algorithm = LBFGSB(),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r; loss = RayleighLoss())
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
            loss = UserDefinedLoss(
                (x, m) -> log(m + 1) - x * log(m + 1e-10);
                deriv = (x, m) -> 1 / (m + 1) - (x / (m + 1e-10)),
                domain = Interval(0.0, +Inf),
            ),
            constraints = (LowerBoundConstraint(0.0),),
            algorithm = LBFGSB(),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r; loss = BernoulliOddsLoss())
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
            loss = UserDefinedLoss(
                (x, m) -> log(1 + exp(m)) - x * m;
                deriv = (x, m) -> exp(m) / (1 + exp(m)) - x,
                domain = Interval(-Inf, +Inf),
            ),
            constraints = (),
            algorithm = LBFGSB(),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r; loss = BernoulliLogitLoss())
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
            loss = UserDefinedLoss(
                (x, m) -> (num_failures + x) * log(1 + m) - x * log(m + 1e-10);
                deriv = (x, m) -> (num_failures + x) / (1 + m) - x / (m + 1e-10),
                domain = Interval(0.0, +Inf),
            ),
            constraints = (LowerBoundConstraint(0.0),),
            algorithm = LBFGSB(),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r; loss = NegativeBinomialOddsLoss(num_failures))
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
            loss = UserDefinedLoss(
                (x, m) -> abs(x - m) <= Δ ? (x - m)^2 : 2 * Δ * abs(x - m) - Δ^2;
                deriv = (x, m) ->
                    abs(x - m) <= Δ ? -2 * (x - m) : -2 * sign(x - m) * Δ * x,
                domain = Interval(-Inf, +Inf),
            ),
            constraints = (),
            algorithm = LBFGSB(),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r; loss = HuberLoss(Δ))
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end

@testitem "BetaDivergence" begin
    using Random, IntervalSets
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r, β=$β" for sz in [(15, 20, 25), (50, 40, 30)],
        r in 1:2,
        β in [0, 0.5, 1]

        # Generate data:
        # + for β > 0, use Poisson distribution
        # + for β = 0, use Exponential distribution (seems better behaved)
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        dist = iszero(β) ? Exponential : Poisson
        X = [rand(dist(M[I])) for I in CartesianIndices(size(M))]

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
            loss = UserDefinedLoss(
                (x, m) -> beta_value(β, x, m);
                deriv = (x, m) -> beta_deriv(β, x, m),
                domain = Interval(0.0, +Inf),
            ),
            constraints = (LowerBoundConstraint(0.0),),
            algorithm = LBFGSB(),
        )

        # Test 
        Random.seed!(0)
        Mh = gcp(X, r; loss = BetaDivergenceLoss(β))
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
                loss = UserDefinedLoss(
                    (x, m) -> (x - m)^2;
                    deriv = (x, m) -> 2 * (m - x),
                    domain = Interval(-Inf, +Inf),
                ),
                constraints = (),
                algorithm = LBFGSB(),
            )

            # Test
            Random.seed!(0)
            Mh = gcp(X, r; loss = UserDefinedLoss((x, m) -> (x - m)^2))
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
                loss = UserDefinedLoss(
                    (x, m) -> m - x * log(m + 1e-10);
                    deriv = (x, m) -> 1 - x / (m + 1e-10),
                    domain = Interval(0.0, +Inf),
                ),
                constraints = (LowerBoundConstraint(0.0),),
                algorithm = LBFGSB(),
            )

            # Test
            Random.seed!(0)
            Mh = gcp(
                X,
                r;
                loss = UserDefinedLoss(
                    (x, m) -> m - x * log(m + 1e-10);
                    domain = 0.0 .. Inf,
                ),
            )
            @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
        end
    end
end

@testitem "GCP-Adam" begin
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
            loss = UserDefinedLoss(
                (x, m) -> log(m + 1) - x * log(m + 1e-10);
                deriv = (x, m) -> 1 / (m + 1) - (x / (m + 1e-10)),
                domain = Interval(0.0, +Inf),
            ),
            constraints = (LowerBoundConstraint(0.0),),
            algorithm = LBFGSB(),
        )

        # Uniform sampling with dense data tensor
        Random.seed!(0)
        Mh = gcp(
            X,
            r;
            loss = BernoulliOddsLoss(),
            algorithm = Adam(;
                α = 0.01,
                epochiters = 100,
                fsampler = UniformSampler(10^5),
                gsampler = UniformSampler(10^4),
            ),
        )
        @test sum(abs2, Array(Mh) - Array(Mr)) / sum(abs2, Array(Mr)) < 0.1

        # Stratified sampling with sparse data tensor
        Random.seed!(0)
        Mh = gcp(
            SparseArrayCOO(X),
            r;
            loss = BernoulliOddsLoss(),
            algorithm = Adam(;
                α = 0.01,
                epochiters = 100,
                fsampler = StratifiedSampler(10^4, 10^4),
                gsampler = StratifiedSampler(10^3, 10^1),
            ),
        )
        @test sum(abs2, Array(Mh) - Array(Mr)) / sum(abs2, Array(Mr)) < 0.1

        # Semistratified sampling with sparse data tensor
        Random.seed!(0)
        Mh = gcp(
            SparseArrayCOO(X),
            r;
            loss = BernoulliOddsLoss(),
            algorithm = Adam(;
                α = 0.01,
                epochiters = 100,
                fsampler = SemistratifiedSampler(10^4, 10^4),
                gsampler = SemistratifiedSampler(10^3, 10^3),
            ),
        )
        @test sum(abs2, Array(Mh) - Array(Mr)) / sum(abs2, Array(Mr)) < 0.1
    end
end

@testitem "stochastic obj / grad" begin
    using Random, IntervalSets
    using Distributions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (50, 40, 30)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [rand(Bernoulli(M[I] / (M[I] + 1))) for I in CartesianIndices(size(M))]
        Xs = SparseArrayCOO(X)

        # Compute references
        Fr = gcp_objective(M, X, LeastSquaresLoss())
        Gr = gcp_grad_U!(similar.(M.U), M, X, LeastSquaresLoss())

        # Dense data samplers
        @testset "sampler=$sampler" for sampler in [UniformSampler(10)]
            Random.seed!(0)
            F = mean(1:100) do _
                return gcp_stoch_objective(M, X, LeastSquaresLoss(), sampler)
            end
            G = mean(1:10000) do _
                Gtuple = gcp_stoch_grad_U!(similar.(M.U), M, X, LeastSquaresLoss(), sampler)
                return collect(Gtuple)
            end
            @test abs2(F - Fr) / abs2(Fr) < 1e-2
            @test maximum(sum.(abs2, G .- Gr) ./ sum.(abs2, Gr)) < 1e-1
        end

        # Sparse data samplers
        @testset "sampler=$sampler" for sampler in [
            StratifiedSampler(10, 2),
            SemistratifiedSampler(10, 10),
        ]
            Random.seed!(0)
            F = mean(1:100) do _
                return gcp_stoch_objective(M, Xs, LeastSquaresLoss(), sampler)
            end
            G = mean(1:10000) do _
                Gtuple =
                    gcp_stoch_grad_U!(similar.(M.U), M, Xs, LeastSquaresLoss(), sampler)
                return collect(Gtuple)
            end
            @test abs2(F - Fr) / abs2(Fr) < 1e-2
            @test maximum(sum.(abs2, G .- Gr) ./ sum.(abs2, Gr)) < 1e-1
        end
    end
end
