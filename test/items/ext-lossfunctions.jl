## LossFunctionsExt

# DistanceLoss
@testitem "LossFunctions: DistanceLoss" begin
    using Random
    using LossFunctions

    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh = gcp(X, r; loss = L2DistLoss())
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end
end

# MarginLoss
@testitem "LossFunctions: MarginLoss" begin
    using Random, IntervalSets
    using LossFunctions

    @testset "size(X)=$sz" for sz in [(15, 20, 25), (30, 40, 50)]
        Random.seed!(0)
        M = CPD([1], rand.(Ref([-1, 1]), sz, 1))
        X = [M[I] for I in CartesianIndices(size(M))]

        # Compute reference
        Random.seed!(10)
        Mr = gcp(
            X,
            1;
            loss = GCPLosses.UserDefinedLoss(
                (x, m) -> exp(-x * m);
                deriv = (x, m) -> -x * exp(-x * m),
                domain = Interval(-Inf, +Inf),
            ),
            constraints = (),
            algorithm = GCPAlgorithms.LBFGSB(),
        )

        # Test
        Random.seed!(10)
        Mh = gcp(X, 1; loss = ExpLoss())
        @test maximum(I -> abs(Mh[I] - Mr[I]), CartesianIndices(X)) <= 1e-5
    end
end
