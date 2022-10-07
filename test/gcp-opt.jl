## GCP decomposition - full optimization

@testset "least squares" begin
    @testset "size(X)=$sz, rank(X)=$r" for sz in [(15, 20, 25), (30, 40, 50)], r in 1:2
        Random.seed!(0)
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh = gcp(X, r, (x, m) -> (x - m)^2, (x, m) -> 2 * (m - x))
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Xm = convert(Array{Union{Missing,eltype(X)}}, X)
        Xm[1, 1, 1] = missing
        Mm = gcp(Xm, r, (x, m) -> (x - m)^2, (x, m) -> 2 * (m - x))
        @test maximum(I -> abs(Mm[I] - X[I]), CartesianIndices(X)) <= 1e-5

        Mh = gcp(X, r) # test default (least-squares) loss
        @test maximum(I -> abs(Mh[I] - X[I]), CartesianIndices(X)) <= 1e-5
    end
end
