## GCP decomposition - full optimization

Random.seed!(0)
@testset "least squares" begin
    @testset "size(X)=$sz, rank(X)=$r" for sz in [(3, 4, 5), (30, 40, 50)], r in 1:2
        M = CPD(ones(r), rand.(sz, r))
        X = [M[I] for I in CartesianIndices(size(M))]
        Mh = gcp(X, r, (x, m) -> (x - m)^2, (x, m) -> 2 * (m - x))
        @test maximum(I -> abs(M[I] - X[I]), CartesianIndices(X)) <= 1e-6

        Xm = convert(Array{Union{Missing,eltype(X)}}, X)
        Xm[1, 1, 1] = missing
        Mm = gcp(Xm, r, (x, m) -> (x - m)^2, (x, m) -> 2 * (m - x))
        @test maximum(I -> abs(Mm[I] - X[I]), CartesianIndices(X)) <= 1e-4
    end
end
