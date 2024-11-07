## checksym function

@testitem "checksym" begin

    using GCPDecompositions.TensorKernels
    
    U1 = randn(10, 5)
    U2 = randn(20, 5)
    U3 = randn(30, 5)

    # Nonsymmetric case
    X = zeros(10,20,30)
    for i1 in axes(U1, 1), i2 in axes(U2, 1), i3 in axes(U3, 1)
        X[i1, i2, i3] = sum(U1[i1, :] .* U2[i2, :] .* U3[i3, :])
    end 
    
    @test_throws DimensionMismatch checksym(X, (1,2))
    @test_throws DimensionMismatch checksym(X, (1,2,3,4))
    @test_throws DimensionMismatch checksym(X, (2,3,4))
    @test_throws DimensionMismatch checksym(X, (1,2,4))

    @test checksym(X, (1,2,3)) == true
    @test checksym(X, (1,1,1)) == false
    @test checksym(X, (1,2,2)) == false
    @test checksym(X, (2,1,2)) == false

    # Fully symmetric
    X = zeros(10,10,10)
    for i1 in axes(U1, 1), i2 in axes(U1, 1), i3 in axes(U1, 1)
        X[i1, i2, i3] = sum(U1[i1, :] .* U1[i2, :] .* U1[i3, :])
    end 

    @test checksym(X, (1,1,1)) == true
    @test checksym(X, (1,2,3)) == true  # Ignoring symmetry
    @test checksym(X, (1,2,2)) == true  # Fully symmetric -> partially symmetric
    @test checksym(X, (2,2,1)) == true  # Fully symmetric -> partially symmetric

    # Partially symmetric
    X = zeros(10,20,10)
    for i1 in axes(U1, 1), i2 in axes(U2, 1), i3 in axes(U1, 1)
        X[i1, i2, i3] = sum(U1[i1, :] .* U2[i2, :] .* U1[i3, :])
    end

    @test checksym(X, (1,2,1)) == true
    @test checksym(X, (2,1,2)) == true
    @test checksym(X, (1,2,3)) == true
    @test checksym(X, (1,1,1)) == false
    @test checksym(X, (1,1,2)) == false
    @test checksym(X, (1,2,2)) == false

    # 4-way partially symmetric case
    X = zeros(10,10,20,10)
    for i1 in axes(U1, 1), i2 in axes(U1, 1), i3 in axes(U2, 1), i4 in axes(U1, 1)
        X[i1, i2, i3, i4] = sum(U1[i1, :] .* U1[i2, :] .* U2[i3, :] .* U1[i4, :])
    end

    @test checksym(X, (1,1,2,1)) == true
    @test checksym(X, (2,2,1,2)) == true
    @test checksym(X, (1,2,3,4)) == true
    @test checksym(X, (1,1,1,1)) == false
    @test checksym(X, (1,1,2,2)) == false
    @test checksym(X, (2,2,1,1)) == false
    @test checksym(X, (1,2,1,1)) == false
    
end