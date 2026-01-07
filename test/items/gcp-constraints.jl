## Constraint types

@testitem "constraint constructors" begin
    # LowerBound constraint
    @test LowerBoundConstraint(0) isa LowerBoundConstraint
    @test LowerBoundConstraint(-Inf) isa LowerBoundConstraint
end

@testitem "satisfies/project! methods" begin
    using InteractiveUtils: subtypes
    @testset "type=$type" for type in subtypes(AbstractConstraint)
        @test hasmethod(satisfies, Tuple{CPD,type})
        @test hasmethod(project!, Tuple{CPD,type})
    end

    # LowerBound constraint
    M = CPD(ones(2), ([-ones(3, 1) ones(3, 1)], ones(4, 2), ones(5, 2)))
    @test !satisfies(M, LowerBoundConstraint(0))
    project!(M, LowerBoundConstraint(0))
    @test satisfies(M, LowerBoundConstraint(0))
end
