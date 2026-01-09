# ExplicitImports

@testitem "Implicit imports" begin
    using ExplicitImports
    @test check_no_implicit_imports(GCPDecompositions) === nothing
end

@testitem "Stale imports" begin
    using ExplicitImports
    @test check_no_stale_explicit_imports(GCPDecompositions) === nothing
end

@testitem "Non-owner imports" begin
    using ExplicitImports
    @test check_all_explicit_imports_via_owners(GCPDecompositions) === nothing
end

@testitem "Non-public imports" begin
    using ExplicitImports
    if VERSION >= v"1.11-"  # public only declared on Julia 1.11+
        @test check_all_explicit_imports_are_public(GCPDecompositions) === nothing
    end
end

@testitem "Non-owner qualified accesses" begin
    using ExplicitImports
    @test check_all_qualified_accesses_via_owners(GCPDecompositions) === nothing
end

@testitem "Non-public qualified accesses" begin
    using ExplicitImports
    if VERSION >= v"1.11-"  # public only declared on Julia 1.11+
        ignore = (
            :checkbounds_indices,         # from Base
            :require_one_based_indexing,  # from Base
            :throw_boundserror,           # from Base
            :derivative,                  # from ForwardDiff
            :DistanceLoss,                # from LossFunctions
            :MarginLoss,                  # from LossFunctions
            :deriv,                       # from LossFunctions
        )
        @test check_all_qualified_accesses_are_public(GCPDecompositions; ignore) === nothing
    end
end

@testitem "Self-qualified accesses" begin
    using ExplicitImports
    @test check_no_self_qualified_accesses(GCPDecompositions) === nothing
end
