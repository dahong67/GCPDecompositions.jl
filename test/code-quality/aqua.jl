# Aqua

@testitem "Method ambiguities" begin
    using Aqua: test_ambiguities
    test_ambiguities(GCPDecompositions)
end

@testitem "Unbound type parameters [CURRENTLY BROKEN]" begin
    using Aqua: test_unbound_args
    test_unbound_args(GCPDecompositions; broken = true)
end

@testitem "Undefined exports" begin
    using Aqua: test_undefined_exports
    test_undefined_exports(GCPDecompositions)
end

@testitem "Stale dependencies" begin
    using Aqua: test_stale_deps
    test_stale_deps(GCPDecompositions)
end

@testitem "Compat entries" begin
    using Aqua: test_deps_compat
    test_deps_compat(GCPDecompositions)
end

@testitem "Type piracy" begin
    using Aqua: test_piracies
    test_piracies(GCPDecompositions)
end

@testitem "Persistent tasks" begin
    using Aqua: test_persistent_tasks
    test_persistent_tasks(GCPDecompositions)
end

@testitem "Undocumented names" begin
    using Aqua: test_undocumented_names
    test_undocumented_names(GCPDecompositions)
end
