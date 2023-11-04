# Select which benchmark suites to run
const SELECTED_SUITES = nothing     # set to `nothing` to run all suites
# const SELECTED_SUITES = ["mttkrp"]

# Load packages
import Pkg
Pkg.activate(@__DIR__)
using GCPDecompositions, PkgBenchmark

# Run benchmarks and save results
results =
    isnothing(SELECTED_SUITES) ? benchmarkpkg(GCPDecompositions) :
    benchmarkpkg(
        GCPDecompositions,
        BenchmarkConfig(; env = Dict("GCP_BENCHMARK_SUITES" => join(SELECTED_SUITES, ' '))),
    )
writeresults(joinpath(@__DIR__, "results.json"), results)

# Generate report and save
report = sprint(export_markdown, results)
write(joinpath(@__DIR__, "report.md"), report)
