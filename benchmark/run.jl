# Script to run benchmarks and export a report
# 
# To run this script from the package root directory:
# > julia benchmark/run.jl
# 
# By default it will run all of the benchmark suites.
# To select a subset, pass them in as a command line arg:
# > julia benchmark/run.jl --suite mttkrp
#
# To compare against a previous commit, pass the commit git id as a command line arg:
# > julia benchmark/run.jl --compare eca7cb4
# 
# This script produces/overwrites two files:
# + `benchmark/results.json` : results from the benchmark run
# + `benchmark/report.md` : report summarizing the results

## Make sure the benchmark environment is activated
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
include("utils.jl")

## Run benchmarks
using GCPDecompositions, PkgBenchmark, ArgParse

settings = ArgParseSettings()
@add_arg_table settings begin
    "--suite"
        help = "which suite to run benchmarks for" 
        arg_type = String
        default = "all"
    "--compare"
        help = "git id for previous commit to compare current version against"
        arg_type = String
        default = "none"
end
parsed_args = parse_args(settings)
compare = parsed_args["compare"]
suite = parsed_args["suite"]

if (compare == "none")
    results =
        suite == "all" ? benchmarkpkg(GCPDecompositions) :
        benchmarkpkg(
            GCPDecompositions,
            BenchmarkConfig(; env = Dict("GCP_BENCHMARK_SUITES" => suite)),
        )
    writeresults(joinpath(@__DIR__, "results.json"), results)
else
    judgement =
        suite == "all" ? judge(GCPDecompositions, compare) :
        judge(
            GCPDecompositions,
            BenchmarkConfig(; env = Dict("GCP_BENCHMARK_SUITES" => suite)),
            BenchmarkConfig(; id=compare, env = Dict("GCP_BENCHMARK_SUITES" => suite)),
        )
    results = PkgBenchmark.target_result(judgement)
    results_baseline = PkgBenchmark.baseline_result(judgement)
    writeresults(joinpath(@__DIR__, "results.json"), results)
end

## Generate report and save

### Create initial/base report
report = compare != "none" ? sprint(export_markdown, judgement) : sprint(export_markdown, results)
report *= "\n\n" * (compare != "none" ? GCPBenchmarkUtils.export_mttkrp_sweep(judgement) : GCPBenchmarkUtils.export_mttkrp_sweep(results))

### Save report
write(joinpath(@__DIR__, "report.md"), report)
