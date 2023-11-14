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
# + `benchmark/results.json` : results from a benchmark run w/o comparison
# + `benchmark/results-target.json` : target results from a benchmark run w/ comparison
# + `benchmark/results-baseline.json` : baseline results from a benchmark run w/ comparison
# + `benchmark/report.md` : report summarizing the results

## Make sure the benchmark environment is activated and load utilities
@info("Loading benchmark environment")
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
include("utils.jl")

## Parse command line arguments
@info("Parsing arguments")
using ArgParse
settings = ArgParseSettings()
@add_arg_table settings begin
    "--suite"
    help = "which suite to run benchmarks for"
    arg_type = String
    "--compare"
    help = "git id for previous commit to compare current version against"
    arg_type = String
end
parsed_args = parse_args(settings)

## Run benchmarks
@info("Running benchmarks")
using GCPDecompositions, PkgBenchmark

### Create ENV for BenchmarkConfig
GCP_ENV =
    isnothing(parsed_args["suite"]) ? Dict{String,Any}() :
    Dict("GCP_BENCHMARK_SUITES" => parsed_args["suite"])

### Run benchmark and save
if isnothing(parsed_args["compare"])
    results = benchmarkpkg(GCPDecompositions, BenchmarkConfig(; env = GCP_ENV))
    writeresults(joinpath(@__DIR__, "results.json"), results)
else
    results = judge(
        GCPDecompositions,
        BenchmarkConfig(; env = GCP_ENV),
        BenchmarkConfig(; env = GCP_ENV, id = parsed_args["compare"]),
    )
    writeresults(
        joinpath(@__DIR__, "results-target.json"),
        PkgBenchmark.target_result(results),
    )
    writeresults(
        joinpath(@__DIR__, "results-baseline.json"),
        PkgBenchmark.baseline_result(results),
    )
end

## Generate report and save
@info("Generating report")

### Build report
report = sprint(export_markdown, results)
report *= "\n\n" * GCPBenchmarkUtils.export_mttkrp_sweep(results)

### Save report
write(joinpath(@__DIR__, "report.md"), report)
