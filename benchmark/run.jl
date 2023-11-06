# Script to run benchmarks and export a report
# 
# To run this script from the package root directory:
# > julia benchmark/run.jl
# 
# By default it will run all of the benchmark suites.
# To select a subset, pass them in as ARGS:
# > julia benchmark/run.jl mttkrp
# 
# This script produces/overwrites two files:
# + `benchmark/results.json` : results from the benchmark run
# + `benchmark/report.md` : report summarizing the results

## Make sure the benchmark environment is activated
import Pkg
Pkg.activate(@__DIR__)

## Run benchmarks
using GCPDecompositions, PkgBenchmark
results =
    isempty(ARGS) ? benchmarkpkg(GCPDecompositions) :
    benchmarkpkg(
        GCPDecompositions,
        BenchmarkConfig(; env = Dict("GCP_BENCHMARK_SUITES" => join(ARGS, ' '))),
    )
writeresults(joinpath(@__DIR__, "results.json"), results)

## Generate report and save
using BenchmarkTools, Dictionaries, SplitApplyCombine, UnicodePlots

### Create initial/base report
report = sprint(export_markdown, results)

### Add plots for MTTKRP sweeps
# NOTE:
# > Using a separate plot for each curve since it would be great
# > to be able to copy into GitHub comments (e.g., within PRs)
# > but GitHub doesn't currently handle ANSI color codes
# > (https://github.com/github/markup/issues/1538).
# > Tried working around by converting to HTML with ANSIColoredPrinters.jl
# > (https://github.com/JuliaDocs/ANSIColoredPrinters.jl)
# > which is the package that Documenter.jl uses to support ANSI color codes
# > (https://github.com/JuliaDocs/Documenter.jl/pull/1441/files).
# > However, GitHub also doesn't currently support coloring the text
# > (https://github.com/github/markup/issues/1440).
if haskey(PkgBenchmark.benchmarkgroup(results), "mttkrp")
    # Load results into a dictionary
    mttkrp_results = PkgBenchmark.benchmarkgroup(results)["mttkrp"]
    mttkrp_dict = (sortkeys ∘ dictionary ∘ map)(mttkrp_results) do (key_str, result)
        key_vals = match(
            r"^size=\((?<size>[0-9, ]*)\), rank=(?<rank>[0-9]+), mode=(?<mode>[0-9]+)$",
            key_str,
        )
        key = (;
            size = Tuple(parse.(Int, split(key_vals[:size], ','))),
            rank = parse(Int, key_vals[:rank]),
            mode = parse(Int, key_vals[:mode]),
        )
        return key => result
    end

    # Runtime vs. size (for square tensors)
    size_sweeps = (sortkeys ∘ group)(
        ((key, _),) -> (; ndims = length(key.size), rank = key.rank, mode = key.mode),
        ((key, result),) -> ((only ∘ unique)(key.size), result),
        filter(((key, _),) -> allequal(key.size), pairs(mttkrp_dict)),
    )
    size_plts = map(pairs(size_sweeps)) do (key, sweep)
        return lineplot(
            getindex.(sweep, 1),
            getproperty.(median.(getindex.(sweep, 2)), :time) ./ 1e6;
            title = string(key)[begin+1:end-1],
            xlabel = "Size",
            ylabel = "Time (ms)",
            canvas = DotCanvas,
            width = 30,
            height = 10,
            margin = 0,
        )
    end
    size_report = """
    ## Runtime vs. size (for square tensors)
    <table>
    <tr>
    $(join(["<th>$(string(key)[begin+1:end-1])</th>" for key in keys(size_plts)], '\n'))
    </tr>
    <tr>
    $(join(["<td>\n\n```\n$(string(plt; color=false))\n```\n\n</td>" for plt in size_plts], '\n'))
    </tr>
    </table>
    """

    # Runtime vs. rank
    rank_sweeps = (sortkeys ∘ group)(
        ((key, _),) -> (; size = key.size, mode = key.mode),
        ((key, result),) -> (key.rank, result),
        pairs(mttkrp_dict),
    )
    rank_plts = map(pairs(rank_sweeps)) do (key, sweep)
        return lineplot(
            getindex.(sweep, 1),
            getproperty.(median.(getindex.(sweep, 2)), :time) ./ 1e6;
            title = string(key)[begin+1:end-1],
            xlabel = "Rank",
            ylabel = "Time (ms)",
            canvas = DotCanvas,
            width = 30,
            height = 10,
            margin = 0,
        )
    end
    rank_report = """
    ## Runtime vs. rank
    <table>
    <tr>
    $(join(["<th>$(string(key)[begin+1:end-1])</th>" for key in keys(rank_plts)], '\n'))
    </tr>
    <tr>
    $(join(["<td>\n\n```\n$(string(plt; color=false))\n```\n\n</td>" for plt in rank_plts], '\n'))
    </tr>
    </table>
    """

    # Add to the report
    report *= "\n\n" * """
    # MTTKRP benchmark plots

    $size_report
    $rank_report
    """
end

### Save report
write(joinpath(@__DIR__, "report.md"), report)
