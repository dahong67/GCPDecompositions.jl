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
            compare,
            BenchmarkConfig(; env = Dict("GCP_BENCHMARK_SUITES" => suite)),
        )
    results = PkgBenchmark.target_result(judgement)
    results_baseline = PkgBenchmark.baseline_result(judgement)
    writeresults(joinpath(@__DIR__, "results.json"), results)
end

## Generate report and save
using BenchmarkTools, Dictionaries, SplitApplyCombine, UnicodePlots

### Create initial/base report
report = compare != "none" ? sprint(export_markdown, judgement) : sprint(export_markdown, results)

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
    # Load result (possibly from 2 different commits) into a dictionary
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

    if compare != "none"
        mttkrp_results_baseline = PkgBenchmark.benchmarkgroup(results_baseline)["mttkrp"]
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
    end


    plot_vars = ["size"]

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
    if compare != "none"
        size_sweeps_baseline = (sortkeys ∘ group)(
            ((key, _),) -> (; ndims = length(key.size), rank = key.rank, mode = key.mode),
            ((key, result),) -> ((only ∘ unique)(key.size), result),
            filter(((key, _),) -> allequal(key.size), pairs(mttkrp_dict_baseline)),
        )
        size_plts_baseline = map(pairs(size_sweeps_baseline)) do (key, sweep)
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
    end
    size_report = """
    ## Runtime vs. size (for square tensors)
    Below are plots showing the runtime in miliseconds of MTTKRP as a function of the size of the square tensor, for varying ranks and modes:
    <table>
    <tr>
    $(join(["<th>$(string(key)[begin+1:end-1])</th>" for key in keys(size_plts)], '\n'))
    </tr>
    <tr>
    $(join(["<td>\n\n```\n$(string(plt; color=false))\n```\n\n</td>" for plt in size_plts], '\n'))
    </tr>
    <tr>
    $(join(["<td>\n\n```\n$(string(plt; color=false))\n```\n\n</td>" for plt in size_plts_baseline], '\n'))
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

    if compare != "none"
        rank_sweeps_baseline = (sortkeys ∘ group)(
            ((key, _),) -> (; size = key.size, mode = key.mode),
            ((key, result),) -> (key.rank, result),
            pairs(mttkrp_dict_baseline),
        )
        rank_plts = map(pairs(rank_sweeps_baseline)) do (key, sweep)
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
    end

    rank_report = """
    ## Runtime vs. rank
    Below are plots showing the runtime in miliseconds of MTTKRP as a function of the size of the rank, for varying sizes and modes:
    <table>
    <tr>
    $(join(["<th>$(string(key)[begin+1:end-1])</th>" for key in keys(rank_plts)], '\n'))
    </tr>
    <tr>
    $(join(["<td>\n\n```\n$(string(plt; color=false))\n```\n\n</td>" for plt in rank_plts], '\n'))
    </tr>
    <tr>
    $(join(["<td>\n\n```\n$(string(plt; color=false))\n```\n\n</td>" for plt in rank_plts_baseline], '\n'))
    </tr>
    </table>
    """

    # Runtime vs. mode
    mode_sweeps = (sortkeys ∘ group)(
        ((key, _),) -> (; size = key.size, rank = key.rank),
        ((key, result),) -> ("mode $(key.mode)", result),
        pairs(mttkrp_dict),
    )
    mode_plts = map(pairs(mode_sweeps)) do (key, sweep)
        return boxplot(
            getindex.(sweep, 1),
            getproperty.(getindex.(sweep, 2), :times) ./ 1e6;
            title = string(key)[begin+1:end-1],
            xlabel = "Time (ms)",
            canvas = DotCanvas,
            width = 30,
            height = 10,
            margin = 0,
        )
    end

    if compare != "none"
        mode_sweeps = (sortkeys ∘ group)(
            ((key, _),) -> (; size = key.size, rank = key.rank),
            ((key, result),) -> ("mode $(key.mode)", result),
            pairs(mttkrp_dict),
        )
        mode_plts = map(pairs(mode_sweeps_baseline)) do (key, sweep)
            return boxplot(
                getindex.(sweep, 1),
                getproperty.(getindex.(sweep, 2), :times) ./ 1e6;
                title = string(key)[begin+1:end-1],
                xlabel = "Time (ms)",
                canvas = DotCanvas,
                width = 30,
                height = 10,
                margin = 0,
            )
        end
    end

    mode_report = """
    ## Runtime vs. mode
    Below are plots showing the runtime in miliseconds of MTTKRP as a function of the mode, for varying sizes and ranks:
    <table>
    <tr>
    $(join(["<th>$(string(key)[begin+1:end-1])</th>" for key in keys(mode_plts)], '\n'))
    </tr>
    <tr>
    $(join(["<td>\n\n```\n$(string(plt; color=false))\n```\n\n</td>" for plt in mode_plts], '\n'))
    </tr>
    <tr>
    $(join(["<td>\n\n```\n$(string(plt; color=false))\n```\n\n</td>" for plt in mode_plts_baseline], '\n'))
    </tr>
    </table>
    """

    # Add to the report
    report *= "\n\n" * """
    # MTTKRP benchmark plots

    $size_report
    $rank_report
    $mode_report
    """
end

### Save report
write(joinpath(@__DIR__, "report.md"), report)
