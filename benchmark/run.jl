# Script to run benchmarks and export a report

## Make sure the benchmark environment is activated
import Pkg
Pkg.activate(@__DIR__)

## Run benchmarks
using GCPDecompositions, PkgBenchmark

const SELECTED_SUITES = nothing     # set to `nothing` to run all suites
# const SELECTED_SUITES = ["mttkrp"]

results =
    isnothing(SELECTED_SUITES) ? benchmarkpkg(GCPDecompositions) :
    benchmarkpkg(
        GCPDecompositions,
        BenchmarkConfig(; env = Dict("GCP_BENCHMARK_SUITES" => join(SELECTED_SUITES, ' '))),
    )
writeresults(joinpath(@__DIR__, "results.json"), results)

## Generate report and save
using BenchmarkTools, UnicodePlots

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
    mttkrp_results = PkgBenchmark.benchmarkgroup(results)["mttkrp"]

    szs = [10, 30, 50, 80, 120, 200]
    rs = 20:20:200

    plts = map(rs) do r
        med_times = map(szs) do sz
            return median(mttkrp_results["mttkrp-size(X)=$((sz, sz, sz)), rank(X)=$r"]).time / 1e6
        end
        return lineplot(
            szs,
            med_times;
            title = "MTTKRP runtime vs. size (r = $r)",
            xlabel = "Size",
            ylabel = "Time (ms)",
            canvas = DotCanvas,
            width = 30,
            height = 10,
            margin = 0,
        )
    end
    report *= "\n\n" * """
    # MTTKRP benchmark plots
    <table>
    <tr>
    $(join(["<th>r = $r</th>" for r in rs], ' '))
    </tr>
    <tr>
    $(join(["<td>\n\n```\n$(string(plt; color=false))\n```\n\n</td>" for plt in plts],'\n'))
    </tr>
    </table>
    """
end

### Save report
write(joinpath(@__DIR__, "report.md"), report)
