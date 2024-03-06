# Utility functions for benchmarking

module GCPBenchmarkUtils

using BenchmarkTools, PkgBenchmark
using Dictionaries, SplitApplyCombine, UnicodePlots

## Insert <details> tags to get collapsible sections on GitHub

"""
    collapsible_details(markdown; header_level=2)

Modify `markdown` to collapse at header level `header_level`.
"""
function collapsible_details(markdown; header_level = 2)
    # Extract all the lines and find the header lines
    lines = split(markdown, '\n')
    header_lines = map(findall(contains(r"^(?<tag>#+) "), lines)) do idx
        line = lines[idx]
        level = length(match(r"^(?<tag>#+) ", line)[:tag])
        return idx => level
    end

    # Filter out subheaders (level above `header_level`)
    header_lines = filter(header_lines) do (_, level)
        return level <= header_level
    end

    # Loop through and insert "<details>" tags
    tag_opened = false
    for (idx, level) in header_lines
        # Close prior opened tag
        if tag_opened
            lines[idx] = string("\n</details>\n", '\n', lines[idx])
            tag_opened = false
        end

        # Insert/open new tag for headers at `header_level`
        if level == header_level
            lines[idx] = string(lines[idx], '\n', "\n<details>\n")
            tag_opened = true
        end
    end
    if tag_opened
        lines[end] = string(lines[end], '\n', "\n</details>\n")
        tag_opened = false
    end

    # Form full string
    new_markdown = join(lines, '\n')

    # Tidy up spacing with "<details>" tags
    new_markdown = replace(
        new_markdown,
        r"<details>\n\n(?<extra>\n+)" => s"\g<extra><details>\n\n",
        r"(?<extra>\n+)\n\n<\/details>" => s"\n\n</details>\g<extra>",
    )

    return new_markdown
end

## MTTKRP sweep

"""
    export_mttkrp_sweep(results::BenchmarkResults)
    export_mttkrp_sweep(results::BenchmarkJudgement)
    export_mttkrp_sweep(results::Vector{Pair{String,BenchmarkResults}})
    export_mttkrp_sweep(results::Vector{Pair{String,BenchmarkGroup}})

Generate a markdown report for the MTTKRP sweeps from `results`.

!!! note

    Using a separate plot for each curve since it would be great
    to be able to copy into GitHub comments (e.g., within PRs)
    but GitHub doesn't currently handle ANSI color codes
    (https://github.com/github/markup/issues/1538).
    Tried working around by converting to HTML with ANSIColoredPrinters.jl
    (https://github.com/JuliaDocs/ANSIColoredPrinters.jl)
    which is the package that Documenter.jl uses to support ANSI color codes
    (https://github.com/JuliaDocs/Documenter.jl/pull/1441/files).
    However, GitHub also doesn't currently support coloring the text
    (https://github.com/github/markup/issues/1440).
"""
export_mttkrp_sweep(results::BenchmarkResults) = export_mttkrp_sweep(["" => results])
export_mttkrp_sweep(results::BenchmarkJudgement) = export_mttkrp_sweep([
    "Target" => PkgBenchmark.target_result(results),
    "Baseline" => PkgBenchmark.baseline_result(results),
])
function export_mttkrp_sweep(results_list::Vector{Pair{String,BenchmarkResults}})
    # Extract the mttkrp suites
    results_list = map(results_list) do (name, results)
        bg = PkgBenchmark.benchmarkgroup(results)
        return haskey(bg, "mttkrp") ? name => bg["mttkrp"] : nothing
    end
    results_list = filter(!isnothing, results_list)
    isempty(results_list) && return ""

    # Call the main method
    return export_mttkrp_sweep(results_list)
end
function export_mttkrp_sweep(results_list::Vector{Pair{String,BenchmarkGroup}})
    isempty(results_list) && return ""

    # Load the results into dictionaries
    result_names = first.(results_list)
    result_dicts = map(last.(results_list)) do results
        return (sortkeys ∘ dictionary ∘ map)(results) do (key_str, result)
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
    all_keys = (sort ∘ unique ∘ mapmany)(keys, result_dicts)

    # Runtime vs. size (for square tensors)
    size_group_keys = group(
        key -> (; ndims = length(key.size), rank = key.rank, mode = key.mode),
        filter(key -> allequal(key.size), all_keys),
    )
    size_groups = collect(keys(size_group_keys))
    size_plots = product(result_dicts, size_groups) do result_dict, size_group
        sweep_keys = filter(in(keys(result_dict)), size_group_keys[size_group])
        isempty(sweep_keys) && return nothing
        return lineplot(
            only.(unique.(getindex.(sweep_keys, :size))),
            getproperty.(median.(getindices(result_dict, sweep_keys)), :time) ./ 1e6;
            title = pretty_str(size_group),
            xlabel = "Size",
            ylabel = "Time (ms)",
            canvas = DotCanvas,
            width = 30,
            height = 10,
            margin = 0,
        )
    end

    # Runtime vs. rank
    rank_group_keys = group(key -> (; size = key.size, mode = key.mode), all_keys)
    rank_groups = collect(keys(rank_group_keys))
    rank_plots = product(result_dicts, rank_groups) do result_dict, rank_group
        sweep_keys = filter(in(keys(result_dict)), rank_group_keys[rank_group])
        isempty(sweep_keys) && return nothing
        return lineplot(
            getindex.(sweep_keys, :rank),
            getproperty.(median.(getindices(result_dict, sweep_keys)), :time) ./ 1e6;
            title = pretty_str(rank_group),
            xlabel = "Rank",
            ylabel = "Time (ms)",
            canvas = DotCanvas,
            width = 30,
            height = 10,
            margin = 0,
        )
    end

    # Runtime vs. mode
    mode_group_keys = group(key -> (; size = key.size, rank = key.rank), all_keys)
    mode_groups = collect(keys(mode_group_keys))
    mode_plots = product(result_dicts, mode_groups) do result_dict, mode_group
        sweep_keys = filter(in(keys(result_dict)), mode_group_keys[mode_group])
        isempty(sweep_keys) && return nothing
        return boxplot(
            string.("mode ", getindex.(sweep_keys, :mode)),
            getproperty.(getindices(result_dict, sweep_keys), :times) ./ 1e6;
            title = pretty_str(mode_group),
            xlabel = "Time (ms)",
            canvas = DotCanvas,
            width = 30,
            height = 10,
            margin = 0,
        )
    end

    return """
    # MTTKRP benchmark plots

    ## Runtime vs. size (for square tensors)
    Below are plots showing the runtime in miliseconds of MTTKRP as a function of the size of the square tensor, for varying ranks and modes:
    $(plot_table(size_plots, pretty_str.(size_groups), result_names))

    ## Runtime vs. rank
    Below are plots showing the runtime in miliseconds of MTTKRP as a function of the size of the rank, for varying sizes and modes:
    $(plot_table(rank_plots, pretty_str.(rank_groups), result_names))

    ## Runtime vs. mode
    Below are plots showing the runtime in miliseconds of MTTKRP as a function of the mode, for varying sizes and ranks:
    $(plot_table(mode_plots, pretty_str.(mode_groups), result_names))
    """
end

# Create pretty string for plot titles, etc.
pretty_str(group::NamedTuple) = string(group)[begin+1:end-1]

# Create HTML table of plots
function plot_table(plots, colnames, rownames)
    # Create strings for each plot
    plot_strs = map(plots) do plot
        return isnothing(plot) ? "NO RESULTS" : """```
                                                $(string(plot; color=false))
                                                ```"""
    end

    # Create matrix of cell strings
    cell_strs = [
        "<th></th>" permutedims(string.("<th>", colnames, "</th>"))
        string.("<th>", rownames, "</th>") string.("<td>\n\n", plot_strs, "\n\n</td>")
    ]

    # Create vector of row strings
    row_strs = string.("<tr>\n", join.(eachrow(cell_strs), '\n'), "\n</tr>")

    # Return full table (with blank rows "<tr></tr>" to work around GitHub styling)
    return string(
        "<table>\n",
        row_strs[1],
        '\n',
        join(row_strs[2:end], "\n<tr></tr>\n"),
        "\n</table>",
    )
end

end
