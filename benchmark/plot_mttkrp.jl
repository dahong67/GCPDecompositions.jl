using Random
using GCPDecompositions
using BenchmarkTools
using UnicodePlots

# Load benchmark results from benchmark_results.txt
benchmark_results = readresults("benchmark_results.txt")
mttkrp_results = benchmark_results.benchmarkgroup["mttkrp"]

szs = [10,30,50,80,120,200]
shapes = [(sz, sz, sz) for sz in szs]
rs = 20:20:200
results = zeros((size(szs)[1], size(rs)[1]))

# Collect results in array where columns are ranks and rows are sizes
for (col_idx, r) in enumerate(rs)
    for (row_idx, sz) in enumerate(szs)
        # Get median runtime (in milliseconds)
        median_time = median(mttkrp_results["mttkrp-size(X)=($sz, $sz, $sz), rank(X)=$r"]).time / 10^6
        results[row_idx, col_idx] = median_time
    end
end

first_r = rs[1]
plt = lineplot(szs, results[:, 1], title="MTTKRP runtime vs. size", 
            xlabel="Size", ylabel="Runtime(ms)", name="r = $first_r")
for (idx, r) in enumerate(rs[2:end])
    lineplot!(plt, szs, results[:, idx + 1], name="r = $r")
end
print(plt)

