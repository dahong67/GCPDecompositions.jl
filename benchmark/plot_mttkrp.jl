using Random
using GCPDecompositions
using BenchmarkTools
using UnicodePlots

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

szs = [15, 30, 45, 60, 75, 90]
shapes = [(sz, sz, sz) for sz in szs]
n = 1
rs = 1:5
results = zeros((size(szs)[1], size(rs)[1]))

for (idx, r) in enumerate(rs)
    Random.seed!(0)
    Xs = [randn(sz) for sz in shapes];
    Us = [[randn(Ik,r) for Ik in sz] for sz in shapes];
    times = [@belapsed GCPDecompositions.mttkrp($X, $U, $n) for (X,U) in zip(Xs, Us)] 
    results[:, idx] = times
end

first_r = rs[1]
plt = lineplot(szs, results[:, 1], title="MTTKRP runtime vs. size", 
            xlabel="Size", ylabel="Runtime(s)", name="r = $first_r");
for (idx, r) in enumerate(rs[2:end])
    lineplot!(plt, szs, results[:, idx + 1], name="r = $r")
end
print(plt)

