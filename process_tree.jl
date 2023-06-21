using POMDPs, QuickPOMDPs, POMDPTools
using DiscreteValueIteration
using Combinatorics
using Distributions
using LsqFit
using LaTeXStrings
using StatsBase

include("utils.jl")
include("figure.jl")


function build_tree(depth, rdist, p_slip; final_only=false)
    levels = map(0:depth) do d
        join.(multiset_permutations("01"^d, d))
    end
    push!(levels, ["TERM"])
    states = reduce(vcat, levels)
    actions = "01"
    rewards = final_only ?
        Dict(l => float(rand(rdist)) for l in levels[end-1]) :
        Dict(l => float(rand(rdist)) for l in states)
    QuickMDP(;
        states,
        actions,
        transition = function (s, a)
            if length(s) == depth
                Deterministic("TERM")
            else
                SparseCat([s*a, s*rand(actions)], [1-p_slip, p_slip])
            end
        end,
        reward = function(s, a)
            s in keys(rewards) ? rewards[s] : 0.
        end,
        initialstate = Deterministic(states[1]),
        discount = 1.,
        isterminal = s -> s == "TERM"
    )
end
DEPTH = 16

# mdp = build_tree(DEPTH, Bernoulli(0.5), 0.5)
mdp = build_tree(DEPTH, Normal(0, 5), 0.5; final_only=true)

solver = SparseValueIterationSolver(max_iterations=1000, belres=1e-6, verbose=false) # creates the solver
println("solving")
policy = solve(solver, mdp)

# %% --------
N = 100000

println("rollouts")
rs = RolloutSimulator()
smart = map(1:N) do i
    simulate(rs, mdp, policy)
end

dumb = map(1:N) do i
    simulate(rs, mdp, RandomPolicy(mdp))
end

# %% --------

println("plotting")

lim = maximum(abs.((
    quantile(dumb, .01),
    quantile(smart, .99),
))) |> round |> Int
g = -lim:lim

s = normalize(counts(Int.(round.(smart)), g))
d = normalize(counts(Int.(round.(dumb)), g))
d[d .< 30 / N] .= NaN
d

@. exponential(x, (x0, β)) = exp(β * (x .- x0))

function exponential_fit(x, y, p0=[0., 1.,])
    fit = curve_fit(exponential, x, y, p0)
    exponential(x, fit.param)
end

function plot_exponential_fit!(x, y, p0=[0., 1.,])
    res = curve_fit(exponential, x, y, p0)
    plot!(x, y, color=:black)
    pred = exponential(x, res.param)
    plot!(x, pred, linestyle=:dot, color=:red)
    x0, β = round.(res.param, digits=2)
    expr = L"e^{%$β(x - %$x0)}"
    annotate!(:topleft, text(expr, 14, :left))
end

figure() do
    p1 = plot()
    plot!(g, d)
    plot!(g, s)

    p2 = plot()
    keep = isfinite.(s ./ d)
    x = g[keep]; y = (s ./ d)[keep]
    plot_exponential_fit!(x, y)
    plot(p1, p2, size=(500, 200))
end

