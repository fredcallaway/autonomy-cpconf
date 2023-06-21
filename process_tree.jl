using POMDPs, QuickPOMDPs, POMDPTools
using DiscreteValueIteration
using Combinatorics
using Distributions
using LsqFit
using LaTeXStrings

include("utils.jl")
include("figure.jl")


function build_tree(depth, rdist, p_slip)
    levels = map(0:depth) do d
        join.(multiset_permutations("01"^d, d))
    end
    push!(levels, ["TERM"])
    states = reduce(vcat, levels)
    actions = "01"
    rewards = Dict(l => float(rand(rdist)) for l in states)
    # rewards = Dict(l => rand(rdist) for l in levels[end-1])
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

mdp = build_tree(DEPTH, Bernoulli(0.5), 0.5)

solver = SparseValueIterationSolver(max_iterations=1000, belres=1e-6, verbose=false) # creates the solver
println("solving")
policy = solve(solver, mdp)

reward(mdp, "0", '0')

# %% --------

println("rollouts")
rs = RolloutSimulator()
smart = map(1:1000000) do i
    simulate(rs, mdp, policy)
end

dumb = map(1:1000000) do i
    simulate(rs, mdp, RandomPolicy(mdp))
end

# %% --------

println("plottings")

g = 0:DEPTH
s = normalize(counts(Int.(smart), g))
d = normalize(counts(Int.(dumb), g))

@. exponential(x, (x0, β)) = exp(β * (x .- x0))

function exponential_fit(x, y, p0=[0., 1.,])
    fit = curve_fit(exponential, x, y, p0)
    exponential(x, fit.param)
end

function plot_exponential_fit!(x, y, p0=[0., 1.,])
    res = curve_fit(exponential, x, y, p0)
    plot!(x, y)
    pred = exponential(x, res.param)
    plot!(x, pred, linestyle=:dot)
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

