using POMDPs, QuickPOMDPs, POMDPTools
using DiscreteValueIteration
using Combinatorics
using Distributions
using LsqFit
using LaTeXStrings
using StatsBase
using ProgressMeter

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

# %% --------
rdist = Normal(0, 5)

function simulate_outcomes(p_slip; depth=10, n_mdp=100, n_roll=10000)
    solver = SparseValueIterationSolver(max_iterations=1000, belres=1e-6)
    outcomes = @showprogress pmap(1:n_mdp) do _
        mdp = build_tree(depth, rdist, p_slip; final_only=true)
        policy = solve(solver, mdp);
        rsim = RolloutSimulator()
        map(1:n_roll) do i
            simulate(rsim, mdp, policy)
        end
    end
    reduce(vcat, outcomes)
end

outcomes = simulate_outcomes(0.5)

out = Int.(round.(outcomes))

lim = maximum(abs.((
    quantile(rdist, .01),
    quantile(out, .99),
))) |> round |> Int

x = -lim:lim

y1 = normalize(counts(out, x))
y0 = pdf.(rdist, x)

# %% --------


@. exponential(x, (C, x0, β)) = C + exp(β * (x - x0))

function plot_fit!(x, y, p0=[1., 0., 1.])
    res = curve_fit(exponential, x, y, p0)
    plot!(x, y, color=:lightgray)
    pred = exponential(x, res.param)
    @show res.param
    plot!(x, pred, linestyle=:dot, color=:black)
    # C, x0, β = round.(res.param, digits=2)
    # expr = L"%$C + e^{%$β(x - %$x0)}"
    # annotate!(:topleft, text(expr, 14, :left))
end

y = y1 ./ y0

figure() do
    p1 = plot()
    plot!(x, y0, color=:gray)
    plot!(x, y1, color=groups.high.color)

    p2 = plot()
    plot_fit!(x, y1 ./ y0)
    plot(p1, p2, size=(500, 200))
end


# %% --------

figure() do
    x = -15:.01:15
    plot(x, power(x, (0, -.5, 1.01, 100)), yaxis=:log)
end