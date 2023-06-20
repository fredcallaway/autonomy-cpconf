include("utils.jl")
include("figure.jl")

using Random
using StatsBase
using Distributions
using Memoize

# %% --------

function prob_consider(u, d::Distribution; β=1., u0=0., C=0.2)
    pdf(d, u) * (C + exp(β * u) * abs(u - u0))
end

function subjective_value(u::Vector{Float64}; β=0., n_sample=1, u0=0.)
    w_soft = exp.(u .* β)
    w_uws = abs.(u .- u0)

    sampled = sample(eachindex(u), Weights(w_soft .* w_uws), n_sample)

    mean(u[sampled], Weights(1 ./ w_uws[sampled]))
end

@memoize function prob_accept(n_sample, d; β)
    p = map(n_sample) do n_sample
        monte_carlo() do
            u = rand(d, 100)
            subjective_value(u; β, n_sample) > 0
        end
    end
end

# %% --------

function fig(f, name; kws...)
    figure(name; pdf=true, size=(200, 200), widen=false, framestyle=:origin, yticks=false, xticks=false, kws...) do
        f()
    end
end

groups = (
    high = (
        color="#D57A72",
        β = 2
    ),
    low = (
        color="#70D6FF",
        β = 0.2
    )
)

function groupfig(f, name; kws...)
    for (grp, def) in pairs(groups)
        fig(()->f(def), "$(name)_$(grp)"; kws...)
    end
end

function testfig(f, name="tmp"; kws...)
    ps = map(collect(pairs(groups))) do (grp, def)
        plot(;kws...)
        f(def)
    end
    fig("tmp") do
        plot!(ps..., size=(400,200))
    end
end

# %% --------
# d = SkewNormal(.5, 1, -5)
d = Normal(-1, 1)
u = -4:.01:4

fig("possible") do
    plot!(u, pdf.(d, u), color=:gray)
end

fig("bear") do
    plot!(u, exp.(u), color=:black)
end

fig("lieder") do
    plot!(u, abs.(u), color=:black)
end

groupfig("considered") do (;β, color)
    plot!(u, prob_consider.(u, d; β); color)
end

groupfig("choice", ylim=(0, 1), framestyle=:axes, yticks=false) do (;β, color)
    p = prob_accept(1, d; β)
    bar!([1-p, p]; color, w=1)
    # plot!(n_sample, prob_accept(n_sample, d; high.β); high.color)
end
