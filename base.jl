include("utils.jl")
include("figure.jl")

using SplitApplyCombine
using Random
using StatsBase
using Distributions
using Memoize
using QuadGK
using LaTeXStrings

integrate(f, lo, hi) = first(quadgk(f, lo, hi))

# %% --------

function prob_consider(u, d::Distribution; β, u0=0., C=0.)
    # WARNING: not normalized!
    pdf(d, u) * (C + exp(β * u) * abs(u - u0))
end

function prob_occur(u, d::Distribution; β)
    # WARNING: not normalized!
    pdf(d, u) * exp(β * u)
end

function subjective_value(u, p; β=0., n_sample=1, u0=0.)
    w_soft = exp.(u .* β)
    w_uws = abs.(u .- u0)
    # w_uws = ones(length(u))
    sampled = sample(eachindex(u), Weights(p .* w_soft .* w_uws), n_sample)
    mean(u[sampled], Weights(1 ./ w_uws[sampled]))
end

subjective_value(u; kws...) = subjective_value(u, 1; kws...)

function prob_accept(n_sample, d::ContinuousDistribution; β, u0=0, N=100)
    n_sample == 1 && return prob_accept_one(d; β, u0)
    monte_carlo() do
        u = rand(d, N)
        subjective_value(u; β, u0, n_sample) > 0
    end
end

function prob_accept(n_sample, d::DiscreteDistribution; β, u0=0, N=100)
    @assert false
    u = support(d)
    p = pdf.(d, u)
    monte_carlo() do
        subjective_value(u, p; β, u0, n_sample) > rand((-eps(), eps()))
    end
end

function prob_accept_one(d; β, u0)
    lo, hi = integration_limits(d, β)
    good = integrate(0, hi) do u
        prob_consider(u, d; β, u0)
    end
    bad = integrate(lo, 0) do u
        prob_consider(u, d; β, u0)
    end
    good / (bad + good)
end


function integration_limits(d, β; tol=1e-10)
    lo = d.µ
    while prob_occur(lo, d; β) > tol
        lo -= d.σ
    end
    hi = d.µ
    while prob_occur(hi, d; β) > tol
        hi += d.σ
    end
    lo, hi
end

@memoize function expected_value(d::ContinuousDistribution; β)
    lo, hi = integration_limits(d, β)
    unnorm = integrate(lo, hi) do u
        p = pdf(d, u) * exp(β * u)
        p * u
    end
    Z = integrate(lo, hi) do u
        pdf(d, u) * exp(β * u)
    end
    unnorm / Z
end

function expected_value(d::DiscreteDistribution; β)
    u = support(d)
    p = pdf.(d, u)
    sum(normalize!(p .* exp.(β .* u)) .* u)
end

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