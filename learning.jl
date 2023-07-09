include("tree.jl")
include("utils.jl")
include("figure.jl")
include("base.jl")

using DataStructures
using Optim

function softmax(d::VDist; β)
    p = normalize(d.p .* exp.(β * d.v))
    VDist(p, d.v)
end

subjective_value(d::VDist; kws...) = subjective_value(d.v, d.p; kws...)

function act(n::Node; β, p_slip, N=1)
    f = rand(Bernoulli(p_slip)) ? argmin : argmax
    f(n.children) do n1
        isleaf(n1) && return n1.r
        baseline = VDist(n1, p_slip=0.5)
        # d = softmax(baseline; β)
        # mean(sample(d.v, Weights(d.p), N))

        subjective_value(baseline; β)
    end
end

function rollout(n::Node; β, p_slip, N=1)
    while !isempty(n.children)
        n = act(n; β, p_slip, N)
    end
    n.r
end

function fit_β(baseline::VDist, d::VDist)
    optimize(-50, 50) do β
        pred = normalize(baseline.p .* exp.(β .* d.v))
        crossentropy(d.p, pred)
    end |> Optim.minimizer
end

function learn(n::Node, experience, n_trial::Int; p_slip, N)
    baseline = VDist(n, p_slip=0.5)
    d = VDist(normalize(experience), baseline.v)
    β = fit_β(baseline, d)

    map(1:n_trial) do _
        r = rollout(n; β, p_slip)
        i = findfirst(isequal(r), baseline.v)
        experience[i] += 1
        d = VDist(normalize(experience), baseline.v)
        β = fit_β(baseline, d)
        (r, β)
    end
end

# %% --------

d = SkewNormal(0.5, 1, -5)
u = -3:.01:3
p0 = pdf.(d, u)

fig("learning_baseline") do
    plot!(u, p0, color=CTRUE)
end

fig("learning_experienced") do
    foreach(groups) do (;β, color)
        plot!(u, exp.(β .* u) .* p0; color)
    end
end

fig("learning_bias") do
    foreach(groups) do (;β, color)
        # β = min(.7, β)
        p = @. abs(u) .* exp(β * u)
        plot!(u, p; color, ylim=(0, 10))
    end
end

# %% --------


traces = cache("cache/learning") do
    repeatedly(10) do
        n_trial = 300
        n = tree(14, SkewNormal(0, 1, -5));
        baseline = VDist(n, p_slip=0.5)

        good = 10 * normalize(exp.(3 .* baseline.v))
        bad = 10 * normalize(exp.(-3 .* baseline.v))

        high = learn(n, good, n_trial; p_slip=0.02, N=1)
        low = learn(n, bad, n_trial; p_slip=0.02, N=1)
        (high, low)
    end
end

figure("learning_control", pdf=true) do
    for (high, low) in traces
        plot!(invert(high)[2]; groups.high.color)
        plot!(invert(low)[2]; groups.low.color)
    end
    plot!(size=(300, 200), grid=false, ticks=false, framestyle=:origin, widen=false)
end

figure("learning_reward", pdf=true) do
    for (high, low) in traces
        plot!(rollingmean(invert(high)[1], 100); groups.high.color)
        plot!(rollingmean(invert(low)[1], 100); groups.low.color)
    end
    plot!(size=(300, 200), grid=false, ticks=false, framestyle=:origin, widen=false)
end

