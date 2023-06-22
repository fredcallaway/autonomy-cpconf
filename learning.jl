include("tree.jl")
include("utils.jl")
include("figure.jl")
include("base.jl")


# %% --------

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

using DataStructures
using Optim

function fit_β(baseline::VDist, d::VDist)
    optimize(-5, 5) do β
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

t = Node(
    Node(0.),
    Node(
        Node(-2),
        Node(1)
    )
)

figure("learning", xlab="Trial", ylab="Perceived Control") do
    high = [0, 0, 5]
    low = [0, 5, 0]
    for i in 1:10
        trace = learn(n, copy(high), 1000; p_slip=0.02, N=1)
        plot!(invert(trace)[2], w=1; groups.high.color)

        trace = learn(n, copy(low), 1000; p_slip=0.02, N=1)
        plot!(invert(trace)[2], w=1; groups.low.color)
    end
end

