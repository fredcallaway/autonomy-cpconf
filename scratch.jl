import Base: @kwdef

abstract type Weighter end

struct Softmax{T<:Real} <: Weighter
    β::T
end

function score(weighter::Softmax, u)
    @. exp(u * weighter.β)
end

@kwdef struct SoftmaxUWS <: Weighter
    β::Real
    u0::Real = 0
end
SoftmaxUWS(β::Real) = SoftmaxUWS(;β)

function score(weighter::SoftmaxUWS, u)
    (;β, u0) = weighter
    @. exp(u * β) * abs(u - u0)
end

weight(w::Weighter, u) = normalize(score(w, u))
sample(w::Weighter, u, n=1) = sample(u, Weights(weight(w, u)), n)

function subjective_value(weighter::Weighter, u::Vector{Float64}, n_sample::Int)
    w = weight(weighter, u)
    mean(sample(w, u, n_sample) ./ w)
end

