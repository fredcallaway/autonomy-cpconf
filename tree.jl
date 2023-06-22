using Distributions
import Statistics: mean
using LinearAlgebra: dot
using StatsBase


struct Node{N}
    children::NTuple{N, Node}
    r::Float64
end

Node(children::Node...) = Node(children, 0.)
Node(r::Float64) = Node((), r)

isleaf(n::Node{N}) where N = N == 0

function tree(depth, rdist)
    if depth == 0
        Node((), rand(rdist))
    else
        Node((
            tree(depth-1, rdist),
            tree(depth-1, rdist),
        ), 0.)
    end
end

struct VDist
    p::Vector{Float64}
    v::Vector{Float64}
end

mean((;v,p)::VDist) = dot(p, v)

function mix(d1, d2, p1)
    VDist(
        [p1 .* d1.p; (1-p1) .* d2.p],
        [d1.v; d2.v],
    )
end

function VDist(n::Node; p_slip)
    isempty(n.children) && return VDist([1.], [n.r])
    v1 = VDist(n.children[1]; p_slip)
    v2 = VDist(n.children[2]; p_slip)
    p1 = mean(v1) > mean(v2) ? 1 - p_slip : p_slip
    mix(v1, v2, p1)
end
