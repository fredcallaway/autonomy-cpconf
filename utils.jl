using Serialization
using AxisKeys
import Base.Iterators: product

function mutate(x::T; kws...) where T
    for field in keys(kws)
        if !(field in fieldnames(T))
            error("$(T.name) has no field $field")
        end
    end
    return T([get(kws, fn, getfield(x, fn)) for fn in fieldnames(T)]...)
end

function grid(;kws...)
    X = map(Iterators.product(values(kws)...)) do x
        (; zip(keys(kws), x)...)
    end
    KeyedArray(X; kws...)
end

function keyed(name, xs)
    KeyedArray(xs; Dict(name => xs)...)
end

keymax(X::KeyedArray) = (; (d=>x[i] for (d, x, i) in zip(dimnames(X), axiskeys(X), argmax(X).I))...)
keymax(x::KeyedArray{<:Real, 1}) = axiskeys(x, 1)[argmax(x)]

Base.dropdims(idx::Union{Symbol,Int}...) = X -> dropdims(X, dims=idx)

function table(X::KeyedArray)
    map(collect(pairs(X))) do (idx, v)
        keyvals = (name => keys[i] for (name, keys, i) in zip(dimnames(X), axiskeys(X), idx.I))
        (;keyvals..., value=v)
    end[:]
end

function table(X::KeyedArray{<:NamedTuple})
    map(collect(pairs(X))) do (idx, v)
        keyvals = (name => keys[i] for (name, keys, i) in zip(dimnames(X), axiskeys(X), idx.I))
        (;keyvals..., v...,)
    end[:]
end

function monte_carlo(f, N=10000)
    N \ mapreduce(+, 1:N) do i
        f()
    end
end

linscale(x, low, high) = low + x * (high-low)
logscale(x, low, high) = exp(log(low) + x * (log(high) - log(low)))
unlinscale(x, low, high) = (x - low) / (high-low)
unlogscale(x, low, high) = (log(x) - log(low)) / (log(high) - log(low))

juxt(fs...) = x -> Tuple(f(x) for f in fs)

nanreduce(f, x) = f(filter(!isnan, x))
nanmean(x) = nanreduce(mean, x)
nanstd(x) = nanreduce(std, x)
normalize(x) = x ./ sum(x)
normalize!(x) = x ./= sum(x)