include("figure.jl")
include("curve_fitting.jl")
include("utils.jl")
include("tree.jl")
using ProgressMeter

hists = map(0.:.1:.4) do p_slip
    ds = @showprogress map(1:5000) do i
        VDist(tree(14, Normal(0, 1)); p_slip);
    end

    p = reduce(vcat, map(get(:p), ds))
    v = reduce(vcat, map(get(:v), ds))
    hist = fit(Histogram, v, Weights(p), -4:.1:4)
end

# %% --------

pdfs = map(hists) do hist
    StatsBase.normalize(hist, mode=:pdf).weights
end
x = centers(hists[1].edges[1])
y0 = pdf.(Normal(0,1), x)

params = map(pdfs) do y1
    fit_exponential(x, y1 ./ y0)
end

figure() do
    plot(0:.1:.4, invert(params)[3])
end

figure("tree_returns") do
    for y1 in pdfs
        plot!(x, y1)
    end
    plot!(x, y0, color=:black)
end

# %% --------

figure("tree_fits") do
    ps = map(pdfs) do y1
        plot()
        i = max(1, findfirst(y1 ./ y0 .> 0))
        plot_fit!(x, y1 ./ y0)
        # plot!(xlim=(-2, 5), ylim=(-3, 10))
    end
    plot(ps..., size=(800,200), layout=(1,5))
end

