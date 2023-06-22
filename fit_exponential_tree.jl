include("figure.jl")
include("curve_fitting.jl")
include("utils.jl")
include("tree.jl")
using ProgressMeter

# %% --------


function control_ratio(t, p_slip)
    d1 = VDist(t; p_slip)
    d0 = VDist(t; p_slip=.5)

    h1 = fit(Histogram, d1.v, Weights(d1.p), -3.25:.5:3.25)
    h0 = fit(Histogram, d0.v, Weights(d0.p), -3.25:.5:3.25)

    x = centers(h0.edges[1])
    p0 = StatsBase.normalize(h0, mode=:pdf).weights
    p1 = StatsBase.normalize(h1, mode=:pdf).weights
    p1 ./ p0
end


# %% --------
slips = .1:.1:.4

curves = map() do p_slip
    repeatedly(30) do
        t = tree(14, Normal(0, 1));
        control_ratio(t, p_slip)
    end
end

# %% --------
x = -3:.5:3

figure() do
    plots = map(curves, slips) do curve, p_slip
        p = plot(title="p_slip = $p_slip", titlefontsize=8)
        xs = reduce(vcat, fill(x, 30))
        ys = reduce(vcat, curve)
        plot!(x, reduce(hcat, curve), color=:black, alpha=0.3, w=1)
        plot!(-3:.01:3, exponential(-3:.01:3, fit_exponential(xs, ys)), color=:red, ls=:dot)
        p
    end
    plot(plots..., size=(400, 400))
end



# %% --------

figure() do
    plots = map(.1:.1:.4) do p_slip
        p = plot()
        repeatedly(30) do
            t = tree(14, Normal(0, 1));
            plot!(control_ratio(t, p_slip)..., w=1, color=:black, alpha=0.3)
        end
        p
    end
    plot(plots...)
end

# %% --------

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

