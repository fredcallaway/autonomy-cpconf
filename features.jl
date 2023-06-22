include("base.jl")
include("curve_fitting.jl")

function sample_outcomes(R, control; n=100000, rdist=Normal(0,1))
    repeatedly(n) do
        sum(1:size(R, 2)) do i
            if rand(Bernoulli(control))
                maximum(R[:, i])
            else
                rand(R[:, i])
            end
        end
    end
end

# %% --------

figure() do
    k = 2; N = 100
    R = rand(Normal(0, 1), (k, N))
    x = sample_outcomes(R, .95)
    histogram!(x, w=0, normalize=:pdf)
    plot!(fit(Normal, x))
end

# %% --------


k = 2; N = 100
R = rand(Normal(0, 1), (k, N))
d0 = fit(Normal, sample_outcomes(R, 0.))
d1 = fit(Normal, sample_outcomes(R, 0.5))
# %% --------

figure() do
    d0 = Normal(0, 1)
    d1 = Normal(-1.5, 1.2)
    x = -3:.1:10
    plot!(x, pdf.(d1, x) ./ pdf.(d0, x))
end

# %% --------

figure() do
    hist = fit(Histogram, sample_outcomes(.2), -30:2:30)
    keep = hist.weights .> 100
    x = centers(hist.edges[1])
    y1 = StatsBase.normalize(hist, mode=:pdf).weights

    hist = fit(Histogram, sample_outcomes(0.), -30:2:30)
    y0 = StatsBase.normalize(hist, mode=:pdf).weights


    plot_fit!(x[keep], (y1 ./ y0)[keep])
    # plot_fit!(x, y1 ./ y0)
    # plot!(xlim=(-10, 10), ylim=(-1, 5))
end