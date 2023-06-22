include("base.jl")
include("curve_fitting.jl")

function sample_outcomes(control; n=100000, k=2, N=30, rdist=Normal(0,1))
    repeatedly(n) do
        R = rand(rdist, (k, N))
        sum(1:N) do i
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
    x = sample_outcomes(1; N=100)
    histogram!(x, w=0, normalize=:pdf)
    plot!(fit(Normal, x))
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