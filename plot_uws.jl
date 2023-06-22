include("base.jl")

d = 0.6 * TDist(1)
u = -8:.01:8

fig("uws") do
    plot!(u, pdf.(d, u), color=:gray)
end

fig("considered") do
    plot!(u, prob_consider.(u, d; β=0); color=:black)
end

# %% --------
d = MixtureModel([Normal(.5, .2), Normal(-3, .2)], [0.8, 0.2])
u = -6:.01:2

fig("uws_skew") do
    plot!(u, pdf.(d, u), color=:gray)
end

fig("considered_skew") do
    plot!(u, prob_consider.(u, d; β=0); color=:black)
end

# %% --------

d = 3.3 + -1 * Gamma(1.1, 3)
u = -10:.01:6
mean(d)

fig("uws_skew") do
    plot!(u, pdf.(d, u), color=:gray)
end
# %% --------

fig("considered_skew") do
    plot!(u, prob_consider.(u, d; β=0); color=:black)
end