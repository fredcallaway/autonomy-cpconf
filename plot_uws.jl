include("base.jl")

d = TDist(1)
u = -7:.01:7

fig("uws") do
    plot!(u, pdf.(d, u), color=CTRUE)
end

fig("lieder") do
    plot!(u, abs.(u), color=CBIAS)
end

fig("considered") do
    plot!(u, prob_consider.(u, d; β=0); color=CSAMP)
end

# %% --------
d = MixtureModel([Normal(.5, .2), Normal(-3, .2)], [0.8, 0.2])
u = -4:.01:4

fig("uws_skew") do
    plot!(u, pdf.(d, u), color=CTRUE)
end

fig("considered_skew") do
    plot!(u, prob_consider.(u, d; β=0); color=CSAMP)
end

# %% --------

fig("bear") do
    plot!(u, exp.(u), color=CBIAS, size=1.2 .* (200, 150))
end

fig("bear_small") do
    plot!(u, exp.(u), color=CBIAS)
end

fig("norm01") do
    u = -4:.01:4
    plot!(u, pdf.(Normal(0,1), u), color=CTRUE)
    plot!(size=(100,100))
end
