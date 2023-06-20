include("base.jl")

testfig("considered") do (;β, color)
    plot!(u, prob_consider.(u, d; β, u0=1); color)
end

# %% --------

d = Normal(-1, 1)
u = -4:.01:4

fig("possible") do
    plot!(u, pdf.(d, u), color=:gray)
end

fig("bear") do
    plot!(u, exp.(u), color=:black)
end

fig("lieder") do
    plot!(u, abs.(u), color=:black)
end

groupfig("considered") do (;β, color)
    plot!(u, prob_consider.(u, d; β); color)
end

groupfig("choice", ylim=(0, 1), framestyle=:axes, yticks=false) do (;β, color)
    p = prob_accept(1, d; β)
    bar!([1-p, p]; color, w=1)
end

groupfig("n_sample", ylim=(0, 1), framestyle=:axes, yticks=false) do (;β, color)
    n_sample = 2 .^ (0:1:5)
    plot!(n_sample, prob_accept.(n_sample, d; β); color)
end
