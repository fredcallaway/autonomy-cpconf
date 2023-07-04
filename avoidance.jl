include("base.jl")

d = SkewNormal(0.5, 1, -5)
u = -3:.01:3

fig("possible") do
    plot!(u, pdf.(d, u), color=CTRUE)
end
# %% --------

groups = (
    high = (
        color=CHIGH,
        β = 3,
        i = 2
    ),
    low = (
        color=CLOW,
        β = 0.2,
        i = 1
    )
)


fig("full_bias") do
    foreach(groups) do (;β, color)
        # β = min(.7, β)
        p = @. exp(β * u) * abs(u)
        plot!(u, p; color, ylim=(0, 10))
    end
    plot!(size=(180,150))
end

fig("considered_control") do
    foreach(groups) do (;β, color)
        plot!(u, prob_consider.(u, d; β); color)
    end
    plot!(size=(180,150))

end

fig("choice_control") do
    foreach(groups) do (;β, color, i)
        p = 1 - prob_accept(1, d; β)
        bar!([i], [p]; color, w=1)
    end
    plot!(xlim=(0.5, 2.5), ylim=(0, 1), framestyle=:axes)
    plot!(size=(150,150))
end

# %% --------

groupfig("considered") do (;β, color)
    plot!(u, prob_consider.(u, d; β); color)
end

groupfig("choice", ylim=(0, 1), framestyle=:axes, yticks=false) do (;β, color)
    p = prob_accept(1, d; β)
    bar!([1-p, p]; color, w=1, )
end

groupfig("n_sample", ylim=(0, 1), framestyle=:axes, yticks=false) do (;β, color)
    n_sample = 2 .^ (0:1:5)
    plot!(n_sample, prob_accept.(n_sample, d; β); color)
end

