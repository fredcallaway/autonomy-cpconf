include("base.jl")

# %% --------

figure() do
    n = 20; p = 0.8
    d1 = Binomial(n, p)
    d0 = Binomial(n, 0.5)
    x = support(d0)
    plot!(x, pdf.(d0, x), lab="random")
    plot!(x, pdf.(d1, x), lab="controlled")
end

# %% --------


cases = [
    (10, 0.6),
    (20, 0.6),
    (30, 0.55),
    (10, 0.8)
]


figure() do
    ps = map(cases) do (n, p)
        plot()
        d1 = Binomial(n, p)
        d0 = Binomial(n, 0.5)
        x = support(d0)
        # plot!(x, pdf.(d1, x))
        # plot!(x, pdf.(d0, x))
        y = pdf.(d1, x) ./ pdf.(d0, x)
        plot_fit!(x, y)
    end
    plot(ps..., size=(500,300))
end

# %% --------



# %% --------

figure() do
    p = .6
    plot_exponential_fit!(0:20, p .^ (20:-1:0))
end

# %% --------
figure() do
    n = 60; p1 = 0.55; p0 = 0.5
    d1 = Binomial(n, p1)
    d0 = Binomial(n, p0)
    x = support(d0)
    y = pdf.(d1, x) ./ pdf.(d0, x)
    plot!(x, y)

    r = @. exp(
        (log(p1) - log(p0)) * x +
        (log(1-p1) - log(1-p0)) * (n-x)
    )
    plot!(x, r, ls=:dot)
end

