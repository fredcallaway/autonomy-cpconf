
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
    (40, 0.51)
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
        plot_exponential_fit!(x, y)
    end
    plot(ps..., size=(500,300))
end

# %% --------

figure() do
    p = .6
    plot_exponential_fit!(0:20, p .^ (20:-1:0))
end