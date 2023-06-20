u = -8:.01:4

x, y, z = map(-1:.1:3) do µ
    β = -.1
    d = SkewNormal(µ, 3, -20)
    # d = Normal(µ, 1.5)
    z = monte_carlo() do
        mean(rand(d, 3)) > 0
    end
    expected_value(d; β), prob_accept(3, d; β, N=100), z
end |> invert

figure() do
    plot!(x, y)
    # plot!(x, z)
    vline!([0], color=:black)
    hline!([0.5], color=:black)
end

# %% --------

figure() do
    n_sample = 2 .^ (0:1:5)
    plot(n_sample, prob_accept.(n_sample, d; β=0.))
end

# %% --------

