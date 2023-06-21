
u = randn(10)
p = normalize(rand(10))
d = DiscreteNonParametric(u, p)
u .-= mean(d)

x, y, z = map(-1:.05:1) do µ
    n_sample = 1
    d = DiscreteNonParametric(u .+ µ, p)
    β = 0.
    z = monte_carlo() do
        mean(rand(d, n_sample)) > 0
    end
    expected_value(d; β), prob_accept(n_sample, d; β, N=100), z
end |> invert

figure() do
    plot!(x, y)
    plot!(x, z)
    vline!([0], color=:black)
    hline!([0.5], color=:black)
end


