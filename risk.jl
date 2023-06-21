include("base.jl")

using Optim

σs = .1:.1:3
x = map(σs) do σ
    µ = optimize(-5, 1) do µ
        p_high = prob_accept(1, Normal(µ, σ); groups.high.β)
        abs(p_high - 0.8)
    end |> Optim.minimizer
    map(groups) do (;β)
        prob_accept(1, Normal(µ, σ); β)
    end
end

res = invert(x)
figure() do
    plot!(σs, res.low)
    plot!(σs, res.high)
end

# %% --------

figure() do
    x = 0:.05:3
    foreach(groups) do (;β, color)
        y = map(x) do σ
            µ = -1
            d = Normal(µ, σ + 1e-3)
            prob_accept(1, d; β)
        end
        plot!(x, y; color, xlab="σ")
    end
end
