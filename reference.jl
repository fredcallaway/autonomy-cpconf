# include("base.jl")

figure() do
    x = -4:.1:4
    µ = -2
    foreach(groups) do (;β, color)
        y = map(x) do u0
            prob_accept(2, Normal(µ, 1); β, u0)
        end
        # y .-= prob_accept(1, Normal(µ, 1); β, u0=0)
        plot!(x, y; color)
    end

    plot!(x, pdf.(Normal(µ, 1), x), color=:gray)
end
