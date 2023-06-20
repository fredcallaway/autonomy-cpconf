g = grid(
    µ=-2:.05:2,
    σ=0:.05:4
)

low = map(g) do (;µ, σ)
    d = Normal(µ, σ + 1e-3)
    prob_accept(1, d; groups.low.β)
end

high = map(g) do (;µ, σ)
    d = Normal(µ, σ + 1e-3)
    prob_accept(1, d; groups.high.β)
end

# %% --------

function heatfig(X, name="tmp")
    figure(name; pdf=true) do
        heatmap(collect(X),
            bottom_margin=-4mm,
            left_margin=-4mm,
            size=(200,200),
            # framestyle=:none,
            xaxis=false, yaxis=false,
            cbar=false, clim=(0,1),
            # xlab=L"\sigma", ylab=L"\mu"
        )
    end
end

heatfig(low, "low")
heatfig(high, "high")
heatfig(high .- low, "diff")
# %% --------


figure() do
    heatmap(high, size=(300,200), xlab=L"\sigma", ylab=L"\mu")
end

figure() do
    heatmap(high .- low, size=(300,200), xlab=L"\sigma", ylab=L"\mu")
end
