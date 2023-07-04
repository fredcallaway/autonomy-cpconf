using LsqFit

@. exponential(x, (β, a)) = exp(β * x + a)

fit_exponential(x, y) = curve_fit(exponential, x, y, [1., 0.]).param

function plot_fit!(x, y)
    keep = isfinite.(x) .& isfinite.(y)
    param = fit_exponential(x[keep], y[keep])
    plot!(x, y, color=:lightgray)
    pred = exponential(x, param)
    plot!(x, pred, linestyle=:dot, color=:black, w=2)
    # C, x0, β = round.(res.param, digits=2)
    # expr = L"%$C + e^{%$β(x - %$x0)}"
    # annotate!(:topleft, text(expr, 14, :left))
end