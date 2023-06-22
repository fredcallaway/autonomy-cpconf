using LsqFit

@. exponential(x, (C, x0, β)) = C + exp(β * (x - x0))

fit_exponential(x, y) = curve_fit(exponential, x, y, [1., 0., 1.]).param

function plot_fit!(x, y)
    param = fit_exponential(x, y)
    plot!(x, y, color=:lightgray)
    pred = exponential(x, param)
    plot!(x, pred, linestyle=:dot, color=:black)
    # C, x0, β = round.(res.param, digits=2)
    # expr = L"%$C + e^{%$β(x - %$x0)}"
    # annotate!(:topleft, text(expr, 14, :left))
end