using PlotlyJS

function violinplot(values::AbstractMatrix; plotstep::Integer, kwargs...)
    N, M = size(values)
    n = plotstep:plotstep:N # the first one is too concentrated
    steps = Matrix{Int}(undef, N, M)
    for i in axes(steps, 1)
        @inbounds steps[i, :] .= i
    end
    values = values[n, :]
    steps = steps[n, :]
    return violin(; x=vec(steps), y=vec(values), kwargs...)
end
function violinplot(values1::AbstractMatrix, name1::AbstractString, values2::AbstractMatrix, name2::AbstractString; plotstep::Integer, kwargs...)
    plt1 = violinplot(values1; plotstep, side="negative", legendgroup=name1, scalegroup=name1, name=name1, kwargs...)
    plt2 = violinplot(values2; plotstep, side="positive", legendgroup=name2, scalegroup=name2, name=name2, kwargs...)
    return plot([plt1, plt2])
end

function violinplot(results::Tuple{AbstractVector, AbstractString}...; title=nothing, yaxis_title=nothing, width=nothing, height=nothing,kwargs...)
    data = hcat((values for (values, name) in results)...)
    N = size(data, 1)
    names = hcat((fill(name, N) for (values, name) in results)...)
    plt = violin(; x=vec(names), y=vec(data), kwargs...)
    layout = Layout(; width, height, title, yaxis_title)
    return plot(plt, layout)
end
