using .PlotlyJS

export plot_llh_vs_nparticles
export plot_filter

# Color scheme: ColorSchems.tol_muted
using .PlotlyJS.PlotlyBase.Colors: RGB, N0f8
const COLORS = [
    RGB{N0f8}(0.533,0.8,0.933),
    RGB{N0f8}(0.267,0.667,0.6),
    RGB{N0f8}(0.067,0.467,0.2),
    RGB{N0f8}(0.2,0.133,0.533),
    RGB{N0f8}(0.867,0.8,0.467),
    RGB{N0f8}(0.6,0.6,0.2),
    RGB{N0f8}(0.8,0.4,0.467),
    RGB{N0f8}(0.533,0.133,0.333),
    RGB{N0f8}(0.667,0.267,0.6),
    RGB{N0f8}(0.867,0.867,0.867),
]

function plotly_rgba(c::RGB, alpha::Real)
    0 ≤ alpha ≤ 1 || throw(ArgumentError("alpha not in [0, 1]"))
    return "rgba($(round(Int, 255*c.r)), $(round(Int, 255*c.g)), $(round(Int, 255*c.b)), $alpha)"
end

function plot_llh_vs_nparticles(ssm::StateSpaceModel, parameters, obs, nparticles::AbstractVector{<:Integer}; nruns::Integer=10, kwargs...)
    x = Vector{String}(undef, length(nparticles) * nruns)
    y = Vector{Float64}(undef, length(nparticles) * nruns)
    k = 1
    for n in nparticles
        llh = LogLikelihood_NoGradient(ssm, obs; nparticles=n)
        for _ in 1:nruns
            @inbounds x[k] = string(convert(Int, n))
            @inbounds y[k] = llh(parameters)
            k += 1
        end
    end
    return plot(violin(; x, y, kwargs...))
end

function plot_filter(ssm::StateSpaceModel, parameters, components::AbstractVector{<:Integer}; steps::Integer, kwargs...)
    hidden, obs = rand(ssm, parameters, steps)
    return plot_filter(ssm, parameters, hidden, obs, components; kwargs...)
end
function plot_filter(ssm::StateSpaceModel, parameters, hidden, obs, components::AbstractVector{<:Integer}; nparticles::Integer, nsigmas::Real=2)
    length(components) > length(COLORS) && error("too many components, not enough colors")
    nsigmas > 0 || throw(ArgumentError("nsigmas must be positive"))
    # Run the particle filter, saving mean and variance of the filter distribution
    bf = BootstrapFilter(ssm, obs)
    pf = SMC(
        bf, parameters, nparticles,
        (filter=RunningSummary(MeanAndVariance(), FullHistory()), ),
    )
    offlinefilter!(pf)
    # Plot each filter distribution
    traces = GenericTrace{Dict{Symbol, Any}}[]
    for (cmp, color) in zip(components, COLORS)
        mean = map(mv -> mv.mean[cmp], pf.history_run.filter)
        sigma = map(mv -> sqrt(mv.var[cmp]), pf.history_run.filter)
        plot_bands!(traces, 1:length(hidden), mean .- nsigmas .* sigma, mean .+ nsigmas .* sigma, mean; color, midname="Comp. #$cmp filter ($(nsigmas)σ bands)", lowname=nothing, upname=nothing)
    end
    # Plot true hidden state
    for (cmp, color) in zip(components, COLORS)
        push!(traces, scatter(;
            x=1:length(hidden),
            y=getindex.(hidden, cmp),
            mode="markers",
            marker_color=color,
            name="Comp. #$cmp",
        ))
    end
    return plot(traces)
end

function plot_bands!(traces::AbstractVector{<:GenericTrace}, x, lowband, upband, midline; color, alpha=0.3, midname, lowname, upname)
    append!(traces, [
        scatter(;
            x, y=midline,
            name=midname,
            mode="lines",
            line_color=color,
        ),
        scatter(;
            x, y=upband,
            name=upname,
            mode="lines",
            # marker=attr(color="#444"),
            line_width=0,
            showlegend=false,
        ),
        scatter(;
            x, y=lowband,
            name=lowname,
            mode="lines",
            # marker=attr(color="#444"),
            line_width=0,
            fillcolor=plotly_rgba(color, alpha),
            fill="tonexty",
            showlegend=false,
        ),
    ])
    return traces
end
