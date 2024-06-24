struct MeanAndVariance <: ImmutableSummary end

function make_history(rs::RunningSummary{MeanAndVariance}, fkm::FeynmanKacModel{T}, nparticles::Integer) where {T}
    S = typeof((zero(T) + zero(T)) / 1.0)
    storage = make_history(rs.Lhistory, NamedTuple{(:mean, :var), Tuple{S, S}})
    return SummaryHistory(storage; computable_from=1)
end

Base.@propagate_inbounds function compute_summary(smc::SMC, s::MeanAndVariance, ::SameTypeNamedTuple{SummaryHistory, 0}, ::Integer, index_in_history::EndMinus)
    particles = smc.history_pf.particles[index_in_history]
    weights = smc.history_pf.weights[index_in_history]
    mean, var = weighted_mean_and_var(particles, weights)
    return (; mean, var)
end
