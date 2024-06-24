struct UniqueAncestors <: ImmutableSummary end

# TODO Make Enoch indices an AmortizedComputation (they are also used by AdaSmooth)
# TODO Allow RunningSummaries to map directly to a temporary/amortized computation

function offline_computation_starts_from(::OfflineSummary{UniqueAncestors}, fkm::FeynmanKacModel, t::Integer)
    return initial_time(fkm)
end

function make_history(::Temporaries{UniqueAncestors}, fkm::FeynmanKacModel, nparticles::Integer)
    L = StaticFiniteHistory{2}()
    enoch = make_history(L, Int, nparticles)
    return (; enoch=SummaryHistory(enoch))
end

function make_history(rs::RunningSummary{UniqueAncestors}, fkm::FeynmanKacModel{T}, nparticles::Integer) where {T}
    storage = make_history(rs.Lhistory, Int)
    return SummaryHistory(storage; computable_from=1)
end

Base.@propagate_inbounds function compute_temporaries!(smc::SMC, ::Temporaries{UniqueAncestors}, history_tmp::SameTypeNamedTuple{SummaryHistory, 1, (:enoch, )}, cache::Nothing, t::Integer, index_in_history::EndMinus, ::Val{:initial})
    enoch = next!(history_tmp.enoch; return_current=false)
    nparticles = length(enoch)
    @turbo enoch .= Base.OneTo(nparticles)
    return nothing
end
Base.@propagate_inbounds function compute_temporaries!(smc::SMC, ::Temporaries{UniqueAncestors}, history_tmp::SameTypeNamedTuple{SummaryHistory, 1, (:enoch, )}, cache::Nothing, t::Integer, index_in_history::EndMinus)
    enoch, enoch_prev = next!(history_tmp.enoch; return_current=true)
    ancestors = smc.history_pf.ancestors[index_in_history]
    @turbo for i in eachindex(enoch)
        @inbounds enoch[i] = enoch_prev[ancestors[i]]
    end
    @assertx issorted(enoch)
    return nothing
end

Base.@propagate_inbounds function compute_summary(smc::SMC, ::UniqueAncestors, history_tmp::SameTypeNamedTuple{SummaryHistory, 1, (:enoch, )}, t::Integer, index_in_history::EndMinus)
    enoch = history_tmp.enoch[index_in_history]
    # NB at least one particle is always present
    # NB enoch is sorted by construction, so counting unique elements is easy
    n_unique = 1
    @inbounds for k in (firstindex(enoch)+1):lastindex(enoch)
        enoch[k-1] == enoch[k] || (n_unique += 1)
    end
    return n_unique
end
