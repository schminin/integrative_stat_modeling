# This had to be separated from summary_core.jl
# because some default methods use the History type

###########################################################################################
# Summary methods

"""
    check(s::Summary, smc::SMC)
Check whether the given `Summary` can be computed by the given particle filter `smc`
(by throwing an exception if that is not the case).
"""
check(::Summary, ::SMC) = nothing

"""
    ParticleHistoryLength(tmp::Temporaries, phl::ParticleHistoryLength)
Upgrades the given `ParticleHistoryLength` in order to be able to compute the given summary temporaries.
The default is that no upgrade is necessary.
"""
ParticleHistoryLength(::Temporaries, phl::ParticleHistoryLength) = phl

"""
    ParticleHistoryLength(s::Summary, phl::ParticleHistoryLength)
Upgrades the given `ParticleHistoryLength` in order to be able to compute the given `Summary` and its temporaries.
Defaults to `ParticleHistoryLength(Temporaries(s), phl)`.
"""
ParticleHistoryLength(s::Summary, phl::ParticleHistoryLength) = ParticleHistoryLength(Temporaries(s), phl)

"""
    make_history(tmp::Temporaries, fkm::FeynmanKacModel, nparticles::Integer)
Allocates the storage required to compute and keep track of the temporaries of a `Summary`.
Must be a `NamedTuple`, whose keys are the names of the temporaries
and whose values are `SummaryHistories`.
"""
make_history(tmp::Temporaries, fkm::FeynmanKacModel, nparticles::Integer) = NamedTuple()
make_history(tmp::Temporaries, smc::SMC) = make_history(tmp, model(smc), nparticles(smc))

"""
    make_cache(tmp::Temporaries, fkm::FeynmanKacModel, nparticles::Integer)
Allocates a cache for computing the temporaries of a `Summary`.
Defaults to `nothing`.
"""
make_cache(tmp::Temporaries, fkm::FeynmanKacModel, nparticles::Integer) = nothing
make_cache(tmp::Temporaries, smc::SMC) = make_cache(tmp, model(smc), nparticles(smc))

"""
    compute_summary(smc::SMC, s::Summary, history_tmp::SameTypeNamedTuple{SummaryHistory}, t::Integer, index_in_history::EndMinus)
Compute the summary value for the given Feynman-Kac at step `t`,
using the particle filter `smc` and the temporary histories `history_tmp`.
The `index_in_history` specifies which index in the history objects (both `history_tmp` and `smc.history_pf`) correspond to the given step `t`.
Consider using `Base.@propagate_inbounds`; the particle filter will use `@inbounds` when `index_in_history` is a valid index.
"""
compute_summary(::SMC, ::Summary, ::SameTypeNamedTuple{SummaryHistory}, ::Integer, ::EndMinus)

compute_summary(::SMC, ::AmortizedComputation, ::SameTypeNamedTuple{SummaryHistory}, ::Integer, ::EndMinus) = throw(ArgumentError("AmortizedComputations have no values, only temporaries"))

"""
    compute_summary!(out, smc::SMC, s::MutableSummary, history_tmp::SameTypeNamedTuple{SummaryHistory}, t::Integer, index_in_history::EndMinus)
Compute the summary value for the given Feynman-Kac at step `t`,
using the particle filter `smc` and the temporary histories `history_tmp`.
The summary value is stored in `out`.
The `index_in_history` specifies which index in the history objects (both `history_tmp` and `smc.history_pf`) correspond to the given step `t`.
Consider using `Base.@propagate_inbounds`; the particle filter will use `@inbounds` when `index_in_history` is a valid index.
"""
compute_summary!(out, ::SMC, ::MutableSummary, ::SameTypeNamedTuple{SummaryHistory}, ::Integer, ::EndMinus)

Base.@propagate_inbounds function compute_summary(smc::SMC, s::MutableSummary, history_tmp::SameTypeNamedTuple{SummaryHistory}, t::Integer, index_in_history::EndMinus)
    out = make_template(s, model(smc))
    compute_summary!(out, smc, s, history_tmp, t, index_in_history)
    return out
end

"""
    compute_temporaries!(smc::SMC, tmp::Temporaries, history_tmp, cache_s, t::Integer, index_in_history::EndMinus)
    compute_temporaries!(smc::SMC, tmp::Temporaries, history_tmp, cache_s, t::Integer, index_in_history::EndMinus, ::Val{:initial})
Compute the summary temporary values for the given Feynman-Kac at step `t`,
using the particle filter `smc` and the temporary histories `history_tmp`.
The `index_in_history` specifies which index in the history objects (both `history_tmp` and `smc.history_pf`) correspond to the given step `t`.
Consider using `Base.@propagate_inbounds`; the particle filter will use `@inbounds` when `index_in_history` is a valid index.
The optional argument `Val(:initial)` denotes that the step is the initial step.
The temporary values are appended to `temporaries_history` and not returned.
"""
function compute_temporaries! end

# No temporaries -> no computation
compute_temporaries!(::SMC, ::Temporaries, ::SameTypeNamedTuple{SummaryHistory, 0}, cache_s, ::Integer, ::EndMinus) = nothing

# No specialized initial step -> use general case
compute_temporaries!(smc::SMC, tmp::Temporaries, history_tmp::SameTypeNamedTuple{SummaryHistory}, cache_s, t::Integer, index_in_history::EndMinus, ::Val{:initial}) = compute_temporaries!(smc, tmp, history_tmp, cache_s, t, index_in_history)

###########################################################################################
# RunningSummary methods

check(rs::RunningSummary, smc::SMC) = check(rs.summary, smc)

"""
    ParticleHistoryLength(rs::RunningSummary, phl::ParticleHistoryLength)
Upgrades the given `ParticleHistoryLength` in order to be able to compute the given `RunningSummary` and its temporaries.
Defaults to `ParticleHistoryLength(rs.summary, phl)`.
"""
ParticleHistoryLength(rs::RunningSummary, phl::ParticleHistoryLength) = ParticleHistoryLength(rs.summary, phl)

"""
    make_history(rs::RunningSummary, fkm::FeynmanKacModel, nparticles::Integer)
Allocates the storage required to compute and keep track of a `RunningSummary`.
Must be a `SummaryHistory`.
"""
function make_history end

"""
    compute_running!(smc::SMC, rs::RunningSummary, history_tmp, history_run)
Compute the summary value at the current step of the particle filter `smc`,
using the particle filter `smc` and the temporary histories `history_tmp`.
The index in the history objects corresponding to the given step `t` is the last one.
Consider using `Base.@propagate_inbounds`; the particle filter will use `@inbounds` when it knows the histories contain enough elements.
The value is appended to `running_history` and not returned.
"""
function compute_running! end

Base.@propagate_inbounds function compute_running!(smc::SMC, rs::RunningSummary{<:ImmutableSummary}, history_tmp::SameTypeNamedTuple{SummaryHistory}, history_run::SummaryHistory)
    value = compute_summary(smc, rs.summary, history_tmp, smc.tcur, END)
    push!(history_run, value)
    return nothing
end

Base.@propagate_inbounds function compute_running!(smc::SMC, rs::RunningSummary{<:MutableSummary}, history_tmp::SameTypeNamedTuple{SummaryHistory}, history_run::SummaryHistory)
    value = next!(history_run)
    compute_summary!(value, smc, rs.summary, history_tmp, smc.tcur, END)
    return nothing
end

###########################################################################################
# OfflineSummary methods

check(os::OfflineSummary, smc::SMC) = check(os.summary, smc)

"""
    ParticleHistoryLength(os::OfflineSummary, phl::ParticleHistoryLength)
Upgrades the given `ParticleHistoryLength` in order to be able to compute the given `OfflineSummary` and its temporaries.
Defaults to `ParticleHistoryLength(os.summary, phl)`.
"""
ParticleHistoryLength(os::OfflineSummary, phl::ParticleHistoryLength) = ParticleHistoryLength(os.summary, phl)

"""
    ParticleHistoryLength(os::OfflineSummary)
Returns the minal `ParticleHistoryLength` required to compute the given `OfflineSummary` and its temporaries.
"""
ParticleHistoryLength(os::OfflineSummary; kwargs...) = ParticleHistoryLength(os, ParticleHistoryLength(; kwargs...))

"""
    offline_computation_starts_from(os::OfflineSummary, fkm::FeynmanKacModel, t::Integer)
The time step at which one should start the computation for obtaining the given `OfflineSummary` at time `t`.
"""
offline_computation_starts_from(os::OfflineSummary, fkm::FeynmanKacModel, t::Integer) = t

###########################################################################################
# AmortizedComputation methods

finalize_amortized!(::SMC, ::AmortizedComputation, history, cache) = nothing

remove_redundant_amortized(::AmortizedComputationTuple{0}) = ()
remove_redundant_amortized(amortized::AmortizedComputationTuple{1}) = amortized
function remove_redundant_amortized(amortized::AmortizedComputationTuple{2})
    amortized_1, amortized_2 = amortized
    if amortized_1 == amortized_2
        return (amortized_1, )
    elseif isa(amortized_1, OfflineAC) && amortized_1.amortized == amortized_2
        return (amortized_1, )
    elseif isa(amortized_2, OfflineAC) && amortized_2.amortized == amortized_1
        return (amortized_2, )
    else
        return (amortized_1, amortized_2)
    end
end
function remove_redundant_amortized(amortized::AmortizedComputationTuple{N}) where {N}
    amortized_1, amortized_rest... = amortized
    amortized_rest_filtered = remove_redundant_amortized(amortized_rest)
    if length(amortized_rest_filtered) == N - 1
        amortized_2, amortized_rest²... = amortized_rest_filtered
        amortized_12 = remove_redundant_amortized((amortized_1, amortized_2))
        if length(amortized_12) == 2
            amortized_1rest² = remove_redundant_amortized((amortized_1, amortized_rest²...))
            if length(amortized_1rest²) == N - 1
                return amortized
            else
                return (amortized_2, amortized_1rest²...)
            end
        else
            return (amortized_12..., amortized_rest²...)
        end
    else
        return remove_redundant_amortized((amortized_1, amortized_rest_filtered...))
    end
end

@generated function required_amortized_computations(summaries::SummaryTuple{N, NAMES}) where {N, NAMES}
    lines = [
        :($(Symbol(:amortized_, name)) = required_amortized_computations(getproperty(summaries, $(QuoteNode(name)))))
        for name in NAMES
    ]
    return_value = Expr(:tuple, (:($(Symbol(:amortized_, name))...) for name in NAMES)...)
    return Expr(:block, lines..., :(return $return_value))
end

function get_required_amortized_history(smc::SMC, summary::Union{AmortizedComputation, Temporaries, RunningSummary, OfflineSummary})
    return get_amortized_history(smc, required_amortized_computations(summary))
end

function get_amortized_history(smc::SMC, amortized::AmortizedComputationTuple{N}) where {N}
    return ntuple(
        i -> _get_amortized_history(smc.amortized, smc.history_amortized, amortized[i]), Val(N)
    )
end

# This function should reduce to trivial code in most cases 
@generated function _get_amortized_history(all_amortized::AmortizedComputationTuple{N}, history::NTuple{N, SameTypeNamedTuple{SummaryHistory}}, amortized::AmortizedComputation) where {N}
    # NB we ensured in the constructor for SMC that elements of all_amortized are all unique
    generate = (n, N) -> begin
        head = isone(n) ? :if : :elseif
        condition = quote
            let amortized_n = all_amortized[$n]
                amortized_n == amortized || (isa(amortized_n, OfflineAC) && amortized_n.amortized == amortized)
            end
        end
        then_block = :(return history[$n])
        else_block = if n == N
            :(error("AmortizedComputation $(amortized) has not been computed"))
        else
            generate(n+1, N)
        end
        return Expr(head, condition, then_block, else_block)
    end
    return generate(1, N)
end
