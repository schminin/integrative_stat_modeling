const SummaryTuple{N, NAMES} = SameTypeNamedTuple{Union{Summary, RunningSummary}, N, NAMES}
const AmortizedComputationTuple{N} = NTuple{N, AmortizedComputation}

mutable struct SMC{T, FKM <: FeynmanKacModel{T}, PARAMS, SCHEME <: ResamplingScheme, SUM <: SummaryTuple, AMO <: AmortizedComputationTuple, H <: ParticleHistory, HTMP <: SameTypeNamedTuple{SameTypeNamedTuple{SummaryHistory}}, HRUN <: SameTypeNamedTuple{SummaryHistory}, HAMO}
    # Feynman-Kac model
    const fkm::FKM
    const θ::PARAMS
    # Algorithm settings
    const resampling_scheme::SCHEME
    const summaries::SUM
    const amortized::AMO
    # Current state
    tcur::Int # the current particles approximate the filter distribution at this time;
              # if tcur == initial_time(fkm) - 1 the filter has not been started yet
    const history_pf::H # history for particle filter
    const history_tmp::HTMP # history for summary temporaries
    const history_run::HRUN # history for running summaries
    const history_amortized::HAMO # history for amortized computations

    function SMC(
        fkm::FeynmanKacModel{T}, θ,
        nparticles::Integer,
        Lhistory::ParticleHistoryLength=ParticleHistoryLength(),
        summaries::SummaryTuple{S_N, S_NAMES}=NamedTuple(),
        amortized::AmortizedComputationTuple=required_amortized_computations(summaries),
        resampling_scheme::ResamplingScheme=AdaptiveResampling(SystematicResampling(), 0.5),
    ) where {T, S_N, S_NAMES}
        # Check arguments
        nparticles > 0 || throw(ArgumentError("at least one particle must be used"))
        isparameter(fkm, θ) || throw(ArgumentError("θ::$(typeof(θ)) is not a valid parameter object for the given FeynmanKacModel"))
        
        tcur = initial_time(fkm) - 1
        amortized_reduced = remove_redundant_amortized(amortized)
        
        # Upgrade ParticleHistoryLength according to Summaries and AmortizedComputations
        Lhistory_up = static_foldr(ParticleHistoryLength, summaries, Lhistory)
        Lhistory_upup = static_foldr(ParticleHistoryLength, amortized_reduced, Lhistory_up)
        history_pf = ParticleHistory(T, Lhistory_upup, nparticles)

        history_tmp = map(
            s -> make_history(Temporaries(s), fkm, nparticles),
            summaries
        )::SameTypeNamedTuple{SameTypeNamedTuple{SummaryHistory}, S_N, S_NAMES}
        history_run = map(
            s -> make_history(s, fkm, nparticles),
            filter_NamedTuple_by_type(RunningSummary, summaries)
        )::SameTypeNamedTuple{SummaryHistory}
        history_amortized = map(
            s -> make_history(Temporaries(s), fkm, nparticles),
            amortized_reduced
        )::NTuple{length(amortized_reduced), SameTypeNamedTuple{SummaryHistory}}

        smc = new{
            T,
            typeof(fkm),
            typeof(θ),
            typeof(resampling_scheme),
            typeof(summaries),
            typeof(amortized_reduced),
            typeof(history_pf),
            typeof(history_tmp),
            typeof(history_run),
            typeof(history_amortized),
        }(
            fkm,
            θ,
            resampling_scheme,
            summaries,
            amortized_reduced,
            tcur,
            history_pf,
            history_tmp,
            history_run,
            history_amortized,
        )

        # Check whether summaries can actually be computed
        foreach(s -> check(s, smc), smc.summaries)
        foreach(s -> check(s, smc), smc.amortized)
        
        return smc
    end
    function SMC(smc::SMC{T, FKM, PARAMS, SCHEME, SUM, AMO, H, HTMP, HRUN, HAMO}, fkm::FKM, params; recycle::Bool=false) where {T, FKM, PARAMS, SCHEME, SUM, AMO, H, HTMP, HRUN, HAMO}
        if recycle
            reset!(smc) # empty histories
            return new{T, FKM, PARAMS, SCHEME, SUM, AMO, H, HTMP, HRUN, HAMO}(
                fkm,
                params,
                smc.resampling_scheme,
                smc.summaries,
                smc.amortized,
                initial_time(fkm) - 1,
                smc.history_pf,
                smc.history_tmp,
                smc.history_run,
                smc.history_amortized,
            )
        else
            return SMC(
                fkm,
                params,
                nparticles(smc),
                ParticleHistoryLength(smc.history_pf),
                smc.summaries,
                smc.amortized,
                smc.resampling_scheme,
            )::SMC{T, FKM, PARAMS, SCHEME, SUM, AMO, H, HTMP, HRUN, HAMO}
        end
    end
end

SMC(smc::SMC{T, FKM}, params; recycle::Bool=false) where {T, FKM} = SMC(smc, smc.fkm, params; recycle)
function SMC(smc::SMC{T, FKM}, fkm::FKM; recycle::Bool=false) where {T, FKM}
    params = recycle ? smc.θ : deepcopy(smc.θ)
    params::typeof(smc.θ)
    return SMC(smc, fkm, params; recycle)
end

# Coversion from HistoryLength to ParticleHistoryLength
function SMC(
    fkm::FeynmanKacModel, θ,
    nparticles::Integer,
    Lhistory::HistoryLength,
    summaries::SummaryTuple=NamedTuple(),
    amortized::AmortizedComputationTuple=required_amortized_computations(summaries),
    resampling_scheme::ResamplingScheme=AdaptiveResampling(SystematicResampling(), 0.5),
)
    return SMC(fkm, θ, nparticles, ParticleHistoryLength(Lhistory), summaries, amortized, resampling_scheme)
end

# One argument missing
function SMC(
    fkm::FeynmanKacModel, θ,
    nparticles::Integer,
    summaries::SummaryTuple,
    amortized::AmortizedComputationTuple=required_amortized_computations(summaries),
    resampling_scheme::ResamplingScheme=AdaptiveResampling(SystematicResampling(), 0.5),
)
    return SMC(fkm, θ, nparticles, ParticleHistoryLength(), summaries, amortized, resampling_scheme)
end
function SMC(
    fkm::FeynmanKacModel, θ,
    nparticles::Integer,
    Lhistory::Union{ParticleHistoryLength, HistoryLength},
    summaries::SummaryTuple,
    resampling_scheme::ResamplingScheme=AdaptiveResampling(SystematicResampling(), 0.5),
)
    return SMC(fkm, θ, nparticles, Lhistory, summaries, required_amortized_computations(summaries), resampling_scheme)
end
# NB even if summaries are not presente, AmortizedComputations may be needed for OfflineSummaries
function SMC(
    fkm::FeynmanKacModel, θ,
    nparticles::Integer,
    Lhistory::Union{ParticleHistoryLength, HistoryLength},
    amortized::AmortizedComputationTuple=(),
    resampling_scheme::ResamplingScheme=AdaptiveResampling(SystematicResampling(), 0.5),
)
    return SMC(fkm, θ, nparticles, Lhistory, NamedTuple(), amortized, resampling_scheme)
end

# Two arguments missing
function SMC(
    fkm::FeynmanKacModel, θ,
    nparticles::Integer,
    summaries::SummaryTuple,
    resampling_scheme::ResamplingScheme,
)
    return SMC(fkm, θ, nparticles, ParticleHistoryLength(), summaries, required_amortized_computations(summaries), resampling_scheme)
end
function SMC(
    fkm::FeynmanKacModel, θ,
    nparticles::Integer,
    Lhistory::Union{ParticleHistoryLength, HistoryLength},
    resampling_scheme::ResamplingScheme,
)
    return SMC(fkm, θ, nparticles, Lhistory, NamedTuple(), (), resampling_scheme)
end
# NB even if summaries are not presente, AmortizedComputations may be needed for OfflineSummaries
function SMC(
    fkm::FeynmanKacModel, θ,
    nparticles::Integer,
    amortized::AmortizedComputationTuple,
    resampling_scheme::ResamplingScheme=AdaptiveResampling(SystematicResampling(), 0.5),
)
    return SMC(fkm, θ, nparticles, ParticleHistoryLength(), NamedTuple(), amortized, resampling_scheme)
end

# Three-arguments missing
function SMC(
    fkm::FeynmanKacModel, θ,
    nparticles::Integer,
    resampling_scheme::ResamplingScheme,
)
    return SMC(fkm, θ, nparticles, ParticleHistoryLength(), NamedTuple(), (), resampling_scheme)
end

##########################################################################################################
# Properties

"""
    StateSpaceSMC{T_X, T_Y}
`SMC` algorithm object associated to a `StateSpaceFeynmanKacModel`.
"""
const StateSpaceSMC{T_X, T_Y} = SMC{T_X, <:StateSpaceFeynmanKacModel{T_X, T_Y}}

"""
    model(smc::SMC)
Return the Feynman-Kac model associated to this SMC object.
"""
model(smc::SMC) = smc.fkm

"""
    statespacemodel(smc::StateSpaceSMC)
Return the state-space model associated to this SMC object.
"""
statespacemodel(smc::StateSpaceSMC) = statespacemodel(model(smc))

"""
    nparticles(smc::SMC)
Return the number of particles used in this SMC object.
"""
nparticles(smc::SMC) = nparticles(smc.history_pf)

"""
    started(smc::SMC)
Check whether the filter has started (i.e., if the starting time has been processed).
"""
started(smc::SMC) = smc.tcur ≥ initial_time(smc)

initial_time(smc::SMC) = initial_time(smc.fkm)
isready(smc::SMC) = isready(smc.fkm, smc.tcur + 1)
readyupto(smc::SMC) = readyupto(smc.fkm)

index2time(smc::SMC, i::EndMinus) = smc.tcur - i.k

##########################################################################################################

function reset!(smc::SMC, parameters)
    # ismutable(smc.θ) || ArrayInterfaceCore.ismutable(smc.θ) || error("reset!(smc, parameter) cannot be used for immutable parameters")
    copy!(smc.θ, parameters)
    return reset!(smc)
end
reset!(smc::SMC{T, FKM, Nothing}, ::Nothing) where {T, FKM <: FeynmanKacModel{T}} = reset!(smc)
function reset!(smc::SMC)
    smc.tcur = initial_time(smc.fkm) - 1
    empty!(smc.history_pf)
    foreach(smc.history_tmp) do h
        foreach(empty!, h)
    end
    foreach(empty!, smc.history_run)
    foreach(smc.history_amortized) do h
        foreach(empty!, h)
    end
    return smc
end

# Pre-allocate history according to the number of available steps
function preallocate!(smc::SMC, n::Integer)
    preallocate!(smc.history_pf, n)
    preallocate!(smc.history_tmp, n)
    preallocate!(smc.history_run, n)
    preallocate!(smc.history_amortized, n)
    return smc
end
function preallocate!(smc::SMC)
    n = readyupto(smc) - initial_time(smc) + 1
    return preallocate!(smc, n)
end

struct SMCCache{T_SMC <: SMC, RC, SC, AC}
    cache_resampling::RC
    cache_summaries::SC
    cache_amortized::AC
    function SMCCache(smc::SMC)
        cache_resampling = make_cache(smc.resampling_scheme, nparticles(smc))
        cache_summaries = map(s -> make_cache(Temporaries(s), model(smc), nparticles(smc)), smc.summaries)
        cache_amortized = map(s -> make_cache(Temporaries(s), model(smc), nparticles(smc)), smc.amortized)
        return new{typeof(smc), typeof(cache_resampling), typeof(cache_summaries), typeof(cache_amortized)}(cache_resampling, cache_summaries, cache_amortized)
    end
end

##########################################################################################################
# Filter algorithm

function stepforward!(smc::T_SMC, cache::SMCCache{T_SMC}=SMCCache(smc)) where {T_SMC <: SMC}
    isready(smc) || error("Feynman-Kac model not ready for time step $(smc.tcur + 1) (e.g., data for the associated state-space model is not yet available)")
    preallocate!(smc)
    if smc.tcur == initial_time(smc) - 1
        _stepforward!(smc, cache, Val(:initial))
    else
        _stepforward!(smc, cache)
    end
    return smc
end

function onlinefilter!(smc::T_SMC, cache::SMCCache{T_SMC}=SMCCache(smc)) where {T_SMC <: SMC}
    # NB In case we are doing online filtering and the data gathering is done in another thread,
    #    the number of available data points can increase with time.
    #    We cannot thus preallocate only once before the loop, but do it instead each time in _stepforward!
    if isready(smc)
        # Do first step (may be the initial one)
        preallocate!(smc)
        if smc.tcur == initial_time(smc) - 1
            _stepforward!(smc, cache, Val(:initial))
        else
            _stepforward!(smc, cache)
        end
        # Do other steps
        while isready(smc)
            preallocate!(smc)
            _stepforward!(smc, cache)
        end
    end
    return smc
end

function offlinefilter!(smc::T_SMC, cache::SMCCache{T_SMC}=SMCCache(smc)) where {T_SMC <: SMC}
    tend = readyupto(smc)
    if smc.tcur < tend
        # Allocate beforehand
        n = tend - initial_time(smc) + 1
        preallocate!(smc, n)
        # Do first step (may be the initial one)
        @assertx isready(smc)
        if smc.tcur == initial_time(smc) - 1
            _stepforward!(smc, cache, Val(:initial))
        else
            _stepforward!(smc, cache)
        end
        # Do other steps
        for _ in smc.tcur+1:tend
            @assertx isready(smc)
            _stepforward!(smc, cache)
        end
    end
    # HACK
    # Update amortized computations that cannot be computed efficiently step by step
    for (ac, history_ac, cache_ac) in zip(smc.amortized, smc.history_amortized, cache.cache_amortized)
        finalize_amortized!(smc, ac, history_ac, cache_ac)
    end
    # END HACK
    return smc
end

function _stepforward!(smc::SMC, cache::SMCCache, ::Val{:initial})
    smc.tcur += 1

    # Initialize particles/weights according to Feynman-Kac model's initial distribution
    # NB we can assume isready(smc) is true here, hence the @inbounds for the fkm_* functions
    firststep = next!(smc.history_pf, Val(:initial))
    M0 = @inbounds fkm_M0(smc.fkm, smc.θ)
    @threadsx pf for i in eachindex(firststep.particles)
        rng = TaskLocalRNG()
        x0 = rand(rng, M0)
        logG0 = @inbounds fkm_logG0(smc.fkm, smc.θ, x0)
        @inbounds firststep.particles[i] = x0
        @inbounds firststep.logweights[i] = logG0
    end

    # Linearize weights and compute normalizing constant
    _, lwmax, wsum = linweights!(firststep.weights, firststep.logweights)
    if maxlength(smc.history_pf.logCnorm) > 0
        lmw = lwmax + log(wsum / length(firststep.weights)) # log mean weight
        push!(smc.history_pf.logmeanw, lmw)
        push!(smc.history_pf.logCnorm, lmw)
    end

    collect_summaries!(smc, cache, Val(:initial))

    return nothing
end

function _stepforward!(smc::SMC, cache::SMCCache)
    smc.tcur += 1

    # Get storage space for this step and the values at the previous step
    # NB @inbounds because we are sure that a previous step exists
    # NB newstep.(log)weights and prevstep.(log)weights may alias,
    #    but it should not be a problem
    newstep, prevstep = @inbounds next!(smc.history_pf)

    # Resample particles writing the results in the ancestors vector.
    # In the case of adaptive resampling, the return value indicates whether resampling actually occurred
    didresample = resample!(TaskLocalRNG(), newstep.ancestors, smc.resampling_scheme, prevstep.weights, cache.cache_resampling)
    push!(smc.history_pf.didresample, didresample)

    # Evolve sampled particles according to the transition kernel and update their weight
    # NB we can assume isready(smc) is true here, hence the @inbounds for the fkm_* functions
    @threadsx pf for i in eachindex(newstep.particles)
        iancestor = @inbounds newstep.ancestors[i]
        xp = @inbounds prevstep.particles[iancestor]
        Mt = @inbounds fkm_Mt(smc.fkm, smc.θ, smc.tcur, xp)
        rng = TaskLocalRNG()
        x = rand(rng, Mt)
        @inbounds newstep.particles[i] = x
        logGt = @inbounds fkm_logGt(smc.fkm, smc.θ, smc.tcur, xp, x)
        if isa(smc.resampling_scheme, AdaptiveResampling) && !didresample
            @inbounds newstep.logweights[i] = prevstep.logweights[i] + logGt
        else
            @inbounds newstep.logweights[i] = logGt
        end
    end

    # Linearize weights and compute normalizing constant
    _, lwmax, wsum = linweights!(newstep.weights, newstep.logweights)
    if maxlength(smc.history_pf.logCnorm) > 0
        @assertx length(smc.history_pf.logmeanw) == 1
        lmw = lwmax + log(wsum / length(newstep.weights)) # log mean weight
        ΔlogCnorm = didresample ? lmw : lmw - @inbounds smc.history_pf.logmeanw[1]
        push!(smc.history_pf.logmeanw, lmw)
        push!(smc.history_pf.logCnorm, smc.history_pf.logCnorm[end] + ΔlogCnorm)
    end

    collect_summaries!(smc, cache)

    return nothing
end

##########################################################################################################
# Summary computation

function collect_summaries!(smc::T_SMC, name::Union{Symbol, Integer}, cache::SMCCache{T_SMC}, ::Val{STEP}=Val(:later)) where {T_SMC <: SMC, STEP}
    # Unpack differently depending on whether the summary is AmortizedComputation or not
    summary, history_tmp, cache_s = if isa(name, Symbol)
        getproperty(smc.summaries, name), getproperty(smc.history_tmp, name), getproperty(cache.cache_summaries, name)
    else
        smc.amortized[name], smc.history_amortized[name], cache.cache_amortized[name]
    end
    # Compute temporaries
    if STEP === :initial
        @inbounds compute_temporaries!(smc, Temporaries(summary), history_tmp, cache_s, smc.tcur, END, Val(:initial))
    else
        @inbounds compute_temporaries!(smc, Temporaries(summary), history_tmp, cache_s, smc.tcur, END)
    end
    # Compute running summary
    if isa(summary, RunningSummary)
        history_run = getproperty(smc.history_run, name)
        @inbounds compute_running!(smc, summary, history_tmp, history_run)
    end
    return nothing
end

Base.@propagate_inbounds function compute_summary(smc::SMC, name::Symbol)
    s = getproperty(smc.summaries, name)
    isa(s, RunningSummary) && error("values of RunningSummaries can be accessed through smc.history_run")
    history_tmp = getproperty(smc.history_tmp, name)
    return compute_summary(smc, s, history_tmp, smc.tcur, END)
end

Base.@propagate_inbounds function compute_summary!(out, smc::SMC, name::Symbol)
    s = getproperty(smc.summaries, name)
    isa(s, RunningSummary) && error("values of RunningSummaries can be accessed through smc.history_run")
    isa(s, ImmutableSummary) && error("values of ImmutableSummaries must be computed with compute_summary(smc::SMC, name::Symbol)")
    history_tmp = getproperty(smc.history_tmp, name)
    return compute_summary!(out, smc, s, history_tmp, smc.tcur, END)
end

function compute_summary(smc::SMC, os::OfflineSummary, step::EndMinus=END)
    t, history_tmp = _compute_summary(smc, os, step)
    return compute_summary(smc, os.summary, history_tmp, t, step)
end
function compute_summary!(out, smc::SMC, os::OfflineSummary{<:MutableSummary}, step::EndMinus=END)
    t, history_tmp = _compute_summary(smc, os, step)
    return compute_summary!(out, smc, os.summary, history_tmp, t, step)
end
function _compute_summary(smc::SMC, os::OfflineSummary, step::EndMinus=END)
    t = index2time(smc, step)
    # Check whether enough history was saved to compute the summary offline
    let phl = ParticleHistoryLength(smc.history_pf)
        ParticleHistoryLength(os, phl) == phl || error("ParticleHistoryLength of current ParticleHistory is not sufficient to compute the given OfflineSummary")
    end
    # Check whether required AmortizedComputations have been computed
    # NB The check occurring inside compute_temporaries! forgets about the offline qualities of this summary,
    #    and so it is insufficient
    get_required_amortized_history(smc, os)
    # Check other requirements (will throw if not satisfied)
    check(os, smc)
    # Create storage for temporaries
    history_tmp = make_history(Temporaries(os), smc)
    cache_s = make_cache(Temporaries(os), smc)
    # From which time should we start?
    t_initial = initial_time(smc)
    tcur = offline_computation_starts_from(os, model(smc), t)
    t_initial ≤ tcur ≤ t || error("check definition of offline_computation_starts_from for summary type $(typeof(os.summary))")
    stepcur = EndMinus(step.k + (t - tcur); check=false)
    # Are we starting at the beginning?
    if tcur == t_initial
        @inbounds compute_temporaries!(smc, Temporaries(os), history_tmp, cache_s, tcur, stepcur, Val(:initial))
        tcur == t && return t, history_tmp
        tcur += 1
        stepcur = EndMinus(stepcur.k - 1; check=false)
    end
    # Loop
    while true
        @inbounds compute_temporaries!(smc, Temporaries(os), history_tmp, cache_s, tcur, stepcur)
        tcur == t && break
        tcur += 1
        stepcur = EndMinus(stepcur.k - 1; check=false)
    end
    return t, history_tmp
end

@inline function collect_summaries!(smc::T_SMC, cache::SMCCache{T_SMC}, ::Val{STEP}=Val(:later)) where {STEP, T_SMC <: SMC}
    _collect_summaries!(smc, Val(keys(smc.summaries)), smc.amortized, cache, Val(STEP))
end
@generated function _collect_summaries!(smc::SMC, ::Val{NAMES}, amortized::AmortizedComputationTuple{N}, cache::SMCCache, ::Val{STEP}) where {NAMES, N, STEP}
    if length(NAMES) == 1 && N == 0
        return quote
            collect_summaries!(smc, $(QuoteNode(NAMES[1])), cache, Val(STEP))
            return nothing
        end
    elseif length(NAMES) == 0 && N == 1
        return quote
            collect_summaries!(smc, 1, cache, Val(STEP))
            return nothing
        end
    elseif SUMMARY_OUTER_THREADING
        # First compute AmortizedComputations, then Summary temporaries.
        # This is because the latter may depend on the results of the former.
        # If that is not the case, then we are losing an opportunity to parallelize. (TODO?)
        lines = []
        for idx in 1:N
            push!(lines, :(
                $(Symbol(:task_amortized_, idx)) = spawn_function(CollectSummariesClosure{$idx, STEP}(smc, cache))
            ))
        end
        for idx in 1:N
            push!(lines, :(
                wait($(Symbol(:task_amortized_, idx)))
            ))
        end
        for name in NAMES
            push!(lines, :(
                $(Symbol(:task_, name)) = spawn_function(CollectSummariesClosure{$(QuoteNode(name)), STEP}(smc, cache))
            ))
        end
        for name in NAMES
            push!(lines, :(
                wait($(Symbol(:task_, name)))
            ))
        end
        return Expr(:block, lines..., :(return nothing))
    elseif length(NAMES) == 1 && N == 1
        return quote
            collect_summaries!(smc, 1, cache, Val(STEP))
            collect_summaries!(smc, $(QuoteNode(NAMES[1])), cache, Val(STEP))
            return nothing
        end
    else
        lines = []
        for idx in 1:N
            push!(lines, :(
                collect_summaries!(smc, $idx, cache, Val(STEP))
            ))
        end
        for name in NAMES
            push!(lines, :(
                collect_summaries!(smc, $(QuoteNode(name)), cache, Val(STEP))
            ))
        end
        return Expr(:block, lines..., :(return nothing))
    end
end

if SUMMARY_OUTER_THREADING
    struct CollectSummariesClosure{NAME, STEP, T <: SMC, C <: SMCCache{T}}
        smc::T
        cache::C
        CollectSummariesClosure{NAME, STEP}(smc::T_SMC, cache::SMCCache{T_SMC}) where {NAME, STEP, T_SMC <: SMC} = new{NAME, STEP, typeof(smc), typeof(cache)}(smc, cache)
    end
    (cl::CollectSummariesClosure{NAME, STEP})() where {NAME, STEP} = collect_summaries!(cl.smc, NAME, cl.cache, Val(STEP))
end
