struct AdaSmoothAC{USE_REJECTION} <: AmortizedComputation
    beta::Float64
    function AdaSmoothAC(beta::Real=0.5; use_rejection::Bool=true)
        0 < beta < 1 || throw(ArgumentError("AdaSmooth beta parameter must be in (0, 1)"))
        return new{use_rejection}(beta)
    end
end

function ParticleHistoryLength(::Temporaries{<:MaybeOfflineAC{AdaSmoothAC{true}}}, phl::ParticleHistoryLength)
    # rejection
    return ParticleHistoryLength(
        phl;
        particles=max(phl.particles, StaticFiniteHistory{2}()),
        didresample=max(phl.didresample, StaticFiniteHistory{1}()),
    )
end
function ParticleHistoryLength(::Temporaries{<:MaybeOfflineAC{AdaSmoothAC{false}}}, phl::ParticleHistoryLength)
    # exact
    return ParticleHistoryLength(
        phl;
        particles=max(phl.particles, StaticFiniteHistory{2}()),
        logweights=max(phl.logweights, StaticFiniteHistory{2}()),
        didresample=max(phl.didresample, StaticFiniteHistory{1}()),
    )
end

function make_history(tmp::Temporaries{AdaSmoothAC{true}}, fkm::FeynmanKacModel, nparticles::Integer)
    # rejection, online
    L1, L2 = StaticFiniteHistory{1}(), StaticFiniteHistory{2}()
    enoch = make_history(L2, Int, nparticles)
    mustmix = make_history(L1, Bool)
    sup_logpdf = make_history(L1, Float64, nparticles)
    return (
        enoch=SummaryHistory(enoch),
        mustmix=SummaryHistory(mustmix),
        sup_logpdf=SummaryHistory(sup_logpdf),
    )
end
function make_history(tmp::Temporaries{OfflineAC{AdaSmoothAC{true}}}, fkm::FeynmanKacModel, nparticles::Integer)
    # rejection, offline
    L2, Lf = StaticFiniteHistory{2}(), FullHistory()
    enoch = make_history(L2, Int, nparticles)
    mustmix = make_history(Lf, Bool)
    sup_logpdf = make_history(Lf, Float64, nparticles)
    return (
        enoch=SummaryHistory(enoch),
        mustmix=SummaryHistory(mustmix),
        sup_logpdf=SummaryHistory(sup_logpdf),
    )
end
function make_history(tmp::Temporaries{AdaSmoothAC{false}}, fkm::FeynmanKacModel, nparticles::Integer)
    # exact, online
    L1, L2 = StaticFiniteHistory{1}(), StaticFiniteHistory{2}()
    enoch = make_history(L2, Int, nparticles)
    mustmix = make_history(L1, Bool)
    cumweights = make_history(L1, Float64, (nparticles, nparticles))
    return (
        enoch=SummaryHistory(enoch),
        mustmix=SummaryHistory(mustmix),
        cumweights=SummaryHistory(cumweights),
    )
end
function make_history(tmp::Temporaries{OfflineAC{AdaSmoothAC{false}}}, fkm::FeynmanKacModel, nparticles::Integer)
    # exact, offline
    L2, Lf = StaticFiniteHistory{2}(), FullHistory()
    enoch = make_history(L2, Int, nparticles)
    mustmix = make_history(Lf, Bool)
    cumweights = make_history(Lf, Float64, (nparticles, nparticles))
    return (
        enoch=SummaryHistory(enoch),
        mustmix=SummaryHistory(mustmix),
        cumweights=SummaryHistory(cumweights),
    )
end

Base.@propagate_inbounds function compute_temporaries!(smc::SMC, tmp::Temporaries{<:MaybeOfflineAC{AdaSmoothAC}}, history_tmp::SameTypeNamedTuple{SummaryHistory}, cache::Nothing, t::Integer, index_in_history::EndMinus, ::Val{:initial})
    enoch = next!(history_tmp.enoch; return_current=false)
    nparticles = length(enoch)
    @turbo enoch .= Base.OneTo(nparticles) 
    return nothing
end

Base.@propagate_inbounds function compute_temporaries!(smc::SMC, tmp::Temporaries{<:MaybeOfflineAC{AdaSmoothAC}}, history_tmp::SameTypeNamedTuple{SummaryHistory}, cache::Nothing, t::Integer, index_in_history::EndMinus)
    enoch, enoch_prev = next!(history_tmp.enoch; return_current=true)
    ancestors = smc.history_pf.ancestors[index_in_history]
    didresample = smc.history_pf.didresample[index_in_history]

    # Update Enoch indices
    @turbo for i in eachindex(enoch)
        @inbounds enoch[i] = enoch_prev[ancestors[i]]
    end
    @assertx issorted(enoch)
    
    mustmix = false
    if didresample
        num_unique_ancestors = count_unique_sorted_nonempty(enoch)
        beta = isa(tmp.summary, OfflineAC) ? tmp.summary.amortized.beta : tmp.summary.beta
        if num_unique_ancestors < beta * nparticles(smc)
            @turbo enoch .= Base.OneTo(nparticles(smc))
            mustmix = true
        end
    end
    _compute_temporaries!(smc, mustmix, tmp, history_tmp, t, index_in_history)
    return nothing
end

Base.@propagate_inbounds function _compute_temporaries!(smc::SMC, mustmix::Bool, tmp::Temporaries{<:MaybeOfflineAC{AdaSmoothAC{true}}}, history_tmp, t::Integer, index_in_history::EndMinus)
    fkm, θ = model(smc), smc.θ
    
    sup_logpdf = next!(history_tmp.sup_logpdf; return_current=false)
    particles_prev = smc.history_pf.particles[index_in_history - 1]
    particles = smc.history_pf.particles[index_in_history]
    
    push!(history_tmp.mustmix, mustmix)
    mustmix || return nothing
    # When mustmix is false, sup_logpdf will never be used, but still must be added
    # TODO: can this be made more memory-efficient?

    @threadsx summary for i in eachindex(particles)
        x = @inbounds particles[i]
        lwmax = -Inf64
        for xp in particles_prev
            Mt = fkm_Mt(fkm, θ, t, xp)
            lw = convert(Float64, logpdf(Mt, x))
            lwmax = max(lwmax, lw)
        end
        @inbounds sup_logpdf[i] = lwmax
    end

    return nothing
end

Base.@propagate_inbounds function _compute_temporaries!(smc::SMC, mustmix::Bool, tmp::Temporaries{<:MaybeOfflineAC{AdaSmoothAC{false}}}, history_tmp, t::Integer, index_in_history::EndMinus)
    fkm, θ = model(smc), smc.θ
    
    cumweights = next!(history_tmp.cumweights; return_current=false)
    particles_prev = smc.history_pf.particles[index_in_history - 1]
    particles = smc.history_pf.particles[index_in_history]
    logweights_prev = smc.history_pf.logweights[index_in_history - 1]
    
    push!(history_tmp.mustmix, mustmix)
    mustmix || return nothing
    # When mustmix is false, cumweights will never be used, but still must be added
    # TODO: can this be made more memory-efficient?

    @threadsx summary for i in eachindex(particles)
        x = @inbounds particles[i]
        cw = @inbounds view(cumweights, :, i)
        lwmax = -Inf64
        @inbounds for a in eachindex(particles_prev)
            xp = particles_prev[a]
            Mt = fkm_Mt(fkm, θ, t, xp)
            lw = logweights_prev[a] + convert(Float64, logpdf(Mt, x))
            cw[a] = lw
            lwmax = max(lwmax, lw)
        end
        @inbounds cw[1] = exp(cw[1] - lwmax)
        @inbounds for i in 2:length(cw)
            cw[i] = cw[i-1] + exp(cw[i] - lwmax)
        end
    end

    return nothing
end

##########################################################################################

struct AmortizedAdaSmooth{USE_REJECTION} <: AfsMethod
    beta::Float64
    # Suggested values for alpha (adaptive resampling threshold) and beta
    # are such that alpha ≥ beta and alpha ≈ beta ≈ 0.5
    # Default values in the paper are alpha=0.6, beta=0.5
    function AmortizedAdaSmooth(beta::Real=0.5; use_rejection::Bool=true)
        0 < beta < 1 || throw(ArgumentError("AdaSmooth beta parameter must be in (0, 1)"))
        return new{use_rejection}(beta)
    end
end

AdaSmoothAC(method::AmortizedAdaSmooth{REJ}) where {REJ} = AdaSmoothAC(method.beta; use_rejection=REJ)
required_amortized_computations(s::AbstractAdditiveFunctionSmoother{<:AmortizedAdaSmooth}) = (AdaSmoothAC(s.method), )

function ParticleHistoryLength(::AfsTmp{AmortizedAdaSmooth{true}}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=max(phl.particles, StaticFiniteHistory{2}()),
        weights=max(phl.weights, StaticFiniteHistory{2}()),
    )
end
function ParticleHistoryLength(::AfsOff{AmortizedAdaSmooth{true}}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=FullHistory(),
        weights=FullHistory(),
        ancestors=FullHistory(),
    )
end
function ParticleHistoryLength(::AfsTmp{AmortizedAdaSmooth{false}}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=max(phl.particles, StaticFiniteHistory{2}()),
    )
end
function ParticleHistoryLength(::AfsOff{AmortizedAdaSmooth{false}}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=FullHistory(),
        ancestors=FullHistory(),
    )
end

function make_cache(tmp::Temporaries{<:ImmutableAdditiveFunctionSmoother{AmortizedAdaSmooth{true}}}, fkm::FeynmanKacModel, nparticles::Integer)
    addfun = tmp.summary.addfun
    return AutoCache(
        () -> (
            cache_af=make_cache(addfun, fkm),
            queue=MultinomialQueue(nparticles),
        ),
        nparticles
    )
end
function make_cache(tmp::Temporaries{<:MutableAdditiveFunctionSmoother{AmortizedAdaSmooth{true}}}, fkm::FeynmanKacModel, nparticles::Integer)
    addfun = tmp.summary.addfun
    mkvalue = template_maker(addfun, fkm)
    return AutoCache(
        () -> (
            cache_af=make_cache(addfun, fkm),
            queue=MultinomialQueue(nparticles),
            tmp=mkvalue(),
        ),
        nparticles
    )
end
function make_cache(tmp::Temporaries{<:ImmutableAdditiveFunctionSmoother{AmortizedAdaSmooth{false}}}, fkm::FeynmanKacModel, nparticles::Integer)
    addfun = tmp.summary.addfun
    return AutoCache(
        () -> (
            cache_af=make_cache(addfun, fkm),
        ),
        nparticles
    )
end
function make_cache(tmp::Temporaries{<:MutableAdditiveFunctionSmoother{AmortizedAdaSmooth{false}}}, fkm::FeynmanKacModel, nparticles::Integer)
    addfun = tmp.summary.addfun
    mkvalue = template_maker(addfun, fkm)
    return AutoCache(
        () -> (
            cache_af=make_cache(addfun, fkm),
            tmp=mkvalue(),
        ),
        nparticles
    )
end

Base.@propagate_inbounds function compute_temporaries!(smc::SMC, tmp::AfsTmp{AmortizedAdaSmooth{USE_REJECTION}}, history_tmp::SameTypeNamedTuple{SummaryHistory}, autocache::AutoCache, t::Integer, index_in_history::EndMinus) where {USE_REJECTION}
    addfun = tmp.summary.addfun
    history_amortized = get_required_amortized_history(smc, tmp)[1]
    
    if history_amortized.mustmix[index_in_history]
        if USE_REJECTION
            adasmooth_reject(smc, addfun, history_tmp, history_amortized, autocache, t, index_in_history)
        else
            adasmooth_exact(smc, addfun, history_tmp, history_amortized, autocache, t, index_in_history)
        end
    else
        adasmooth_poor(smc, addfun, history_tmp, autocache, t, index_in_history)
    end

    return nothing
end

Base.@propagate_inbounds function adasmooth_reject(smc::SMC, addfun::AdditiveFunction, history_tmp, history_amortized, autocache::AutoCache, t::Integer, index_in_history::EndMinus)
    fkm, θ = model(smc), smc.θ
    
    sup_logpdf = history_amortized.sup_logpdf[index_in_history]
    Phi, Phi_prev = next!(history_tmp.Phi; return_current=true)
    particles_prev = smc.history_pf.particles[index_in_history - 1]
    particles = smc.history_pf.particles[index_in_history]
    weights_prev = smc.history_pf.weights[index_in_history - 1]
    ancestors = smc.history_pf.ancestors[index_in_history]
    
    # Initialize multinomial queues
    initqueue! = cache -> reset!(cache.queue, weights_prev, Val(:noresize))
    foreach(initqueue!, autocache)
    
    @threadsx summary for i in eachindex(Phi)
        cache = @inbounds getindex(autocache, i, initqueue!)
        x = @inbounds particles[i]
        if isa(addfun, MutableAdditiveFunction)
            Phi_i = @inbounds Phi[i]
        end

        let sup_logpdf = @inbounds sup_logpdf[i]
            @inbounds while true
                a = pop!(cache.queue)::Int
                xp = particles_prev[a]
                Mt = fkm_Mt(fkm, θ, t, xp)
                prob_acceptance = convert(Float64, exp(logpdf(Mt, x) - sup_logpdf))
                @assertx prob_acceptance ≤ 1.0 # by construction
                if rand(TaskLocalRNG(), Float64) < prob_acceptance
                    if isa(addfun, ImmutableAdditiveFunction)
                        sumPhi = Phi_prev[a] + addfun(fkm, θ, t, xp, x, cache.cache_af)
                    else
                        @assert isa(addfun, MutableAdditiveFunction)
                        addfun(cache.tmp, fkm, θ, t, xp, x, cache.cache_af)
                        af_set!(addfun, Phi_i, Phi_prev[a], cache.tmp)
                        # The above line is for ArrayAdditiveFunctions equivalent to
                        #   Phi_i .= Phi_prev[a] .+ cache.tmp
                    end
                    break
                end
            end
        end

        a = @inbounds ancestors[i]::Int
        xp = @inbounds particles_prev[a]
        @inbounds if isa(addfun, ImmutableAdditiveFunction)
            sumPhi += Phi_prev[a] + addfun(fkm, θ, t, xp, x, cache.cache_af)
        else
            @assert isa(addfun, MutableAdditiveFunction)
            addfun(cache.tmp, fkm, θ, t, xp, x, cache.cache_af)
            af_add!(addfun, Phi_i, Phi_prev[a], cache.tmp)
            # The above line is for ArrayAdditiveFunctions equivalent to
            #   Phi_i .+= Phi_prev[a] .+ cache.tmp
        end

        release!(autocache, i)

        if isa(addfun, ImmutableAdditiveFunction)
            @inbounds Phi[i] = sumPhi / 2
        else
            @assert isa(addfun, MutableAdditiveFunction)
            af_div!(addfun, Phi_i, 2)
            # The above line is for ArrayAdditiveFunctions equivalent to
            #   @. Phi_i /= 2
        end
    end

    return nothing
end

Base.@propagate_inbounds function adasmooth_exact(smc::SMC, addfun::AdditiveFunction, history_tmp, history_amortized, autocache::AutoCache, t::Integer, index_in_history::EndMinus)
    fkm, θ = model(smc), smc.θ
    
    cumweights = history_amortized.cumweights[index_in_history]
    Phi, Phi_prev = next!(history_tmp.Phi; return_current=true)
    particles_prev = smc.history_pf.particles[index_in_history - 1]
    particles = smc.history_pf.particles[index_in_history]
    ancestors = smc.history_pf.ancestors[index_in_history]
    
    @threadsx summary for i in eachindex(Phi)
        cache = @inbounds autocache[i]
        x = @inbounds particles[i]
        if isa(addfun, MutableAdditiveFunction)
            Phi_i = @inbounds Phi[i]
        end

        # Sample ancestor from backward kernel
        a = _sample_one_multinomial(@inbounds view(cumweights, :, i))::Int
        xp = particles_prev[a]

        @inbounds if isa(addfun, ImmutableAdditiveFunction)
            sumPhi = Phi_prev[a] + addfun(fkm, θ, t, xp, x, cache.cache_af)
        else
            @assert isa(addfun, MutableAdditiveFunction)
            addfun(cache.tmp, fkm, θ, t, xp, x, cache.cache_af)
            af_set!(addfun, Phi_i, Phi_prev[a], cache.tmp)
            # The above line is for ArrayAdditiveFunctions equivalent to
            #   Phi_i .= Phi_prev[a] .+ cache.tmp
        end

        a = @inbounds ancestors[i]::Int
        xp = @inbounds particles_prev[a]
        @inbounds if isa(addfun, ImmutableAdditiveFunction)
            sumPhi += Phi_prev[a] + addfun(fkm, θ, t, xp, x, cache.cache_af)
        else
            @assert isa(addfun, MutableAdditiveFunction)
            addfun(cache.tmp, fkm, θ, t, xp, x, cache.cache_af)
            af_add!(addfun, Phi_i, Phi_prev[a], cache.tmp)
            # The above line is for ArrayAdditiveFunctions equivalent to
            #   Phi_i .+= Phi_prev[a] .+ cache.tmp
        end

        release!(autocache, i)

        if isa(addfun, ImmutableAdditiveFunction)
            @inbounds Phi[i] = sumPhi / 2
        else
            @assert isa(addfun, MutableAdditiveFunction)
            af_div!(addfun, Phi_i, 2)
            # The above line is for ArrayAdditiveFunctions equivalent to
            #   @. Phi_i /= 2
        end
    end

    return nothing
end
