struct AdaSmooth{USE_REJECTION} <: AfsMethod
    beta::Float64
    # Suggested values for alpha (adaptive resampling threshold) and beta
    # are such that alpha ≥ beta and alpha ≈ beta ≈ 0.5
    # Default values in the paper are alpha=0.6, beta=0.5
    function AdaSmooth(beta::Real=0.5; use_rejection::Bool=true)
        0 < beta < 1 || throw(ArgumentError("AdaSmooth beta parameter must be in (0, 1)"))
        return new{use_rejection}(beta)
    end
end

function ParticleHistoryLength(::AfsTmp{AdaSmooth{true}}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=max(phl.particles, StaticFiniteHistory{2}()),
        weights=max(phl.weights, StaticFiniteHistory{2}()),
        didresample=max(phl.didresample, StaticFiniteHistory{1}()),
    )
end

function ParticleHistoryLength(::AfsOff{AdaSmooth{true}}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=FullHistory(),
        weights=FullHistory(),
        ancestors=FullHistory(),
        didresample=FullHistory(),
    )
end

function ParticleHistoryLength(::AfsTmp{AdaSmooth{false}}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=max(phl.particles, StaticFiniteHistory{2}()),
        logweights=max(phl.logweights, StaticFiniteHistory{2}()),
        didresample=max(phl.didresample, StaticFiniteHistory{1}()),
    )
end

function ParticleHistoryLength(::AfsOff{AdaSmooth{false}}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=FullHistory(),
        logweights=FullHistory(),
        ancestors=FullHistory(),
        didresample=FullHistory(),
    )
end

function make_cache(tmp::Temporaries{<:ImmutableAdditiveFunctionSmoother{AdaSmooth{true}}}, fkm::FeynmanKacModel, nparticles::Integer)
    addfun = tmp.summary.addfun
    return AutoCache(
        () -> (
            cache_af=make_cache(addfun, fkm),
            queue=MultinomialQueue(nparticles),
        ),
        nparticles
    )
end

function make_cache(tmp::Temporaries{<:MutableAdditiveFunctionSmoother{AdaSmooth{true}}}, fkm::FeynmanKacModel, nparticles::Integer)
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

function make_cache(tmp::Temporaries{<:ImmutableAdditiveFunctionSmoother{AdaSmooth{false}}}, fkm::FeynmanKacModel, nparticles::Integer)
    addfun = tmp.summary.addfun
    return AutoCache(
        () -> (
            cache_af=make_cache(addfun, fkm),
            w=Vector{Float64}(undef, nparticles),
        ),
        nparticles
    )
end

function make_cache(tmp::Temporaries{<:MutableAdditiveFunctionSmoother{AdaSmooth{false}}}, fkm::FeynmanKacModel, nparticles::Integer)
    addfun = tmp.summary.addfun
    mkvalue = template_maker(addfun, fkm)
    return AutoCache(
        () -> (
            cache_af=make_cache(addfun, fkm),
            w=Vector{Float64}(undef, nparticles),
            tmp=mkvalue(),
        ),
        nparticles
    )
end

function make_extra_temporaries(::AfsTmp{<:AdaSmooth}, ::FeynmanKacModel, nparticles::Integer)
    L = StaticFiniteHistory{2}()
    enoch = make_history(L, Int, nparticles)
    return (; enoch=SummaryHistory(enoch))
end

function initialize_extra_temporaries!(::SMC, ::AfsTmp{<:AdaSmooth}, history_tmp::SameTypeNamedTuple{SummaryHistory}, ::AutoCache, t::Integer, index_in_history::EndMinus)
    enoch = next!(history_tmp.enoch; return_current=false)
    nparticles = length(enoch)
    @turbo enoch .= Base.OneTo(nparticles) 
    return nothing
end

Base.@propagate_inbounds function compute_temporaries!(smc::SMC, tmp::AfsTmp{AdaSmooth{USE_REJECTION}}, history_tmp::SameTypeNamedTuple{SummaryHistory, 2, (:Phi, :enoch)}, autocache::AutoCache, t::Integer, index_in_history::EndMinus) where {USE_REJECTION}
    addfun = tmp.summary.addfun
    enoch, enoch_prev = next!(history_tmp.enoch; return_current=true)
    ancestors = smc.history_pf.ancestors[index_in_history]
    didresample = smc.history_pf.didresample[index_in_history]

    # Update Enoch indices
    @turbo for i in eachindex(enoch)
        @inbounds enoch[i] = enoch_prev[ancestors[i]]
    end
    @assertx issorted(enoch)
    
    if didresample
        num_unique_ancestors = count_unique_sorted_nonempty(enoch)
        if num_unique_ancestors < tmp.summary.method.beta * nparticles(smc)
            @turbo enoch .= Base.OneTo(nparticles(smc))
            if USE_REJECTION
                adasmooth_reject(smc, addfun, history_tmp, autocache, t, index_in_history)
            else
                adasmooth_exact(smc, addfun, history_tmp, autocache, t, index_in_history)
            end
            return nothing
        end
    end
    adasmooth_poor(smc, addfun, history_tmp, autocache, t, index_in_history)
    return nothing
end

Base.@propagate_inbounds function adasmooth_poor(smc::SMC, addfun::AdditiveFunction, history_tmp, autocache::AutoCache, t::Integer, index_in_history::EndMinus)
    fkm, θ = model(smc), smc.θ

    Phi, Phi_prev = next!(history_tmp.Phi; return_current=true)
    particles_prev = smc.history_pf.particles[index_in_history - 1]
    particles = smc.history_pf.particles[index_in_history]
    ancestors = smc.history_pf.ancestors[index_in_history]
    
    @threadsx summary for i in eachindex(Phi)
        x = @inbounds particles[i]
        k = @inbounds ancestors[i]
        xp = @inbounds particles_prev[k]
        cache = @inbounds autocache[i]
        if isa(addfun, ImmutableAdditiveFunction)
            @inbounds Phi[i] = Phi_prev[k] + addfun(fkm, θ, t, xp, x, cache.cache_af)
        else
            @assert isa(addfun, MutableAdditiveFunction)
            Phi_i = @inbounds Phi[i]
            @inbounds addfun(Phi_i, fkm, θ, t, xp, x, cache.cache_af)
            af_add!(addfun, Phi_i, @inbounds Phi_prev[k])
            # The above line is for ArrayAdditiveFunctions equivalent to
            #   Phi_i .+= @inbounds Phi_prev[k]
        end
        release!(autocache, i)
    end

    return nothing
end

Base.@propagate_inbounds function adasmooth_reject(smc::SMC, addfun::AdditiveFunction, history_tmp, autocache::AutoCache, t::Integer, index_in_history::EndMinus)
    fkm, θ = model(smc), smc.θ
    
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
        sup_logpdf = fkm_sup_logpdf_Mt(fkm, θ, t, x) # upper bound for rejection sampling
        if isa(addfun, MutableAdditiveFunction)
            Phi_i = @inbounds Phi[i]
        end

        @inbounds while true
            a = pop!(cache.queue)::Int
            xp = particles_prev[a]
            Mt = fkm_Mt(fkm, θ, t, xp)
            prob_acceptance = convert(Float64, exp(logpdf(Mt, x) - sup_logpdf))
            if prob_acceptance > 1.0 || !isfinite(prob_acceptance)
                error("invalid acceptance probability $prob_acceptance (fkm_sup_logpdf_Mt may not be correctly defined [$sup_logpdf])")
            end
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

Base.@propagate_inbounds function adasmooth_exact(smc::SMC, addfun::AdditiveFunction, history_tmp, autocache::AutoCache, t::Integer, index_in_history::EndMinus)
    fkm, θ = model(smc), smc.θ
    
    Phi, Phi_prev = next!(history_tmp.Phi; return_current=true)
    particles_prev = smc.history_pf.particles[index_in_history - 1]
    particles = smc.history_pf.particles[index_in_history]
    logweights_prev = smc.history_pf.logweights[index_in_history - 1]
    ancestors = smc.history_pf.ancestors[index_in_history]
    
    @threadsx summary for i in eachindex(Phi)
        cache = @inbounds autocache[i]
        x = @inbounds particles[i]
        if isa(addfun, MutableAdditiveFunction)
            Phi_i = @inbounds Phi[i]
        end

        # Sample ancestor from backward kernel
        lwmax = -Inf64
        @inbounds for a in eachindex(particles_prev)::Base.OneTo
            xp = particles_prev[a]
            Mt = fkm_Mt(fkm, θ, t, xp)
            lw = logweights_prev[a] + convert(Float64, logpdf(Mt, x))
            cache.w[a] = lw
            lwmax = max(lwmax, lw)
        end
        a = _sample_one_multinomial!(cache.w, lwmax)::Int
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

function _sample_one_multinomial!(logw::AbstractVector{<:Real}, logw_max::Real)
    ifirst, ilast = firstindex(logw), lastindex(logw)
    @inbounds logw[ifirst] = exp(logw[ifirst] - logw_max)
    @inbounds for i in (ifirst + 1):ilast
        logw[i] = logw[i-1] + exp(logw[i] - logw_max)
    end
    return _sample_one_multinomial(logw)
end

function _sample_one_multinomial(cumweights::AbstractVector{<:Real})
    ilast = lastindex(cumweights)
    wsum = @inbounds cumweights[ilast]
    z = rand(TaskLocalRNG(), Float64)
    return searchsorted(cumweights, wsum * z).start
end
