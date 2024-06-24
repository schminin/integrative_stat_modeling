struct ParisAfsMethod <: AfsMethod
    n::Int
end
ParisAfsMethod() = ParisAfsMethod(2)

function ParticleHistoryLength(::AfsTmp{ParisAfsMethod}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=max(phl.particles, StaticFiniteHistory{2}()),
        weights=max(phl.weights, StaticFiniteHistory{2}()),
    )
end

function ParticleHistoryLength(::AfsOff{ParisAfsMethod}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=FullHistory(),
        weights=FullHistory(),
    )
end

function make_cache(tmp::Temporaries{<:ImmutableAdditiveFunctionSmoother{ParisAfsMethod}}, fkm::FeynmanKacModel, nparticles::Integer)
    addfun = tmp.summary.addfun
    return AutoCache(
        () -> (
            cache_af=make_cache(addfun, fkm),
            queue=MultinomialQueue(nparticles),
        ),
        nparticles
    )
end

function make_cache(tmp::Temporaries{<:MutableAdditiveFunctionSmoother{ParisAfsMethod}}, fkm::FeynmanKacModel, nparticles::Integer)
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

Base.@propagate_inbounds function compute_temporaries!(smc::SMC, tmp::AfsTmp{ParisAfsMethod}, history_tmp::SameTypeNamedTuple{SummaryHistory}, autocache::AutoCache, t::Integer, index_in_history::EndMinus)
    fkm, θ = model(smc), smc.θ
    addfun, nsamples = tmp.summary.addfun, tmp.summary.method.n
    
    Phi, Phi_prev = next!(history_tmp.Phi; return_current=true)
    particles_prev = smc.history_pf.particles[index_in_history - 1]
    particles = smc.history_pf.particles[index_in_history]
    weights_prev = smc.history_pf.weights[index_in_history - 1]
    
    # Initialize multinomial queues
    initqueue! = cache -> reset!(cache.queue, weights_prev, Val(:noresize))
    foreach(initqueue!, autocache)
    
    @threadsx summary for i in eachindex(Phi)
        cache = @inbounds getindex(autocache, i, initqueue!)
        X = @inbounds particles[i]
        sup_logpdf = fkm_sup_logpdf_Mt(fkm, θ, t, X) # upper bound for rejection sampling
        countdown = nsamples # target number of draws by rejection sampling
        if isa(addfun, ImmutableAdditiveFunction)
            sumPhi = zero(eltype(Phi))
        else
            @assert isa(addfun, MutableAdditiveFunction)
            Phi_i = @inbounds Phi[i]
            af_zero!(addfun, Phi_i)
            # The above line is for ArrayAdditiveFunctions equivalent to
            #   fill!(Phi_i, zero(eltype(Phi_i)))
        end
        while true
            a = pop!(cache.queue)
            Xprev = @inbounds particles_prev[a]
            Mt = @inbounds fkm_Mt(fkm, θ, t, Xprev)
            prob_acceptance = convert(Float64, exp(logpdf(Mt, X) - sup_logpdf))
            if prob_acceptance > 1.0 || !isfinite(prob_acceptance)
                error("invalid acceptance probability $prob_acceptance (fkm_sup_logpdf_Mt may not be correctly defined [$sup_logpdf])")
            end
            if rand(TaskLocalRNG(), Float64) < prob_acceptance
                if isa(addfun, ImmutableAdditiveFunction)
                    sumPhi += @inbounds Phi_prev[a] + addfun(fkm, θ, t, Xprev, X, cache.cache_af)
                else
                    @assert isa(addfun, MutableAdditiveFunction)
                    addfun(cache.tmp, fkm, θ, t, Xprev, X, cache.cache_af)
                    af_add!(addfun, Phi_i, @inbounds(Phi_prev[a]), cache.tmp)
                    # The above line is for ArrayAdditiveFunctions equivalent to
                    #   Phi_i .+= Phi_prev[a] .+ cache.tmp
                end
                countdown -= 1
                iszero(countdown) && break
            end
        end
        release!(autocache, i)
        if isa(afs.addfun, ImmutableAdditiveFunction)
            @inbounds Phi[i] = sumPhi / nsamples
        else
            @assert isa(addfun, MutableAdditiveFunction)
            af_div!(addfun, Phi_i, nsamples)
            # The above line is for ArrayAdditiveFunctions equivalent to
            #   @. Phi_i /= nsamples
        end
    end

    return nothing
end
