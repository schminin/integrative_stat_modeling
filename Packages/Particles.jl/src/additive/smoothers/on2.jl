struct ON2AfsMethod <: AfsMethod end

function ParticleHistoryLength(::AfsTmp{ON2AfsMethod}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=max(phl.particles, StaticFiniteHistory{2}()),
        logweights=max(phl.logweights, StaticFiniteHistory{2}()),
    )
end

function ParticleHistoryLength(::AfsOff{ON2AfsMethod}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=FullHistory(),
        logweights=FullHistory(),
    )
end

function make_cache(tmp::Temporaries{<:ImmutableAdditiveFunctionSmoother{ON2AfsMethod}}, fkm::FeynmanKacModel, nparticles::Integer)
    addfun = tmp.summary.addfun
    return AutoCache(
        () -> (
            cache_af=make_cache(addfun, fkm),
            w=Vector{Float64}(undef, nparticles),
        ),
        nparticles
    )
end

function make_cache(tmp::Temporaries{<:MutableAdditiveFunctionSmoother{ON2AfsMethod}}, fkm::FeynmanKacModel, nparticles::Integer)
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

Base.@propagate_inbounds function compute_temporaries!(smc::SMC, tmp::AfsTmp{ON2AfsMethod}, history_tmp::SameTypeNamedTuple{SummaryHistory}, autocache::AutoCache, t::Integer, index_in_history::EndMinus)
    fkm, θ = model(smc), smc.θ
    addfun = tmp.summary.addfun
    
    Phi, Phi_prev = next!(history_tmp.Phi; return_current=true)
    particles_prev = smc.history_pf.particles[index_in_history - 1]
    particles = smc.history_pf.particles[index_in_history]
    logweights_prev = smc.history_pf.logweights[index_in_history - 1]
    
    @threadsx summary for i in eachindex(Phi)
        # Compute weights
        X = @inbounds particles[i]
        cache = @inbounds autocache[i]
        lwmax = -Inf64
        @inbounds for k in eachindex(cache.w)
            Xprev = particles_prev[k]
            Mt = fkm_Mt(fkm, θ, t, Xprev)
            lw = logweights_prev[k] + convert(Float64, logpdf(Mt, X))
            cache.w[k] = lw
            lwmax = max(lwmax, lw)
        end
        # Weighted average
        if isa(addfun, ImmutableAdditiveFunction)
            tot = zero(eltype(Phi))
        else
            @assert isa(addfun, MutableAdditiveFunction)
            Phi_i = @inbounds Phi[i]
            af_zero!(addfun, Phi_i)
            # The above line is for ArrayAdditiveFunctions equivalent to
            #   fill!(Phi_i, zero(eltype(Phi_i)))
        end
        wsum = 0.0
        @inbounds for k in eachindex(cache.w)
            Xprev = particles_prev[k]
            w = exp(cache.w[k] - lwmax)
            wsum += w
            if isa(addfun, ImmutableAdditiveFunction)
                tot += w * (Phi_prev[k] + addfun(fkm, θ, t, Xprev, X, cache.cache_af))
            else
                @assert isa(addfun, MutableAdditiveFunction)
                addfun(cache.tmp, fkm, θ, t, Xprev, X, cache.cache_af)
                af_add!(addfun, Phi_i, w, Phi_prev[k], cache.tmp)
                # The above line is for ArrayAdditiveFunctions equivalent to
                #   Phi_i .= muladd.(w, Phi_prev[k] .+ cache.tmp, Phi_i)
                #   i.e.
                #   Phi_i .+= w .* (Phi_prev[k] .+ cache.tmp)
            end
        end
        release!(autocache, i)
        if isa(addfun, ImmutableAdditiveFunction)
            @inbounds Phi[i] = tot / wsum
        else
            @assert isa(addfun, MutableAdditiveFunction)
            af_div!(addfun, Phi_i, wsum)
            # The above line is for ArrayAdditiveFunctions equivalent to
            #   @. Phi_i /= wsum
        end
    end

    return nothing
end
