struct NaiveAfsMethod <: AfsMethod end

function ParticleHistoryLength(::AfsTmp{NaiveAfsMethod}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=max(phl.particles, StaticFiniteHistory{2}()),
    )
end

function ParticleHistoryLength(::AfsOff{NaiveAfsMethod}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=FullHistory(),
        ancestors=FullHistory(),
    )
end

Base.@propagate_inbounds function compute_temporaries!(smc::SMC, tmp::AfsTmp{NaiveAfsMethod}, history_tmp::SameTypeNamedTuple{SummaryHistory}, autocache, t::Integer, index_in_history::EndMinus)
    fkm, θ = model(smc), smc.θ
    addfun = tmp.summary.addfun
    Phi, Phi_prev = next!(history_tmp.Phi; return_current=true)
    particles_prev = smc.history_pf.particles[index_in_history - 1]
    particles = smc.history_pf.particles[index_in_history]
    ancestors = smc.history_pf.ancestors[index_in_history]
    @threadsx summary for i in eachindex(Phi)
        X = @inbounds particles[i]
        k = @inbounds ancestors[i]
        Xprev = @inbounds particles_prev[k]
        cache = @inbounds autocache[i]
        if isa(addfun, ImmutableAdditiveFunction)
            @inbounds Phi[i] = Phi_prev[k] + addfun(fkm, θ, t, Xprev, X, cache.cache_af)
        else
            @assert isa(addfun, MutableAdditiveFunction)
            Phi_i = @inbounds Phi[i]
            @inbounds addfun(Phi_i, fkm, θ, t, Xprev, X, cache.cache_af)
            af_add!(addfun, Phi_i, @inbounds Phi_prev[k])
            # The above line is for ArrayAdditiveFunctions equivalent to
            #   Phi_i .+= @inbounds Phi_prev[k]
        end
        release!(autocache, i)
    end
    return nothing
end
