# TODO this whole file is hacky

struct LiveParticlesAC <: AmortizedComputation end

function ParticleHistoryLength(::Temporaries{<:MaybeOfflineAC{LiveParticlesAC}}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(phl; ancestors=FullHistory())
end

function make_history(tmp::Temporaries{<:MaybeOfflineAC{LiveParticlesAC}}, fkm::FeynmanKacModel, nparticles::Integer)
    alive = make_history(FullHistory(), Bool, nparticles)
    return (alive=SummaryHistory(alive), )
end

Base.@propagate_inbounds function compute_temporaries!(smc::SMC, tmp::Temporaries{<:MaybeOfflineAC{LiveParticlesAC}}, history_tmp::SameTypeNamedTuple{SummaryHistory}, cache::Nothing, t::Integer, index_in_history::EndMinus)
    alive = next!(history_tmp.alive; return_current=false)
    fill!(alive, false)
    return nothing
end

function finalize_amortized!(smc::SMC, ::MaybeOfflineAC{LiveParticlesAC}, history_ac::SameTypeNamedTuple{SummaryHistory}, cache::Nothing)
    alive = history_ac.alive
    ancestors = smc.history_pf.ancestors
    for i in Base.OneTo(nparticles(smc))
        k = i
        s_alive = lastindex(alive)
        s_ancestors = lastindex(ancestors)
        while true
            @inbounds alive[s_alive][k] = true
            isone(s_alive) && break
            k = @inbounds ancestors[s_ancestors][k]
            s_alive -= 1
            @inbounds alive[s_alive][k] && break
            s_ancestors -= 1
        end
    end
    return nothing
end

#########################################################################################################

struct AmortizedNaiveAfsMethod <: AfsMethod end

required_amortized_computations(s::AbstractAdditiveFunctionSmoother{<:AmortizedNaiveAfsMethod}) = (LiveParticlesAC(), )

function ParticleHistoryLength(::AfsTmp{AmortizedNaiveAfsMethod}, phl::ParticleHistoryLength)
    error("AmortizedNaiveAfsMethod cannot be run online")
end
function ParticleHistoryLength(::AfsOff{AmortizedNaiveAfsMethod}, phl::ParticleHistoryLength)
    return ParticleHistoryLength(
        phl;
        particles=FullHistory(),
        ancestors=FullHistory(),
    )
end

Base.@propagate_inbounds function compute_temporaries!(smc::SMC, tmp::AfsTmp{AmortizedNaiveAfsMethod}, history_tmp::SameTypeNamedTuple{SummaryHistory}, autocache, t::Integer, index_in_history::EndMinus)
    fkm, θ = model(smc), smc.θ
    addfun = tmp.summary.addfun
    alive = get_required_amortized_history(smc, tmp)[1].alive[index_in_history]
    Phi, Phi_prev = next!(history_tmp.Phi; return_current=true)
    particles_prev = smc.history_pf.particles[index_in_history - 1]
    particles = smc.history_pf.particles[index_in_history]
    ancestors = smc.history_pf.ancestors[index_in_history]
    @threadsx summary for i in eachindex(Phi)
        @inbounds alive[i] || continue
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
