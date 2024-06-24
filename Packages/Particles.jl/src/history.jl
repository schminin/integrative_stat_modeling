struct ParticleHistory{T, P, LW, W, A, R, C, LMW}
    particles::P
    logweights::LW
    weights::W
    ancestors::A
    didresample::R
    logCnorm::C
    logmeanw::LMW
    function ParticleHistory(::Type{T}, Lhistory::ParticleHistoryLength, nparticles::Integer) where {T}
        particles = make_history(Lhistory.particles, T, nparticles)
        logweights = make_history(Lhistory.logweights, Float64, nparticles)
        weights = make_history(Lhistory.weights, Float64, nparticles)
        ancestors = make_history(Lhistory.ancestors, Int, nparticles)
        didresample = make_history(Lhistory.didresample, Bool)
        logCnorm = make_history(Lhistory.logCnorm, Float64)
        Lhistory_logmeanw = FiniteHistory(min(maxlength(Lhistory.logCnorm), static(1))) # only needed if computing logCnorm
        logmeanw = make_history(Lhistory_logmeanw, Float64)
        return new{T, typeof(particles), typeof(logweights), typeof(weights), typeof(ancestors), typeof(didresample), typeof(logCnorm), typeof(logmeanw)}(
            particles, logweights, weights, ancestors, didresample, logCnorm, logmeanw
        )
    end
end

function ParticleHistoryLength(h::ParticleHistory)
    return ParticleHistoryLength(
        HistoryLength(h.particles),
        HistoryLength(h.logweights),
        HistoryLength(h.weights),
        HistoryLength(h.ancestors),
        HistoryLength(h.didresample),
        HistoryLength(h.logCnorm),
    )
end

nparticles(h::ParticleHistory) = first(arraysize(h.particles))

function Base.empty!(h::ParticleHistory)
    empty!(h.particles)
    empty!(h.logweights)
    empty!(h.weights)
    empty!(h.ancestors)
    empty!(h.didresample)
    empty!(h.logCnorm)
    empty!(h.logmeanw)
    @assertx isempty(h.particles)
    @assertx isempty(h.logweights)
    @assertx isempty(h.weights)
    @assertx isempty(h.ancestors)
    @assertx isempty(h.didresample)
    @assertx isempty(h.logCnorm)
    @assertx isempty(h.logmeanw)
    return h
end

function next!(h::ParticleHistory, ::Val{:initial})
    return (
        particles=next!(h.particles),
        logweights=next!(h.logweights),
        weights=next!(h.weights),
    )
end

Base.@propagate_inbounds function next!(h::ParticleHistory{T}) where {T}
    # @inbounds should be used if we are sure that the initial step was already done (so that the previous step can be returned)
    new_particles,  prev_particles  = next!(h.particles;  return_current=true)
    new_logweights, prev_logweights = next!(h.logweights; return_current=true)
    new_weights,    prev_weights    = next!(h.weights;    return_current=true)
    new_ancestors = next!(h.ancestors)
    newstep = (
        particles=new_particles,
        logweights=new_logweights,
        weights=new_weights,
        ancestors=new_ancestors,
    )
    prevstep = (
        particles=prev_particles,
        logweights=prev_logweights,
        weights=prev_weights,
    )
    return newstep, prevstep
end

function preallocate!(h::ParticleHistory, n::Integer)
    preallocate!(h.particles, n)
    preallocate!(h.logweights, n)
    preallocate!(h.weights, n)
    preallocate!(h.ancestors, n - 1) # ancestors cannot be computed at the initial step
    preallocate!(h.didresample, n - 1) # no resampling ever at the initial step
    preallocate!(h.logCnorm, n)
    return h
end
