"""
    BootstrapFilter{T_X, T_Y, SSM <: StateSpaceModel{T_X, T_Y}, OBS <: Observations{T_Y}} <: StateSpaceFeynmanKacModel{T_X, T_Y, SSM, OBS}
Fenyman-Kac model associated to the Boootstrap Filter for a state-space model.
"""
struct BootstrapFilter{T_X, T_Y, SSM <: StateSpaceModel{T_X, T_Y}, OBS <: Observations{T_Y}} <: StateSpaceFeynmanKacModel{T_X, T_Y, SSM, OBS}
    ssm::SSM
    observations::OBS
    function BootstrapFilter(ssm::StateSpaceModel{T_X, T_Y}, observations::Observations{T_Y}) where {T_X, T_Y}
        Base.has_offset_axes(observations) && throw(ArgumentError("observations vector indexing must be 1-based"))
        return new{T_X, T_Y, typeof(ssm), typeof(observations)}(ssm, observations)
    end
end

statespacemodel(bf::BootstrapFilter) = bf.ssm
observations(bf::BootstrapFilter) = bf.observations

function fkm_M0(bf::BootstrapFilter, θ)
    if convert(Bool, has_proposal(bf.ssm)) # maybe static
        t = initial_time(bf.ssm)
        i = observations_t2i(bf, t)
        y = bf.observations[i]
        if ismissing(y)
            return ssm_PX0(bf.ssm, θ)
        else
            # TODO not type stable! we really should give up on Feynman-Kac formulation
            return ssm_proposal0(bf.ssm, θ, proposal0_parameters(bf.ssm), y)
        end
    else
        return ssm_PX0(bf.ssm, θ)
    end
end

Base.@propagate_inbounds function  fkm_Mt(bf::BootstrapFilter, θ, t::Integer, xp)
    if convert(Bool, has_proposal(bf.ssm)) # maybe static
        i = observations_t2i(bf, t)
        y = bf.observations[i]
        if ismissing(y)
            return ssm_PX(bf.ssm, θ, t, xp)
        else
            return ssm_proposal(bf.ssm, θ, proposal_parameters(bf.ssm), t, xp, y)
        end
    else
        return ssm_PX(bf.ssm, θ, t, xp)
    end
end

Base.@propagate_inbounds fkm_sup_logpdf_Mt(bf::BootstrapFilter, θ, t::Integer, x) = ssm_sup_logpdf_PX(bf.ssm, θ, t, x)

Base.@propagate_inbounds function fkm_logG0(bf::BootstrapFilter{T_X, T_Y}, θ, x) where {T_X, T_Y}
    t = initial_time(bf.ssm)
    i = observations_t2i(bf, t)
    y = bf.observations[i]
    if ismissing(y)
        return 0.0
    elseif convert(Bool, has_proposal(bf.ssm)) # maybe static
        # TODO proposal computed twice! we really should give up on Feynman-Kac formulation
        PX = ssm_PX0(bf.ssm, θ)
        PX_proposal = ssm_proposal0(bf.ssm, θ, proposal0_parameters(bf.ssm), y)
        obs_dist = ssm_PY(bf.ssm, θ, t, x)
        return convert(Float64,
            logpdf(obs_dist, y::T_Y) + logpdf(PX, x) - logpdf(PX_proposal, x)
        )
    else
        obs_dist = ssm_PY(bf.ssm, θ, t, x)
        return convert(Float64, logpdf(obs_dist, y::T_Y))
    end
end

Base.@propagate_inbounds function fkm_logGt(bf::BootstrapFilter{T_X, T_Y}, θ, t::Integer, xp, x) where {T_X, T_Y}
    i = observations_t2i(bf, t)
    y = bf.observations[i]
    if ismissing(y)
        return 0.0
    elseif convert(Bool, has_proposal(bf.ssm)) # maybe static
        PX = ssm_PX(bf.ssm, θ, t, xp)
        PX_proposal = ssm_proposal(bf.ssm, θ, proposal_parameters(bf.ssm), t, xp, y)
        obs_dist = ssm_PY(bf.ssm, θ, t, x)
        return convert(Float64,
            logpdf(obs_dist, y::T_Y) + logpdf(PX, x) - logpdf(PX_proposal, x)
        )
    else
        obs_dist = ssm_PY(bf.ssm, θ, t, x)
        return convert(Float64, logpdf(obs_dist, y::T_Y))
    end
end
