struct ScoreAf{AD <: ADBackend} <: ArrayAdditiveFunction
    ad_backend::AD
end

function check(::ScoreAf, fkm::FeynmanKacModel)
    isa(fkm, StateSpaceFeynmanKacModel) || error("ScoreAf additive function can only be computed for StateSpaceFeynmanKacModels")
    parameter_type(fkm) <: AbstractVector || error("In order to use the ScoreAf additive function, the parameter vector must be an AbstractVector. This is to make automatic differentiation work.")
    return nothing
end

return_type(addfun::ScoreAf, fkm::StateSpaceFeynmanKacModel) = return_type(addfun, parameter_type(fkm))
return_type(::ScoreAf, ::Type{V}) where {V <: AbstractVector} = V
return_type(::ScoreAf, ::Type{V}) where {N, T, V <: SVector{N, T}} = MVector{N, T}

function template_maker(addfun::ScoreAf, fkm::FeynmanKacModel)
    T = return_type(addfun, fkm)
    if T <: MVector
        return () -> similar(T)::T
    else
        return () -> parameter_template(fkm)::T
    end
end

Base.@propagate_inbounds function (addfun::ScoreAf)(out::AbstractVector, fkm::StateSpaceFeynmanKacModel, θ, x, cache::Nothing)
    ssm = fkm.ssm
    t = initial_time(ssm)
    i = observations_t2i(fkm, t)
    y = fkm.observations[i]
    if ismissing(y)
        let f = θ -> logpdf(ssm_PX0(ssm, θ), x)
            gradient!(out, f, θ, addfun.ad_backend)
        end
    else
        let f = θ -> logpdf(ssm_PX0(ssm, θ), x) + logpdf(ssm_PY(ssm, θ, t, x), y)
            gradient!(out, f, θ, addfun.ad_backend)
        end
    end
    return out
end

Base.@propagate_inbounds function (addfun::ScoreAf)(out::AbstractVector, fkm::StateSpaceFeynmanKacModel, θ, t::Integer, xp, x, cache::Nothing)
    ssm = fkm.ssm
    i = observations_t2i(fkm, t)
    y = fkm.observations[i]
    if ismissing(y)
        let f = θ -> logpdf(ssm_PX(ssm, θ, t, xp), x)
            gradient!(out, f, θ, addfun.ad_backend)
        end
    else
        let f = θ -> logpdf(ssm_PX(ssm, θ, t, xp), x) + logpdf(ssm_PY(ssm, θ, t, x), y)
            gradient!(out, f, θ, addfun.ad_backend)
        end
    end
    return out
end

Score(ad_backend::ADBackend, method::AfsMethod=DEFAULT_AFSMETHOD()) = AdditiveFunctionSmoother(ScoreAf(ad_backend), method)
