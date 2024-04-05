struct QFunctionAf{V} <: ImmutableAdditiveFunction
    p::V
end
QFunctionAf() = QFunctionAf(nothing)

function check(addfun::QFunctionAf, fkm::FeynmanKacModel)
    isa(fkm, StateSpaceFeynmanKacModel) || error("QFunctionAf additive function can only be computed for StateSpaceFeynmanKacModels")
    isnothing(addfun.p) || isparameter(fkm, addfun.p) || error("If a parameter set is provided to QFunctionAf, it must be compatible with the StateSpaceFeynmanKacModels")
    return nothing
end

return_type(::QFunctionAf, ::StateSpaceFeynmanKacModel) = Float64

Base.@propagate_inbounds function (addfun::QFunctionAf)(fkm::StateSpaceFeynmanKacModel, θ, x, cache::Nothing)
    ssm = fkm.ssm
    p = isnothing(addfun.p) ? θ : addfun.p
    value = convert(Float64, logpdf(ssm_PX0(ssm, p), x))
    t = initial_time(ssm)
    i = observations_t2i(fkm, t)
    y = fkm.observations[i]
    if !ismissing(y)
        value += convert(Float64, logpdf(ssm_PY(ssm, p, t, x), y))
    end
    return value
end

Base.@propagate_inbounds function (addfun::QFunctionAf)(fkm::StateSpaceFeynmanKacModel, θ, t::Integer, xp, x, cache::Nothing)
    ssm = fkm.ssm
    p = isnothing(addfun.p) ? θ : addfun.p
    value = convert(Float64, logpdf(ssm_PX(ssm, p, t, xp), x))
    i = observations_t2i(fkm, t)
    y = fkm.observations[i]
    if !ismissing(y)
        value += convert(Float64, logpdf(ssm_PY(ssm, p, t, x), y))
    end
    return value
end

QFunction(p=nothing, method::AfsMethod=DEFAULT_AFSMETHOD()) = AdditiveFunctionSmoother(QFunctionAf(p), method)
QFunction(method::AfsMethod) = QFunction(nothing, method)

############################################################################

struct QGradientAf{V, AD <: ADBackend} <: MutableAdditiveFunction
    p::V
    ad_backend::AD
end
QGradientAf(ad_backend::ADBackend) = QGradientAf(nothing, ad_backend) # In this case the gradient of the Q-function is the score

const QResult{V} = GradientDiffResult{Float64, V}
const make_QResult = make_DiffResult

function check(addfun::QGradientAf, fkm::FeynmanKacModel)
    isa(fkm, StateSpaceFeynmanKacModel) || error("QGradientAf additive function can only be computed for StateSpaceFeynmanKacModels")
    parameter_type(fkm) <: AbstractVector || error("In order to use the QGradientAf additive function, the parameter vector must be an AbstractVector. This is to make automatic differentiation work.")
    isnothing(addfun.p) || isparameter(fkm, addfun.p) || error("If a parameter set is provided to QFunctionAf, it must be compatible with the StateSpaceFeynmanKacModels")
    return nothing
end

function return_type(addfun::QGradientAf, fkm::StateSpaceFeynmanKacModel)
    V = _return_type(addfun, fkm)
    return QResult{V}
end
_return_type(addfun::QGradientAf, fkm::StateSpaceFeynmanKacModel) = _return_type(addfun, parameter_type(fkm))
_return_type(::QGradientAf, ::Type{V}) where {V <: AbstractVector} = V
_return_type(::QGradientAf, ::Type{V}) where {N, T, V <: SVector{N, T}} = MVector{N, T}

function template_maker(addfun::QGradientAf, fkm::FeynmanKacModel)
    V = _return_type(addfun, fkm)
    if V <: MVector
        return () -> DiffResults.DiffResult(0.0, similar(V))::QResult{V}
    else
        return () -> DiffResults.DiffResult(0.0, parameter_template(fkm))::QResult{V}
    end
end

Base.@propagate_inbounds function (addfun::QGradientAf)(out::QResult, fkm::StateSpaceFeynmanKacModel, θ, x, cache::Nothing)
    ssm = fkm.ssm
    p = isnothing(addfun.p) ? θ : addfun.p
    t = initial_time(ssm)
    i = observations_t2i(fkm, t)
    y = fkm.observations[i]
    if ismissing(y)
        let f = θ -> logpdf(ssm_PX0(ssm, θ), x)
            gradient!(out, f, p, addfun.ad_backend)
        end
    else
        let f = θ -> logpdf(ssm_PX0(ssm, θ), x) + logpdf(ssm_PY(ssm, θ, t, x), y)
            gradient!(out, f, p, addfun.ad_backend)
        end
    end
    return out
end

Base.@propagate_inbounds function (addfun::QGradientAf)(out::QResult, fkm::StateSpaceFeynmanKacModel, θ, t::Integer, xp, x, cache::Nothing)
    ssm = fkm.ssm
    p = isnothing(addfun.p) ? θ : addfun.p
    i = observations_t2i(fkm, t)
    y = fkm.observations[i]
    if ismissing(y)
        let f = θ -> logpdf(ssm_PX(ssm, θ, t, xp), x)
            gradient!(out, f, p, addfun.ad_backend)
        end
    else
        let f = θ -> logpdf(ssm_PX(ssm, θ, t, xp), x) + logpdf(ssm_PY(ssm, θ, t, x), y)
            gradient!(out, f, p, addfun.ad_backend)
        end
    end
    return out
end

QGradient(p, ad_backend::ADBackend, method::AfsMethod=DEFAULT_AFSMETHOD()) = AdditiveFunctionSmoother(QGradientAf(p, ad_backend), method)
QGradient(ad_backend::ADBackend, method::AfsMethod=DEFAULT_AFSMETHOD()) = QGradient(nothing, ad_backend, method)

# Boring QResult methods below

@inline function af_zero!(::QGradientAf, x::QResult)
    x.value = 0.0
    T = eltype(x.derivs[1])
    fill!(x.derivs[1], zero(T))
    return x
end

@inline function af_add!(::QGradientAf, x::QResult, y::QResult)
    # @. x += y
    x.value += y.value
    @turbo @. x.derivs[1] += y.derivs[1]
    return x
end

@inline function af_add!(::QGradientAf, x::QResult, alpha::Real, y1::QResult, y2::QResult)
    # @. x = muladd(alpha, y1 + y2, x)
    x.value = muladd(alpha, y1.value + y2.value, x.value)
    @turbo @. x.derivs[1] = muladd(alpha, y1.derivs[1] + y2.derivs[1], x.derivs[1])
    return x
end

@inline function af_add!(::QGradientAf, x::QResult, alpha::Real, y::QResult, ::Nothing)
    # @. x = muladd(alpha, y, x)
    x.value = muladd(alpha, y.value, x.value)
    @turbo @. x.derivs[1] = muladd(alpha, y.derivs[1], x.derivs[1])
    return x
end

@inline function af_add!(::QGradientAf, x::QResult, y1::QResult, y2::QResult)
    # @. x += y1 + y2
    x.value += y1.value + y2.value
    @turbo @. x.derivs[1] += y1.derivs[1] + y2.derivs[1]
    return x
end

@inline function af_set!(::QGradientAf, x::QResult, y1::QResult, y2::QResult)
    # @. x = y1 + y2
    x.value = y1.value + y2.value
    @turbo @. x.derivs[1] = y1.derivs[1] + y2.derivs[1]
    return x
end

@inline function af_div!(::QGradientAf, x::QResult, alpha::Real)
    # @. x /= alpha
    x.value /= alpha
    @turbo @. x.derivs[1] /= alpha
    return x
end
