"""
    FeynmanKacModel{T}
Abstract type for Feynman-Kac models with state of type `T`.
"""
abstract type FeynmanKacModel{T} end

"""
    parameter_type(m::FeynmanKacModel)
Returns the type of the parameters object of the given Feynman-Kac model.
Defaults to `Nothing`.
"""
parameter_type(::FeynmanKacModel) = Nothing

"""
    parameter_template(m::FeynmanKacModel)
Returns a template for the parameters object of the given Feynman-Kac model.
Defaults to `nothing`.
"""
parameter_template(::FeynmanKacModel) = nothing

"""
    isparameter(m::FeynmanKacModel, θ)
Returns `true` if `θ` is a valid parameter object for the given Feynman-Kac model.
"""
isparameter(fkm::FeynmanKacModel, θ) = isa(θ, parameter_type(fkm))

"""
    ADBackend(m::FeynmanKacModel)
Returns the preferred `ADBackend` for the given `FeynmanKacModel`.
"""
ADBackend(::FeynmanKacModel) = error("please specify a preferred ADBackend for this FeynmanKacModel")

"""
    initial_time(m::FeynmanKacModel)
Initial time for the Feynman-Kac model.
Defaults to `1`.
"""
initial_time(::FeynmanKacModel) = 1

"""
    isready(m::FeynmanKacModel, t::Integer)
Returns `true` if the Feynman-Kac model can be run
up to time `t` (included).
"""
function isready end

"""
    readyupto(m::FeynmanKacModel, t::Integer)
Returns the last time index `t` until which the Feynman-Kac model can be run.
Returns `initial_time(m) - 1` if no times are ready.
"""
function readyupto(m::FeynmanKacModel)
    t = initial_time(m)::Int
    while isready(m, t)
        t += 1
    end
    return t - 1
end

"""
    fkm_M0(m::FeynmanKacModel, θ)
Distribution of the initial state.
The parameter object `θ` defaults to `nothing` if not specified.
Must return a `Distributions.Sampleable` or a `Distributions.Distribution`.
"""
function fkm_M0 end

"""
    fkm_Mt(m::FeynmanKacModel, θ, t::Integer, xp)
Transition kernel, i.e., distribution of the state at time `t`
conditioned on the value of the state at time `t - 1` being `xp`.
The parameter object `θ` defaults to `nothing` if not specified.
Must return a `Distributions.Sampleable` or a `Distributions.Distribution`.
"""
function fkm_Mt end

"""
    fkm_sup_logpdf_Mt(m::FeynmanKacModel, θ, t::Integer, x)
Returns an upper bound for the transition kernel density at time `t`,
i.e., a value `U` such that `logpdf(fkm_Mt(m, θ, t, xp), x) < U`
for all possible states `xp`.
Required for smoothing algorithms based on rejection.
"""
function fkm_sup_logpdf_Mt end

"""
    fkm_logG0(m::FeynmanKacModel, θ, x0)
Logarithm of weighting function at the initial time,
given initial state `x0`.
The parameter object `θ` defaults to `nothing` if not specified.
Must return a `Distributions.Sampleable` or a `Distributions.Distribution`.
"""
function fkm_logG0 end

"""
    fkm_logGt(m::FeynmanKacModel, θ, t::Integer, xp, x)
Logarithm of weighting function at time `t`,
given current state `x` and previous state `xp`.
The parameter object `θ` defaults to `nothing` if not specified.
Must return a `Distributions.Sampleable` or a `Distributions.Distribution`.
"""
function fkm_logGt end

# No-parameter versions
function fkm_M0(m::FeynmanKacModel)
    parameter_type(m) === Nothing || error("if parameters object is omitted, its type must be Nothing")
    return fkm_M0(m, nothing)
end
function fkm_Mt(m::FeynmanKacModel, t::Integer, xp)
    parameter_type(m) === Nothing || error("if parameters object is omitted, its type must be Nothing")
    return fkm_Mt(m, nothing, t, xp)
end
function fkm_logG0(m::FeynmanKacModel, x0)
    parameter_type(m) === Nothing || error("if parameters object is omitted, its type must be Nothing")
    return fkm_logG0(m::FeynmanKacModel, nothing, x0)
end
function fkm_logGt(m::FeynmanKacModel, t::Integer, xp, x)
    parameter_type(m) === Nothing || error("if parameters object is omitted, its type must be Nothing")
    return fkm_logGt(m::FeynmanKacModel, nothing, t, xp, x)
end
