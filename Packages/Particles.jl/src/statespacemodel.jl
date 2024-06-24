"""
    StateSpaceModel{T_X, T_Y}
Abstract type for state-space models with state of type `T_X` and observations of type `T_Y`.
"""
abstract type StateSpaceModel{T_X, T_Y} end

"""
    parameter_type(m::StateSpaceModel)
Returns the type of the parameters object of the given state-space model.
Defaults to `Nothing`.
"""
parameter_type(::StateSpaceModel) = Nothing

"""
    parameter_template(m::StateSpaceModel)
Returns a template for the parameters object of the given state-space model.
Defaults to `nothing`.
"""
parameter_template(::StateSpaceModel) = nothing

"""
    isparameter(m::StateSpaceModel, θ)
Returns `true` if `θ` is a valid parameter object for the given state-space model.
"""
isparameter(ssm::StateSpaceModel, θ) = isa(θ, parameter_type(ssm))

"""
    ADBackend(m::StateSpaceModel)
Returns the preferred `ADBackend` for the given `StateSpaceModel`.
"""
ADBackend(ssm::StateSpaceModel) = error("please specify a preferred ADBackend for this StateSpaceModel")

"""
   initial_time(m::StateSpaceModel)
Initial time for the state-space model.
Defaults to `1`.
"""
initial_time(::StateSpaceModel) = 1

"""
    ssm_PX0(m::StateSpaceModel, θ)
Distribution of the initial state.
The parameter object `θ` defaults to `nothing` if not specified.
Must return a `Distributions.Sampleable` or a `Distributions.Distribution`.
"""
function ssm_PX0 end

"""
    ssm_PX(m::StateSpaceModel, θ, t::Integer, xp)
Transition kernel, i.e., distribution of the state at time `t`
conditioned on the value of the state at time `t - 1` being `xp`.
The parameter object `θ` defaults to `nothing` if not specified.
Must return a `Distributions.Sampleable` or a `Distributions.Distribution`.
"""
function ssm_PX end

"""
    ssm_sup_logpdf_PX(m::StateSpaceModel, θ, t::Integer, x)
Returns an upper bound for the transition kernel density at time `t`,
i.e., a value `U` such that `logpdf(ssm_PX(m, θ, t, xp), x) < U`
for all possible states `xp`.
Required for smoothing algorithms based on rejection.
"""
function ssm_sup_logpdf_PX end

"""
    ssm_PY(m::StateSpaceModel, θ, t::Integer, x)
Distribution of the observation at time `t` given the hidden state `x`.
The parameter object `θ` defaults to `nothing` if not specified.
Must return a `Distributions.Sampleable` or a `Distributions.Distribution`.
"""
function ssm_PY end

has_proposal(::StateSpaceModel) = static(false)
proposal_parameters(::StateSpaceModel) = nothing
proposal0_parameters(::StateSpaceModel) = nothing
reset_proposal_parameters!(::StateSpaceModel) = nothing
reset_proposal0_parameters!(::StateSpaceModel) = nothing
function ssm_proposal0 end
function ssm_proposal end

# No-parameter versions
function ssm_PX0(m::StateSpaceModel)
    parameter_type(m) === Nothing || error("if parameters object is omitted, its type must be Nothing")
    return fkm_M0(m, nothing)
end
function ssm_PX(m::StateSpaceModel, t::Integer, xp)
    parameter_type(m) === Nothing || error("if parameters object is omitted, its type must be Nothing")
    return ssm_PX(m, nothing, t, xp)
end
function ssm_PY(m::StateSpaceModel, t::Integer, x)
    parameter_type(m) === Nothing || error("if parameters object is omitted, its type must be Nothing")
    return ssm_PY(m, nothing, t, x)
end

########################################################################################################################################

struct ReplacedPX0SSM{T_X, T_Y, SSM <: StateSpaceModel{T_X, T_Y}, F} <: StateSpaceModel{T_X, T_Y}
    ssm::SSM
    PX0::F
end

parameter_type(ssm_wrapped::ReplacedPX0SSM) = parameter_type(ssm_wrapped.ssm)
parameter_template(ssm_wrapped::ReplacedPX0SSM) = parameter_template(ssm_wrapped.ssm)
isparameter(ssm_wrapped::ReplacedPX0SSM, θ) = isparameter(ssm_wrapped.ssm, θ)

ADBackend(ssm_wrapped::ReplacedPX0SSM) = ADBackend(ssm_wrapped.ssm)

initial_time(ssm_wrapped::ReplacedPX0SSM) = initial_time(ssm_wrapped.ssm)

ssm_PX(ssm_wrapped::ReplacedPX0SSM, θ, t::Integer, xp) = ssm_PX(ssm_wrapped.ssm, θ, t, xp)
ssm_sup_logpdf_PX(ssm_wrapped::ReplacedPX0SSM, θ, t::Integer, x) = ssm_sup_logpdf_PX(ssm_wrapped.ssm, θ, t, x)
ssm_PY(ssm_wrapped::ReplacedPX0SSM, θ, t::Integer, x) = ssm_PY(ssm_wrapped.ssm, θ, t, x)

ssm_PX0(ssm_wrapped::ReplacedPX0SSM, θ) = ssm_wrapped.PX0(ssm_wrapped.ssm, θ)

replace_PX0(ssm::StateSpaceModel, PX0) = ReplacedPX0SSM(ssm, PX0)
function replace_PX0(ssm::StateSpaceModel{T_X}, particles::AbstractVector{T_X}, weights::AbstractVector{<:Real}) where {T_X}
    PX0 = CategoricalDistribution(particles, weights)
    return replace_PX0(ssm, (ssm, θ) -> PX0)
end
function replace_PX0(ssm::StateSpaceModel{T_X}, value::T_X) where {T_X}
    PX0 = Deterministic(value)
    return replace_PX0(ssm, (ssm, θ) -> PX0)
end

struct CategoricalDistribution{T, W <: Real, VT <: AbstractVector{T}, VW <: AbstractVector{W}}
    values::VT
    cumweights::VW
    function CategoricalDistribution(values::AbstractVector{T}, weights::AbstractVector{<:Real}) where {T}
        isempty(values) && throw(ArgumentError("at least one value should be given"))
        length(values) == length(weights) || throw(ArgumentError("values and weights should have the same length"))
        all(≥(0), weights) || throw(ArgumentError("all weights must be ≥ 0"))
        W = typeof(inv(one(eltype(weights)))) # ensure division is possible
        cumweights = similar(weights, W) # ensures that cumweights is mutable (e.g., when weights is a static array)
        cumsum!(cumweights, weights)
        wsum = @inbounds(cumweights[end])
        iszero(wsum) && throw(ArgumentError("weights total must be > 0"))
        isfinite(wsum) || throw(ArgumentError("weights total must be finite"))
        cumweights ./= @inbounds(cumweights[end]) # normalize (may be needed, may be not)
        @assert isone(@inbounds(cumweights[end]))
        return new{T, W, typeof(values), typeof(cumweights)}(values, cumweights)
    end
end

function Random.rand(rng::AbstractRNG, catd::CategoricalDistribution)
    idx = searchsorted(catd.cumweights, rand(rng)).start
    return @inbounds catd.values[idx]
end
Random.rand(catd::CategoricalDistribution) = rand(Random.default_rng(), catd)

########################################################################################################################################

"""
    StateSpaceFeynmanKacModel{T_X, T_Y, SSM <: StateSpaceModel{T_X, T_Y}, OBS <: ObservedTrajectory{T_Y}} <: FeynmanKacModel{T_X}
Fenyman-Kac model associated to a state-space model of type `SMM` and to observations of type `Observations`.
"""
abstract type StateSpaceFeynmanKacModel{T_X, T_Y, SSM <: StateSpaceModel{T_X, T_Y}, OBS <: Observations{T_Y}} <: FeynmanKacModel{T_X} end

"""
    statespacemodel(m::StateSpaceFeynmanKacModel)
Return the state-space model associated to this Feynman-Kac model.
"""
function statespacemodel end

"""
    observations(m::StateSpaceFeynmanKacModel)
Return the observations for the state-space model associated to this Feynman-Kac model.
"""
function observations end

parameter_type(fkm::StateSpaceFeynmanKacModel) = parameter_type(statespacemodel(fkm))
parameter_template(fkm::StateSpaceFeynmanKacModel) = parameter_template(statespacemodel(fkm))
isparameter(fkm::StateSpaceFeynmanKacModel, θ) = isparameter(statespacemodel(fkm), θ)
ADBackend(fkm::StateSpaceFeynmanKacModel) = ADBackend(statespacemodel(fkm))
initial_time(fkm::StateSpaceFeynmanKacModel) = initial_time(statespacemodel(fkm))
observations_t2i(fkm::StateSpaceFeynmanKacModel, t::Integer) = t - (initial_time(fkm) - 1)
observations_i2t(fkm::StateSpaceFeynmanKacModel, i::Integer) = i + (initial_time(fkm) - 1)
function isready(fkm::StateSpaceFeynmanKacModel, t::Integer)
    obs = observations(fkm)
    i = observations_t2i(fkm, t)
    return checkbounds(Bool, obs, i)
end
function readyupto(fkm::StateSpaceFeynmanKacModel)
    obs = observations(fkm)
    i = lastindex(obs)
    return observations_i2t(fkm, i)
end
