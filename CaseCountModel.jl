using Particles 
using Distributions
using Random
using StaticDistributions
using StaticArrays

include("CaseCountDistributions.jl")

# set dimension of the state space
const m_Λ = 4
const state_space_dimension = 3*m_Λ + 2


struct CaseCountModel <: StateSpaceModel{SVector{state_space_dimension, Float64}, SVector{1, Float64}}
    m_Λ::Int
end

Particles.parameter_type(::LotkaVolterra) = Vector{Float64} # TODO allow different parameter type: add a function that only checks type isparametertypecorrect or something
Particles.parameter_template(::LotkaVolterra) = Float64[log(2), log(2)]
Particles.isparameter(::LotkaVolterra, θ) = isa(θ, Vector{Float64}) && length(θ) == 2



function Particles.ssm_PX0(ssm::CaseCountModel, θ::AbstractVector{<:Real})
    shape, scale = 1+exp(θ[1]), exp(θ[2])
    initial_state = SVector{state_space_dimension, Float64}(4.0, 8.0, 7.0, 15.0, 3.0, 2.0, 3.0, 6.0, 2.0, 2.0, 7.0, 6.0, 2.0, 1.0)
    return SIndependent(
        Deterministic(initial_state)
    )
end

function Particles.ssm_PX(ssm::CaseCountModel, θ::AbstractVector{<:Real}, t::Integer, xp::SVector{state_space_dimension, Float64})
    shape, scale = 1+exp(θ[1]), exp(θ[2])

    return CaseCountDistribution(xp, SVector{2, Float64}(shape, scale))
end

function Particles.ssm_PY(ssm::CaseCountModel, θ::AbstractVector{<:Real}, t::Integer, x::SVector{state_space_dimension, Float64})
    
    return 
end