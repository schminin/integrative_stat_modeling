using Particles 
using Distributions
using Random
using StaticDistributions
using StaticArrays

include("CaseCountDistributions.jl")

# set dimension of the state space
const m_Λ = 20
const state_space_dimension = 3*m_Λ + 2


struct CaseCountModel <: StateSpaceModel{SVector{state_space_dimension, Float64}, SVector{1, Int64}}
    m_Λ::Int
    I_init::Int
    function CaseCountModel(;m_Λ::Int, I_init::Int)
        new(m_Λ, I_init)
    end
end

Particles.parameter_type(::CaseCountModel) = Vector{Float64} # TODO allow different parameter type: add a function that only checks type isparametertypecorrect or something
Particles.parameter_template(::CaseCountModel) = Float64[log(2), log(2), log(0.5)]
Particles.isparameter(::CaseCountModel, θ) = isa(θ, Vector{Float64}) && length(θ) == 3



function Particles.ssm_PX0(ssm::CaseCountModel, θ::AbstractVector{<:Real})
    """
    pi_ua: binomial parameter for the udnerascertainment distribution in PY
    """
    m_Λ = ssm.m_Λ
    state_space_dimension = 3*m_Λ + 2
    shape, scale = 1+exp(θ[1]), exp(θ[2])
    pi_ua = exp(θ[3])
    Y_init = round(Int,(1+rand(Uniform(0, 1)))*ssm.I_init)
    X_init = zeros(state_space_dimension)
    X_init[1:m_Λ+1] = rand(Multinomial(Y_init, fill(1/(m_Λ+1), m_Λ+1)))
    X_init[end] = rand(Uniform(0.8, 1.5)) # initial reproduction number
    X = X_init
    for i in 1:m_Λ
        X = rand(CaseCountDistribution(SVector(X...), SVector{3, Float64}(shape, scale, pi_ua)))
    end

    return Deterministic(X)
end

function Particles.ssm_PX(ssm::CaseCountModel, θ::AbstractVector{<:Real}, t::Integer, xp::SVector{state_space_dimension, Float64})
    shape, scale = 1+exp(θ[1]), exp(θ[2])
    pi_ua = exp(θ[3])

    return CaseCountDistribution(xp, SVector{3, Float64}(shape, scale, pi_ua))
end

function Particles.ssm_PY(ssm::CaseCountModel, θ::AbstractVector{<:Real}, t::Integer, x::SVector{state_space_dimension, Float64})
    pi = exp(θ[3])
    M_t  = sum(x[ssm.m_Λ+2:2*(ssm.m_Λ+1)]) 
    return SIndependent(
        Binomial(M_t, pi)
    )
end


ssm = CaseCountModel(m_Λ = 20, I_init = 100)
theta = Particles.parameter_template(ssm)

# test
x0_gen = Particles.ssm_PX0(ssm, theta)
x0 = rand(x0_gen)


T = 20
xtrue, data_full = rand(ssm, theta, T)

xtrue

m_Λ