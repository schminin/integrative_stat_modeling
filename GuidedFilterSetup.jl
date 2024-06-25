using Particles 
using Distributions
using Random
using StaticDistributions
using StaticArrays
using Plots
using StatsFuns
using StatsPlots
using DataFrames
using AdvancedMH
using MCMCChains
using MCMCChainsStorage
using JLD2
using HDF5
using PyCall
using BenchmarkTools



# define the linear Gauss model without proposal distributions
struct LinearGaussBF <: StateSpaceModel{Float64, Float64}
    rho::Float64
    sigmaX::Float64
    sigmaY::Float64
    sigma0::Float64
end
LinearGaussBF(; rho=0.9, sigmaX=1.0, sigmaY=0.2, sigma0=sigmaX/sqrt(1-rho^2)) = LinearGaussBF(rho, sigmaX, sigmaY, sigma0)

Particles.ssm_PX0(ssm::LinearGaussBF, θ::AbstractVector{<:Real}) = Normal(0.0, θ[2]/(1-θ[1]^2))
Particles.ssm_PX(ssm::LinearGaussBF, θ::AbstractVector{<:Real}, t::Integer, xp::Real) = Normal(θ[1]*xp, θ[2])
Particles.ssm_PY(ssm::LinearGaussBF, θ::AbstractVector{<:Real}, t::Integer, x::Real) = Normal(x, θ[3])

Particles.parameter_type(::LinearGaussBF) = Vector{Float64} # TODO allow different parameter type: add a function that only checks type isparametertypecorrect or something
Particles.parameter_template(::LinearGaussBF) = Float64[0.9, 1.0, 0.2]
Particles.isparameter(::LinearGaussBF, θ) = isa(θ, Vector{Float64}) && length(θ) == 3

# define the linear Gauss model with proposal distributions
struct LinearGaussGuided <: StateSpaceModel{Float64, Float64}
    rho::Float64
    sigmaX::Float64
    sigmaY::Float64
    sigma0::Float64
end
LinearGaussGuided(; rho=0.9, sigmaX=1.0, sigmaY=0.2, sigma0=sigmaX/sqrt(1-rho^2)) = LinearGaussGuided(rho, sigmaX, sigmaY, sigma0)

Particles.parameter_type(::LinearGaussGuided) = Vector{Float64} # TODO allow different parameter type: add a function that only checks type isparametertypecorrect or something
Particles.parameter_template(::LinearGaussGuided) = Float64[0.9, 1.0, 0.2]
Particles.isparameter(::LinearGaussGuided, θ) = isa(θ, Vector{Float64}) && length(θ) == 3

Particles.ssm_PX0(ssm::LinearGaussGuided, θ::AbstractVector{<:Real}) = Normal(0.0, θ[2]/(1-θ[1]^2))
Particles.ssm_PX(ssm::LinearGaussGuided, θ::AbstractVector{<:Real}, t::Integer, xp::Real) = Normal(θ[1]*xp, θ[2])
Particles.ssm_PY(ssm::LinearGaussGuided, θ::AbstractVector{<:Real}, t::Integer, x::Real) = Normal(x, θ[3])

has_proposal(::LinearGaussGuided) = static(true)
proposal_parameters(::LinearGaussGuided) = nothing
proposal0_parameters(::LinearGaussGuided) = nothing

function Particles.ssm_proposal0(ssm::LinearGaussGuided, θ::AbstractVector{<:Real}, y::Real)
    sig2post = 1. / (1. / (θ[2]/(1-θ[1]^2))^2 + 1. / θ[3]^2)
    mupost = sig2post * (y / θ[3]^2)
    return Normal(mupost, np.sqrt(sig2post))
end

function Particles.ssm_proposal(ssm::LinearGaussGuided, θ::AbstractVector{<:Real}, t::Integer, xp::Real, y::Real)
    sig2post = 1. / (1. / θ[2]^2 + 1. / θ[3]^2)
    mupost = sig2post * (θ[1] * xp / θ[2]^2 + y / θ[3]^2)
    return Normal(mupost, np.sqrt(sig2post))
end