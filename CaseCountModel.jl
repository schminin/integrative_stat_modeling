using Particles 
using Distributions
using Random
using StaticDistributions
using StaticArrays
using StatsFuns
using StatsPlots
using Plots
using AdvancedMH
using MCMCChains
using DataFrames

include("CaseCountDistributions.jl")

# set dimension of the state space
const m_Λ = 13
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
    pi_ua: binomial parameter for the underascertainment distribution in PY
    """
    m_Λ = ssm.m_Λ
    state_space_dimension = 3*m_Λ + 2
    shape, scale = 1+exp(θ[1]), exp(θ[2])
    pi_ua = StatsFuns.logistic(θ[3])
    Y_init = ssm.I_init # initiallay infected people
    X_init = zeros(state_space_dimension)
    # X_init[1:m_Λ+1] = rand(Multinomial(Y_init, fill(1/(m_Λ+1), m_Λ+1)))
    X_init[1:m_Λ+1] = repeat([Int(round(Y_init/(m_Λ+1)))], m_Λ+1)
    X_init[1] += Y_init-sum(X_init[1:m_Λ+1])
    X_init[end] = 1 # initial reproduction number
    X = X_init
    for i in 1:m_Λ
        X = rand(CaseCountDistribution(SVector(X...), SVector{3, Float64}(shape, scale, pi_ua)))
    end

    return Deterministic(X)
end

function Particles.ssm_PX(ssm::CaseCountModel, θ::AbstractVector{<:Real}, t::Integer, xp::SVector{state_space_dimension, Float64})
    shape, scale = 1+exp(θ[1]), exp(θ[2])
    pi_ua = StatsFuns.logistic(θ[3])

    return CaseCountDistribution(xp, SVector{3, Float64}(shape, scale, pi_ua))
end

function Particles.ssm_PY(ssm::CaseCountModel, θ::AbstractVector{<:Real}, t::Integer, x::SVector{state_space_dimension, Float64})
    pi_ua = StatsFuns.logistic(θ[3])
    M_t  = sum(x[ssm.m_Λ+2:2*(ssm.m_Λ+1)]) 
    return SIndependent(
        Binomial(M_t, pi_ua)
    )
end


struct LogPosterior{T_SMC <: SMC, T_CACHE}
    pf::T_SMC
    cache::T_CACHE
    function LogPosterior(ssm::StateSpaceModel, data, nparticles::Integer)
        bf = BootstrapFilter(ssm, data)
        pf = SMC(
            bf, Particles.parameter_template(ssm), nparticles,
            ParticleHistoryLength(; logCnorm=StaticFiniteHistory{1}()),
            NamedTuple(),
            AdaptiveResampling(SystematicResampling(), 0.5),
        )
        cache = Particles.SMCCache(pf)
        return new{typeof(pf), typeof(cache)}(pf, cache)
    end
end

function (logp::LogPosterior)(theta) #::Float64
    reset!(logp.pf, theta)
    offlinefilter!(logp.pf, logp.cache)
    if isnan(logp.pf.history_pf.logCnorm[end])
        println("NaN posterior")
        # return logp.pf
        return -Inf
    end
    return logp.pf.history_pf.logCnorm[end]
end

function logp_vs_nparticles(ssm::StateSpaceModel, data, nparticles::AbstractVector{<:Integer}, theta; nruns::Integer=10, kwargs...)
    x = Vector{String}(undef, length(nparticles) * nruns)
    y = Vector{Float64}(undef, length(nparticles) * nruns)
    k = 1
    for n in nparticles
        logp = LogPosterior(ssm, data, n)
        for _ in 1:nruns
            @inbounds x[k] = string(convert(Int, n))
            @inbounds y[k] = logp(theta)
            k += 1
        end
    end
    plt_df = DataFrame(nparticles=x, logp=y)
    display(@df plt_df violin(:nparticles, :logp))
    return plt_df
end

