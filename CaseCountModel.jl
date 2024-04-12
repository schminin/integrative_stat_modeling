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


ssm = CaseCountModel(m_Λ = 20, I_init = 100)
theta0 = Particles.parameter_template(ssm)

# test
x0_gen = Particles.ssm_PX0(ssm, theta0)
x0 = rand(x0_gen)

x_fail = zeros(state_space_dimension)
x_fail[end] = 0.2
rand(Particles.ssm_PX(ssm, theta0, 1, SVector(x_fail...)))
rand(Particles.ssm_PY(ssm, theta0, 1, SVector(x_fail...)))

T = 20
xtrue, data_full = rand(ssm, theta0, T)

xtrue

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
        return logp.pf
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
    @df plt_df violin(:nparticles, :logp)
    return plt_df
end


plt_df = logp_vs_nparticles(ssm, data_full, [100, 150, 200, 250, 300, 500, 1000], theta0; nruns=200)

findall(isnan.(y_plt))


violin(;x_plt, y_plt)

logp = LogPosterior(ssm, data_full, 100);
@time post_val = logp(theta0)

failed_hist = post_val

fieldnames(typeof(failed_hist))
failed_hist.history_pf.weights
findall(isnan.(failed_hist))
fieldnames(typeof(failed_hist.history_pf))

[sum(failed_hist.history_pf.particles[end][i] .> 0.0) for i in 1:100]




last_state = failed_hist.history_pf.particles[end][end]

logpdf(Particles.ssm_PY(ssm, theta0, 1, last_state), data_full[2])



model = DensityModel(logp)

spl = RWMH([Normal(0.0, 0.05), Normal(0.0, 0.05), Normal(0.0, 0.05)])
chain = sample(model, spl, 100; init_params=theta0, param_names=["shape", "scale", "pi_ua"], chain_type=Chains)