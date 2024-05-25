#=
Try out the guided filter in Julia using the easy example of LinearGauss from Lorenzos example notebooks.
=#

using Particles 
using Distributions
using Random
using StaticDistributions
using StaticArrays
using Plots
using StatsFuns
using StatsPlots
using DataFrames



# define the linear Gauss model without proposal distributions
struct LinearGaussBF <: StateSpaceModel{Float64, Float64}
    rho::Float64
    sigmaX::Float64
    sigmaY::Float64
    sigma0::Float64
end
LinearGaussBF(; rho=0.9, sigmaX=1.0, sigmaY=0.2, sigma0=sigmaX/sqrt(1-rho^2)) = LinearGaussBF(rho, sigmaX, sigmaY, sigma0)

# Particles.parameter_type(::LinearGaussBF) = Vector{Float64}
# Particles.parameter_template(::LinearGaussBF) = Float64[0.9, 1.0, 0.2]
# Particles.isparameter(::LinearGaussBF, θ) = isa(θ, Vector{Float64}) && length(θ) == 3

Particles.ssm_PX0(ssm::LinearGaussBF, ::Nothing) = Normal(0.0, ssm.sigma0)
Particles.ssm_PX(ssm::LinearGaussBF, ::Nothing, t::Integer, xp::Real) = Normal(ssm.rho*xp, ssm.sigmaX)
Particles.ssm_PY(ssm::LinearGaussBF, ::Nothing, t::Integer, x::Real) = Normal(x, ssm.sigmaY)


# define the linear Gauss model with proposal distributions
struct LinearGaussGuided <: StateSpaceModel{Float64, Float64}
    rho::Float64
    sigmaX::Float64
    sigmaY::Float64
    sigma0::Float64
end
LinearGaussGuided(; rho=0.9, sigmaX=1.0, sigmaY=0.2, sigma0=sigmaX/sqrt(1-rho^2)) = LinearGaussGuided(rho, sigmaX, sigmaY, sigma0)

# Particles.parameter_type(::LinearGaussGuided) = Vector{Float64}
# Particles.parameter_template(::LinearGaussGuided) = Float64[0.9, 1.0, 0.2]
# Particles.isparameter(::LinearGaussGuided, θ) = isa(θ, Vector{Float64}) && length(θ) == 3

Particles.ssm_PX0(ssm::LinearGaussGuided, ::Nothing) = Normal(0.0, ssm.sigma0)
Particles.ssm_PX(ssm::LinearGaussGuided, ::Nothing, t::Integer, xp::Real) = Normal(ssm.rho*xp, ssm.sigmaX)
Particles.ssm_PY(ssm::LinearGaussGuided, ::Nothing, t::Integer, x::Real) = Normal(x, ssm.sigmaY)

has_proposal(::LinearGaussGuided) = static(true)
proposal_parameters(::LinearGaussGuided) = nothing
proposal0_parameters(::LinearGaussGuided) = nothing

function Particles.ssm_proposal0(ssm::LinearGaussGuided, ::Nothing, y::Real)
    sig2post = 1. / (1. / ssm.sigma0^2 + 1. / ssm.sigmaY^2)
    mupost = sig2post * (y / ssm.sigmaY^2)
    return Normal(mupost, np.sqrt(sig2post))
end

function Particles.ssm_proposal(ssm::LinearGaussGuided, ::Nothing, t::Integer, xp::Real, y::Real)
    sig2post = 1. / (1. / ssm.sigmaX^2 + 1. / ssm.sigmaY^2)
    mupost = sig2post * (ssm.rho * xp / ssm.sigmaX^2
                            + y / ssm.sigmaY^2)
    return Normal(mupost, np.sqrt(sig2post))
end


# define StateSpaceModels
ssm_bf = LinearGaussBF(sigmaX=1., sigmaY=2.0, rho=.9)
x_bf, data_bf = rand(ssm_bf, 100)
bf = BootstrapFilter(ssm_bf, data_bf)

ssm_guided = LinearGaussGuided(sigmaX=1., sigmaY=2.0, rho=.9)
x_guided, data_guided = rand(ssm_guided, 100)
gf = BootstrapFilter(ssm_guided, data_guided)

# plot data against each other to compare
sim_bf = repeat([rand(ssm_bf, 100)[2] for i in 1:1000])
sim_guided = repeat([rand(ssm_guided, 100)[2] for i in 1:1000])

plot(mean(sim_bf), label="Data BF")
plot!(mean(sim_guided), label="Data Guided")

# not completely agreeing but should be the same process, all centered at 0.

# run Bf vs. Guided Filter and plot violin Plots
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

nruns = 200
nparticles = [10, 20, 50, 100, 200, 500, 1000]

plt_df_bf = logp_vs_nparticles(ssm_bf, data_bf, nparticles, nothing; nruns=nruns)
plt_df_guided = logp_vs_nparticles(ssm_guided, data_guided, nparticles, nothing; nruns=nruns)

var_bf = [var(plt_df_bf.logp[plt_df_bf.nparticles .== "$n"]) for n in nparticles]
var_guided = [var(plt_df_guided.logp[plt_df_guided.nparticles .== "$n"]) for n in nparticles]

findall(var_bf .> var_guided)
# variances are similar or even better for the Bootstrapfilter, so how to test whether guided really works better.

