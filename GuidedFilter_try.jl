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
using AdvancedMH
using MCMCChains
using BenchmarkTools



# define the linear Gauss model without proposal distributions
struct LinearGaussBF <: StateSpaceModel{Float64, Float64}
    rho::Float64
    sigmaX::Float64
    sigmaY::Float64
    sigma0::Float64
end
LinearGaussBF(; rho=0.9, sigmaX=1.0, sigmaY=0.2, sigma0=sigmaX/sqrt(1-rho^2)) = LinearGaussBF(rho, sigmaX, sigmaY, sigma0)

Particles.ssm_PX0(ssm::LinearGaussBF, θ::AbstractVector{<:Real}) = Normal(0.0, θ[2]/sqrt(1-θ[1]^2))
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

Particles.ssm_PX0(ssm::LinearGaussGuided, θ::AbstractVector{<:Real}) = Normal(0.0, θ[2]/sqrt(1-θ[1]^2))
Particles.ssm_PX(ssm::LinearGaussGuided, θ::AbstractVector{<:Real}, t::Integer, xp::Real) = Normal(θ[1]*xp, θ[2])
Particles.ssm_PY(ssm::LinearGaussGuided, θ::AbstractVector{<:Real}, t::Integer, x::Real) = Normal(x, θ[3])

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

T = 1000
θ_true = Particles.parameter_template(ssm_bf)
x_bf, data = rand(ssm_bf, θ_true, T)
# hopefully this is a valid data creation
bf = BootstrapFilter(ssm_bf, data)

ssm_guided = LinearGaussGuided(sigmaX=1., sigmaY=2.0, rho=.9)
x_guided, data_guided = rand(ssm_guided, θ_true, 1000)
gf = BootstrapFilter(ssm_guided, data)

# plot data

plt = scatter(1:T, x_bf, mode="markers", name="state")
scatter!(plt, 1:T, data, mode="markers", name="observation")



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

plt_df_bf = logp_vs_nparticles(ssm_bf, data, nparticles, nothing; nruns=nruns)
plt_df_guided = logp_vs_nparticles(ssm_guided, data, nparticles, nothing; nruns=nruns)

var_bf = [var(plt_df_bf.logp[plt_df_bf.nparticles .== "$n"]) for n in nparticles]
var_guided = [var(plt_df_guided.logp[plt_df_guided.nparticles .== "$n"]) for n in nparticles]

findall(var_bf .> var_guided)
# variances are similar or even better for the Bootstrapfilter, so how to test whether guided really works better.

# perform inference with both filters
function run_smc_jl(fk::FeynmanKacModel{T}, N::Integer) where {T}
    f = SMC(
        fk, Particles.parameter_template(fk), N,
        
        ParticleHistoryLength(FullHistory()),
        
        (mean_and_var=RunningSummary(MeanAndVariance(), FullHistory()), ),
        #(mean_and_var=RunningSummary(MeanAndVariance(), StaticFiniteHistory{3}()), ),
        #(mean_and_var=MeanAndVariance(), ),
        #NamedTuple(),
        
        AdaptiveResampling(SystematicResampling(), 0.5),
    )
    offlinefilter!(f)
    return f
end

nparticles = 1000
bootstrap = @btime run_smc_jl(bf, nparticles)

guided = @btime run_smc_jl(gf, nparticles)

bootstrap.history_pf.logCnorm[end], guided.history_pf.logCnorm[end]
# log-likelihood and speed are the same basically.

# Offline summary computation
let step = END-2
    offline_value = Particles.compute_summary(bootstrap, OfflineSummary(MeanAndVariance()), step)
    if hasproperty(bootstrap.history_run, :mean_and_var)
        @assert offline_value == bootstrap.history_run.mean_and_var[step]
    end
    offline_value
end

let step = END-2
    offline_value = Particles.compute_summary(guided, OfflineSummary(MeanAndVariance()), step)
    if hasproperty(guided.history_run, :mean_and_var)
        @assert offline_value == guided.history_run.mean_and_var[step]
    end
    offline_value
end

if hasproperty(bootstrap.history_run, :mean_and_var)
    bootstrap_means = getproperty.(bootstrap.history_run.mean_and_var, :mean)
    bootstrap_vars = getproperty.(bootstrap.history_run.mean_and_var, :var)
end;
if hasproperty(guided.history_run, :mean_and_var)
    guided_means = getproperty.(guided.history_run.mean_and_var, :mean)
    guided_vars = getproperty.(guided.history_run.mean_and_var, :var)
end;

# plot the means and variances

plot((1:T)[end-length(bootstrap_means)+1:end], bootstrap_means, mode="lines", line_color="blue", name="particle (jl) filter mean")
plot!((1:T)[end-length(bootstrap_means)+1:end], bootstrap_means.+sqrt.(bootstrap_vars), mode="lines", line_color="lightblue", name="particle (jl) filter mean + 1sd")
plot!((1:T)[end-length(bootstrap_means)+1:end], bootstrap_means.-sqrt.(bootstrap_vars), mode="lines", line_color="lightblue", name="particle (jl) filter mean - 1sd")
plot!((1:T)[end-length(guided_means)+1:end], guided_means, mode="lines", line_color="red", name="particle (jl) filter mean")
plot!((1:T)[end-length(guided_means)+1:end], guided_means.+sqrt.(guided_vars), mode="lines", line_color="orange", name="particle (jl) filter mean + 1sd")
plot!((1:T)[end-length(guided_means)+1:end], guided_means.-sqrt.(guided_vars), mode="lines", line_color="orange", name="particle (jl) filter mean - 1sd")
# scatter!(1:T, x_bf, mode="markers", name="state")
scatter!(1:T, data, mode="markers", name="observation", alpha=0.5)

# Both filters produce similar trajectories, so we should look at inference I guess

nparticles = 1000
true_parameter = [ssm_bf.rho, ssm_bf.sigmaX, ssm_bf.sigmaY]
initial_parameter = [1.0, 0.9, 0.15]
parameter_names = ["rho", "sigmaX", "sigmaY"]

llh_bf = LogLikelihood_NoGradient(ssm_bf, data, nparticles=nparticles)
model_bf = DensityModel(llh_bf)
llh_guided = LogLikelihood_NoGradient(ssm_guided, data, nparticles=nparticles)
model_guided = DensityModel(llh_guided)

sampler = RWMH([Normal(0.0, 0.05), Normal(0.0, 0.05), Normal(0.0, 0.05)])

chain_bf = sample(model_bf, sampler, 100; init_params=initial_parameter, param_names=parameter_names, chain_type=Chains)

chain_guided = sample(model_guided, sampler, 100; init_params=initial_parameter, param_names=parameter_names, chain_type=Chains)
# noch probleme mit negativen Parametern irgendwie....
# Wie war das nochmal mit Parameter bounds implementieren?


reset!(llh_bf.pf, [1.0, 1.0, 1.0])

test_par = [1.0, 1.0, 1.0]
llh_bf(test_par)

