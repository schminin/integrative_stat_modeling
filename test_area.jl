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
using JLD2

include("CaseCountDistributions.jl")
include("CaseCountModel.jl")

# set dimension of the state space
const m_Λ = 13
const state_space_dimension = 3*m_Λ + 2


# check for the variacne in the initial state vectors

function initial_state_generation(ssm::CaseCountModel, θ::AbstractVector{<:Real})
    """
    pi_ua: binomial parameter for the underascertainment distribution in PY
    """
    m_Λ = ssm.m_Λ
    state_space_dimension = 3*m_Λ + 2
    shape, scale = 1+exp(θ[1]), exp(θ[2])
    pi_ua = exp(θ[3])
    Y_init = ssm.I_init # initiallay infected people
    X_init = zeros(state_space_dimension)
    X_init[1:m_Λ+1] = repeat([Int(round(Y_init/(m_Λ+1)))], m_Λ+1)
    X_init[1] += Y_init-sum(X_init[1:m_Λ+1])
    X_init[end] = 1 # initial reproduction number
    X = X_init
    R_history = [1.0]

    ω, ϕ = case_count_parameter_mapping(SVector(exp.(θ)...), m_Λ)
    Λ = infection_potential(X_init[1:m_Λ+1][1:end-1], ω) # old_Y[1:end-1]
    Λ_history = [Λ]

    for i in 1:m_Λ
        X = rand(CaseCountDistribution(SVector(X...), SVector{3, Float64}(shape, scale, pi_ua)))
        Λ = infection_potential(X[1:m_Λ+1][1:end-1], ω) # old_Y[1:end-1]

        push!(R_history, X[end])
        push!(Λ_history, Λ)
    end

    return X, R_history, Λ_history
end

ssm = CaseCountModel(m_Λ = m_Λ, I_init = 100)
theta0 = Particles.parameter_template(ssm)

initial_states_list = [initial_state_generation(ssm, theta0) for i in 1:10000]
x0_list = [x[1] for x in initial_states_list]
Λ_hist_list = [x[3] for x in initial_states_list]
R_hist_list = [x[2] for x in initial_states_list]
Rmean_list = [mean(x[2]) for x in initial_states_list]
R0_list = [x[end] for x in x0_list]
sum_curr_inf_list = [sum(x[1:m_Λ+1]) for x in x0_list]
Rstd_list = [std(x[2]) for x in initial_states_list]

poi_par_list = [Λ_hist_list[i].*R_hist_list[i] for i in eachindex(Λ_hist_list)]

plot(poi_par_list, legend=false, title = "Poisson Parameters")
plot(R_hist_list, legend=false, title = "R0 History")
plot(Λ_hist_list, legend=false, title = "Infection Potential")

plot(Rstd_list, sum_curr_inf_list, seriestype = :scatter, xlabel = "R0_mean", ylabel = "Current Infections", title = "Initial State Vectors")
plot(Rmean_list, sum_curr_inf_list, seriestype = :scatter, xlabel = "R0_std", ylabel = "Current Infections", title = "Initial State Vectors")

test_idxs = findall(Rmean_list .<1.0002 .&& Rmean_list .>0.9998)
test_histories = initial_states_list[test_idxs][2]
test_states = initial_states_list[test_idxs][1]
sum_curr_inf_test_list = sum_curr_inf_list[test_idxs]
Rmean_test_list = Rmean_list[test_idxs]
Rstd_test_list = Rstd_list[test_idxs]

plot(Rstd_test_list, sum_curr_inf_test_list, seriestype = :scatter, xlabel = "R0_std", ylabel = "Current Infections", title = "Initial State Vectors")
plot(Rmean_test_list, sum_curr_inf_test_list, seriestype = :scatter, xlabel = "R0_mean", ylabel = "Current Infections", title = "Initial State Vectors")




# plot particles vs. log-posterior variance
T = 50
xtrue, data_full = rand(ssm, theta0, T)

# save data
save("synth_casecount_data.jld2", "synth_data", data_full)

plt_df = logp_vs_nparticles(ssm, data_full, [500], theta0; nruns=200)


logp = LogPosterior(ssm, data_full, 500);
@time post_val = logp(theta0)
x = Vector{String}(undef, length(100) * 200)
y = Vector{Float64}(undef, length(100) * 200)
k = 1
for j in 1:200
    println(j)
    @inbounds x[k] = string(convert(Int, 1))
    @inbounds y[k] = logp(theta0)
    k += 1
end

logp.pf.tcur


failed_hist = post_val

fieldnames(typeof(failed_hist))
failed_hist.history_pf.particles[2]
findall(isnan.(failed_hist))
fieldnames(typeof(failed_hist.history_pf))

[sum(failed_hist.history_pf.particles[end][i] .> 0.0) for i in 1:100]


last_state = failed_hist.history_pf.particles[end][end]

logpdf(Particles.ssm_PY(ssm, theta0, 1, last_state), data_full[2])



model = DensityModel(logp)

spl = RWMH([Normal(0.0, 0.05), Normal(0.0, 0.05), Normal(0.0, 0.05)])
chain = sample(model, spl, 100; init_params=theta0, param_names=["shape", "scale", "pi_ua"], chain_type=Chains)

exp(0.16)

theta0
# test sampling with pypestousing PyCall
using PyCall
pypesto = pyimport("pypesto")

# include utilities for pypesto to MCMCChains
include("utilities.jl")

ssm = CaseCountModel(m_Λ = m_Λ, I_init = 100)

# data = load("home/vincent/WasteWater_inference/data/synthetic_casecount_data.jld2", "synth_data")
θ0 = Particles.parameter_template(ssm) 
logp = LogPosterior(ssm, data_full, 500)

# for pypesto we need the negative log-likelihood
neg_llh = let logp =logp
    p -> begin
        logval = logp(p)
        if isnan(logval)
            return -Inf
        end
        return -logval
    end
end

# transform to pypesto objective
objective = pypesto.Objective(fun=neg_llh)

problem = pypesto.Problem(
    objective,
    x_names=["shape", "scale", "pi_ua"],
    lb=[-1, -1, -1], # parameter bounds
    ub=[1, 1, 1], # NB for sampling it is usually better if you remap parameters to (-∞, ∞)
    copy_objective=false, # important
)

# specify sampler
sampler = pypesto.sample.AdaptiveMetropolisSampler()

# sample start value
x0 = θ0

# sample
function pypesto_chain()
    result = pypesto.sample.sample(
                problem,
                n_samples=100,
                x0=x0, # starting point
                sampler=sampler,
                )
    return  Chains_from_pypesto(result)
end

jlchain = pypesto_chain()