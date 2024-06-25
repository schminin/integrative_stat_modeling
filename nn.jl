include("CaseCountModel.jl")
# learn mapping from state to distribution

using Flux
using Random: default_rng

# covariance matrix has shape state_space_dimension x state_space_dimension

# Assumption 1: only Y_ts and A_ts are correlated
# upper diagonal parameters for Y_ts, A_ts: 2*(m_Λ+1)*(2*(m_Λ+1)+1)/2 
# lower diagonal entries for sum Ats: (m_Λ+1)*m_Λ/2
# lower diagonal entries for R_t: 1 (diagonal entry)

# Assumption 2: Everything but R_t is correlated

# Define FFNN hyperparameters
n_input_neurons = state_space_dimension + 3 + 1 # state space (t) + parameters (θ) + obs (t+1)
hidden_layers = 5
hidden_neurons = 50
# For assumption 1:
n_output_neurons = Int((2*(m_Λ+1)*(2*(m_Λ+1)+1)/2) + (m_Λ+1)*m_Λ/2 + 1) + state_space_dimension # covariance matrix parameters + mu parameters
act_fct = Flux.tanh

# Construct FFNN
first_layer = Dense(n_input_neurons, hidden_neurons, act_fct)
intermediate_layers = [Dense(hidden_neurons, hidden_neurons, act_fct) for _ in 1:hidden_layers-1]
last_layer = Dense(hidden_neurons, n_output_neurons)
nn_model = Flux.Chain(first_layer, intermediate_layers..., last_layer)

# sample data
# 1. sample theta (shape, scale = 1+exp(θ[1]), exp(θ[2]), pi_ua = StatsFuns.logistic(θ[3]))
function sample_theta()
    return [rand(Uniform(-1,1)), rand(Uniform(-1,1)), rand(Uniform(2, 4))]
end

# 2. simulate S0
function initial_state_generation(ssm::CaseCountModel, θ::AbstractVector{<:Real})
    m_Λ = ssm.m_Λ
    state_space_dimension = 3*m_Λ + 2
    shape, scale = 1+exp(θ[1]), exp(θ[2])
    pi_ua = StatsFuns.logistic(θ[3])
    Y_init = ssm.I_init # initiallay infected people
    X_init = zeros(state_space_dimension)
    X_init[1:m_Λ+1] = repeat([Int(round(Y_init/(m_Λ+1)))], m_Λ+1)
    X_init[1] += Y_init-sum(X_init[1:m_Λ+1])
    X_init[end] = 1 # initial reproduction number
    X = X_init

    ω, ϕ = case_count_parameter_mapping(SVector{3, Float64}(shape, scale, pi_ua), m_Λ)
    Λ = infection_potential(X_init[1:m_Λ+1][1:end-1], ω) # old_Y[1:end-1]

    for i in 1:m_Λ
        X = rand(CaseCountDistribution(SVector(X...), SVector{3, Float64}(shape, scale, pi_ua)))
        Λ = infection_potential(X[1:m_Λ+1][1:end-1], ω) # old_Y[1:end-1]
    end

    return X
end

function generate_data(n_theta_init, n_timesteps)
    # Generate initial states
    ssm = CaseCountModel(m_Λ = m_Λ, I_init = 100)
    thetas = [sample_theta() for i in 1:n_theta_init]
    initial_states_list = [initial_state_generation(ssm, thetas[i]) for i in 1:n_theta_init]

    X = []
    Y = []
    X_t_new = initial_states_list
    for i in 1:n_timesteps
        initial_states_list = X_t_new
        X_t_new = []
        for i in 1:n_theta_init
            X_t = initial_states_list[i]
            theta = thetas[i]
            shape_par, scale_par = 1+exp(theta[1]), exp(theta[2])
            pi_ua = StatsFuns.logistic(theta[3])
            X_tp1 = rand(CaseCountDistribution(SVector(X_t...), SVector{3, Float64}(shape_par, scale_par, pi_ua)))
            Y_tp1 = rand(Particles.ssm_PY(ssm, theta, 1, X_tp1))
            push!(X, vcat(X_t, theta, Y_tp1)) # NN input
            push!(Y, X_tp1) # NN output
            push!(X_t_new, X_tp1)
        end
    end
    return X, Y
end

X, Y = generate_data(100, 30)

nn_model(X[1])

# optimization routine is based on https://github.com/FluxML/model-zoo/blob/master/vision/mlp_mnist/mlp_mnist.jl

# find normalization constants for the input data 
# for while not converged 
#    sample batch 
#    normalize batch
#    update NN parameters + store
# end


# questions: loss function? based on several samples?