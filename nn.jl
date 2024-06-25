include("CaseCountModel.jl")
# learn mapping from state to distribution

using Flux
using Random: default_rng
using LinearAlgebra: I

# covariance matrix has shape state_space_dimension x state_space_dimension

# Assumption 1: 
# Correlation Y_t_1: 1
# Correlation Y_t_1 with Y_t_2...s: ? -> m_Λ parameters
# Correlation Y_t_1 with A_t: ? -> 0
# Correlation Y_t with sum_A ? -> 0
# Correlation Y_t_1 with R1: ? -> 1 parameters
# Correlation Y_t_2...s: identity
# Correlation Y_t_2...s with A_t: ? -> m_Λ * (m_Λ+1) parameters
# Correlation Y_t_2...s with R1: 0 
# Correlation A_t: (m_Λ)*(m_Λ+1)/2 parameters (n*(n+1)/2)
# Correlation A_t with sum_A, R1: 0 
# Correlation sum_A: (m_Λ-2)*(m_Λ-1)/2 parameters
# Correlation sum_A with R1: 0
# Correlation R1: 1

# Assumption 2: Everything but R_t is correlated

# Define FFNN hyperparameters
n_input_neurons = state_space_dimension + 3 + 1 # state space (t) + parameters (θ) + obs (t+1)
hidden_layers = 5
hidden_neurons = 48
# For assumption 1:
n_output_neurons = Int(m_Λ + 1 + (m_Λ+1)*m_Λ + (m_Λ)*(m_Λ+1)/2 + (m_Λ-2)*(m_Λ-1)/2 + 1) + (state_space_dimension-m_Λ) # covariance matrix parameters + mu parameters
act_fct = Flux.tanh

# Construct FFNN
first_layer = Dense(n_input_neurons, hidden_neurons, act_fct)
intermediate_layers = [Dense(hidden_neurons, hidden_neurons, act_fct) for _ in 1:hidden_layers-1]
last_layer = Dense(hidden_neurons, n_output_neurons)
nn_model = Flux.Chain(first_layer, intermediate_layers..., last_layer)

# sample data
# 1. sample theta (shape, scale = 1+exp(θ[1]), exp(θ[2]), pi_ua = StatsFuns.logistic(θ[3]))
function sample_theta()
    return [rand(Uniform(-1,2)), rand(Uniform(-1,1)), rand(Uniform(2, 4))]
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

    for i in 1:m_Λ
        X = rand(CaseCountDistribution(SVector(X...), SVector{3, Float64}(shape, scale, pi_ua)))
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

function map_nn_output_to_covariance_matrix(nn_output, state_space_dimension)
    m_Λ = Int((state_space_dimension-2)/3)
    cov_parameters = nn_output[1:end-(state_space_dimension-m_Λ)]

    # fill lower triangular part of covariance matrix
    Cov = Matrix{Float32}(I(state_space_dimension))
    # Correlation Y_t_1 with Y_t_2...s: ? -> m_Λ parameters
    Cov[2:m_Λ+1, 1] = cov_parameters[1:m_Λ]
    # Correlation Y_t_1 with A_t: ? -> 0
    # Correlation Y_t with sum_A ? -> 0
    # Correlation Y_t_1 with R1: ? -> 1 parameters
    Cov[end, 1] = cov_parameters[m_Λ+1]
    # Correlation Y_t_2...s: identity
    # Correlation Y_t_2...s with A_t: ? -> m_Λ * (m_Λ+1) parameters
    Cov[m_Λ+2:2*(m_Λ+1), 2:(m_Λ+1)] = reshape(cov_parameters[m_Λ+2:(m_Λ+1)+m_Λ*(m_Λ+1)], m_Λ+1, m_Λ)
    # Correlation Y_t_2...s with R1: 0 
    # Correlation A_t: (m_Λ)*(m_Λ+1)/2 parameters (n*(n+1)/2)
    start_row = m_Λ + 2
    start_col = m_Λ + 2
    start_entry = (m_Λ+1)+m_Λ*(m_Λ+1)
    for i in 1:m_Λ
        Cov[start_row+i, start_col:start_col+i-1] = cov_parameters[start_entry:start_entry+i-1]
        start_entry += i
    end
    # Correlation A_t with sum_A, R1: 0 
    # Correlation sum_A: (m_Λ-2)*(m_Λ-1)/2 parameters
    start_row = 2*(m_Λ+1)+1
    start_col = 2*(m_Λ+1)+1
    # start entry based on the last entry in the previous loop
    for i in 1:m_Λ-2
        Cov[start_row+i, start_col:start_col+i-1] = cov_parameters[start_entry:start_entry+i-1]
        start_entry += i
    end
    # Correlation sum_A with R1: 0
    # Correlation R1: 1
    Cov[end, end] = cov_parameters[end]

    # fill upper triangular part of covariance matrix
    Cov = Cov + Cov' - I(state_space_dimension)
    return Cov
end

function map_nn_output_to_mean_vector(nn_output, state_space_dimension, Y_t)
    m_Λ = Int((state_space_dimension-2)/3)
    mean_parameters = nn_output[end-(state_space_dimension-m_Λ)+1:end]

    # Y_t+1_2...s 
    Y_tp1_rest = Y_t[1:end-1]
    μ = vcat(mean_parameters[1], Y_tp1_rest, mean_parameters[2:end])
    return μ
end

X, Y = generate_data(100, 30)
nn_output = nn_model(X[1])
Cov = map_nn_output_to_covariance_matrix(nn_output, state_space_dimension)
μ = map_nn_output_to_mean_vector(nn_output, state_space_dimension, X[1][1:m_Λ+1])


# optimization routine is based on https://github.com/FluxML/model-zoo/blob/master/vision/mlp_mnist/mlp_mnist.jl

# find normalization constants for the input data 
# for while not converged 
#    sample batch 
#    normalize batch
#    update NN parameters + store
# end

