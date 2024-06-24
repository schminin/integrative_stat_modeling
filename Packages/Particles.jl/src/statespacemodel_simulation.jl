function Random.rand!(rng::AbstractRNG, states::AbstractVector{T_X}, observations::AbstractVector{T_Y}, ssm::StateSpaceModel{T_X, T_Y}, θ) where {T_X, T_Y}
    isparameter(ssm, θ) || throw(ArgumentError("given parameters are incompatible with the given StateSpaceModel"))
    axes(states) == axes(observations) || throw(ArgumentError("states and observations vectors should have the same axes"))
    length(states) ≥ 1 || throw(ArgumentError("trajectory should have a length of at least 1"))
    t, k, klast = initial_time(ssm), firstindex(states), lastindex(states)
    @inbounds states[k] = rand(rng, ssm_PX0(ssm, θ))::T_X
    @inbounds observations[k] = rand(rng, ssm_PY(ssm, θ, t, states[k]))::T_Y
    @inbounds while k < klast
        k += 1
        t += 1
        states[k] = rand(rng, ssm_PX(ssm, θ, t, states[k-1]))::T_X
        observations[k] = rand(rng, ssm_PY(ssm, θ, t, states[k]))::T_Y
    end
    return states, observations
end

function Random.rand!(states::AbstractMatrix{T_X}, observations::AbstractMatrix{T_Y}, ssm::StateSpaceModel{T_X, T_Y}, θ) where {T_X, T_Y}
    axes(states) == axes(observations) || throw(ArgumentError("states and observations matrices should have the same axes"))
    size(states, 1) ≥ 1 || throw(ArgumentError("trajectories should have a length of at least 1"))
    Threads.@threads for n in axes(states, 2)
        trajectory_states = @inbounds view(states, :, n)
        trajectory_observations = @inbounds view(observations, :, n)
        rand!(TaskLocalRNG(), trajectory_states, trajectory_observations, ssm, θ)
    end
    return states, observations
end

function Random.rand(rng::AbstractRNG, ssm::StateSpaceModel{T_X, T_Y}, θ, trajectory_length::Integer) where {T_X, T_Y}
    states = Vector{T_X}(undef, trajectory_length)
    observations = Vector{T_Y}(undef, trajectory_length)
    return rand!(rng, states, observations, ssm, θ)
end

function Random.rand(ssm::StateSpaceModel{T_X, T_Y}, θ, trajectory_length::Integer, ntrajectories::Integer) where {T_X, T_Y}
    states = Matrix{T_X}(undef, (trajectory_length, ntrajectories))
    observations = Matrix{T_Y}(undef, (trajectory_length, ntrajectories))
    return rand!(states, observations, ssm, θ)
end

function Random.rand!(states::AbstractVector{T_X}, observations::AbstractVector{T_Y}, ssm::StateSpaceModel{T_X, T_Y}, θ) where {T_X, T_Y}
    return rand!(Random.default_rng(), states, observations, ssm, θ)
end
Random.rand(ssm::StateSpaceModel, θ, trajectory_length::Integer) = rand(Random.default_rng(), ssm, θ, trajectory_length)

# Shorthands for no parameters case
Random.rand(ssm::StateSpaceModel, trajectory_length::Integer) = rand(ssm, nothing, trajectory_length)
Random.rand(ssm::StateSpaceModel, trajectory_length::Integer, ntrajectories::Integer) = rand(ssm, nothing, trajectory_length, ntrajectories)
