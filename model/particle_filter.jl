include("utils.jl")
using Distributions
using LowLevelParticleFilters
using Random
using Statistics

# todo: 
# tune epsilon for negbin
# checking everything against a very simple ODE model, e.g. SEIR model

##############################################################
# Settings
##############################################################

# general
min_date = Date(2022,1,1)
country = "canada"

# transmission process
R0 = 1.0
const m_Λ = 21
ω =  discr_si(1:m_Λ, 4.7, 2.9) # prob. that time until onset of secondary cases is i days

# particle filter
n = 3*m_Λ + 3   # dimension of state 
                # contains Y_i_t-k for k=0,...,m_Λ -> (m_Λ+1) and
                # A_i, t-k, k for k = 0, ...,m_Λ -> (m_Λ+1)
                # sum_j A_i,t-k,j -> for k=1, ...,m_Λ -> (m_Λ) 
                # one state for R0
                # later, we will add m_Λ days for the concentration?
m = 0       # dimension of input
p = 1       # dimension of measurements
            # later, we will add one dimension for the concentration
N = 500     # number of particles

##############################################################
# Load data
##############################################################
# df = load_data(country, false)
# df = create_time_idx(df, min_date)


p = (Y_avg = 50.0, # number of imported infections 
    ϕ = repeat([1/(m_Λ+1)], m_Λ+1), # parameters of multinomial
    dispersion = 1.1, # Variance mean ratio of NB noise distribution 
    R0_σ_2 = 0.01,
    )

##############################################################
# Define particle filter
##############################################################
"""
x(t+1) = dynamics(x(t), u(t), p, t, w(t))

x:  state vector
u:  input
p:  parameters
t:  time
w:  disturbances (noise)
"""
function dynamics(x, u, p, t, noise=false)
    state = x
    # for k = 0,...,m_Λ
    # Number of infections k days ago, Y_i_t-k
    Y(state, k) = state[k+1]

    # Number of infections k days ago, reported after k days (i.e. reported at current state)
    A(state, k) = state[k+m_Λ+2]

    # sum of reported infections up to state of infections that happened k days ago, for k = 1,...,m_Λ
    # for k = 0 this is A(state, 0)
    sum_A(state, k) = state[2*m_Λ+2+k]

    R_t(state) = state[end]

    # calculate infection potential
    Λ_it = infection_potential(t, [Y(state, k) for k in 1:m_Λ], ω, m_A, p.Y_avg)
    
    new_state = similar(state)

    # Update Yit 
    new_state[2:m_Λ+1] = [Y(state, k) for k in 0:m_Λ-1]
    if noise
        new_state[1] = rand(Distributions.Poisson(Λ_it*R_t(state)), 1)[1]
    end

    # Update sum_A

    # Define helper
    # Ai,t-j,j = x[m_Λ+2, 2*m_Λ+3]
    Aijj = [A(x, k) for k in 0:m_Λ] # m_A+1 elements
    sum_Aik = [sum_A(state, k) for k in 1:m_Λ] # m_A-1 elements
    
    # Update sum_A
    helper = vcat([0.], sum_Aik) + Aijj[1:end]
    new_state[2*m_Λ+3:3*m_Λ+2] = helper[1:end-1]

    # Update Aijj(t) -> Aijj(t+1)
    bin_p = p.ϕ ./ (1 .- vcat([0.], cumsum(p.ϕ))[1:end-1])
    n = new_state[1:m_Λ+1] - vcat([0.], helper[1:end-1])
    new_state[m_Λ+2:2*m_Λ+2] = [rand(Binomial(n[i], bin_p[i])) for i in eachindex(n)]

    # Update R0
    new_state[end] = brownian_reproduction_number(x[end], p.R0_σ_2)

    return new_state
end


"""
Calculates the likelihood based on 

x:  state vector
u:  input 
y:  measurement
p:  parameter 
t: time
"""
function measurement_likelihood(x, u, y, p, t)
    x::AbstractVector{<:Real}
    y::AbstractVector{<:Real}
    A(state, k) = state[k+m_Λ+2]
    Aijj = [A(x, k) for k in 0:m_A] # m_Λ+1 elements
    Mit = sum(Aijj) + 1e-3 # todo
    # log-likelhiood of measurement given the state
    p_s = 1/p.dispersion  # success probability p = mu/σ²
    r = Mit * p_s / (1-p_s) # number of successes
    return logpdf(NegativeBinomial(r, p_s), y[1]) # todo possibly add likelihood with prior on reproduction number
end

"""
y(t) = measurements(x(t), u(t), p, t, e(t))

x:  state vector
u:  input
p:  parameters
t:  time
e:  disturbances (noise)
"""
function measurement(x, u, p, t, noise=false)
    # Number of Reported Infected Individuals
    A(state, k) = state[k+m_Λ+2]
    Aijj = [A(x, k) for k in 0:m_Λ] # m_A+1 elements
    Mit = sum(Aijj) + 1e-3 # todo add small epsilon
    if noise
        p_s = 1/p.dispersion  # success probability p = mu/σ²
        r = Mit * p_s / (1-p_s) # number of successes
        Oit = rand(NegativeBinomial(r, p_s), 1)
    end
    
    # concentration
    # todo

    return Oit
end

struct InitialDistribution 
    n::Int
end

function Random.rand(rng::AbstractRNG, d::InitialDistribution)
    multi = Multinomial(15, repeat([1/(m_Λ+1)], m_Λ+1))
    I_t = rand(rng, multi)
    Aijj = repeat([0], m_Λ+1)
    sum_A = repeat([0.], m_Λ)
    R0 = rand(rng,Weibull(0.5, 1))
    return vcat(I_t, Aijj, sum_A, R0)
end

function Base.length(d::InitialDistribution)
    return d.n
end

#= 
function Statistics.mean(d::InitialDistribution)
    multi_μ = mean(Multinomial(5, repeat([1/(m_Λ+1)], m_Λ+1)))
    Aijj_μ = repeat([0], m_Λ+1)
    sum_A_μ = repeat([0.], m_Λ-1)
    R0_μ = mean(Weibull(1,2))
    return vcat(multi_μ, Aijj_μ, sum_A_μ, R0_μ) 
end
=#

d = InitialDistribution(n)
res = rand(Random.default_rng(), d)
#mean(d)

d0 = InitialDistribution(n)
df = Normal(0.0,0.0) # Dynamics noise Distribution

pf = LowLevelParticleFilters.AdvancedParticleFilter(N, dynamics, measurement, measurement_likelihood, df, d0; p =p)

du = MvNormal(m, 1.0) # Random input distribution for simulation
xs,u,y = LowLevelParticleFilters.simulate(pf, 200, du, p, sample_initial=true) # We can simulate the model that the pf represents
pf(u[1], y[1])               # Perform one filtering step using input u and measurement y
particles(pf)                # Query the filter for particles, try weights(pf) or expweights(pf) as well
x̂ = weighted_mean(pf)        # using the current state

sol = forward_trajectory(pf, u, y, p)