include("utils.jl")
using Distributions
using LowLevelParticleFilters
using Random
using Statistics

# todo: tune epsilon for negbin

##############################################################
# Settings
##############################################################

# general
min_date = Date(2022,1,1)
country = "canada"

# transmission process
R0 = 1.0
const m_A = 21
ω =  discr_si(1:m_A, 4.7, 2.9) # prob. that time until onset of secondary cases is i days

# particle filter
n = 3*m_A + 2   # dimension of state 
                # contains Y_i_t-k for k=0,...,m_A -> (m_A+1)and
                # A_i, t-k, k for k = 0, ...,m_A -> (m_A+1)
                # sum_j A_i,t-k,j -> (m_A-1) 
                # one state for R0
                # later, we will add m_A days for the concentration?
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
    ϕ = repeat([1/(m_A+1)], m_A+1), # parameters of multinomial
    dispersion = 1.0, # Variance mean ratio of NB noise distribution 
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
    Λ_it = infection_potential(t, x[2:m_A+1], ω, m_A, p.Y_avg)
    
    # Update Yit -> Yi,t+1        
    # Yit = x[1:m_A+1]
    x[2:m_A+1] = x[1:m_A]
    if noise
        x[1] = rand(Distributions.Poisson(Λ_it*x[end]), 1)[1]
    end

    # Define helper
    # Ai,t-j,j = x[m_A+2, 2*m_A+3]
    Aijj = x[m_A+2: 2*m_A+2] # m_A+1 elements
    sum_Aik = x[2*m_A+3: 3*m_A+1] # m_A-1 elements
    helper = vcat([0.], sum_Aik) + Aijj[1:end-1]

    # Update sum_Aik(t) -> sum_Aik(t+1)
    x[2*m_A+3 : 3*m_A+1] = helper[1:end-1]

    # Update Aijj(t) -> Aijj(t+1)
    bin_p = p.ϕ ./ (1 .- vcat([0.], cumsum(p.ϕ))[1:end-1])
    n = x[1:m_A+1] - vcat(0, helper)
    x[m_A+2: 2*m_A+2] = [rand(Binomial(n[i], bin_p[i])) for i in eachindex(μ)]

    # Update R0
    x[end] = brownian_reproduction_number(x[end], p.R0_σ_2)

    return x
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
    Aijj = x[m_A+2:2*m_A+3] 
    Mit = sum(Aijj) + 1e-3 # todo
    # log-likelhiood of measurement given the state
    p_s = 1/p.dispersion  # success probability p = mu/σ²
    r = Mit * p_s / (1-p_s) # number of successes
    return logpdf(NegativeBinomial(r, p_s), y) # todo possibly add likelihood with prior on reproduction number
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
    Aijj = x[m_A+2: 2*m_A+3] # see above
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
    multi = Multinomial(5, repeat([1/(m_A+1)], m_A+1))
    I_t = rand(rng, multi)
    Aijj = repeat([0], m_A+1)
    sum_A = repeat([0.], m_A-1)
    R0 = rand(rng,Weibull(1,2))
    return vcat(I_t, Aijj, sum_A, R0)
end

function Base.length(d::InitialDistribution)
    return d.n
end

function Statistics.mean(d::InitialDistribution)
    multi_μ = mean(Multinomial(5, repeat([1/(m_A+1)], m_A+1)))
    Aijj_μ = repeat([0], m_A+1)
    sum_A_μ = repeat([0.], m_A-1)
    R0_μ = mean(Weibull(1,2))
    return vcat(multi_μ, Aijj_μ, sum_A_μ, R0_μ) 
end

d = InitialDistribution(n)
res = rand(Random.default_rng(), d)
mean(d)

d0 = InitialDistribution(n)
df = Normal(0.0,0.0) # Dynamics noise Distribution

pf = AdvancedParticleFilter(N, dynamics, measurement, measurement_likelihood, df, d0;p =p)

du = MvNormal(m, 1.0) # Random input distribution for simulation
xs,u,y = simulate(pf, 200, du, p) # We can simulate the model that the pf represents
pf(u[1], y[1])               # Perform one filtering step using input u and measurement y
particles(pf)                # Query the filter for particles, try weights(pf) or expweights(pf) as well
x̂ = weighted_mean(pf)        # using the current state

sol = forward_trajectory(pf, u, y, p)