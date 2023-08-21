include("utils.jl")
using Distributions
using LowLevelParticleFilters

##############################################################
# Settings
##############################################################

# general
min_date = Date(2022,1,1)
country = "canada"
rolling_avg = true

# transmission process
R0 = 1.0
m_A = 21  
ω =  discr_si(1:m_A, 4.7, 2.9) # prob. that time until onset of secondary cases is i days

# particle filter
n = 2*m_A + 1   # dimension of state 
                # contains Y_i_t-k for k=0,...,m_A (m_A+1)and
                # sum_j A_i,t-k,j (m_A-1) 
                # one state for R0
                # later, we will add m_A days for the concentration?
m = 0       # dimension of input
p = 1       # dimension of measurements
            # later, we will add one dimension for the concentration
N = 500     # number of particles


##############################################################
# Load data
##############################################################
df = load_data(country, false)
df = create_time_idx(df, min_date)


p = (Y_avg = 50.0, # number of imported infections 
    ϕ = repeat([0.047], 21), # parameters of multinomial
    σ_2 = 1.0, # Variance of NB noise distribution
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
    R0 = brownian_reproduction_number(x[end], p.R0_σ_2)
        
    # A_i, t-j, j
    Aijj = get_n_infected_i_days_ago(x, ϕ, m_A)

    # update step
    # Yit 
    x[2:m_A] = x[1:m_A-1]
    if noise
        x[1] = rand(Distributions.Poisson(Λ_it*R0), 1)
    end
    # aggregate A_i for next time step
    x[m_A+1, end-1] = Aijj[3, end-1] + x[m_A+1, end-1] # A_i,t-j,0 + A_i,t-j,1 + ...+A_i,t-j,j
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
    Aijj = get_n_infected_i_days_ago(x, ϕ, m_A)
    Mit = sum(Aijj)
    # log-likelhiood of measurement given the state
    p_s = Mit / p.σ_2 # success probability p = mu/σ²
    r = Mit * p_s / (1-p_s) # number of successes
    return logpdf(NegativeBinomial(r, p_s), y)
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
    Aijj = get_n_infected_i_days_ago(x, p.ϕ, m_A)
    Mit = sum(Aijj)
    if noise
        p_s = Mit / p.σ_2 # success probability p = mu/σ²
        r = Mit * p_s / (1-p_s) # number of successes
        Oit = rand(NegativeBinomial(r, p_s), 1)
    end
    
    # concentration
    # todo

    return Oit
end

# initial state distribution
# todo: define custom multivariate distribution
d0 = MvNormal(randn(n),2.0)
df = Normal(0.0,0.0)

pf = AdvancedParticleFilter(N, dynamics, measurement, measurement_likelihood, df, d0)

du = MvNormal(0, 1.0)
xs,u,y = simulate(pf,200,du, p) # We can simulate the model that the pf represents
pf(u[1], y[1])               # Perform one filtering step using input u and measurement y
particles(pf)                # Query the filter for particles, try weights(pf) or expweights(pf) as well
x̂ = weighted_mean(pf)        # using the current state

sol = forward_trajectory(pf, u, y, p)