##############################################################
# Settings
##############################################################

# general
min_date = Date(2022,1,1)
country = "canada"
rolling_avg = true

# transmission process
R0 = 1.015868
m_a = # todo
ω = # todo

# particle filter
n = m_a   # dimension of state (we track the last m_a timepoints)
m = 1   # dimension of input
p = 1   # dimension of measurements
N = 500 # number of particles


##############################################################
# Load data
##############################################################

df = load_data(canada, true)
df = create_time_idx(df, min_date)


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
function dynamics(x, u, p, t)
    Λ_it = infection_potential(t, Y, ω, m_a)
    if constant_R0
        R = p.R
    else
        R = brownian_reproduction_number(variance)
    end
    Y = Poisson(Λ_it*R)
end

"""
y(t) = measurements(x(t), u(t), p, t, e(t))

x:  state vector
u:  input
p:  parameters
t:  time
e:  disturbances (noise)
"""
measurement(x, u, p, t) = x

# Helper function
vecvec_to_mat(x) = copy(reduce(hcat, x)') # see LowLevelParticleFilters documentation


const df = MvNormal(n,1.0)          # Dynamics noise Distribution
const dg = MvNormal(p,1.0)          # Measurement noise Distribution
const d0 = MvNormal(randn(n),2.0)   # Initial state Distribution

pf = ParticleFilter(N, dynamimcs, measurement, df, dg, d0)