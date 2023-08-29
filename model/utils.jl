using CSV
using DataFrames
using RCall
using Dates

##############################################################
# Load data
##############################################################

"""
load case counts and concentration data 
"""
function load_data(country::String, rolling_avg::Bool)
    # load data
    df = CSV.read("data/preprocessed/$(country).csv", DataFrame)
    
    # select relevant columns
    if rolling_avg
        return df[!,[:date, :viral_load, :cases_7d, :city]]
    else
        return df[!,[:date, :conc_7d_ravg, :cases_7d, :city]]
    end
end

"""
create a day_idx describing the observation day number
"""
function create_time_idx(df::DataFrame, min_date::Date)
    # select relevant dates
    df = df[df.date.>=min_date,:]
    min_year = Dates.year(min_date)
    # create day idx
    df.day_idx = @. Dates.dayofyear(df.date) + (Dates.year(df.date)-min_year)*365*(1-Dates.isleapyear(df.date)) + (Dates.year(df.date)-min_year)*366*(Dates.isleapyear(df.date))
    return df
end


##############################################################
# Transmission Process
##############################################################
"""
estimate discrete serial interval weights ω
(see https://github.com/lin-lab/MERMAID/blob/befea7bac282e030f3811a022216cb3e2610e7d5/US_analysis_scripts/fit_state.R#L40
and https://github.com/mrc-ide/EpiEstim/blob/a0d025b518db8dc9e2ab3da887e54393f7ddf533/R/discr_si.R#L9 )
"""
function discr_si(k=1:30, mu::Float64 = 4.7, sigma::Float64 = 2.9)
    R"""
    vnapply <- function(X, FUN, ...) {
        vapply(X, FUN, numeric(1), ...)
    }

    discr_si <- function(k, mu, sigma) {
        if (sigma < 0) {
            stop("sigma must be >=0.")
        }
        if (mu <= 1) {
            stop("mu must be >1")
        }
        if (any(k < 0)) {
            stop("all values in k must be >=0.")
        }
        
        a <- ((mu - 1) / sigma)^2
        b <- sigma^2 / (mu - 1)
        
        cdf_gamma <- function(k, a, b) stats::pgamma(k, shape = a, scale = b)
        
        res <- k * cdf_gamma(k, a, b) + 
            (k - 2) * cdf_gamma(k - 2, a, b) - 2 * (k - 1) * cdf_gamma(k - 1, a, b)
        res <- res + a * b * (2 * cdf_gamma(k - 1, a + 1, b) - 
                                cdf_gamma(k - 2, a + 1, b) - cdf_gamma(k, a + 1, b))
        res <- vnapply(res, function(e) max(0, e))
        
        return(res)
    }
    """

    return rcopy(R"discr_si($k, mu = $mu, sigma = $sigma)")
end



"""
defines the infection potential used to sample new infections on day t
"""
function infection_potential(t::Int, Y::Vector{T} where T<:Real, ω::Vector{T} where T<:Real, m_Λ::Int=30, Y_avg::Float64=50.0)
    if t<=m_Λ # warmup phase
        Λ_it = sum(ω[1:t].*Y[1:t]) + (1-sum(ω[1:t]))*Y_avg
    else
        Λ_it = sum(ω.*Y) + (1-sum(ω))*Y_avg
    end
    return Λ_it
end


# option 1: constant reproduction number
#function estimate_reproduction_number()
    # todo (implemented in R at the moment)
#end

# option 2:
function brownian_reproduction_number(R0::Float64, variance::Float64)
    return max(0, R0 + rand(Normal(0, variance)))
end

