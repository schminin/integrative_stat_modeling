using CSV
using DataFrames



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
        return df[!,[:date, :conc_7d_ravg, :incidence_7d, :city]]
    else
        return df[!,[:date, :viral_load, :cases_reported_date, :city]]
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
defines the infection potential used to sample new infections on day t
"""
function infection_potential(t::Int, Y::Vector{T} where T<:Real, ω::Vector{T} where T<:Real, m_a::Int=30)
    if t<=m_a
        Λ_it = ω[end-t:end]*Y[end-t:end] + (1-sum(ω[end-t:end]))*Y_avg
    else
        Λ_it = ω*Y[end-m_a:end] 
    end
    return Λ_it
end

# option 1: constant reproduction number
#function estimate_reproduction_number()
    # todo (implemented in R at the moment)
#end

# option 2:
function brownian_reproduction_number(variance::Float)
    # todo
end

