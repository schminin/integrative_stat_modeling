using Distributions
using Random
using StaticDistributions
using StaticArrays
import StaticDistributions: SMultivariateDistribution

# utility functions
function brownian_reproduction_number(Rt::Float64, variance::Float64)
    return max(0, Rt + rand(Normal(0, variance)))
end

function InfectionPotential(Y::Vector{T}, ω::Vector{T}) where {T<:Real}
    Λ_it = sum(ω.*Y) 
    return Λ_it
end

function DiscretizedGamma(shape::T, scale::T, m::Int) where {T<:Real}
    if shape <= 1
        throw(ArgumentError("shape must be >1"))
    end
    if scale < 0
        throw(ArgumentError("scale must be >=0"))
    end
    cdf_eval = cdf(Distributions.Gamma(shape, scale), range(1, m))
    res = zeros(T, m)
    res[1] = cdf_eval[1]
    res[2] = cdf_eval[2] - cdf_eval[1]
    res[3:end] = cdf_eval[3:end] - cdf_eval[2:end-1]
    return res
end

function CaseCountParameterMapping(θ::SVector{2, T}, m::Int) where {T<:Real}
    scale = θ[1]
    shift = θ[2]
    ω = DiscretizedGamma(scale, shift, m+1)
    ϕ = ω./sum(ω)
    return ω[1:end-1], ϕ
end 


# distributions

struct MixedPoiBin{T<:Real} <: SMultivariateDistribution{2, T, Discrete}
    Lambda::T
    R::T
    Phi::T
    function MixedPoiBin(Lambda::T, R::T, Phi::T) where {T <: Real}
        return new{T}(Lambda, R, Phi)
    end
end



mm = MixedPoiBin(2.0, 3.0, 0.5)


function Random.rand(rng::AbstractRNG, d::MixedPoiBin{T})::SVector{2, T} where {T<: Real}
    Y = rand(rng, Poisson(d.Lambda*d.R))
    A = rand(rng, Binomial(Y, d.Phi))

    return SVector(Y, A)
end


Random.rand(d::MixedPoiBin) = rand(Random.default_rng(), d)

Y, A = rand(mm)

function Distributions.logpdf(::MixedPoiBin{T}, x) where {T <: Real}
    # TODO implement correctly, this assumes independence which is not true
    return logpdf(Poisson(mm.Lambda*mm.R), x[1]) + logpdf(Binomial(x[1], mm.Phi), x[2])
end


struct CaseCountDistribution{N, T <: Real, K} <: SMultivariateDistribution{N, T, Continuous}
    State::SVector{N, T}
    θ::SVector{K, T}
    ndim::Int
    m_Λ::Int
    """
    State: Vecor of states with dimension N=3*m_Λ+2 with the following order
    - Y_{t-k} for k =0,...,m_Λ
    - A_{t-j,j} for j=0,...,m_Λ 
    - sum_{i=0}^{k} A_{t-k,i} for k=1,...,m_Λ-1 
    - R_t

    θ: Vector of parameters with dimension K=2 with the following order
    - scale of the discretized gamma distribution
    - shift of the discretized gamma distribution
    """
    function CaseCountDistribution(State::SVector{N, T}, θ::SVector{K, T}) where {N, T <: Real, K}
        ndim = length(State)
        m_Λ = (ndim - 2) ÷ 3
        return new{N, T, K}(State, θ, ndim, m_Λ)
    end
end

function Random.rand(rng::AbstractRNG, d::CaseCountDistribution{N, T, K})::SVector{N, T} where {N, T<: Real, K}

    old_state = d.State

    old_Y = old_state[1:d.m_Λ+1]
    old_A = old_state[d.m_Λ+2:2*(d.m_Λ+1)]
    old_A_sum = old_state[2*d.m_Λ+3:3*d.m_Λ+1]
    old_R = old_state[end]

    new_Y = similar(old_Y)
    new_A = similar(old_A)
    new_A_sum = similar(old_A_sum)

    ω, ϕ = CaseCountParameterMapping(d.θ, d.m_Λ)

    # get binomial parameters
    bin_par = ϕ./ (1 .- vcat([0.], cumsum(ϕ))[1:end-1])
    bin_par = min.(bin_par, 1.0)

    Λ = InfectionPotential(old_Y[1:end-1], ω) 

    # update Y_{t-i} i=m_Λ,...,1 by shifitng
    new_Y[2:end] = old_Y[1:end-1]

    # update Y_{t}, A_{t,0} by MixedPoiBin
    new_Y[1], new_A[1] = rand(rng, MixedPoiBin(Λ, old_R, bin_par[1])) #TODO Parameter ordering

    # update A_{t-j, j} by binomial
    println("n", old_Y[1]-old_A[1], ", p", bin_par[2])
    new_A[2] = rand(rng, Binomial(old_Y[1]-old_A[1], bin_par[2]) ) # n of Binomial is: Y_{t-1}-A_{t-1,0}
    for k in 3:d.m_Λ+1
        # e.g. for k=3 -> A_{t-2,2}
        #         -> n of Binomial is: Y_{t-2}-old_sum_A_{t-2} 
        println("n", old_Y[k]-old_A_sum[k-2], ", p", bin_par[k])
        new_A[k]= rand(rng, Binomial(new_Y[k]-old_A_sum[k-2], bin_par[k])) #TODO Parameter ordering
    end

    # update A_sums by shifting and summation
    new_A_sum[1] = old_A[1]+new_A[2]
    new_A_sum[2:d.m_Λ-1] = new_A[3:d.m_Λ]+old_A_sum[1:d.m_Λ-2]

    # update R by brownian motion
    new_R = brownian_reproduction_number(old_R, 1.0) #TODO Change variance to be optimized?!

    new_state = SVector(vcat(new_Y, new_A, new_A_sum, new_R)...)

    return new_state
end

Random.rand(d::CaseCountDistribution) = rand(Random.default_rng(), d)

