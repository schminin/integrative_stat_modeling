using Distributions
using Random
using StaticDistributions
using StaticArrays
import StaticDistributions: SMultivariateDistribution

# utility functions
function brownian_reproduction_number(Rt::Float64, variance::Float64)
    return max(0, Rt + rand(Normal(0, variance)))
end

function InfectionPotential(Y::Vector{T}, ω::Vector{T}) where {N, T<:Real}
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


    Λ = InfectionPotential(old_Y[1:end-1], ω) 

    # update Y_{t-i} i=m_Λ,...,1 by shifitng
    new_Y[2:end] = old_Y[1:end-1]

    # update Y_{t}, A_{t,0} by MixedPoiBin
    new_Y[1], new_A[1] = rand(rng, MixedPoiBin(Λ, old_R, bin_par[1])) #TODO Parameter ordering

    # update A_{t-j, j} by binomial
    new_A[2] = rand(rng, Binomial(old_Y[1]-old_A[1], bin_par[2]) )
    for k in 3:d.m_Λ+1
        new_A[k]= rand(rng, Binomial(old_Y[k]-old_A_sum[k-2], bin_par[k])) #TODO Parameter ordering
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


######################################################################
# Testing Area 
######################################################################

function test_distribution_update()
    cc = CaseCountDistribution(reverse(SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)), SVector(2.0, 2.0))

    old_state = reverse(SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
    old_Y = old_state[1:cc.m_Λ+1]
    old_A = old_state[cc.m_Λ+2:2*(cc.m_Λ+1)]
    old_A_sum = old_state[2*cc.m_Λ+3:3*cc.m_Λ+1]
    old_R = old_state[end]

    new_state = rand(cc)
    new_Y = new_state[1:cc.m_Λ+1]
    new_A = new_state[cc.m_Λ+2:2*(cc.m_Λ+1)]
    new_A_sum = new_state[2*cc.m_Λ+3:3*cc.m_Λ+1]
    new_R = new_state[end]

    # test 1: Update Y is likely correct
    if ! (new_Y[2:end] == old_Y[1:end-1])
        print("Error in Y update: Values are not shifted correctly")
    end
    if ! (isinteger(new_Y[1]))
        print("Error in Y update: Y[1] is not an integer")
    end
    if (new_Y[2:end] == old_Y[1:end-1]) & isinteger(new_Y[1])
        print("Update of Y is correct")
    end

    # test 2: Update A is likely correct
    # subtest a): A[1] is smaller than Y[1]
    if ! (new_A[1] <= new_Y[1])
        print("Error in A update: A[1] is larger than Y[1]")
    end
    # subtest b): sum of old and new A of one timepoint of infection are smaller than corresponding Y
    if !all(new_A[2:end] + old_A[1:end-1] .<= old_Y[1:end-1])
        print("Error in A update: sum of two A's is larger than corresponding Y")
    end
    
    # test 3: Update of A_sum is likely correct
    if !(new_A_sum[1] == old_A[1] + new_A[2])
        print("Error in A_sum update: A_sum[1] is not correct")
    end
end

test_distribution_update()



ytest = SVector(1.0, 2.0, 3.0)

old_test = ytest[SVector(Array(range(1,3))...)]

new_test = similar(old_test)

om, pi = CaseCountParameterMapping(SVector(2.0, 2.0), 8)
om

pi./ (1 .- vcat([0.], cumsum(pi))[1:end-1])