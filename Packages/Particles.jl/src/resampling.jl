abstract type ResamplingScheme end

make_cache(::ResamplingScheme, nweights::Integer, nchoices::Integer) = nothing
make_cache(s::ResamplingScheme, n::Integer) = make_cache(s, n, n)

function resample!(rng::AbstractRNG, ancestors::AbstractVector{<:Integer}, scheme::ResamplingScheme, weights::Vector{<:Real})
    nweights = length(weights)
    nchoices = length(ancestors)
    cache_r = make_cache(scheme, nweights, nchoices)
    return resample!(rng, ancestors, scheme, weights, cache_r)
end

abstract type NonAdaptiveResampling <: ResamplingScheme end

# Algorithm 9.1 page 112
"""
    inverse_categorical_cdf!(A::Vector{<:Integer}, w::Vector{<:Real}, u::Vector{<:Real})
Evaluate the inverse CDF of the categorical distribution define by the normalized (non-negative and sum to one) weights `w` at the ordered points `u` in the unit interval `[0, 1]`.
The vectors `w` and `u` need not have the same length, but `A` must have the same length as `u` and `w` must be non-empty.
The elements of `A` will belong to the set `{1, 2, ..., length(w)}`.
"""
function inverse_categorical_cdf!(A::AbstractVector{<:Integer}, w::AbstractVector{<:Real}, u::AbstractVector{<:Real})
    isempty(w) && throw(ArgumentError("weights vector is empty"))
    Base.has_offset_axes(A, w, u) && throw(ArgumentError("The indexing of A, w, and u must be 1-based"))
    s = @inbounds w[1]
    m = 1
    @inbounds for n in eachindex(A, u)
        while u[n] > s
            m += 1
            s += w[m]
        end
        A[n] = m
    end
    return A
end

####################################################################################################

struct MultinomialResampling <: NonAdaptiveResampling end

make_cache(::MultinomialResampling, nparticles::Integer, nchoices::Integer) = Vector{Float64}(undef, nchoices)

"""
    uniform_spacing!(rng::AbstractRNG, v::AbstractVector)
Generate \$N\$ ordered uniform variates (where `N = length(v)`) in \$O(N)\$ time.
Equivalent to `sort!(rand!(v))` which has \$O(N\\log(N))\$ complexity.
"""
function uniform_spacing!(rng::AbstractRNG, v::AbstractVector{T}) where {T}
    ifirst, ilast = firstindex(v), lastindex(v)
    v0 = -log(rand(rng, T))
    @turbo for i in axes(v, 1)
        v[i] = -log(rand(rng, T))
    end
    for i in ifirst+1:ilast
        v[i] += v[i-1]
    end
    vsum = @inbounds v[ilast] + v0
    @turbo @. v /= vsum
    return v
end

function resample!(rng::AbstractRNG, ancestors::AbstractVector{<:Integer}, ::MultinomialResampling, weights::Vector{<:Real}, cache_r::Vector{Float64})
    spacings = uniform_spacing!(rng, cache_r)
    inverse_categorical_cdf!(ancestors, weights, spacings)
    return true
end

####################################################################################################

struct SystematicResampling <: NonAdaptiveResampling end

function resample!(rng::AbstractRNG, ancestors::AbstractVector{<:Integer}, ::SystematicResampling, weights::Vector{<:Real}, cache_r::Nothing)
    M = length(ancestors)
    spacings = (rand(rng) .+ (0:M-1)) ./ M # NB spacings is still a range!
    inverse_categorical_cdf!(ancestors, weights, spacings) # TODO a specialized version of inverse_categorical_cdf! for SystematicResampling could be potentially be faster (but I guess probably not by much)
    return true
end

####################################################################################################

struct AdaptiveResampling{T <: NonAdaptiveResampling} <: ResamplingScheme
    scheme::T
    ESSrmin::Float64
    function AdaptiveResampling(scheme::NonAdaptiveResampling, ESSrmin::Real=0.5)
        0 ≤ ESSrmin ≤ 1 || throw(ArgumentError("ESSrmin must be in [0, 1]"))
        return new{typeof(scheme)}(scheme, ESSrmin)
    end
end

make_cache(ar::AdaptiveResampling, nparticles::Integer) = make_cache(ar.scheme, nparticles)

function resample!(rng::AbstractRNG, ancestors::AbstractVector{<:Integer}, ascheme::AdaptiveResampling, weights::Vector{<:Real}, cache_r)
    ESS = ess_of_normalized_weights(weights)
    if ESS < ascheme.ESSrmin * length(weights)
        return resample!(rng, ancestors, ascheme.scheme, weights, cache_r)
    else
        M = length(ancestors)
        @assert length(weights) == M # This is true at the moment, but could fail if the number of particles is allowed to change between steps
        @turbo ancestors .= Base.OneTo(M)
        return false
    end
end
