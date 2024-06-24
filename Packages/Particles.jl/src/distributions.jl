struct Deterministic{T}
    value::T
end
Random.rand(::AbstractRNG, d::Deterministic) = d.value
Random.rand(d::Deterministic) = rand(Random.default_rng(), d)
Distributions.logpdf(::Deterministic, x) = 0.0
function Distributions.logpdf(::Deterministic{T}, x) where {T <: Real}
    S = typeof(inv(one(T)))
    return zero(S)
end

struct ConvertDistribution{T, D}
    dist::D
    ConvertDistribution{T}(dist) where {T} = new{T, typeof(dist)}(dist)
end
Random.rand(rng::AbstractRNG, d::ConvertDistribution{T}) where {T} = convert(T, rand(rng, d.dist))
Random.rand(d::ConvertDistribution) = rand(Random.default_rng(), d)
Distributions.logpdf(d::ConvertDistribution, x) = logpdf(d.dist, x)

struct ClampFiniteDistribution{D}
    dist::D
end
Distributions.logpdf(d::ClampFiniteDistribution, x) = logpdf(d.dist, x)
Base.rand(d::ClampFiniteDistribution) = rand(Random.default_rng(), d)
function Base.rand(rng::AbstractRNG, d::ClampFiniteDistribution)
    x = rand(rng, d.dist)
    any(isnan, x) && error("NaNs detected")
    T = eltype(x)
    return clamp.(x, -floatmax(T), floatmax(T))
end

struct CheckFiniteDistribution{D}
    dist::D
end
Distributions.logpdf(d::CheckFiniteDistribution, x) = logpdf(d.dist, x)
Base.rand(d::CheckFiniteDistribution) = rand(Random.default_rng(), d)
function Base.rand(rng::AbstractRNG, d::CheckFiniteDistribution)
    x = rand(rng, d.dist)
    all(isfinite, x) || error("non-finite values detected in x = $x")
    return x
end

struct Mixture{D1, D2}
    dist1::D1
    dist2::D2
    p1::Float64
    function Mixture(dist1, dist2, p1::Real; check::Bool=true)
        check && !(0 ≤ p1 ≤ 1) && throw(ArgumentError("probability p1 must be in [0, 1]"))
        return new{typeof(dist1), typeof(dist2)}(dist1, dist2, p1)
    end
end
Base.rand(d::Mixture) = rand(Random.default_rng(), d)
function Base.rand(rng::AbstractRNG, d::Mixture)
    if rand(rng) < d.p1
        return rand(rng, d.dist1)
    else
        return rand(rng, d.dist2)
    end
end
function Distributions.logpdf(d::Mixture, x)
    logpdf1 = logpdf(d.dist1, x)
    logpdf2 = logpdf(d.dist2, x)
    return muladd(d.p1, logpdf1 - logpdf2, logpdf2)
end
