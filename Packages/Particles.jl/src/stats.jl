weighted_sum(x::AbstractVector, weights::AbstractVector{<:Real}) = transpose(weights) * x

weighted_mean(x::AbstractVector, w::AbstractVector{<:Real}) = weighted_sum(x, w) / sum(w)

function weighted_var(x::AbstractVector{<:Real}, w::AbstractVector{<:Real}, mean::Real, wsum::Real)
    T = typeof(one(eltype(w)) * (zero(eltype(x)) - mean))
    s = zero(T)
    @turbo for i in eachindex(x, w)
        s += w[i] * (x[i] - mean)^2
    end
    return s / wsum
end

function weighted_mean_and_var(x::AbstractVector, w::AbstractVector{<:Real})
    wsum = sum(w)
    mean = weighted_sum(x, w) / wsum
    var = weighted_var(x, w, mean, wsum)
    return mean, var
end

function weighted_mean!(out::AbstractArray{T1, N}, x::AbstractVector{<:AbstractArray{T2, N}}, w::AbstractVector{Tw}) where {N, T1, T2, Tw <: Real}
    fill!(out, zero(T1))
    wsum = zero(Tw)
    for (xi, wi) in zip(x, w)
        @turbo @. out = muladd(wi, xi, out)
        wsum += wi
    end
    @turbo @. out /= wsum
    return out
end
