function linweights!(linweights::AbstractVector{<:Real}, logweights::AbstractVector{<:Real})
    lwmax = maximum_of_nonempty(logweights)
    @turbo @. linweights = exp(logweights - lwmax)
    _, wsum = normalize_weights!(linweights)
    return linweights, lwmax, wsum
end

linweights!(logweights::AbstractVector{<:Real}) = linweights!(logweights, logweights)
linweights(logweights::AbstractVector{<:Real}) = linweights!(similar(logweights), logweights)

function maximum_of_nonempty(x::AbstractVector)
    ifirst, ilast = firstindex(x), lastindex(x)
    M = @inbounds x[ifirst]
    @turbo for i in ifirst+1:ilast
        M = max(M, x[i])
    end
    return M
end

function normalize_weights!(weights::AbstractVector{<:Number})
    s = sum(weights)
    @turbo weights ./= s
    return weights, s
end

ess_of_normalized_weights(weights) = inv(sum(abs2, weights))
