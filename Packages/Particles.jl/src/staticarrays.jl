# using .StaticArrays

function weighted_var(x::AbstractVector{<:SVector{N}}, w::AbstractVector{<:Real}, mean::SVector{N}, wsum::Real) where {N}
    T = typeof(one(eltype(w)) .* (zero(eltype(x)) .- mean))
    s = zero(T)
    for (x, w) in zip(x, w)
        s = (@. muladd(w, (x - mean)^2, s))::T
    end
    return @. s / wsum
end
