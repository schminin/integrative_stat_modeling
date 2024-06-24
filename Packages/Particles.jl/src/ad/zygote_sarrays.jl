using .Zygote
using .StaticArraysCore

Zygote.@adjoint (T::Type{<:SVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)
Zygote.@adjoint (T::Type{<:SVector{N}})(x::NTuple{N}) where {N} = T(x), dv -> (nothing, dv)
