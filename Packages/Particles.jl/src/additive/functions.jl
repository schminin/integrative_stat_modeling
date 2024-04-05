"""
    AdditiveFunction
An additive function.
Return type is given by `return_type(f::AdditiveFunction, fkm::FeynmanKacModel)`.
"""
abstract type AdditiveFunction end

"""
    return_type(f::AdditiveFunction, fkm::FeynmanKacModel)
Return the type of `f` applied on the model `fkm`.
"""
function return_type end

"""
    template_maker(f::AdditiveFunction, fkm::FeynmanKacModel)
Return a callable that creates template values of `f` applied on the model `fkm`.
"""
function template_maker end

"""
    check(f::AdditiveFunction, fkm::FeynmanKacModel)
Check whether the given `AdditiveFunction` can be applied to the given `FeynmanKacModel`
(by throwing an exception if that is not the case).
"""
check(::AdditiveFunction, ::FeynmanKacModel) = nothing

make_cache(::AdditiveFunction, ::FeynmanKacModel) = nothing

####################################################################################################

"""
    ImmutableAdditiveFunction <: AdditiveFunction
An additive function returning immutables.
Methods to define are `return_type`, `(f::ImmutableAdditiveFunction)(fkm::FeynmanKacModel, θ, x, cache)` for the initial step
and `(f::ImmutableAdditiveFunction)(fkm::FeynmanKacModel, θ, t, xp, x, cache)` for later steps.
"""
abstract type ImmutableAdditiveFunction <: AdditiveFunction end

template_maker(f::ImmutableAdditiveFunction, fkm::FeynmanKacModel) = return_type(f, fkm)

####################################################################################################

"""
    MutableAdditiveFunction <: AdditiveFunction
An additive function returning mutables.
Methods to define are `template_maker`, `return_type` (optional), `(f::MutableAdditiveFunction)(out, fkm::FeynmanKacModel, θ, x, cache)` for the initial step
and `(f::MutableAdditiveFunction)(out, fkm::FeynmanKacModel, θ, t, xp, x, cache)` for later steps.
"""
abstract type MutableAdditiveFunction <: AdditiveFunction end

return_type(f::MutableAdditiveFunction, fkm::FeynmanKacModel) = typeof(template_maker(f, fkm)())

function (f::MutableAdditiveFunction)(fkm::FeynmanKacModel, θ, x, cache)
    out = template_maker(f, fkm)()
    f(out, fkm, θ, x, cache)
    return out
end
function (f::MutableAdditiveFunction)(fkm::FeynmanKacModel, θ, t::Integer, xp, x, cache)
    out = template_maker(f, fkm)()
    f(out, fkm, θ, t, xp, x, cache)
    return out
end

"""
    af_zero!(f::MutableAdditiveFunction, x)
Implements the equivalent of `fill!(x, zero(eltype(x)))` for the return types of `f`.
"""
af_zero!(::MutableAdditiveFunction, x)

"""
    af_add!(f::MutableAdditiveFunction, x, y)
Implements the equivalent of `x .+= y` for the return types of `f`.
"""
af_add!(::MutableAdditiveFunction, x, y)

"""
    af_add!(f::MutableAdditiveFunction, x, alpha::Real, y1, y2)
Implements the equivalent of `@. x += w * (y1 + y2)` for the return types of `f`.
Pass `y2=nothing` for `@. x += w * y1`.
"""
af_add!(::MutableAdditiveFunction, x, alpha::Real, y1, y2)

"""
    af_add!(f::MutableAdditiveFunction, x, y1, y2)
Implements the equivalent of `@. x += y1 + y2` for the return types of `f`.
"""
af_add!(::MutableAdditiveFunction, x, y1, y2)

"""
    af_set!(f::MutableAdditiveFunction, x, y1, y2)
Implements the equivalent of `@. x = y1 + y2` for the return types of `f`.
"""
af_set!(::MutableAdditiveFunction, x, y1, y2)

"""
    af_div!(f::MutableAdditiveFunction, x, alpha::Real)
Implements the equivalent of `@. x /= alpha` for the return types of `f`.
"""
af_div!(::MutableAdditiveFunction, x, alpha::Real)

function weighted_mean!(addfun::MutableAdditiveFunction, out, x::AbstractVector, w::AbstractVector{T}) where {T <: Real}
    af_zero!(addfun, out)
    wsum = zero(T)
    for (xi, wi) in zip(x, w)
        af_add!(addfun, out, wi, xi, nothing)
        wsum += wi
    end
    af_div!(addfun, out, wsum)
    return out
end

####################################################################################################

"""
    ArrayAdditiveFunction <: AdditiveFunction
An additive function returning arrays.
Methods to define are `template_maker`, `return_type` (optional), `(f::MutableAdditiveFunction)(out, fkm::FeynmanKacModel, θ, x, cache)` for the initial step
and `(f::MutableAdditiveFunction)(out, fkm::FeynmanKacModel, θ, t, xp, x, cache)` for later steps.
"""
abstract type ArrayAdditiveFunction <: MutableAdditiveFunction end

@inline af_zero!(::ArrayAdditiveFunction, x::AbstractArray{T}) where {T} = fill!(x, zero(T))

@inline function af_add!(::ArrayAdditiveFunction, x::AbstractArray{T1, N}, y::AbstractArray{T2, N}) where {N, T1, T2}
    @turbo x .+= y
end

@inline function af_add!(::ArrayAdditiveFunction, x::AbstractArray{T1, N}, alpha::Real, y1::AbstractArray{T2, N}, y2::AbstractArray{T3, N}) where {N, T1, T2, T3}
    @turbo @. x = muladd(alpha, y1 + y2, x)
end

@inline function af_add!(::ArrayAdditiveFunction, x::AbstractArray{T1, N}, alpha::Real, y::AbstractArray{T2, N}, ::Nothing) where {N, T1, T2}
    @turbo @. x = muladd(alpha, y, x)
end

@inline function af_add!(::ArrayAdditiveFunction, x::AbstractArray{T1, N}, y1::AbstractArray{T2, N}, y2::AbstractArray{T3, N}) where {N, T1, T2, T3}
    @turbo @. x += y1 + y2
end

@inline function af_set!(::ArrayAdditiveFunction, x::AbstractArray{T1, N}, y1::AbstractArray{T2, N}, y2::AbstractArray{T3, N}) where {N, T1, T2, T3}
    @turbo @. x = y1 + y2
end

@inline function af_div!(::ArrayAdditiveFunction, x::AbstractArray, alpha::Real)
    @turbo @. x /= alpha
end

####################################################################################################

abstract type AfsMethod end
# Define it here in case it is used by any additive function to create a corresponding Summary

include("functions/score.jl")
include("functions/qfunction.jl")
