abstract type AbstractVectorOfMutables{T} <: AbstractVector{T} end
const AbstractVectorOfArrays{T, N} = AbstractVectorOfMutables{Array{T, N}}
const AbstractVectorOfVectors{T} = AbstractVectorOfArrays{T, 1}

abstract type AbstractCircularVector{T} <: AbstractVector{T} end

abstract type AbstractCircularVectorOfMutables{T} <: AbstractCircularVector{T} end
const AbstractCircularVectorOfArrays{T, N} = AbstractCircularVectorOfMutables{Array{T, N}}
const AbstractCircularVectorOfVectors{T} = AbstractCircularVectorOfArrays{T, 1}

preallocate!(v::AbstractCircularVector, n::Integer) = v # CircularVectors are already preallocated

@generated function preallocate!(components::NAMEDTUPLE, n::Integer) where {NAMES, NAMEDTUPLE <: NamedTuple{NAMES}}
    quote
        $(map(name -> :(preallocate!(components.$name, n)), NAMES)...)
        return components
    end
end

# NB the simple version below is dispatched at runtime according to SnoopCompile,
# hence the need for the @generated version above
# function preallocate!(components::NamedTuple, n::Integer)
#     foreach(v -> preallocate!(v, n), components)
#     return components
# end

function preallocate!(components::Tuple, n::Integer)
    foreach(v -> preallocate!(v, n), components)
    return components
end

###########################################################################
###########################################################################

struct CircularVector0{T} <: AbstractCircularVector{T} end
Base.IndexStyle(::Type{CircularVector0}) = IndexLinear()
maxlength(::CircularVector0) = static(0)
Base.size(::CircularVector0) = (0, )
Base.getindex(v::CircularVector0, i::Integer) = throw(BoundsError(v, i))
Base.push!(cv::CircularVector0, value) = cv
Base.empty!(cv::CircularVector0) = cv

###########################################################################

struct CircularVectorOfArrays0{T, N} <: AbstractCircularVectorOfArrays{T, N}
    arraysize::NTuple{N, Int}
    function CircularVectorOfArrays0{T, N}(arraysize::NTuple{N, Integer}) where {N, T}
        return new{T, N}(arraysize)
    end
end

CircularVectorOfArrays0{T}(arraysize::NTuple{N, Integer}) where {T, N} = CircularVectorOfArrays0{T, N}(arraysize)
CircularVectorOfArrays0{T, N}(arraysize::Vararg{Integer, N}) where {T, N} = CircularVectorOfArrays0{T, N}(arraysize)
CircularVectorOfArrays0{T}(arraysize::Vararg{Integer, N}) where {T, N} = CircularVectorOfArrays0{T, N}(arraysize)

const CircularVectorOfVectors0{T} = CircularVectorOfArrays0{T, 1}

Base.IndexStyle(::Type{CircularVectorOfArrays0}) = IndexLinear()
maxlength(::CircularVectorOfArrays0) = static(0)
Base.size(::CircularVectorOfArrays0) = (0, )
arraysize(cv::CircularVectorOfArrays0) = cv.arraysize
Base.getindex(cv::CircularVectorOfArrays0, i::Integer) = throw(BoundsError(cv, i))
next!(cv::CircularVectorOfArrays0; return_current::Bool=false) = error("cannot call next! on CircularVectorOfArrays0")
Base.empty!(cv::CircularVectorOfArrays0) = cv

###########################################################################

struct CircularVectorOfMutables0{T} <: AbstractCircularVectorOfMutables{T} end

CircularVectorOfMutables0(mkvalue) = CircularVectorOfMutables0{typeof(mkvalue())}()

Base.IndexStyle(::Type{CircularVectorOfMutables0}) = IndexLinear()
maxlength(::CircularVectorOfMutables0) = static(0)
Base.size(::CircularVectorOfMutables0) = (0, )
Base.getindex(cv::CircularVectorOfMutables0, i::Integer) = throw(BoundsError(cv, i))
next!(cv::CircularVectorOfMutables0; return_current::Bool=false) = error("cannot call next! on CircularVectorOfArrays0")
Base.empty!(cv::CircularVectorOfMutables0) = cv

###########################################################################
###########################################################################

mutable struct CircularVector1{T} <: AbstractCircularVector{T}
    value::T
    filled::Bool
    function CircularVector1{T}() where {T}
        cv = new()
        cv.filled = false
        return cv
    end
end

Base.IndexStyle(::Type{CircularVector1}) = IndexLinear()
maxlength(::CircularVector1) = static(1)
Base.size(cv::CircularVector1) = (cv.filled ? 1 : 0, )
Base.@inline function Base.getindex(cv::CircularVector1, i::Integer)
    @boundscheck checkbounds(cv, i)
    return cv.value
end
function Base.push!(cv::CircularVector1, value)
    cv.value = value
    cv.filled = true
    return cv
end
function Base.empty!(cv::CircularVector1)
    cv.filled = false
    return cv
end

###########################################################################

mutable struct CircularVectorOfArrays1{T, N} <: AbstractCircularVectorOfArrays{T, N}
    const value::Array{T, N}
    filled::Bool
    function CircularVectorOfArrays1{T, N}(arraysize::NTuple{N, Integer}) where {N, T}
        value = Array{T}(undef, arraysize)
        return new{T, N}(value, false)
    end
end

CircularVectorOfArrays1{T}(arraysize::NTuple{N, Integer}) where {T, N} = CircularVectorOfArrays1{T, N}(arraysize)
CircularVectorOfArrays1{T, N}(arraysize::Vararg{Integer, N}) where {T, N} = CircularVectorOfArrays1{T, N}(arraysize)
CircularVectorOfArrays1{T}(arraysize::Vararg{Integer, N}) where {T, N} = CircularVectorOfArrays1{T, N}(arraysize)

const CircularVectorOfVectors1{T} = CircularVectorOfArrays1{T, 1}

Base.IndexStyle(::Type{CircularVectorOfArrays1}) = IndexLinear()
maxlength(::CircularVectorOfArrays1) = static(1)
Base.size(cv::CircularVectorOfArrays1) = (cv.filled ? 1 : 0, )
arraysize(cv::CircularVectorOfArrays1) = size(cv.value)
Base.@inline function Base.getindex(cv::CircularVectorOfArrays1, i::Integer)
    @boundscheck checkbounds(cv, i)
    return cv.value
end
@inline function next!(cv::CircularVectorOfArrays1; return_current::Bool=false) # TODO check constant propagation
    @boundscheck if return_current && !cv.filled
        throw(ArgumentError("no current value to return"))
    end
    cv.filled = true
    return return_current ? (cv.value, cv.value) : cv.value
end
function Base.empty!(cv::CircularVectorOfArrays1)
    cv.filled = false
    return cv
end

###########################################################################

mutable struct CircularVectorOfMutables1{T} <: AbstractCircularVectorOfMutables{T}
    const value::T
    filled::Bool
    CircularVectorOfMutables1{T}(mkvalue) where {T} = new{T}(mkvalue(), false)
    function CircularVectorOfMutables1(mkvalue)
        value = mkvalue()
        return new{typeof(value)}(value, false)
    end
end

Base.IndexStyle(::Type{CircularVectorOfMutables1}) = IndexLinear()
maxlength(::CircularVectorOfMutables1) = static(1)
Base.size(cv::CircularVectorOfMutables1) = (cv.filled ? 1 : 0, )
Base.@inline function Base.getindex(cv::CircularVectorOfMutables1, i::Integer)
    @boundscheck checkbounds(cv, i)
    return cv.value
end
@inline function next!(cv::CircularVectorOfMutables1; return_current::Bool=false) # TODO check constant propagation
    @boundscheck if return_current && !cv.filled
        throw(ArgumentError("no current value to return"))
    end
    cv.filled = true
    return return_current ? (cv.value, cv.value) : cv.value
end
function Base.empty!(cv::CircularVectorOfMutables1)
    cv.filled = false
    return cv
end

###########################################################################
###########################################################################

mutable struct CircularVector2{T} <: AbstractCircularVector{T}
    value1::T
    value2::T
    flag::Int # 0: empty, 1: length is one, 2: length is two (first element is value1), 3: length is two (first element is value2)
    function CircularVector2{T}() where {T}
        cv = new()
        cv.flag = 0
        return cv
    end
end

Base.IndexStyle(::Type{CircularVector2}) = IndexLinear()
maxlength(::CircularVector2) = static(2)
Base.size(cv::CircularVector2) = (min(cv.flag, 2), )
Base.@inline function Base.getindex(cv::CircularVector2, i::Integer)
    @boundscheck checkbounds(cv, i)
    if isone(i)
        return cv.flag < 3 ? cv.value1 : cv.value2
    else
        return cv.flag == 2 ? cv.value2 : cv.value1
    end
end
function Base.push!(cv::CircularVector2, value)
    if cv.flag == 0
        cv.value1 = value
        cv.flag = 1
    elseif cv.flag == 1 || cv.flag == 3
        cv.value2 = value
        cv.flag = 2
    else # cv.flag == 2
        cv.value1 = value
        cv.flag = 3
    end
    return cv
end
function Base.empty!(cv::CircularVector2)
    cv.flag = 0
    return cv
end

###########################################################################

mutable struct CircularVectorOfArrays2{T, N} <: AbstractCircularVectorOfArrays{T, N}
    const value1::Array{T, N}
    const value2::Array{T, N}
    flag::Int # 0: empty, 1: length is one, 2: length is two (first element is value1), 3: length is two (first element is value2)
    function CircularVectorOfArrays2{T, N}(arraysize::NTuple{N, Integer}) where {T, N}
        value1 = Array{T}(undef, arraysize)
        value2 = Array{T}(undef, arraysize)
        return new{T, N}(value1, value2, 0)
    end
end

CircularVectorOfArrays2{T}(arraysize::NTuple{N, Integer}) where {T, N} = CircularVectorOfArrays2{T, N}(arraysize)
CircularVectorOfArrays2{T, N}(arraysize::Vararg{Integer, N}) where {T, N} = CircularVectorOfArrays2{T, N}(arraysize)
CircularVectorOfArrays2{T}(arraysize::Vararg{Integer, N}) where {T, N} = CircularVectorOfArrays2{T, N}(arraysize)

const CircularVectorOfVectors2{T} = CircularVectorOfArrays2{T, 1}

Base.IndexStyle(::Type{CircularVectorOfArrays2}) = IndexLinear()
maxlength(::CircularVectorOfArrays2) = static(2)
Base.size(cv::CircularVectorOfArrays2) = (min(cv.flag, 2), )
arraysize(cv::CircularVectorOfArrays2) = size(cv.value1)
Base.@inline function Base.getindex(cv::CircularVectorOfArrays2, i::Integer)
    @boundscheck checkbounds(cv, i)
    if isone(i)
        return cv.flag < 3 ? cv.value1 : cv.value2
    else
        return cv.flag == 2 ? cv.value2 : cv.value1
    end
end
function next!(cv::CircularVectorOfArrays2; return_current::Bool=false) # TODO check constant propagation
    if cv.flag == 0
        return_current && throw(ArgumentError("no current value to return"))
        cv.flag = 1
        return cv.value1
    elseif cv.flag == 1 || cv.flag == 3
        cv.flag = 2
        return return_current ? (cv.value2, cv.value1) : cv.value2
    else # cv.flag == 2
        cv.flag = 3
        return return_current ? (cv.value1, cv.value2) : cv.value1
    end
end
function Base.empty!(cv::CircularVectorOfArrays2)
    cv.flag = 0
    return cv
end

###########################################################################

mutable struct CircularVectorOfMutables2{T} <: AbstractCircularVectorOfMutables{T}
    const value1::T
    const value2::T
    flag::Int # 0: empty, 1: length is one, 2: length is two (first element is value1), 3: length is two (first element is value2)
    function CircularVectorOfMutables2{T}(mkvalue) where {T}
        value1 = mkvalue()
        value2 = mkvalue()
        return new{T}(value1, value2, 0)
    end
    function CircularVectorOfMutables2(mkvalue)
        value1 = mkvalue()
        value2 = mkvalue()
        return new{typeof(value1)}(value1, value2, 0)
    end
end

Base.IndexStyle(::Type{CircularVectorOfMutables2}) = IndexLinear()
maxlength(::CircularVectorOfMutables2) = static(2)
Base.size(cv::CircularVectorOfMutables2) = (min(cv.flag, 2), )
Base.@inline function Base.getindex(cv::CircularVectorOfMutables2, i::Integer)
    @boundscheck checkbounds(cv, i)
    if isone(i)
        return cv.flag < 3 ? cv.value1 : cv.value2
    else
        return cv.flag == 2 ? cv.value2 : cv.value1
    end
end
function next!(cv::CircularVectorOfMutables2; return_current::Bool=false) # TODO check constant propagation
    if cv.flag == 0
        return_current && throw(ArgumentError("no current value to return"))
        cv.flag = 1
        return cv.value1
    elseif cv.flag == 1 || cv.flag == 3
        cv.flag = 2
        return return_current ? (cv.value2, cv.value1) : cv.value2
    else # cv.flag == 2
        cv.flag = 3
        return return_current ? (cv.value1, cv.value2) : cv.value1
    end
end
function Base.empty!(cv::CircularVectorOfMutables2)
    cv.flag = 0
    return cv
end

###########################################################################
###########################################################################

mutable struct CircularVector{T} <: AbstractCircularVector{T}
    const data::Vector{T}
    origin::Int
    function CircularVector{T}(maxlength::Integer) where {T}
        maxlength > 0 || throw(ArgumentError("cannot create a CircularVector with maxlength = 0"))
        data = Vector{T}(undef, maxlength)
        origin = 1 - maxlength
        return new{T}(data, origin)
    end
end

Base.IndexStyle(::Type{CircularVector}) = IndexLinear()
Base.parent(cv::CircularVector) = cv.data
maxlength(cv::CircularVector) = length(cv.data)
Base.size(cv::CircularVector) = (cv.origin > 0 ? length(cv.data) : cv.origin - (1 - length(cv.data)), )
Base.@inline function Base.getindex(cv::CircularVector, i::Integer)
    @boundscheck checkbounds(cv, i)
    if cv.origin > 0
        n = length(cv.data)
        k = cv.origin + i - 1
        return @inbounds cv.data[k ≤ n ? k : k - n]
    else
        return @inbounds cv.data[i]
    end
end
function Base.push!(cv::CircularVector, value)
    if cv.origin > 0 
        @inbounds cv.data[cv.origin] = value
        if cv.origin == length(cv.data)
            cv.origin = 1
        else
            cv.origin += 1
        end
    else
        i = cv.origin + length(cv.data)
        @inbounds cv.data[i] = value
        cv.origin += 1
    end
    return cv
end
function Base.empty!(cv::CircularVector)
    cv.origin = 1 - maxlength(cv)
    return cv
end

###########################################################################

mutable struct CircularVectorOfArrays{T, N} <: AbstractCircularVectorOfArrays{T, N}
    const data::Vector{Array{T, N}}
    origin::Int
    function CircularVectorOfArrays{T, N}(maxlength::Integer, arraysize::NTuple{N, Integer}) where {T, N}
        maxlength > 0 || throw(ArgumentError("cannot create a CircularVectorOfArrays with maxlength = 0"))
        data = [Array{T}(undef, arraysize) in Base.OneTo(maxlength)]
        origin = 1 - maxlength
        return new{T, N}(data, origin)
    end
end

CircularVectorOfArrays{T}(maxlength::Integer, arraysize::NTuple{N, Integer}) where {T, N} = CircularVectorOfArrays{T, N}(maxlength, arraysize)
CircularVectorOfArrays{T, N}(maxlength::Integer, arraysize::Vararg{Integer, N}) where {T, N} = CircularVectorOfArrays{T, N}(maxlength, arraysize)
CircularVectorOfArrays{T}(maxlength::Integer, arraysize::Vararg{Integer, N}) where {T, N} = CircularVectorOfArrays{T, N}(maxlength, arraysize)

const CircularVectorOfVectors{T} = CircularVectorOfArrays{T, 1}

Base.IndexStyle(::Type{CircularVectorOfArrays}) = IndexLinear()
Base.parent(cv::CircularVectorOfArrays) = cv.data
maxlength(cv::CircularVectorOfArrays) = length(cv.data)
Base.size(cv::CircularVectorOfArrays) = (cv.origin > 0 ? length(cv.data) : cv.origin - (1 - length(cv.data)), )
arraysize(cv::CircularVectorOfArrays) = size(@inbounds cv.data[1])
Base.@inline function Base.getindex(cv::CircularVectorOfArrays, i::Integer)
    @boundscheck checkbounds(cv, i)
    if cv.origin > 0
        n = length(cv.data)
        k = cv.origin + i - 1
        return @inbounds cv.data[k ≤ n ? k : k - n]
    else
        return @inbounds cv.data[i]
    end
end
@inline function next!(cv::CircularVectorOfArrays; return_current::Bool=false) # TODO check constant propagation
    if cv.origin > 0 
        value = @inbounds cv.data[cv.origin]
        if return_current
            value_cur = @inbounds cv.data[isone(cv.origin) ? length(cv.data) : cv.origin - 1]
        end
        if cv.origin == length(cv.data)
            cv.origin = 1
        else
            cv.origin += 1
        end
    else
        i = cv.origin + length(cv.data)
        value = @inbounds cv.data[i]
        if return_current
            @boundscheck if isone(i)
                throw(ArgumentError("no current value to return"))
            end
            value_cur = @inbounds cv.data[i - 1]
        end
        cv.origin += 1
    end
    return return_current ? (value, value_cur) : value
end
function Base.empty!(cv::CircularVectorOfArrays)
    cv.origin = 1 - maxlength(cv)
    return cv
end

###########################################################################

mutable struct CircularVectorOfMutables{T} <: AbstractCircularVectorOfMutables{T}
    const data::Vector{T}
    origin::Int
    function CircularVectorOfMutables{T}(maxlength::Integer, mkvalue) where {T}
        maxlength > 0 || throw(ArgumentError("cannot create a CircularVectorOfMutables with maxlength = 0"))
        data = T[mkvalue() in Base.OneTo(maxlength)]
        origin = 1 - maxlength
        return new{T}(data, origin)
    end
    function CircularVectorOfMutables(maxlength::Integer, mkvalue)
        maxlength > 0 || throw(ArgumentError("cannot create a CircularVectorOfMutables with maxlength = 0"))
        data = [mkvalue() in Base.OneTo(maxlength)]
        origin = 1 - maxlength
        return new{eltype(data)}(data, origin)
    end
end

Base.IndexStyle(::Type{CircularVectorOfMutables}) = IndexLinear()
Base.parent(cv::CircularVectorOfMutables) = cv.data
maxlength(cv::CircularVectorOfMutables) = length(cv.data)
Base.size(cv::CircularVectorOfMutables) = (cv.origin > 0 ? length(cv.data) : cv.origin - (1 - length(cv.data)), )
Base.@inline function Base.getindex(cv::CircularVectorOfMutables, i::Integer)
    @boundscheck checkbounds(cv, i)
    if cv.origin > 0
        n = length(cv.data)
        k = cv.origin + i - 1
        return @inbounds cv.data[k ≤ n ? k : k - n]
    else
        return @inbounds cv.data[i]
    end
end
@inline function next!(cv::CircularVectorOfMutables; return_current::Bool=false) # TODO check constant propagation
    if cv.origin > 0 
        value = @inbounds cv.data[cv.origin]
        if return_current
            value_cur = @inbounds cv.data[isone(cv.origin) ? length(cv.data) : cv.origin - 1]
        end
        if cv.origin == length(cv.data)
            cv.origin = 1
        else
            cv.origin += 1
        end
    else
        i = cv.origin + length(cv.data)
        value = @inbounds cv.data[i]
        if return_current
            @boundscheck if isone(i)
                throw(ArgumentError("no current value to return"))
            end
            value_cur = @inbounds cv.data[i - 1]
        end
        cv.origin += 1
    end
    return return_current ? (value, value_cur) : value
end
function Base.empty!(cv::CircularVectorOfMutables)
    cv.origin = 1 - maxlength(cv)
    return cv
end

###########################################################################
###########################################################################

mutable struct PreallocatableVector{T} <: AbstractVector{T}
    const data::Vector{T}
    length::Int
    function PreallocatableVector{T}() where {T}
        data = Vector{T}(undef, 0)
        length = 0
        return new{T}(data, length)
    end
end

Base.IndexStyle(::Type{PreallocatableVector}) = IndexLinear()
Base.parent(v::PreallocatableVector) = v.data
maxlength(v::PreallocatableVector) = static(typemax(Int))
Base.size(v::PreallocatableVector) = (v.length, )
Base.@inline function Base.getindex(v::PreallocatableVector, i::Integer)
    @boundscheck checkbounds(v, i)
    return @inbounds v.data[i]
end
function Base.push!(v::PreallocatableVector, value)
    v.length += 1
    preallocate!(v, v.length)
    @inbounds v.data[v.length] = value
    return v
end
function preallocate!(v::PreallocatableVector, n::Integer)
    length(v.data) < n && resize!(v.data, n)
    return v
end
function Base.empty!(v::PreallocatableVector)
    # NB does not free memory.
    #    This is on purpose, so that we do not need to reallocate it
    #    when the PF is reset and run again)
    v.length = 0
    return v
end

###########################################################################

mutable struct PreallocatableVectorOfArrays{T, N} <: AbstractVectorOfArrays{T, N}
    const data::Vector{Array{T, N}}
    length::Int
    function PreallocatableVectorOfArrays{T, N}(arraysize::NTuple{N, Integer}) where {T, N}
        data = [Array{T}(undef, arraysize)]
        length = 0
        return new{T, N}(data, length)
    end
end

PreallocatableVectorOfArrays{T}(arraysize::NTuple{N, Integer}) where {T, N} = PreallocatableVectorOfArrays{T, N}(arraysize)
PreallocatableVectorOfArrays{T, N}(arraysize::Vararg{Integer, N}) where {T, N} = PreallocatableVectorOfArrays{T, N}(arraysize)
PreallocatableVectorOfArrays{T}(arraysize::Vararg{Integer, N}) where {T, N} = PreallocatableVectorOfArrays{T, N}(arraysize)

const PreallocatableVectorOfVectors{T} = PreallocatableVectorOfArrays{T, 1}

Base.IndexStyle(::Type{PreallocatableVectorOfArrays}) = IndexLinear()
Base.parent(v::PreallocatableVectorOfArrays) = v.data
maxlength(v::PreallocatableVectorOfArrays) = static(typemax(Int))
Base.size(v::PreallocatableVectorOfArrays) = (v.length, )
arraysize(v::PreallocatableVectorOfArrays) = size(@inbounds v.data[1])
Base.@inline function Base.getindex(v::PreallocatableVectorOfArrays, i::Integer)
    @boundscheck checkbounds(v, i)
    return @inbounds v.data[i]
end
function preallocate!(v::PreallocatableVectorOfArrays{T}, n::Integer) where {T}
    if length(v.data) < n
        kfirst = lastindex(v.data) + 1
        resize!(v.data, n)
        S = arraysize(v)
        @inbounds for k in kfirst:n
            v.data[k] = Array{T}(undef, S)
        end
    end
    return v
end
@inline function next!(v::PreallocatableVectorOfArrays; return_current::Bool=false) # TODO check constant propagation
    if return_current
        @boundscheck if iszero(v.length)
            throw(ArgumentError("no current value to return"))
        end
        value_cur = @inbounds v.data[v.length]
    end
    v.length += 1
    preallocate!(v, v.length)
    value = @inbounds v.data[v.length]
    return return_current ? (value, value_cur) : value
end
function Base.empty!(v::PreallocatableVectorOfArrays)
    # NB does not free memory.
    #    This is on purpose, so that we do not need to reallocate it
    #    when the PF is reset and run again)
    v.length = 0
    return v
end

###########################################################################

mutable struct PreallocatableVectorOfMutables{T, F} <: AbstractVectorOfMutables{T}
    const data::Vector{T}
    length::Int
    const mkvalue::F
    function PreallocatableVectorOfMutables{T}(mkvalue) where {T}
        data = T[]
        length = 0
        return new{T, typeof(mkvalue)}(data, length, mkvalue)
    end
    function PreallocatableVectorOfMutables(mkvalue)
        data = [mkvalue()]
        length = 0
        return new{eltype(data), typeof(mkvalue)}(data, length, mkvalue)
    end
end

Base.IndexStyle(::Type{PreallocatableVectorOfMutables}) = IndexLinear()
Base.parent(v::PreallocatableVectorOfMutables) = v.data
maxlength(::PreallocatableVectorOfMutables) = static(typemax(Int))
Base.size(v::PreallocatableVectorOfMutables) = (v.length, )
Base.@inline function Base.getindex(v::PreallocatableVectorOfMutables, i::Integer)
    @boundscheck checkbounds(v, i)
    return @inbounds v.data[i]
end
function preallocate!(v::PreallocatableVectorOfMutables, n::Integer)
    if length(v.data) < n
        kfirst = lastindex(v.data) + 1
        resize!(v.data, n)
        @inbounds for k in kfirst:n
            v.data[k] = v.mkvalue()
        end
    end
    return v
end
@inline function next!(v::PreallocatableVectorOfMutables; return_current::Bool=false) # TODO check constant propagation
    if return_current
        @boundscheck if iszero(v.length)
            throw(ArgumentError("no current value to return"))
        end
        value_cur = @inbounds v.data[v.length]
    end
    v.length += 1
    preallocate!(v, v.length)
    value = @inbounds v.data[v.length]
    return return_current ? (value, value_cur) : value
end
function Base.empty!(v::PreallocatableVectorOfMutables)
    # NB does not free memory.
    #    This is on purpose, so that we do not need to reallocate it
    #    when the PF is reset and run again)
    v.length = 0
    return v
end
