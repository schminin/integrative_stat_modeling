const SameTypeNamedTuple{T, N, NAMES} = NamedTuple{NAMES, <:NTuple{N, T}}

########################################################################################################

@inline function spawn_function(fun)
    task = Task(fun)
    task.sticky = false
    schedule(task)
    return task
end

########################################################################################################

static_foldl(fun, objects::NamedTuple, init) = static_foldl(fun, values(objects), init)
@generated function static_foldl(fun, objects::Tuple, init)
    N = length(objects.parameters)
    iszero(N) && return :(return init)
    lines = [
        :(x1 = fun(init, objects[1]))
    ]
    for i in 2:N
        push!(lines, :($(Symbol(:x, i)) = fun($(Symbol(:x, i-1)), objects[$i])))
    end
    push!(lines, :(return $(Symbol(:x, N))))
    return Expr(:block, lines...)
end

static_foldr(fun, objects::NamedTuple, init) = static_foldr(fun, values(objects), init)
@generated function static_foldr(fun, objects::Tuple, init)
    N = length(objects.parameters)
    iszero(N) && return :(return init)
    lines = [
        :($(Symbol(:x, N)) = fun(objects[$N], init))
    ]
    for i in N-1:-1:1
        push!(lines, :($(Symbol(:x, i)) = fun(objects[$i], $(Symbol(:x, i+1)))))
    end
    push!(lines, :(return x1))
    return Expr(:block, lines...)
end

########################################################################################################

@generated function filter_NamedTuple_by_type(::Type{T}, objects::NamedTuple{NAMES, TUPLE}) where {T, NAMES, TUPLE}
    NEW_NAMES = [NAME for (NAME, S) in zip(NAMES, TUPLE.parameters) if S <: T]
    new_values = Expr(
        :tuple,
        (:(objects.$NAME) for NAME in NEW_NAMES)...
    )
    return :(NamedTuple{$(Tuple(NEW_NAMES))}($new_values))
end

########################################################################################################

struct EndMinus
    k::Int
    function EndMinus(k::Integer; check::Bool=true)
        !check || k ≥ 0 || throw(ArgumentError("EndMinus cannot represent points after then end (last index is EndMinus(0))"))
        return new(k)
    end
end

const END = EndMinus(0)

Base.show(io::IO, i::EndMinus) = iszero(i.k) ? print(io, "end") : print(io, "end-", i.k)
Base.isless(i::EndMinus, j::EndMinus) = i.k > j.k
Base.:+(i::EndMinus, j::Integer) = EndMinus(i.k - j)
Base.:-(i::EndMinus, j::Integer) = EndMinus(i.k + j)
Base.@propagate_inbounds Base.getindex(v::AbstractVector, i::EndMinus) = v[end - i.k]

abstract type AbstractEndMinusRange <: AbstractRange{EndMinus} end

struct EndMinusUnitRange <: AbstractEndMinusRange
    start::EndMinus
    stop::EndMinus
    EndMinusUnitRange(start::EndMinus, stop::EndMinus) = new(start, max(stop, start - 1))
end

(::Colon)(i::EndMinus, j::EndMinus) = EndMinusUnitRange(i, j)

Base.length(r::EndMinusUnitRange) = max(r.start.k - r.stop.k + 1, 0)
Base.step(r::EndMinusUnitRange) = 1
Base.show(io::IO, r::EndMinusUnitRange) = print(io, r.start, ':', r.stop)

@inline function Base.getindex(r::EndMinusUnitRange, i::Integer)
    @boundscheck checkbounds(r, i)
    return EndMinus(r.start.k - (i - 1); check=false)
end

Base.@propagate_inbounds function Base.getindex(v::AbstractVector, i::EndMinusUnitRange)
    ilast = lastindex(v)
    i1 = ilast - i.start.k
    i2 = ilast - i.stop.k
    return v[i1:i2]
end

Base.@propagate_inbounds function Base.view(v::AbstractVector, i::EndMinusUnitRange)
    ilast = lastindex(v)
    i1 = ilast - i.start.k
    i2 = ilast - i.stop.k
    return view(v, i1:i2)
end

########################################################################################################

function count_unique_sorted_nonempty(x::AbstractVector)
    count = 1
    ifirst, ilast = firstindex(x), lastindex(x)
    @inbounds for i in (ifirst+1:ilast)
        x[i] == x[i-1] && continue
        count += 1
    end
    @assertx count == length(Set(x))
    return count
end
