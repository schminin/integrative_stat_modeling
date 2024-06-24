if SUMMARY_INNER_THREADING

const AutoCacheLock = ReentrantLock
# const AutoCacheLock = Threads.SpinLock
# TODO compare ReentrantLock vs SpinLock

struct AutoCache{T, F}
    mkobj::F
    mapping::Vector{Int}
    objects::Vector{T}
    available::Vector{Bool}
    lock::AutoCacheLock
    function AutoCache{T}(mkobj, num_consumers::Int) where {T}
        mapping = zeros(Int, num_consumers)
        objects = T[]
        available = Bool[]
        return new{T, typeof(mkobj)}(mkobj, mapping, objects, available, AutoCacheLock())
    end
    function AutoCache(mkobj, num_consumers::Int)
        mapping = zeros(Int, num_consumers)
        objects = [mkobj()]
        available = [true]
        return new{eltype(objects), typeof(mkobj)}(mkobj, mapping, objects, available, AutoCacheLock())
    end
end

Base.eltype(::AutoCache{T}) where {T} = T
Base.length(cache::AutoCache) = length(cache.mapping)
cachelength(cache::AutoCache) = lock(cache.lock) do; length(cache.objects) end

Base.summary(io::IO, cache::AutoCache{T}) where {T} = print(io, length(cache), "-element AutoCache{", T, '}')

function Base.checkbounds(cache::AutoCache, i::Int)
    1 ≤ i ≤ length(cache) || throw(BoundsError(cache, i))
    return nothing
end

@inline function Base.getindex(cache::AutoCache{T}, i::Int, init! = nothing) where {T}
    @boundscheck checkbounds(cache, i)
    return lock(cache.lock) do
        k = @inbounds cache.mapping[i]
        if iszero(k)
            j = findfirst(cache.available)
            if isnothing(j)
                N = length(cache.objects) + 1
                resize!(cache.objects, N)
                push!(cache.available, false)
                obj = cache.mkobj()
                isnothing(init!) || init!(obj)
                @inbounds cache.objects[N] = obj
                @inbounds cache.mapping[i] = N
                return obj::T
            else
                j::Int
                @inbounds cache.available[j] = false
                @inbounds cache.mapping[i] = j
                return @inbounds cache.objects[j]
            end
        else
            return @inbounds cache.objects[k]
        end
    end
end

@inline function release!(cache::AutoCache, i::Int)
    @boundscheck checkbounds(cache, i)
    return lock(cache.lock) do
        k = @inbounds cache.mapping[i]
        if !iszero(k)
            @inbounds cache.available[k] = true
            @inbounds cache.mapping[i] = 0
        end
        return nothing
    end
end

function Base.foreach(fun, cache::AutoCache)
    lock(cache.lock) do
        foreach(fun, cache.objects)
    end
end

else # if SUMMARY_INNER_THREADING == false

struct AutoCache{T}
    obj::T
    num_consumers::Int
    AutoCache{T}(mkobj, num_consumers::Int) where {T} = new{T}(mkobj(), num_consumers)
    function AutoCache(mkobj, num_consumers::Int)
        obj = mkobj()
        T = typeof(obj)
        return new{T}(obj, num_consumers)
    end
end

Base.eltype(::AutoCache{T}) where {T} = T
Base.length(cache::AutoCache) = cache.num_consumers
cachelength(cache::AutoCache) = 1

Base.summary(io::IO, cache::AutoCache{T}) where {T} = print(io, length(cache), "-element AutoCache{", T, '}')

function Base.checkbounds(cache::AutoCache, i::Int)
    1 ≤ i ≤ length(cache) || throw(BoundsError(cache, i))
    return nothing
end

@inline function Base.getindex(cache::AutoCache{T}, i::Int, init! = nothing) where {T}
    @boundscheck checkbounds(cache, i)
    return cache.obj
end

@inline function release!(cache::AutoCache, i::Int)
    @boundscheck checkbounds(cache, i)
    return nothing
end

function Base.foreach(fun, cache::AutoCache)
    fun(cache.obj)
    return nothing
end

end # if SUMMARY_INNER_THREADING
