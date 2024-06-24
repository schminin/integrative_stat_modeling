mutable struct MultinomialQueue
    const queue::Vector{Int}
    const perm::Vector{Int}
    const cache::Vector{Float64}
    weights::Vector{Float64}
    k::Int
    function MultinomialQueue(queue_length::Int)
        return new(
            Vector{Int}(undef, queue_length),
            Vector{Int}(undef, queue_length),
            Vector{Float64}(undef, queue_length),
        )
    end
end

function reset!(q::MultinomialQueue, weights::Vector{Float64}, queue_length::Int=length(weights))
    resize!(q.queue, queue_length)
    resize!(q.perm, queue_length)
    resize!(q.cache, queue_length)
    return reset!(q, weights, Val(:noresize))
end
function reset!(q::MultinomialQueue, weights::Vector{Float64}, ::Val{:noresize})
    q.weights = weights
    q.k = length(q.queue) + 1 # delay drawing from multinomial until necessary
    return q
end

function Base.pop!(q::MultinomialQueue)
    if q.k ≤ length(q.queue)
        draw = @inbounds q.queue[q.perm[q.k]]
        q.k += 1
        return draw
    else
        rng = TaskLocalRNG()
        resample!(rng, q.queue, MultinomialResampling(), q.weights, q.cache)
        randperm!(rng, q.perm)
        q.k = 2
        return @inbounds q.queue[q.perm[1]] # if queue is empty than resample! would have failed
    end
end
