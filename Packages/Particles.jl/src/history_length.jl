abstract type HistoryLength end

struct FullHistory <: HistoryLength end

abstract type FiniteHistory <: HistoryLength end

struct DynamicFiniteHistory <: FiniteHistory
    len::Int
    function DynamicFiniteHistory(len::Integer)
        len ≥ 0 || throw(ArgumentError("history length must be ≥ 0"))
        return new(len)
    end
end

struct StaticFiniteHistory{L} <: FiniteHistory
    len::StaticInt{L}
    function StaticFiniteHistory{L}() where {L}
        L ≥ 0 || throw(ArgumentError("history length must be ≥ 0"))
        return new{L}()
    end
end
StaticFiniteHistory(::StaticInt{L}) where {L} = StaticFiniteHistory{L}()
StaticFiniteHistory(len::Integer) = StaticFiniteHistory{convert(Int, len)}()

const NoHistory = StaticFiniteHistory{0}

FiniteHistory(i::Integer) = DynamicFiniteHistory(i)
FiniteHistory(::StaticInt{L}) where {L} = StaticFiniteHistory{L}()

maxlength(::FullHistory) = static(typemax(Int))
maxlength(fh::FiniteHistory) = fh.len

Base.:+(::FullHistory, ::FiniteHistory) = FullHistory()
Base.:+(::FiniteHistory, ::FullHistory) = FullHistory()
Base.:+(::FullHistory, ::FullHistory) = FullHistory()
Base.:+(fh1::FiniteHistory, fh2::FiniteHistory) = FiniteHistory(fh1.len + fh2.len)
Base.:+(len::Union{StaticInt, Integer}, h::HistoryLength) = FiniteHistory(len) + h
Base.:+(h::HistoryLength, len::Union{StaticInt, Integer}) = FiniteHistory(len) + h

Base.:-(::FullHistory, ::FiniteHistory) = FullHistory()
Base.:-(::HistoryLength, ::FullHistory) = error("cannot subtract FullHistory()")
Base.:-(fh1::FiniteHistory, fh2::FiniteHistory) = FiniteHistory(fh1.len - fh2.len)
Base.:-(len::Union{StaticInt, Integer}, h::HistoryLength) = FiniteHistory(len) - h
Base.:-(h::HistoryLength, len::Union{StaticInt, Integer}) = h - FiniteHistory(len)

Base.max(::FullHistory, ::FiniteHistory) = FullHistory()
Base.max(::FiniteHistory, ::FullHistory) = FullHistory()
Base.max(::FullHistory, ::FullHistory) = FullHistory()
Base.max(fh1::FiniteHistory, fh2::FiniteHistory) = FiniteHistory(max(fh1.len, fh2.len))
Base.max(len::Union{StaticInt, Integer}, h::HistoryLength) = max(FiniteHistory(len), h)
Base.max(h::HistoryLength, len::Union{StaticInt, Integer}) = max(FiniteHistory(len), h)

Base.min(::FullHistory, fh::FiniteHistory) = fh
Base.min(fh::FiniteHistory, ::FullHistory) = fh
Base.min(::FullHistory, ::FullHistory) = FullHistory()
Base.min(fh1::FiniteHistory, fh2::FiniteHistory) = FiniteHistory(min(fh1.len, fh2.len))
Base.min(len::Union{StaticInt, Integer}, h::HistoryLength) = min(FiniteHistory(len), h)
Base.min(h::HistoryLength, len::Union{StaticInt, Integer}) = min(FiniteHistory(len), h)

Base.convert(::Type{T}, fh::FiniteHistory) where {T <: Integer} = convert(T, fh.len)
Base.convert(::Type{T}, fh::FullHistory) where {T <: Integer} = error("cannot convert FullHistory to Integer")
Base.convert(::Type{FiniteHistory}, len::Union{Integer, StaticInt}) = FiniteHistory(len)
Base.convert(::Type{HistoryLength}, len::Union{Integer, StaticInt}) = FiniteHistory(len)

##########################################################################################################

make_history(L::HistoryLength, cmps::NamedTuple) = map(cmp -> make_history(L, cmp), cmps)

make_history(L::HistoryLength, mkvalue) = make_history(L, mkvalue, ())
make_history(L::HistoryLength, mkvalue, size::Vararg{Int}) = make_history(L, mkvalue, size)

make_history(::FullHistory, ::Type{T}, ::Tuple{}) where {T} = PreallocatableVector{T}()
make_history(::FullHistory, ::Type{T}, size::Dims) where {T} = PreallocatableVectorOfArrays{T}(size)
make_history(::FullHistory, mkvalue, ::Tuple{}) = PreallocatableVectorOfMutables(mkvalue)

make_history(fh::FiniteHistory, ::Type{T}, ::Tuple{}) where {T} = CircularVector{T}(convert(Int, fh.len))
make_history(fh::FiniteHistory, ::Type{T}, size::Dims) where {T} = CircularVectorOfArrays{T}(convert(Int, fh.len), size)
make_history(fh::FiniteHistory, mkvalue, ::Tuple{}) = CircularVectorOfMutables(convert(Int, fh.len), mkvalue)

make_history(::StaticFiniteHistory{0}, ::Type{T}, ::Tuple{}) where {T} = CircularVector0{T}()
make_history(::StaticFiniteHistory{0}, ::Type{T}, size::Dims) where {T} = CircularVectorOfArrays0{T}(size)
make_history(::StaticFiniteHistory{0}, mkvalue, ::Tuple{}) = CircularVectorOfMutables0(mkvalue)

make_history(::StaticFiniteHistory{1}, ::Type{T}, ::Tuple{}) where {T} = CircularVector1{T}()
make_history(::StaticFiniteHistory{1}, ::Type{T}, size::Dims) where {T} = CircularVectorOfArrays1{T}(size)
make_history(::StaticFiniteHistory{1}, mkvalue, ::Tuple{}) = CircularVectorOfMutables1(mkvalue)

make_history(::StaticFiniteHistory{2}, ::Type{T}, ::Tuple{}) where {T} = CircularVector2{T}()
make_history(::StaticFiniteHistory{2}, ::Type{T}, size::Dims) where {T} = CircularVectorOfArrays2{T}(size)
make_history(::StaticFiniteHistory{2}, mkvalue, ::Tuple{}) = CircularVectorOfMutables2(mkvalue)

##########################################################################################################

HistoryLength(::PreallocatableVector) = FullHistory()
HistoryLength(::PreallocatableVectorOfArrays) = FullHistory()
HistoryLength(::PreallocatableVectorOfMutables) = FullHistory()

HistoryLength(cv::CircularVector) = DynamicFiniteHistory(maxlength(cv))
HistoryLength(cv::CircularVectorOfArrays) = DynamicFiniteHistory(maxlength(cv))
HistoryLength(cv::CircularVectorOfMutables) = DynamicFiniteHistory(maxlength(cv))

HistoryLength(::CircularVector0) = StaticFiniteHistory{0}()
HistoryLength(::CircularVector1) = StaticFiniteHistory{1}()
HistoryLength(::CircularVector2) = StaticFiniteHistory{2}()

HistoryLength(::CircularVectorOfArrays0) = StaticFiniteHistory{0}()
HistoryLength(::CircularVectorOfArrays1) = StaticFiniteHistory{1}()
HistoryLength(::CircularVectorOfArrays2) = StaticFiniteHistory{2}()

HistoryLength(::CircularVectorOfMutables0) = StaticFiniteHistory{0}()
HistoryLength(::CircularVectorOfMutables1) = StaticFiniteHistory{1}()
HistoryLength(::CircularVectorOfMutables2) = StaticFiniteHistory{2}()

##########################################################################################################

struct ParticleHistoryLength{L_P <: HistoryLength, L_LW <: HistoryLength, L_W <: HistoryLength, L_A <: HistoryLength, L_R <: HistoryLength, L_C <: HistoryLength}
    particles::L_P
    logweights::L_LW
    weights::L_W
    ancestors::L_A
    didresample::L_R
    logCnorm::L_C
    function ParticleHistoryLength(particles::HistoryLength, logweights::HistoryLength, weights::HistoryLength, ancestors::HistoryLength, didresample::HistoryLength, logCnorm::HistoryLength)
        maxlength(particles)  ≥ 2 || throw(ArgumentError("particle history length should be at least 2"))
        maxlength(logweights) ≥ 1 || throw(ArgumentError("logweights history length should be at least 1"))
        maxlength(weights)    ≥ 1 || throw(ArgumentError("weights history length should be at least 1"))
        maxlength(ancestors)  ≥ 1 || throw(ArgumentError("ancestors history length should be at least 1"))
        return new{typeof(particles), typeof(logweights), typeof(weights), typeof(ancestors), typeof(didresample), typeof(logCnorm)}(particles, logweights, weights, ancestors, didresample, logCnorm)
    end
end

ParticleHistoryLength(; particles=static(2), logweights=static(1), weights=static(1), ancestors=static(1), didresample=static(0), logCnorm=static(0)) = ParticleHistoryLength(particles, logweights, weights, ancestors, didresample, logCnorm)
ParticleHistoryLength(phl::ParticleHistoryLength; particles=phl.particles, logweights=phl.logweights, weights=phl.weights, ancestors=phl.ancestors, didresample=phl.didresample, logCnorm=phl.logCnorm) = ParticleHistoryLength(particles, logweights, weights, ancestors, didresample, logCnorm)

ParticleHistoryLength(h::HistoryLength) = ParticleHistoryLength(max(h, FiniteHistory(static(2))), max(h, FiniteHistory(static(1))), max(h, FiniteHistory(static(1))), max(h, FiniteHistory(static(1))), h, h)
ParticleHistoryLength(L::Union{StaticInt, Integer}) = ParticleHistoryLength(FiniteHistory(L))

function ParticleHistoryLength(particles::Union{HistoryLength, StaticInt, Integer}, logweights::Union{HistoryLength, StaticInt, Integer}, weights::Union{HistoryLength, StaticInt, Integer}, ancestors::Union{HistoryLength, StaticInt, Integer}, didresample::Union{HistoryLength, StaticInt, Integer}, logCnorm::Union{HistoryLength, StaticInt, Integer})
    return ParticleHistoryLength(
        convert(HistoryLength, particles),
        convert(HistoryLength, logweights),
        convert(HistoryLength, weights),
        convert(HistoryLength, ancestors),
        convert(HistoryLength, didresample),
        convert(HistoryLength, logCnorm),
    )
end

Base.max(phs1::ParticleHistoryLength, phs2::ParticleHistoryLength) = ParticleHistoryLength(
    max(phs1.particles, phs2.particles),
    max(phs1.logweights, phs2.logweights),
    max(phs1.weights, phs2.weights),
    max(phs1.ancestors, phs2.ancestors),
    max(phs1.didresample, phs2.didresample),
    max(phs1.logCnorm, phs2.logCnorm),
)
