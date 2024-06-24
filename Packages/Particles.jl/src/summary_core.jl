struct SummaryHistory{T, V <: AbstractVector{T}} <: AbstractVector{T}
    v::V
    computable_from::Int
    """
        SummaryHistory(v::AbstractVector, computable_from::Integer)
    Wraps a vector in order to store the history for a `RunningSummary` or a `Summary` temporary.
    The positive integer `computable_from` is the step of the particle filter at which the relevant quantity can be computed for the first time.
    """
    function SummaryHistory(v::AbstractVector{T}; computable_from::Int=1) where {T}
        computable_from > 0 || throw(ArgumentError("computable_from must be > 0"))
        return new{T, typeof(v)}(v, computable_from)
    end
end

Base.IndexStyle(::Type{SummaryHistory{T, V}}) where {T, V} = IndexStyle(V)
Base.parent(h::SummaryHistory) = h.v
Base.size(h::SummaryHistory) = size(parent(h))
Base.@propagate_inbounds Base.getindex(h::SummaryHistory, i::Integer) = getindex(parent(h), i)
Base.push!(h::SummaryHistory, value) = push!(parent(h), value)
Base.@propagate_inbounds next!(h::SummaryHistory; return_current::Bool=false) = next!(parent(h); return_current)

function preallocate!(h::SummaryHistory, n::Int)
    m = max(0, n - (h.computable_from - 1))
    return preallocate!(h.v, m)
end

function Base.empty!(h::SummaryHistory)
    empty!(h.v)
    @assertx isempty(h.v)
    return h
end

###########################################################################################

abstract type Summary end
abstract type ImmutableSummary <: Summary end
abstract type MutableSummary <: Summary end
# abstract type ArraySummary{N} <: MutableSummary end
# const VectorSummary = ArraySummary{1}

"""
    make_template(s::MutableSummary, fkm::FeynmanKacModel)
Create a template for the value of a `MutableSummary`.
"""
make_template(::MutableSummary, ::FeynmanKacModel)

abstract type AmortizedComputation <: Summary end

struct OfflineAC{AMO <: AmortizedComputation} <: AmortizedComputation
    amortized::AMO
end
OfflineAC(oac::OfflineAC) = throw(ArgumentError("OfflineAC cannot be applied twice"))

# NB
#     Temporaries(oac::OfflineAC) != Temporaries(oac.amortized)
# We have to keep track of the offline status in order to create temporaries of the correct length

const MaybeOfflineAC{T <: AmortizedComputation} = Union{OfflineAC{<:T}, T}

"""
    required_amortized_computations(s::Summary)
Return a tuple of `AmortizedComputations` that are required to compute the given `Summary`.
"""
required_amortized_computations(::Summary) = ()

"""
    Temporaries{S <: Summary}
`Temporaries(s)` is syntactic sugar to denote the temporaries of a `Summary`.
"""
struct Temporaries{S <: Summary}
    summary::S
end

required_amortized_computations(tmp::Temporaries) = required_amortized_computations(tmp.summary)

struct RunningSummary{S <: Summary, HL <: HistoryLength}
    summary::S
    Lhistory::HL
    function RunningSummary(s::Summary, L::HistoryLength)
        isa(L, FullHistory) || L.len > 0 || throw(ArgumentError("history length must be at least one (otherwise why having a running summary at all?)"))
        return new{typeof(s), typeof(L)}(s, L)
    end
end
RunningSummary(s::Summary, L::StaticInt=static(1)) = RunningSummary(s, convert(HistoryLength, L))
RunningSummary(s::AmortizedComputation, L::HistoryLength=static(1)) = throw(ArgumentError("AmortizedComputations have no values, only temporaries"))

Temporaries(rs::RunningSummary) = Temporaries(rs.summary)
required_amortized_computations(rs::RunningSummary) = required_amortized_computations(rs.summary)

struct OfflineSummary{S <: Summary}
    summary::S
end
OfflineSummary(s::AmortizedComputation) = throw(ArgumentError("AmortizedComputations have no values, only temporaries"))

Temporaries(os::OfflineSummary) = Temporaries(os.summary)
required_amortized_computations(os::OfflineSummary) = map(OfflineAC, required_amortized_computations(os.summary))
