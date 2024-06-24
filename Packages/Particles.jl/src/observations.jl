const Observations{T} = AbstractVector{<:Union{T, Missing}}
const FullObservations{T} = AbstractVector{T}
const DenseObservations{T} = AbstractVector{Union{T, Missing}}

# struct DenseObservedTrajectory{T, S <: Union{T, Missing}} <: ObservedTrajectory{T}
#     observations::Vector{S}
#     function DenseObservedTrajectory(observations::AbstractVector{S}) where {T, S <: Union{T, Missing}}
#         isempty(observations) && throw(ArgumentError("empty trajectory"))
#         T === Missing && throw(ArgumentError("observations are missing at all time points"))
#         return new{T, S}(observations)
#     end
# end
# Base.parent(trajectory::DenseObservedTrajectory) = trajectory.observations
# Base.size(trajectory::DenseObservedTrajectory) = size(parent(trajectory))
# Base.@propagate_inbounds Base.getindex(trajectory::DenseObservedTrajectory, i::Integer) = getindex(parent(trajectory), i)
# Base.IndexStyle(::Type{DenseObservedTrajectory}) = IndexLinear()
