module Particles

using Random
using Statistics

using Requires
using StaticArraysCore

using Static
using LoopVectorization
using Distributions

using DiffResults

export EndMinus, END

export FeynmanKacModel
export ObservedTrajectory
export StateSpaceModel
export BootstrapFilter
export replace_PX0

export FullHistory
export FiniteHistory
export DynamicFiniteHistory, StaticFiniteHistory
export ParticleHistoryLength

export SystematicResampling
export AdaptiveResampling

export SMC
export reset!
export preallocate!
export onlinefilter!
export offlinefilter!
export compute_summary, compute_summary!

export Summary
export RunningSummary
export OfflineSummary

export MeanAndVariance
export UniqueAncestors

export AdditiveFunction
export AdditiveFunctionSmoother
export NaiveAfsMethod, ON2AfsMethod
export AmortizedNaiveAfsMethod
export AdaSmooth
export AmortizedAdaSmooth

export Score
export QFunction, QGradient

export LogLikelihood_NoGradient, NegLogLikelihood_NoGradient, BatchedLogLikelihood_NoGradient
export LogLikelihood, NegLogLikelihood, BatchedLogLikelihood, ThreadedBatchedLogLikelihood
export QFunctor, BatchedQFunctor, ThreadedBatchedQFunctor
export SummaryFunctor, BatchedSummaryFunctor
export set_parameters!, set_batch!

export Deterministic
export ConvertDistribution
export ClampFiniteDistribution
export CheckFiniteDistribution
export Mixture

function __init__()
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("ad/forwarddiff.jl")
    @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" include("ad/reversediff.jl")
    @require Enzyme="7da242da-08ed-463a-9acd-ee780be4f1d9" include("ad/enzyme.jl")
    @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin
        include("ad/zygote.jl")
        @require StaticArraysCore="1e83bf80-4336-4d27-bf5d-d5a4f845583c" include("ad/zygote_sarrays.jl")
    end
    @require PlotlyJS="f0f68f2c-4968-5e81-91da-67840de0976a" include("plots.jl")
end

const ENABLE_ASSERTX = false
macro assertx(expr)
    ENABLE_ASSERTX ? esc(:(@assert $expr)) : nothing
end

const SUMMARY_OUTER_THREADING = true
const SUMMARY_INNER_THREADING = true
const PF_THREADING = true
macro threadsx(which::Symbol, loop::Expr)
    loop.head === :for || throw(ArgumentError("second argument to @threadsx must be a for loop"))
    threaded = if which === :pf
        PF_THREADING
    elseif which === :summary
        SUMMARY_INNER_THREADING
    else
        error("which = $which in @threadsx")
    end
    threaded_loop = threaded ? :(Threads.@threads $loop) : loop
    return esc(threaded_loop)
end

include("utils.jl")
include("stats.jl")
include("distributions.jl")
include("staticarrays.jl")
include("circular.jl")
include("autocache.jl")
include("ad/adbackend.jl")

include("feynmankac.jl")
include("observations.jl")
include("statespacemodel.jl")
include("statespacemodel_simulation.jl")
include("bootstrap.jl")

include("history_length.jl")
include("summary_core.jl")
include("history.jl")
include("weights.jl")
include("resampling.jl")
include("smc.jl")

include("summary.jl")
include("summary_moments.jl")
include("summary_ancestors.jl")

include("additive/functions.jl")
include("additive/smoothers.jl")

include("functors.jl")

end
