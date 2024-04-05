struct ImmutableAdditiveFunctionSmoother{METHOD <: AfsMethod, F <: ImmutableAdditiveFunction} <: ImmutableSummary
    addfun::F
    method::METHOD
    function ImmutableAdditiveFunctionSmoother(addfun::ImmutableAdditiveFunction, method::AfsMethod=DEFAULT_AFSMETHOD())
        return new{typeof(method), typeof(addfun)}(addfun, method)
    end
end

struct MutableAdditiveFunctionSmoother{METHOD <: AfsMethod, F <: MutableAdditiveFunction} <: MutableSummary
    addfun::F
    method::METHOD
    function MutableAdditiveFunctionSmoother(addfun::MutableAdditiveFunction, method::AfsMethod=DEFAULT_AFSMETHOD())
        return new{typeof(method), typeof(addfun)}(addfun, method)
    end
end

AdditiveFunctionSmoother(::AdditiveFunction, ::AfsMethod=DEFAULT_AFSMETHOD()) = error("AdditiveFunctions must derived from either ImmutableAdditiveFunction or MutableAdditiveFunction")
AdditiveFunctionSmoother(addfun::ImmutableAdditiveFunction, method::AfsMethod=DEFAULT_AFSMETHOD()) = ImmutableAdditiveFunctionSmoother(addfun, method)
AdditiveFunctionSmoother(addfun::MutableAdditiveFunction, method::AfsMethod=DEFAULT_AFSMETHOD()) = MutableAdditiveFunctionSmoother(addfun, method)

# TODO The explicit division in two types could be solved by turning Mutable/Immutable in a type parameter of Summary

const AbstractAdditiveFunctionSmoother{METHOD <: AfsMethod} = Union{ImmutableAdditiveFunctionSmoother{METHOD}, MutableAdditiveFunctionSmoother{METHOD}}
const ArrayAdditiveFunctionSmoother{METHOD <: AfsMethod} = MutableAdditiveFunctionSmoother{METHOD, <: ArrayAdditiveFunction}

const AfsTmp{METHOD} = Temporaries{<:AbstractAdditiveFunctionSmoother{METHOD}}
const AfsOff{METHOD} = OfflineSummary{<:AbstractAdditiveFunctionSmoother{METHOD}}
# const ImmAfsTmp{METHOD} = Temporaries{<:ImmutableAdditiveFunctionSmoother{METHOD}}
# const ImmAfsOff{METHOD} = OfflineSummary{<:ImmutableAdditiveFunctionSmoother{METHOD}}
# const MutAfsTmp{METHOD} = Temporaries{<:MutableAdditiveFunctionSmoother{METHOD}}
# const MutAfsOff{METHOD} = OfflineSummary{<:MutableAdditiveFunctionSmoother{METHOD}}
# TODO cleanup and make uniform this notation

function offline_computation_starts_from(::AfsOff, fkm::FeynmanKacModel, t::Integer)
    return initial_time(fkm)
end

check(afs::AbstractAdditiveFunctionSmoother, smc::SMC) = check(afs.addfun, model(smc))

function make_template(summary::MutableAdditiveFunctionSmoother, fkm::FeynmanKacModel)
    mkvalue = template_maker(summary.addfun, fkm)
    return mkvalue()
end
function make_template(os::OfflineSummary{<:MutableAdditiveFunctionSmoother}, fkm::FeynmanKacModel)
    return make_template(os.summary, fkm)
end

function make_cache(tmp::Temporaries{<:AbstractAdditiveFunctionSmoother}, fkm::FeynmanKacModel, nparticles::Integer)
    addfun = tmp.summary.addfun
    return AutoCache(() -> (cache_af=make_cache(addfun, fkm), ), nparticles)
end

function make_history(tmp::Temporaries{<:ImmutableAdditiveFunctionSmoother}, fkm::FeynmanKacModel, nparticles::Integer)
    extra_history = make_extra_temporaries(tmp, fkm, nparticles)
    L = StaticFiniteHistory{2}()
    T = return_type(tmp.summary.addfun, fkm)
    Phi = make_history(L, T, nparticles)
    return (; Phi=SummaryHistory(Phi), extra_history...)
end

function make_history(tmp::Temporaries{<:MutableAdditiveFunctionSmoother}, fkm::FeynmanKacModel, nparticles::Integer)
    extra_history = make_extra_temporaries(tmp, fkm, nparticles)
    L = StaticFiniteHistory{2}()
    T = return_type(tmp.summary.addfun, fkm)
    mkvalue = template_maker(tmp.summary.addfun, fkm)
    mkvector = () -> [mkvalue()::T for _ in Base.OneTo(nparticles)]
    Phi = make_history(L, mkvector)
    return (; Phi=SummaryHistory(Phi), extra_history...)
end

make_extra_temporaries(::AfsTmp, ::FeynmanKacModel, nparticles::Integer) = NamedTuple()

function make_history(rs::RunningSummary{<:AbstractAdditiveFunctionSmoother}, fkm::FeynmanKacModel, nparticles::Integer)
    mkvalue = template_maker(rs.summary.addfun, fkm)
    storage = make_history(rs.Lhistory, mkvalue)
    return SummaryHistory(storage; computable_from=1)
end

Base.@propagate_inbounds function compute_summary(smc::SMC, ::ImmutableAdditiveFunctionSmoother, history_tmp::SameTypeNamedTuple{SummaryHistory}, t::Integer, index_in_history::EndMinus)
    Phi = history_tmp.Phi[index_in_history]
    weights = smc.history_pf.weights[index_in_history]
    return weighted_mean(Phi, weights)
end

Base.@propagate_inbounds function compute_summary!(out, smc::SMC, s::MutableAdditiveFunctionSmoother, history_tmp::SameTypeNamedTuple{SummaryHistory}, t::Integer, index_in_history::EndMinus)
    Phi = history_tmp.Phi[index_in_history]
    weights = smc.history_pf.weights[index_in_history]
    return weighted_mean!(s.addfun, out, Phi, weights)
end

Base.@propagate_inbounds function compute_temporaries!(smc::SMC, tmp::AfsTmp, history_tmp::SameTypeNamedTuple{SummaryHistory}, autocache::AutoCache, t::Integer, index_in_history::EndMinus, ::Val{:initial})
    fkm, θ = model(smc), smc.θ
    afs = tmp.summary
    Phi = next!(history_tmp.Phi; return_current=false)
    particles = smc.history_pf.particles[index_in_history]
    @threadsx summary for i in eachindex(Phi)
        X = @inbounds particles[i]
        cache = @inbounds autocache[i]
        if isa(afs.addfun, ImmutableAdditiveFunction)
            @inbounds Phi[i] = afs.addfun(fkm, θ, X, cache.cache_af)
        else
            @inbounds afs.addfun(Phi[i], fkm, θ, X, cache.cache_af)
        end
        release!(autocache, i)
    end
    initialize_extra_temporaries!(smc, tmp, history_tmp, autocache, t, index_in_history)
    return nothing
end

initialize_extra_temporaries!(::SMC, ::AfsTmp, ::SameTypeNamedTuple{SummaryHistory}, ::AutoCache, t::Integer, index_in_history::EndMinus) = nothing

##############################################################################################

include("smoothers/naive.jl")
include("smoothers/naive_amortized.jl")
include("smoothers/on2.jl")
include("smoothers/multinomial_queue.jl") # needed for Pairs smoother
include("smoothers/paris.jl")
include("smoothers/adasmooth.jl")
include("smoothers/adasmooth_amortized.jl")

const DEFAULT_AFSMETHOD = NaiveAfsMethod
