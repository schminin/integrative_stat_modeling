using Particles

compute_addfun(ssm::StateSpaceModel, data::AbstractVector, addfun::Particles.AdditiveFunction, method::Particles.AfsMethod; kwargs...) = compute_addfun(ssm, nothing, data, addfun, method; kwargs...)
function compute_addfun(ssm::StateSpaceModel, theta, data::AbstractVector, addfun::Particles.AdditiveFunction, method::Particles.AfsMethod; alpha::Real=0.5, nparticles::Integer, nruns::Integer)
    bf = BootstrapFilter(ssm, data)
    pf = SMC(
        bf, theta, nparticles,
        (addfun=RunningSummary(AdditiveFunctionSmoother(addfun, method), FullHistory()), ),
        AdaptiveResampling(SystematicResampling(), alpha),
    )
    T = Particles.return_type(addfun, bf)
    values = Matrix{T}(undef, length(data), nruns)
    for k in Base.OneTo(nruns)
        Particles.reset!(pf)
        offlinefilter!(pf)
        values[:, k] = map(copy, pf.history_run.addfun)
    end
    return values
end

compute_addfun_last(ssm::StateSpaceModel, data::AbstractVector, addfun::Particles.AdditiveFunction, method::Particles.AfsMethod, offline::Val=Val(false); kwargs...) = compute_addfun_last(ssm, nothing, data, addfun, method, offline; kwargs...)
function compute_addfun_last(ssm::StateSpaceModel, theta, data::AbstractVector, addfun::Particles.AdditiveFunction, method::Particles.AfsMethod, ::Val{offline}=Val(false); alpha::Real=0.5, nparticles::Integer, nruns::Integer) where {offline}
    bf = BootstrapFilter(ssm, data)
    afs = AdditiveFunctionSmoother(addfun, method)
    summary = offline ? OfflineSummary(afs) : afs
    amortized = Particles.required_amortized_computations(summary)
    summaries = offline ? NamedTuple() : (addfun=summary, )
    phistorylen = offline ? ParticleHistoryLength(summary) : ParticleHistoryLength()
    pf = SMC(
        bf, theta, nparticles,
        phistorylen,
        summaries, amortized,
        AdaptiveResampling(SystematicResampling(), alpha),
    )
    T = Particles.return_type(addfun, bf)
    values = Vector{T}(undef, nruns)
    for k in Base.OneTo(nruns)
        Particles.reset!(pf)
        offlinefilter!(pf)
        values[k] = offline ? compute_summary(pf, summary) : compute_summary(pf, :addfun)
    end
    return values
end
