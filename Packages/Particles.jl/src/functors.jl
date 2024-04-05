struct LogLikelihood_NoGradient{PF <: SMC, CACHE}
    pf::PF
    cache::CACHE
    function LogLikelihood_NoGradient(ssm::StateSpaceModel, data; nparticles::Integer)
        bf = BootstrapFilter(ssm, data)
        pf = SMC(
            bf, parameter_template(ssm), nparticles,
            ParticleHistoryLength(; logCnorm=StaticFiniteHistory{1}()),
        )
        cache = SMCCache(pf)
        return new{typeof(pf), typeof(cache)}(pf, cache)
    end
    function LogLikelihood_NoGradient(llh::LogLikelihood_NoGradient, data::Observations; recycle::Bool=false)
        bf = model(llh.pf)::BootstrapFilter
        ssm = statespacemodel(bf)
        eltype(data) == eltype(bf.observations) || throw(ArgumentError("observation type must coincide with the one for the given llh"))
        bf = BootstrapFilter(ssm, data)::typeof(model(llh.pf))
        pf = SMC(llh.pf, bf; recycle)
        cache = recycle ? llh.cache : deepcopy(llh.cache)
        return new{typeof(pf), typeof(cache)}(pf, cache)
    end
end

model(fun::LogLikelihood_NoGradient) = model(fun.pf)
statespacemodel(fun::LogLikelihood_NoGradient{<:StateSpaceSMC}) = statespacemodel(fun.pf)
nparticles(fun::LogLikelihood_NoGradient) = nparticles(fun.pf)

function (llh::LogLikelihood_NoGradient)(p)::Float64
    reset!(llh.pf, p)
    offlinefilter!(llh.pf, llh.cache)
    return llh.pf.history_pf.logCnorm[end]
end

################################################################################################################################

struct NegLogLikelihood_NoGradient{LLH <: LogLikelihood_NoGradient}
    llh::LLH
end

function NegLogLikelihood_NoGradient(ssm::StateSpaceModel, args...; kwargs...)
    llh = LogLikelihood_NoGradient(ssm, args...; kwargs...)
    return NegLogLikelihood_NoGradient(llh)
end

model(fun::NegLogLikelihood_NoGradient) = model(fun.llh)
statespacemodel(fun::NegLogLikelihood_NoGradient{<:LogLikelihood_NoGradient{<:StateSpaceSMC}}) = statespacemodel(fun.llh)
nparticles(fun::NegLogLikelihood_NoGradient) = nparticles(fun.llh)

(nllh::NegLogLikelihood_NoGradient)(p) = -nllh.llh(p)

################################################################################################################################

struct LogLikelihood{PF1 <: SMC, PF2 <: SMC, CACHE1, CACHE2, SCORE}
    pf::PF1
    pf_grad::PF2
    cache::CACHE1
    cache_grad::CACHE2
    score_summary::SCORE
    function LogLikelihood(ssm::StateSpaceModel, data::Observations, method::AfsMethod=DEFAULT_AFSMETHOD(), ad_backend::ADBackend=ADBackend(ssm); nparticles::Integer, full_ancestors_history::Bool=false)
        bf = BootstrapFilter(ssm, data)
        pf = SMC(
            bf, parameter_template(ssm), nparticles,
            full_ancestors_history ? ParticleHistoryLength(; ancestors=FullHistory(), logCnorm=StaticFiniteHistory{1}()) : ParticleHistoryLength(; logCnorm=StaticFiniteHistory{1}()),
        )
        cache = SMCCache(pf)
        if isa(method, AmortizedNaiveAfsMethod)
            score_summary = OfflineSummary(Score(ad_backend, method))
            pf_grad = SMC(
                bf, parameter_template(ssm), nparticles,
                ParticleHistoryLength(score_summary; logCnorm=StaticFiniteHistory{1}()),
                required_amortized_computations(score_summary),
            )
        else
            score_summary = Score(ad_backend, method)
            pf_grad = SMC(
                bf, parameter_template(ssm), nparticles,
                ParticleHistoryLength(; logCnorm=StaticFiniteHistory{1}()),
                (score=score_summary, ),
            )
        end
        cache_grad = SMCCache(pf_grad)
        return new{typeof(pf), typeof(pf_grad), typeof(cache), typeof(cache_grad), typeof(score_summary)}(pf, pf_grad, cache, cache_grad, score_summary)
    end
    function LogLikelihood(llh::LogLikelihood, data::Observations; recycle::Bool=false)
        bf = model(llh.pf)::BootstrapFilter
        ssm = statespacemodel(bf)
        eltype(data) == eltype(bf.observations) || throw(ArgumentError("observation type must coincide with the one for the given llh"))
        bf = BootstrapFilter(ssm, data)::typeof(model(llh.pf))
        pf = SMC(llh.pf, bf; recycle)
        pf_grad = SMC(llh.pf_grad, bf; recycle)
        cache = recycle ? llh.cache : deepcopy(llh.cache)
        cache_grad = recycle ? llh.cache_grad : deepcopy(llh.cache_grad)
        return new{typeof(pf), typeof(pf_grad), typeof(cache), typeof(cache_grad), typeof(llh.score_summary)}(pf, pf_grad, cache, cache_grad, llh.score_summary)
    end
end

model(fun::LogLikelihood) = model(fun.pf)
statespacemodel(fun::LogLikelihood{<:StateSpaceSMC}) = statespacemodel(fun.pf)
nparticles(fun::LogLikelihood) = nparticles(fun.pf)

function make_template(llh::LogLikelihood{PF1, PF2}, ::Val{:gradient}) where {PF1, PF2}
    pf = llh.pf_grad::PF2 # helps with type instability of the return type (but property access is still unstable?!?)
    return make_template(llh.score_summary, pf.fkm)
end

function (llh::LogLikelihood)(gradient, p)
    reset!(llh.pf_grad, p)
    offlinefilter!(llh.pf_grad, llh.cache_grad)
    if isa(llh.score_summary, OfflineSummary)
        compute_summary!(gradient, llh.pf_grad, llh.score_summary)
    else
        compute_summary!(gradient, llh.pf_grad, :score)
    end
    return llh.pf_grad.history_pf.logCnorm[end]::Float64
end
function (llh::LogLikelihood)(::Val{:return}, p)
    gradient = similar(p, Float64)
    y = llh(gradient, p)
    return y::Float64, gradient
end
function (llh::LogLikelihood)(p)
    reset!(llh.pf, p)
    offlinefilter!(llh.pf, llh.cache)
    return llh.pf.history_pf.logCnorm[end]::Float64
end

################################################################################################################################

struct NegLogLikelihood{LLH <: LogLikelihood}
    llh::LLH
end

function NegLogLikelihood(ssm::StateSpaceModel, args...; kwargs...)
    llh = LogLikelihood(ssm, args...; kwargs...)
    return NegLogLikelihood(llh)
end

model(fun::NegLogLikelihood) = model(fun.llh)
statespacemodel(fun::NegLogLikelihood{<:LogLikelihood{<:StateSpaceSMC}}) = statespacemodel(fun.llh)
nparticles(fun::NegLogLikelihood) = nparticles(fun.llh)

make_template(nllh::NegLogLikelihood, ::Val{:gradient}) = make_template(nllh.llh, Val(:gradient))

function (nllh::NegLogLikelihood)(gradient, p)
    llh = nllh.llh(gradient, p)
    @turbo @. gradient = -gradient
    return -llh
end
function (nllh::NegLogLikelihood)(::Val{:return}, p)
    llh, gradient = nllh.llh(Val(:return), p)
    @turbo @. gradient = -gradient
    return -llh, gradient
end
(nllh::NegLogLikelihood)(p) = -nllh.llh(p)

################################################################################################################################

struct BatchedLogLikelihood{TRAJ <: Observations, use_median, LLH <: LogLikelihood, TMP}
    llh::LLH
    batch::Vector{TRAJ}
    tmp::TMP
    function BatchedLogLikelihood(ssm::StateSpaceModel{TX, TY}, data::Observations{TY}, ::Val{use_median}, args...; kwargs...) where {TX, TY, use_median}
        isa(use_median, Bool) || throw(ArgumentError("use_median must be a Bool"))
        TRAJ = typeof(data)
        llh = LogLikelihood(ssm, data, args...; kwargs...)
        grad_template = make_template(llh, Val(:gradient))
        tmp = use_median ? [grad_template] : grad_template
        use_median && Base.has_offset_axes(grad_template) && throw(ArgumentError("when use_median=true, gradients with offset axes are not supported"))
        return new{TRAJ, use_median, typeof(llh), typeof(tmp)}(llh, TRAJ[], tmp)
    end
end

function BatchedLogLikelihood(ssm::StateSpaceModel{TX, TY}, data::Observations{TY}, args...; use_median::Bool=false, kwargs...) where {TX, TY}
    return BatchedLogLikelihood(ssm, data, Val(use_median), args...; kwargs...)
end
function BatchedLogLikelihood(ssm::StateSpaceModel{TX, TY}, data::AbstractVector{<:Observations{TY}}, args...; kwargs...) where {TX, TY}
    return BatchedLogLikelihood(ssm, first(data), args...; kwargs...)
end
function BatchedLogLikelihood(ssm::StateSpaceModel{TX, TY}, data::AbstractVector{<:AbstractVector{<:Observations{TY}}}, args...; kwargs...) where {TX, TY}
    return BatchedLogLikelihood(ssm, first(first(data)), args...; kwargs...)
end

model(fun::BatchedLogLikelihood) = model(fun.llh)
statespacemodel(fun::BatchedLogLikelihood{<:Observations, <:LogLikelihood{<:StateSpaceSMC}}) = statespacemodel(fun.llh)
nparticles(fun::BatchedLogLikelihood) = nparticles(fun.llh)

make_template(llhb::BatchedLogLikelihood, ::Val{:gradient}) = make_template(llhb.llh, Val(:gradient))

function set_batch!(llhb::BatchedLogLikelihood{TRAJ, use_median}, data::AbstractVector{TRAJ}) where {TRAJ, use_median}
    copy!(llhb.batch, data)
    if use_median
        n = length(llhb.tmp)
        m = length(data)
        resize!(llhb.tmp, m)
        @inbounds for k in (n+1):m
            llhb.tmp[k] = make_template(llhb, Val(:gradient))
        end
    end
    return llhb
end

function (llhb::BatchedLogLikelihood{TRAJ, false})(gradient, p) where {TRAJ}
    tmp = llhb.tmp
    llh = 0.0
    fill!(gradient, zero(eltype(gradient)))
    for data in llhb.batch
        llh_fun = LogLikelihood(llhb.llh, data; recycle=true)
        llh += llh_fun(tmp, p)::Float64
        @turbo @. gradient += tmp
    end
    return llh
end
function (llhb::BatchedLogLikelihood{TRAJ, true})(gradient, p) where {TRAJ}
    Base.has_offset_axes(gradient) && throw(ArgumentError("offset axes not supported"))
    llh = median(
        LogLikelihood(llhb.llh, data; recycle=true)(tmp, p)::Float64
        for (data, tmp) in zip(llhb.batch, llhb.tmp)
    )
    Threads.@threads for k in eachindex(gradient)
        @inbounds let k = k
            gradient[k] = median(tmp[k] for tmp in llhb.tmp)
        end
    end
    return llh
end

function (llhb::BatchedLogLikelihood)(llhs, gradients, p) 
    batchsize = length(llhb.batch)
    length(llhs) == length(gradients) == batchsize || throw(ArgumentError("storage given has not the same size of the current batch"))
    @assert !Base.has_offset_axes(llhb.batch)
    Base.has_offset_axes(llhs, gradients) && throw(ArgumentError("storage given must not have offset axes"))
    for i in Base.OneTo(batchsize)
        llh_fun = LogLikelihood(llhb.llh, @inbounds(llhb.batch[i]); recycle=true)
        llh = llh_fun(@inbounds(gradients[i]), p)
        @inbounds llhs[i] = llh
    end
    return llhs, gradients
end

# NB double method definition in order to resolve ambiguities
function (llhb::BatchedLogLikelihood{TRAJ, true})(::Val{:return}, p) where {TRAJ}
    gradient = similar(p, Float64)
    y = llhb(gradient, p)
    return y::Float64, gradient
end
function (llhb::BatchedLogLikelihood{TRAJ, false})(::Val{:return}, p) where {TRAJ}
    gradient = similar(p, Float64)
    y = llhb(gradient, p)
    return y::Float64, gradient
end

function (llhb::BatchedLogLikelihood{TRAJ, false})(p) where {TRAJ}
    llh = 0.0
    for data in llhb.batch
        llh_fun = LogLikelihood(llhb.llh, data; recycle=true)
        llh += llh_fun(p)::Float64
    end
    return llh
end
function (llhb::BatchedLogLikelihood{TRAJ, true})(p) where {TRAJ}
    return median(
        LogLikelihood(llhb.llh, data; recycle=true)(p)::Float64
        for data in llhb.batch
    )
end

################################################################################################################################

struct ThreadedBatchedLogLikelihood{TRAJ <: Observations, LLH <: LogLikelihood, TMP}
    llhs::Vector{LLH}
    llh_template::LLH # NB llh_template and llhs[1] will share caches and history vectors
    tmps::Vector{TMP} # TODO an autocache could be used, but would require synchronization when growing the gradient
    function ThreadedBatchedLogLikelihood(ssm::StateSpaceModel{TX, TY}, data::Observations{TY}, args...; kwargs...) where {TX, TY}
        llh = LogLikelihood(ssm, data, args...; kwargs...)
        tmp = make_template(llh, Val(:gradient))
        TRAJ, LLH, TMP = typeof(data), typeof(llh), typeof(tmp)
        return new{TRAJ, typeof(llh), TMP}(LLH[], llh, TMP[])
    end
end

function ThreadedBatchedLogLikelihood(ssm::StateSpaceModel{TX, TY}, data::AbstractVector{<:Observations{TY}}, args...; kwargs...) where {TX, TY}
    return ThreadedBatchedLogLikelihood(ssm, first(data), args...; kwargs...)
end
function ThreadedBatchedLogLikelihood(ssm::StateSpaceModel{TX, TY}, data::AbstractVector{<:AbstractVector{<:Observations{TY}}}, args...; kwargs...) where {TX, TY}
    return ThreadedBatchedLogLikelihood(ssm, first(first(data)), args...; kwargs...)
end

model(fun::ThreadedBatchedLogLikelihood) = model(fun.llh_template)
statespacemodel(fun::ThreadedBatchedLogLikelihood{<:Observations, <:LogLikelihood{<:StateSpaceSMC}}) = statespacemodel(fun.llh_template)
nparticles(fun::ThreadedBatchedLogLikelihood) = nparticles(fun.llh_template)

make_template(llhb::ThreadedBatchedLogLikelihood, ::Val{:gradient}) = make_template(llhb.llh_template, Val(:gradient))

function set_batch!(llhb::ThreadedBatchedLogLikelihood{TRAJ}, data::AbstractVector{TRAJ}) where {TRAJ}
    Base.has_offset_axes(data) && throw(ArgumentError("data cannot have offset axes (this could be relaxed)"))
    batchsize = length(data)
    batchsize_before = length(llhb.llhs)
    @assert length(llhb.tmps) == batchsize_before
    resize!(llhb.llhs, batchsize)
    resize!(llhb.tmps, batchsize)
    @inbounds if iszero(batchsize_before)
        llhb.llhs[1] = LogLikelihood(llhb.llh_template, data[1]; recycle=true)
        llhb.tmps[1] = make_template(llhb, Val(:gradient))
        for i in 2:batchsize
            llhb.llhs[i] = LogLikelihood(llhb.llh_template, data[i]; recycle=false)
            llhb.tmps[i] = make_template(llhb, Val(:gradient))
        end
    elseif batchsize ≤ batchsize_before
        for i in 1:batchsize
            llhb.llhs[i] = LogLikelihood(llhb.llhs[i], data[i]; recycle=true)
        end
    else
        for i in 1:batchsize_before
            llhb.llhs[i] = LogLikelihood(llhb.llhs[i], data[i]; recycle=true)
        end
        for i in batchsize_before+1:batchsize
            llhb.llhs[i] = LogLikelihood(llhb.llh_template, data[i]; recycle=false)
            llhb.tmps[i] = make_template(llhb, Val(:gradient))
        end
    end
    return llhb
end

function (llhb::ThreadedBatchedLogLikelihood)(gradient, p)
    llh = Threads.Atomic{Float64}(0.0)
    fill!(gradient, zero(eltype(gradient)))
    Threads.@threads for i in eachindex(llhb.llhs)
        llh_fun = @inbounds llhb.llhs[i]
        tmp = @inbounds llhb.tmps[i]
        Threads.atomic_add!(llh, llh_fun(tmp, p)::Float64)
        # @turbo @. gradient += tmp # TODO could be done here, with additional synchronization
    end
    for tmp in llhb.tmps
        @turbo @. gradient += tmp
    end
    return llh[]
end
function (llhb::ThreadedBatchedLogLikelihood)(::Val{:return}, p)
    gradient = similar(p, Float64)
    y = llhb(gradient, p)
    return y::Float64, gradient
end
function (llhb::ThreadedBatchedLogLikelihood)(p)
    llh = Threads.Atomic{Float64}(0.0)
    Threads.@threads for llh_fun in llhb.llhs
        Threads.atomic_add!(llh, llh_fun(p)::Float64)
    end
    return llh[]
end

################################################################################################################################

struct BatchedLogLikelihood_NoGradient{TRAJ <: Observations, use_median, LLH <: LogLikelihood_NoGradient}
    llh::LLH
    batch::Vector{TRAJ}
    function BatchedLogLikelihood_NoGradient(ssm::StateSpaceModel{TX, TY}, data::Observations{TY}, ::Val{use_median}; kwargs...) where {TX, TY, use_median}
        isa(use_median, Bool) || throw(ArgumentError("use_median must be a Bool"))
        TRAJ = typeof(data)
        llh = LogLikelihood_NoGradient(ssm, data; kwargs...)
        return new{TRAJ, use_median, typeof(llh)}(llh, TRAJ[])
    end
end

function BatchedLogLikelihood_NoGradient(ssm::StateSpaceModel{TX, TY}, data::Observations{TY}; use_median::Bool=false, kwargs...) where {TX, TY}
    return BatchedLogLikelihood_NoGradient(ssm, data, Val(use_median); kwargs...)
end
function BatchedLogLikelihood_NoGradient(ssm::StateSpaceModel{TX, TY}, data::AbstractVector{<:Observations{TY}}; kwargs...) where {TX, TY}
    return BatchedLogLikelihood_NoGradient(ssm, first(data); kwargs...)
end
function BatchedLogLikelihood_NoGradient(ssm::StateSpaceModel{TX, TY}, data::AbstractVector{<:AbstractVector{<:Observations{TY}}}; kwargs...) where {TX, TY}
    return BatchedLogLikelihood_NoGradient(ssm, first(first(data)); kwargs...)
end

model(fun::BatchedLogLikelihood_NoGradient) = model(fun.llh)
statespacemodel(fun::BatchedLogLikelihood_NoGradient{<:Observations, <:LogLikelihood_NoGradient{<:StateSpaceSMC}}) = statespacemodel(fun.llh)
nparticles(fun::BatchedLogLikelihood_NoGradient) = nparticles(fun.llh)

function set_batch!(llhb::BatchedLogLikelihood_NoGradient{TRAJ, use_median}, data::AbstractVector{TRAJ}) where {TRAJ, use_median}
    copy!(llhb.batch, data)
    return llhb
end

function (llhb::BatchedLogLikelihood_NoGradient{TRAJ, false})(p) where {TRAJ}
    llh = 0.0
    for data in llhb.batch
        llh_fun = LogLikelihood_NoGradient(llhb.llh, data; recycle=true)
        llh += llh_fun(p)::Float64
    end
    return llh
end
function (llhb::BatchedLogLikelihood_NoGradient{TRAJ, true})(p) where {TRAJ}
    return median(
        LogLikelihood_NoGradient(llhb.llh, data; recycle=true)(p)::Float64
        for data in llhb.batch
    )
end

################################################################################################################################

struct QFunctor{PF <: SMC, METHOD <: AfsMethod, AD <: ADBackend, CACHE}
    pf::PF
    method::METHOD
    ad_backend::AD
    cache::CACHE
    function QFunctor(ssm::StateSpaceModel, data, method::AfsMethod=DEFAULT_AFSMETHOD(), ad_backend::ADBackend=ADBackend(ssm); nparticles::Integer)
        summary = OfflineSummary(QGradient(ad_backend, method))
        bf = BootstrapFilter(ssm, data)
        pf = SMC(
            bf, parameter_template(ssm), nparticles,
            ParticleHistoryLength(summary; logCnorm=StaticFiniteHistory{1}()),
            required_amortized_computations(summary),
        )
        cache = SMCCache(pf)
        return new{typeof(pf), typeof(method), typeof(ad_backend), typeof(cache)}(pf, method, ad_backend, cache)
    end
    function QFunctor(qfun::QFunctor, data::Observations; recycle::Bool=false)
        bf = model(qfun.pf)::BootstrapFilter
        ssm = statespacemodel(bf)
        eltype(data) == eltype(bf.observations) || throw(ArgumentError("observation type must coincide with the one for the given qfun"))
        bf = BootstrapFilter(ssm, data)::typeof(model(qfun.pf))
        pf = SMC(qfun.pf, bf; recycle)
        cache = recycle ? qfun.cache : deepcopy(qfun.cache)
        return new{typeof(pf), typeof(qfun.method), typeof(qfun.ad_backend), typeof(cache)}(pf, qfun.method, qfun.ad_backend, cache)
    end
end

model(fun::QFunctor) = model(fun.pf)
statespacemodel(fun::QFunctor{<:StateSpaceSMC}) = statespacemodel(fun.pf)
nparticles(fun::QFunctor) = nparticles(fun.pf)

function make_template(f::QFunctor{PF}) where {PF}
    pf = f.pf::PF # helps with type instability of the return type (but property access is still unstable?!?)
    return template_maker(QGradientAf(f.ad_backend), pf.fkm)()
end
function make_template(f::QFunctor, ::Val{:gradient})
    return make_template(f).derivs[1]
end

function set_parameters!(f::QFunctor, p)
    reset!(f.pf, p)
    offlinefilter!(f.pf, f.cache)
    llh = f.pf.history_pf.logCnorm[end]
    return llh::Float64
end

function (f::QFunctor)(gradient, p)
    summary = OfflineSummary(QGradient(p, f.ad_backend, f.method))
    out = make_QResult(gradient)
    compute_summary!(out, f.pf, summary)
    return out.value
end
function (f::QFunctor)(::Val{:return}, p)
    summary = OfflineSummary(QGradient(p, f.ad_backend, f.method))
    out = make_template(f)
    compute_summary!(out, f.pf, summary)
    return out.value, out.derivs[1]
end
function (f::QFunctor)(p)
    summary = OfflineSummary(QFunction(p, f.method))
    return compute_summary(f.pf, summary)
end

################################################################################################################################

struct BatchedQFunctor{TRAJ <: Observations, QFUN <: QFunctor, TMP}
    qfuns::Vector{QFUN}
    qfun_template::QFUN # NB qfun_template and qfuns[1] will share caches and history vectors
    tmp::TMP
    function BatchedQFunctor(ssm::StateSpaceModel{TX, TY}, data::Observations{TY}, args...; kwargs...) where {TX, TY}
        TRAJ = typeof(data)
        qfun = QFunctor(ssm, data, args...; kwargs...)
        QFUN = typeof(qfun)
        tmp = make_template(qfun, Val(:gradient))
        return new{TRAJ, QFUN, typeof(tmp)}(QFUN[], qfun, tmp)
    end
end

function BatchedQFunctor(ssm::StateSpaceModel{TX, TY}, data::AbstractVector{<:Observations{TY}}, args...; kwargs...) where {TX, TY}
    return BatchedQFunctor(ssm, first(data), args...; kwargs...)
end
function BatchedQFunctor(ssm::StateSpaceModel{TX, TY}, data::AbstractVector{<:AbstractVector{<:Observations{TY}}}, args...; kwargs...) where {TX, TY}
    return BatchedQFunctor(ssm, first(first(data)), args...; kwargs...)
end

model(fun::BatchedQFunctor) = model(fun.qfun_template)
statespacemodel(fun::BatchedQFunctor{<:Observations, <:QFunctor{<:StateSpaceSMC}}) = statespacemodel(fun.qfun_template)
nparticles(fun::BatchedQFunctor) = nparticles(fun.qfun_template)

make_template(qfunb::BatchedQFunctor) = make_template(qfunb.qfun_template)
make_template(qfunb::BatchedQFunctor, ::Val{:gradient}) = make_template(qfunb.qfun_template, Val(:gradient))

function set_batch!(qfunb::BatchedQFunctor{TRAJ}, data::AbstractVector{TRAJ}) where {TRAJ}
    Base.has_offset_axes(data) && throw(ArgumentError("data cannot have offset axes (this could be relaxed)"))
    batchsize = length(data)
    batchsize_before = length(qfunb.qfuns)
    resize!(qfunb.qfuns, batchsize)
    @inbounds if iszero(batchsize_before)
        qfunb.qfuns[1] = QFunctor(qfunb.qfun_template, data[1]; recycle=true)
        for i in 2:batchsize
            qfunb.qfuns[i] = QFunctor(qfunb.qfun_template, data[i]; recycle=false)
        end
    elseif batchsize ≤ batchsize_before
        for i in 1:batchsize
            qfunb.qfuns[i] = QFunctor(qfunb.qfuns[i], data[i]; recycle=true)
        end
    else
        for i in 1:batchsize_before
            qfunb.qfuns[i] = QFunctor(qfunb.qfuns[i], data[i]; recycle=true)
        end
        for i in batchsize_before+1:batchsize
            qfunb.qfuns[i] = QFunctor(qfunb.qfun_template, data[i]; recycle=false)
        end
    end
    return qfunb
end

function set_parameters!(qfunb::BatchedQFunctor, p)
    llh = 0.0
    for qfun in qfunb.qfuns
        llh += set_parameters!(qfun, p)::Float64
    end
    return llh::Float64
end

function (qfunb::BatchedQFunctor)(gradient, p)
    tmp = qfunb.tmp
    qval = 0.0
    fill!(gradient, zero(eltype(gradient)))
    for qfun in qfunb.qfuns
        qval += qfun(tmp, p)::Float64
        @turbo @. gradient += tmp
    end
    return qval
end
function (qfunb::BatchedQFunctor)(::Val{:return}, p)
    grad = similar(p)
    qval = qfunb(grad, p)
    return qval, grad
end
function (qfunb::BatchedQFunctor)(p)
    qval = 0.0
    for qfun in qfunb.qfuns
        qval += qfun(p)::Float64
    end
    return qval
end

################################################################################################################################

struct ThreadedBatchedQFunctor{TRAJ <: Observations, QFUN <: QFunctor, TMP}
    qfuns::Vector{QFUN}
    qfun_template::QFUN # NB qfun_template and qfuns[1] will share caches and history vectors
    tmps::Vector{TMP} # TODO an autocache could be used, but would require synchronization when growing the gradient
    function ThreadedBatchedQFunctor(ssm::StateSpaceModel{TX, TY}, data::Observations{TY}, args...; kwargs...) where {TX, TY}
        qfun = QFunctor(ssm, data, args...; kwargs...)
        tmp = make_template(qfun, Val(:gradient))
        TRAJ, QFUN, TMP = typeof(data), typeof(qfun), typeof(tmp)
        return new{TRAJ, QFUN, typeof(tmp)}(QFUN[], qfun, TMP[])
    end
end

function ThreadedBatchedQFunctor(ssm::StateSpaceModel{TX, TY}, data::AbstractVector{<:Observations{TY}}, args...; kwargs...) where {TX, TY}
    return ThreadedBatchedQFunctor(ssm, first(data), args...; kwargs...)
end
function ThreadedBatchedQFunctor(ssm::StateSpaceModel{TX, TY}, data::AbstractVector{<:AbstractVector{<:Observations{TY}}}, args...; kwargs...) where {TX, TY}
    return ThreadedBatchedQFunctor(ssm, first(first(data)), args...; kwargs...)
end

model(fun::ThreadedBatchedQFunctor) = model(fun.qfun_template)
statespacemodel(fun::ThreadedBatchedQFunctor{<:Observations, <:QFunctor{<:StateSpaceSMC}}) = statespacemodel(fun.qfun_template)
nparticles(fun::ThreadedBatchedQFunctor) = nparticles(fun.qfun_template)

make_template(qfunb::ThreadedBatchedQFunctor) = make_template(qfunb.qfun_template)
make_template(qfunb::ThreadedBatchedQFunctor, ::Val{:gradient}) = make_template(qfunb.qfun_template, Val(:gradient))

function set_batch!(qfunb::ThreadedBatchedQFunctor{TRAJ}, data::AbstractVector{TRAJ}) where {TRAJ}
    Base.has_offset_axes(data) && throw(ArgumentError("data cannot have offset axes (this could be relaxed)"))
    batchsize = length(data)
    batchsize_before = length(qfunb.qfuns)
    @assert length(qfunb.tmps) == batchsize_before
    resize!(qfunb.qfuns, batchsize)
    resize!(qfunb.tmps, batchsize)
    @inbounds if iszero(batchsize_before)
        qfunb.qfuns[1] = QFunctor(qfunb.qfun_template, data[1]; recycle=true)
        qfunb.tmps[1] = make_template(qfunb, Val(:gradient))
        for i in 2:batchsize
            qfunb.qfuns[i] = QFunctor(qfunb.qfun_template, data[i]; recycle=false)
            qfunb.tmps[i] = make_template(qfunb, Val(:gradient))
        end
    elseif batchsize ≤ batchsize_before
        for i in 1:batchsize
            qfunb.qfuns[i] = QFunctor(qfunb.qfuns[i], data[i]; recycle=true)
        end
    else
        for i in 1:batchsize_before
            qfunb.qfuns[i] = QFunctor(qfunb.qfuns[i], data[i]; recycle=true)
        end
        for i in batchsize_before+1:batchsize
            qfunb.qfuns[i] = QFunctor(qfunb.qfun_template, data[i]; recycle=false)
            qfunb.tmps[i] = make_template(qfunb, Val(:gradient))
        end
    end
    return qfunb
end

function set_parameters!(qfunb::ThreadedBatchedQFunctor, p)
    llh = Threads.Atomic{Float64}(0.0)
    for qfun in qfunb.qfuns
        Threads.atomic_add!(llh, set_parameters!(qfun, p)::Float64)
    end
    return llh[]
end

function (qfunb::ThreadedBatchedQFunctor)(gradient, p)
    qval = Threads.Atomic{Float64}(0.0)
    fill!(gradient, zero(eltype(gradient)))
    Threads.@threads for i in eachindex(qfunb.qfuns)
        qfun = @inbounds qfunb.qfuns[i]
        tmp = @inbounds qfunb.tmps[i]
        Threads.atomic_add!(qval, qfun(tmp, p)::Float64)
        # @turbo @. gradient += tmp # TODO could be done here, with additional synchronization
    end
    for tmp in qfunb.tmps
        @turbo @. gradient += tmp
    end
    return qval[]
end
function (qfunb::ThreadedBatchedQFunctor)(::Val{:return}, p)
    grad = similar(p)
    qval = qfunb(grad, p)
    return qval, grad
end
function (qfunb::ThreadedBatchedQFunctor)(p)
    qval = Threads.Atomic{Float64}(0.0)
    Threads.@threads for qfun in qfunb.qfuns
        Threads.atomic_add!(qval, qfun(p)::Float64)
    end
    return qval[]
end

################################################################################################################################

struct SummaryFunctor{NAMES, PF <: SMC, CACHE}
    pf::PF
    cache::CACHE
    function SummaryFunctor(ssm::StateSpaceModel, data::Observations, summaries::SummaryTuple{N, NAMES}; nparticles::Integer) where {N, NAMES}
        bf = BootstrapFilter(ssm, data)
        pf = SMC(bf, parameter_template(ssm), nparticles, summaries)
        cache = SMCCache(pf)
        return new{NAMES, typeof(pf), typeof(cache)}(pf, cache)
    end
    function SummaryFunctor(fun::SummaryFunctor{NAMES}, data::Observations; recycle::Bool=false) where {NAMES}
        bf = model(fun.pf)::BootstrapFilter
        ssm = statespacemodel(bf)
        eltype(data) == eltype(bf.observations) || throw(ArgumentError("observation type must coincide with the one for the given fun"))
        bf = BootstrapFilter(ssm, data)::typeof(model(fun.pf))
        pf = SMC(fun.pf, bf; recycle)
        cache = recycle ? fun.cache : deepcopy(fun.cache)
        return new{NAMES, typeof(pf), typeof(cache)}(pf, cache)
    end
end

model(fun::SummaryFunctor) = model(fun.pf)
statespacemodel(fun::SummaryFunctor{NAMES, <:StateSpaceSMC}) where {NAMES} = statespacemodel(fun.pf)
nparticles(fun::SummaryFunctor) = nparticles(fun.pf)

# NB copy also promotes types to something that can be averaged
@generated function (fun::SummaryFunctor{NAMES})(p; copy::Bool=false) where {NAMES}
    retval = Expr(:tuple, (
        :(
            $name = if isa(fun.pf.summaries.$name, RunningSummary)
                let svalue = fun.pf.history_run.$name
                    if copy
                        T = typeof(one(eltype(svalue))/2)
                        collect(T, svalue)
                    else
                        svalue
                    end
                end
            else
                let svalue = compute_summary(fun.pf, $(QuoteNode(name)))
                    if copy
                        let svalue = deepcopy(svalue)
                            if isa(svalue, Number)
                                T = typeof(one(svalue)/2)
                                convert(T, svalue)
                            else
                                svalue
                            end
                        end
                    else
                        svalue
                    end
                end
            end
        )
        for name in NAMES
    )...)
    return quote
        reset!(fun.pf, p)
        offlinefilter!(fun.pf, fun.cache)
        return $retval
    end
end

################################################################################################################################

struct BatchedSummaryFunctor{TRAJ <: Observations, FUN <: SummaryFunctor}
    fun::FUN
    batch::Vector{TRAJ}
    function BatchedSummaryFunctor(ssm::StateSpaceModel{TX, TY}, data::Observations{TY}, args...; kwargs...) where {TX, TY}
        TRAJ = typeof(data)
        fun = SummaryFunctor(ssm, data, args...; kwargs...)
        return new{TRAJ, typeof(fun)}(fun, TRAJ[])
    end
end

function BatchedSummaryFunctor(ssm::StateSpaceModel{TX, TY}, data::AbstractVector{<:Observations{TY}}, args...; kwargs...) where {TX, TY}
    return BatchedSummaryFunctor(ssm, first(data), args...; kwargs...)
end
function BatchedSummaryFunctor(ssm::StateSpaceModel{TX, TY}, data::AbstractVector{<:AbstractVector{<:Observations{TY}}}, args...; kwargs...) where {TX, TY}
    return BatchedSummaryFunctor(ssm, first(first(data)), args...; kwargs...)
end

model(fun::BatchedSummaryFunctor) = model(fun.fun)
statespacemodel(fun::BatchedSummaryFunctor{<:Observations, <:SummaryFunctor{NAMES, <:StateSpaceSMC}}) where {NAMES} = statespacemodel(fun.fun)
nparticles(fun::BatchedSummaryFunctor) = nparticles(fun.fun)

function set_batch!(funb::BatchedSummaryFunctor{TRAJ}, data::AbstractVector{TRAJ}) where {TRAJ}
    isempty(data) && throw(ArgumentError("empty batch given"))
    copy!(funb.batch, data)
    return funb
end

function (funb::BatchedSummaryFunctor)(p)
    isempty(funb.batch) && error("Before calling a BatchedSummaryFunctor, a batch must be set using set_batch!")
    retvals = _accumulate_summaries(funb, p)
    return diveq!(retvals, length(funb.batch))
end

function (funb::BatchedSummaryFunctor)(p, batches)
    isempty(batches) && error("at least one batch should be given")
    # First batch (allocate output)
    batch = @inbounds batches[begin]
    set_batch!(funb, batch)
    retvals = _accumulate_summaries(funb, p)
    count = length(batch)
    # Other batches
    for k in (firstindex(batches)+1):lastindex(batches)
        batch = @inbounds batches[k]
        set_batch!(funb, batch)
        retvals = _accumulate_summaries!(retvals, funb, p)
        count += length(batch)
    end
    return diveq!(retvals, count)
end

function _accumulate_summaries(funb::BatchedSummaryFunctor, p)
    # First call (to allocate output)
    # NB assumes current batch is not empty
    fun = SummaryFunctor(funb.fun, @inbounds(funb.batch[1]); recycle=true)
    values = fun(p; copy=true)
    # Now loop over the rest of the batch
    for k in 2:length(funb.batch)
        data = @inbounds funb.batch[k]
        fun = SummaryFunctor(funb.fun, data; recycle=true)
        values = addeq!(values, fun(p))
    end
    return values
end

function _accumulate_summaries!(values::NamedTuple, funb::BatchedSummaryFunctor, p)
    for data in funb.batch
        fun = SummaryFunctor(funb.fun, data; recycle=true)
        values = addeq!(values, fun(p))
    end
    return values
end

function addeq!(values_before::NamedTuple{NAMES}, values::NamedTuple{NAMES}) where {NAMES}
    N = length(NAMES)
    new_values = ntuple(
        i -> if isa(values_before[i], AbstractArray)
            values_before[i] .+= values[i]
            values_before[i]
        else
            values_before[i] + values[i]
        end,
        Val(N)
    )
    return NamedTuple{NAMES}(new_values)::typeof(values_before)
end

function diveq!(values::NamedTuple{NAMES}, x::Real) where {NAMES}
    N = length(NAMES)
    new_values = ntuple(
        i -> if isa(values[i], AbstractArray)
            values[i] ./= x
            values[i]
        else
            values[i] / x
        end,
        Val(N)
    )
    return NamedTuple{NAMES}(new_values)::typeof(values)
end
