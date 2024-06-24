import .Zygote

export ZygoteAD

struct ZygoteAD <: ADBackend end

function gradient!(grad::AbstractVector, f, x, ::ZygoteAD)
    res_zygote = Zygote.gradient(f, x)
    _copy_zygote_grad!(grad, res_zygote[1])
    return grad
end
function gradient!(dres::GradientDiffResult, f, x, ::ZygoteAD)
    res_zygote = Zygote.withgradient(f, x)
    dres.value = res_zygote.val
    _copy_zygote_grad!(dres.derivs[1], res_zygote.grad[1])
    return dres
end

@inline function _copy_zygote_grad!(grad::AbstractVector, grad_zygote::AbstractVector)
    copy!(grad, grad_zygote)
    return nothing
end
@inline function _copy_zygote_grad!(grad::AbstractVector, grad_zygote::Nothing)
    # It means the output of f does not depend on x0
    # Sometimes it may happen, e.g., for an initial state distribution independent of the parameters
    fill!(grad, zero(eltype(grad)))
    return nothing
end
