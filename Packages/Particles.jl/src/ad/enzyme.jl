import .Enzyme

export EnzymeAD

struct EnzymeAD <: ADBackend end

function gradient!(grad::AbstractVector, f, x, ::EnzymeAD)
    Enzyme.gradient!(Enzyme.Reverse, grad, f, x)
    return grad
end
function gradient!(dres::GradientDiffResult, f, x, ::EnzymeAD)
    dres.value = withgradient!(dres.derivs[1], f, x, EnzymeAD())
    return dres
end

function withgradient!(grad, f, x, ::EnzymeAD)
    Enzyme.gradient!(Enzyme.Reverse, grad, f, x)
    return f(x)
end
