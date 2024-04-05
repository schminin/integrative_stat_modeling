import .ForwardDiff

export ForwardDiffAD

struct ForwardDiffAD <: ADBackend end

gradient!(grad::AbstractVector, f, x, ::ForwardDiffAD) = ForwardDiff.gradient!(grad, f, x)
gradient!(dres::GradientDiffResult, f, x, ::ForwardDiffAD) = ForwardDiff.gradient!(dres, f, x)
