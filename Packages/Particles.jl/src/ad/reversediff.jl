import .ReverseDiff

export ReverseDiffAD

struct ReverseDiffAD <: ADBackend end

gradient!(grad::AbstractVector, f, x, ::ReverseDiffAD) = ReverseDiff.gradient!(grad, f, x)
gradient!(dres::GradientDiffResult, f, x, ::ReverseDiffAD) = ReverseDiff.gradient!(dres, f, x)
