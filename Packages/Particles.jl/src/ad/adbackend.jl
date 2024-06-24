abstract type ADBackend end

const GradientDiffResult{T, V <: AbstractVector} =  DiffResults.MutableDiffResult{1, T, Tuple{V}}

make_DiffResult(value::Real, grad::AbstractVector) = DiffResults.MutableDiffResult(convert(Float64, value), (grad, ))
make_DiffResult(grad::AbstractVector) = make_DiffResult(0.0, grad)

# Methods to implement are
# gradient!(grad::AbstractVector, f, x, ::ADBackend)
# gradient!(grad::GradientDiffResult, f, x, ::ADBackend)
