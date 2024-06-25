using MCMCChains

# function to transform pypesto chains to MCMCChains
function Chains_from_pypesto(result; kwargs...)
    trace_x = result.sample_result["trace_x"] # parameter values
    trace_neglogp = result.sample_result["trace_neglogpost"] # posterior values
    samples = Array{Float64}(undef, size(trace_x, 2), size(trace_x, 3) + 1, size(trace_x, 1))
    samples[:, begin:end-1, :] .= PermutedDimsArray(trace_x, (2, 3, 1))
    samples[:, end, :] = .-PermutedDimsArray(trace_neglogp, (2, 1))
    param_names = Symbol.(result.problem.x_names)
    return Chains(
        samples,
        vcat(param_names, :lp),
        (parameters = param_names, internals = [:lp]);
        kwargs...
    )
end
