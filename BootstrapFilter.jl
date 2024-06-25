using Distributed


# instantiate and precompile environment in all processes
@everywhere begin
  using Pkg; Pkg.activate("/home/vincent/wastewater_inference/integrative_stat_modelling")
  Pkg.instantiate(); Pkg.precompile()
end

# stuff needed on workers and main
@everywhere begin

    using LinearAlgebra
    using Random
    using StaticArrays
    using Distributions
    using Plots
    # using PEtab # not running on the cluster and I think not needed.
    using CSV
    using DataFrames
    using JLD2
    using MCMCChains
    using MCMCChainsStorage
    using HDF5

    using SBML
    using SBMLToolkit
    using Catalyst
    # using ModelingToolkit

    using Particles
    using StaticDistributions


    # include the ParticleFilter Setup
    include("CaseCountModel.jl")

    # set dimension of the state space
    const m_Λ = 13
    const state_space_dimension = 3*m_Λ + 2

    # Slurm Job-array

    task_id_str = get(ENV, "SLURM_ARRAY_TASK_ID", 0)
    task_id = parse(Int64, task_id_str)
    
    particles_set = [200,500,1000]

    # set hyperparamters
    niter = 100000
    nparticles = particles_set[task_id+1]

end

# stuff only needed on workers
@everywhere workers() begin

    using PyCall
    pypesto = pyimport("pypesto")

    # include utilities for pypesto to MCMCChains
    include("utilities.jl")

    ssm = CaseCountModel(m_Λ = m_Λ, I_init = 100)
    
    data = load("/home/vincent/wastewater_inference/integrative_stat_modelling/data/synth_casecount_data.jld2", "synth_data")
    θ_0 = Particles.parameter_template(ssm) 
    logp = LogPosterior(ssm, data, nparticles)

    # for pypesto we need the negative log-likelihood
    neg_llh = let logp =logp
        p -> begin
            return -logp(p)
        end
    end

    # transform to pypesto objective
    objective = pypesto.Objective(fun=neg_llh)

    problem = pypesto.Problem(
        objective,
        x_names=["shape", "scale", "pi_ua"],
        lb=[-1, -1, -1], # parameter bounds
        ub=[1, 1, 1], # NB for sampling it is usually better if you remap parameters to (-∞, ∞)
        copy_objective=false, # important
    )

    # specify sampler
    sampler = pypesto.sample.AdaptiveMetropolisSampler()

    # sample start value
    x0 = θ_0

    # sample
    function chain()
        result = pypesto.sample.sample(
                    problem,
                    n_samples=niter,
                    x0=x0, # starting point
                    sampler=sampler,
                    )
        return  Chains_from_pypesto(result)
    end
end

jobs = [@spawnat(i, @timed(chain())) for i in workers()]

all_chains = map(fetch, jobs)

chains = all_chains[1].value.value.data

for j in 2:nworkers()
    global chains
    chains = cat(chains, all_chains[j].value.value.data, dims=(3,3))
end

chs = MCMCChains.Chains(chains, [:gamma, :kappa, :beta, :tevent, :scaling, :lp])
complete_chain = set_section(chs, Dict(:parameters => [:gamma, :kappa, :beta, :tevent, :scaling], :internals => [:lp]))
stop_time = mean([all_chains[i].time for i in 1:nworkers()])
complete_chain = setinfo(complete_chain, (start_time=1.0, stop_time=stop_time))

print("Mean duration per chain: ", stop_time)
# store results
h5open("./output/synth_data_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.h5", "w") do f
  write(f, complete_chain)
end

open("./output/time_synth_data_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.txt", "w") do file
    write(file, stop_time)
end
