using Pkg
Pkg.activate(@__DIR__)

Pkg.develop(path=joinpath(@__DIR__, "..", "..", "StaticDistributions"))
Pkg.develop(path=joinpath(@__DIR__, ".."))

Pkg.add("Revise")
Pkg.add("Distributions")
Pkg.add("PlotlyJS")
Pkg.add("StaticArrays")

# sampling notebooks
Pkg.add("AdvancedMH")
Pkg.add("AdvancedHMC")
Pkg.add("MCMCChains")
Pkg.add("StatsPlots")
Pkg.add("Dierckx")

# profiling
Pkg.add("BenchmarkTools")
Pkg.add("ProfileSVG")
Pkg.add("JET")
Pkg.add("SnoopCompile")

PYTHON = strip(read(`which python3`, String))
ENV["PYTHON"] = PYTHON
Pkg.add("PyCall")
Pkg.build("PyCall")

# Create Python venv
VENV = joinpath(@__DIR__, "python_venv")
run(`$PYTHON -m venv $VENV`)

# Install python packages
VPYTHON = joinpath(VENV, "bin", "python3")
run(`$VPYTHON -m pip install particles`)
run(`$VPYTHON -m pip install matplotlib`)

# Create init script for python
write(
    joinpath(@__DIR__, "init_python.jl"),
    "ENV[\"PYCALL_JL_RUNTIME_PYTHON\"] = \"$VPYTHON\"\nusing PyCall\n"
)
