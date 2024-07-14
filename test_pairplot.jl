using Pigeons
using Random
using StatsFuns
using InferenceReport
using PairPlots
using CairoMakie
using Makie, Tables
using MCMCChains
# using DynamicPPL

struct MyLogPotential
    n_trials::Int
    n_successes::Int
end

function (log_potential::MyLogPotential)(x)
    p1, p2 = x
    if !(0 < p1 < 1) || !(0 < p2 < 1)
        return -Inf64
    end
    p = p1 * p2
    return StatsFuns.binomlogpdf(log_potential.n_trials, p, log_potential.n_successes)
end


Pigeons.initialization(::MyLogPotential, ::AbstractRNG, ::Int) = [0.5, 0.5]

pt = pigeons(
        target = MyLogPotential(100, 50),
        reference = MyLogPotential(0, 0),
        record=[traces;record_default()],
        n_rounds = 2
    )

# options = InferenceReport.ReportOptions()
# inference = InferenceReport.Inference(pt)
# src_dir = mkpath("$(options.exec_folder)/src")
# context = InferenceReport.PostprocessContext(inference, src_dir, String[], Dict{String,Any}(), options)
# x = PairPlots.pairplot(InferenceReport.get_chains(context),bins=30)

# file = output_file(context, "pair_plot", "svg")

samples = Chains(pt)
my_plot = PairPlots.pairplot(samples) 

CairoMakie.save("pair_plot.png", my_plot)
