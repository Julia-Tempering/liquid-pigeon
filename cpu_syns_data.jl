using Pigeons
using Random
using Statistics
using Distributions
using CSV
using DataFrames
using InferenceReport
using PairPlots


struct CtDNALogPotential
    ctdna::Vector{Float32}
    clone_cn_profiles::Matrix{Float32}
    num_clones::Int
    n::Int
    scale::Float32
end


function log_t_pdf(x, v)
    result = -((v + 1) / 2) .* log.(1 .+ (x .^ 2) ./ v)
    return result
end
function (log_potential::CtDNALogPotential)(params)

    if any(x -> x < 0 || x > 1, params) || abs(sum(params) - 1) > 1e-5
        return -Inf
    end

    rho = params
    # println("rho:$rho")
    total_sum = log_potential.clone_cn_profiles * rho

    mean_total_sum = mean(total_sum)

    mu = log.(total_sum) .- log(mean_total_sum)

    degrees_of_freedom = 2
    scaled_mu = mu * log_potential.scale
    log_likelihoods = log_t_pdf((log_potential.ctdna .- scaled_mu) / log_potential.scale, degrees_of_freedom)
    log_likelihood = sum(log_likelihoods)
    # println("log_likelihood:$log_likelihood")
    return log_likelihood
end



function Pigeons.initialization(log_potential::CtDNALogPotential, rng::AbstractRNG, ::Int)
    #Random.seed!(1234)
    alpha = 1.0  
    rho = rand(rng, Dirichlet(log_potential.num_clones, alpha))
    println("rho:$rho")
    @assert abs(sum(rho) - 1) < 1e-5 "density not 1!"

    return rho
end

function Pigeons.sample_iid!(log_potential::CtDNALogPotential, replica, shared)
    #Random.seed!(1234)
    rng = replica.rng
    new_state = rand(rng, Dirichlet(log_potential.num_clones, 1.0))

    @assert abs(sum(new_state) - 1) < 1e-5  "density not 1!"

    replica.state = new_state
    
end


function load_data(ctdna_path, clones_path)
    ctdna_data = CSV.read(ctdna_path, DataFrame; delim='\t', header=true)
    clones_data = CSV.read(clones_path, DataFrame; delim='\t', header=true)

    ctdna_data = dropmissing(ctdna_data, :copy)

    joined_data = innerjoin(clones_data, ctdna_data, on = [:chr, :start, :end])

    clone_keys = names(clones_data)
    ctdna_keys = names(ctdna_data)
    clones_data_clean = select(joined_data, clone_keys)

    clones_data_clean.normal = clones_data_clean.E
    clones_data_clean.E = fill(2, nrow(clones_data_clean))

    ctdna_data_clean = select(joined_data, ctdna_keys)

    return ctdna_data_clean, clones_data_clean
end



function default_reference(log_potential::CtDNALogPotential)
    neutral_ctdna = ones(Float32, log_potential.n) * mean(log_potential.ctdna)
    neutral_cn_profiles = ones(size(log_potential.clone_cn_profiles))
    return CtDNALogPotential(neutral_ctdna, neutral_cn_profiles, log_potential.num_clones, log_potential.n, log_potential.scale)
end


function main(ctdna_paths, clones_paths)
    times = Float32[]
    for (ctdna_path, clones_path) in zip(ctdna_paths, clones_paths)
        println("processing: $ctdna_path and $clones_path")
        ctdna_data, clones_data = load_data(ctdna_path, clones_path)

        n = size(clones_data, 1)
        num_clones = size(clones_data, 2) - 3
        clone_cn_profiles = Matrix(clones_data[:, 4:end])
        ctdna = Vector{Float32}(ctdna_data[:, 4])
        scale = 1.0  

        log_potential = CtDNALogPotential(ctdna, clone_cn_profiles, num_clones, n, scale)
        reference_potential = default_reference(log_potential)

        time_taken = @elapsed begin
            pt = pigeons(
                target = log_potential,
                reference = reference_potential,
                record = [traces; record_default()],
                n_rounds = 14
            )
            # InferenceReport.default_postprocessors = [
            #     target_description,
            #     pair_plot,
            #     trace_plot, 
            #     moments,
            #     trace_plot_cumulative,    
            #     mpi_standard_out,
            #     lcb,
            #     gcb_progress,
            #     logz_progress,
            #     round_trip_progress,
            #     pigeons_summary,
            #     reproducibility_info
            # ]
            
         report(pt)    
        end
        push!(times, time_taken)
        println("run complete for $ctdna_path. time taken: $time_taken seconds.")
    end
    return times
end


ctdna_paths = ["syns_data/ctdna.tsv"]
clones_paths = ["syns_data/clone_cn_profiles.tsv"]

times = main(ctdna_paths, clones_paths)