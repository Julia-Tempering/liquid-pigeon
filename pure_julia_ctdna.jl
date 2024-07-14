using Pigeons
using Random
using Statistics
using Distributions
using CSV
using DataFrames
using InferenceReport


struct CtDNALogPotential
    ctdna::Vector{Float32}
    clone_cn_profiles::Matrix{Float32}
    num_clones::Int
    n::Int
    scale::Float32
end

# function (log_potential::CtDNALogPotential)(params)
#     # start_time = time_ns()
#     rho = params
#     if any(x -> x < 0 || x > 1, rho) || abs(sum(rho)  - 1) > 1e-5
#     # if any(x -> x < 0 || x > 1, rho) || sum(rho) != 1
#         return -Inf  #ensure rho is valid
#     end

#     total_sum = log_potential.clone_cn_profiles * rho
#     mean_total_sum = mean(total_sum)

#     mu = log.(total_sum) .- log(mean_total_sum)

#     log_likelihood = 0.0
#     degrees_of_freedom = 2

#     dist = TDist(degrees_of_freedom)

#     for i in 1:log_potential.n
#         dist_i = LocationScale(mu[i], log_potential.scale, dist)
#         log_likelihood += logpdf(dist_i, log_potential.ctdna[i])
#     end

#     return log_likelihood
# end

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
    rho = [0.33, 0.34, 0.33]
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
    ctdna_data = CSV.read(ctdna_path, DataFrame, delim='\t', header=false,types=[Float32])
    clones_data = CSV.read(clones_path, DataFrame, delim='\t')
    return ctdna_data, clones_data
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
        num_clones = size(clones_data, 2) - 1
        clone_cn_profiles = Matrix(clones_data[:, 2:end])
        ctdna = Vector{Float32}(ctdna_data[:, 1])
        scale = 1.0  

        log_potential = CtDNALogPotential(ctdna, clone_cn_profiles, num_clones, n, scale)
        reference_potential = default_reference(log_potential)

        time_taken = @elapsed begin
            pt = pigeons(
                target = log_potential,
                reference = reference_potential,
                record = [traces; record_default()],
                n_rounds = 15
            )
         report(pt)    
        end
        push!(times, time_taken)
        println("run complete for $ctdna_path. time taken: $time_taken seconds.")
    end
    return times
end


ctdna_paths = ["data/ctdna-1000.tsv"]
clones_paths = ["data/3-clones-1000-similar.tsv"]

times = main(ctdna_paths, clones_paths)