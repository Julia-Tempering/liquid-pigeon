using CUDA
using Pigeons
using Random
using Statistics
using Distributions
using CSV
using DataFrames
using InferenceReport
using SpecialFunctions

CUDA.allowscalar(false)  

function log_t_pdf(x, v)
    #if we care about gamma part, use the commented version
    # log_gamma_half_v_plus_1 = lgamma((v + 1) / 2)
    # log_gamma_half_v = lgamma(v / 2)
    # log_v_pi = log(v * π) / 2
    # result = log_gamma_half_v_plus_1 .- log_gamma_half_v .- log_v_pi .- ((v + 1) / 2) .* log.(1 .+ (x .^ 2) ./ v)

    result = - ((v + 1) / 2) .* log.(1 .+ (x .^ 2) ./ v)


    return result
end


struct CtDNALogPotential
    ctdna::CuArray{Float64}
    clone_cn_profiles::CuMatrix{Float64}
    num_clones::Int
    n::Int
    scale::Float64
end


function (log_potential::CtDNALogPotential)(params)
  
    if any(x -> x < 0 || x > 1, params) || abs(sum(params)  - 1) > 1e-5
        # if any(x -> x < 0 || x > 1, rho) || sum(rho) != 1
            return -Inf  #ensure rho is valid
    end

    rho = CuArray(params)
    total_sum = log_potential.clone_cn_profiles * rho
    mean_total_sum = mean(CUDA.reduce(+, total_sum) / length(total_sum))

    mu = log.(total_sum) .- log(mean_total_sum)
    
    degrees_of_freedom = 2
    scaled_mu = mu * log_potential.scale
    log_likelihoods = log_t_pdf((log_potential.ctdna .- scaled_mu) / log_potential.scale, degrees_of_freedom)
    log_likelihood = CUDA.reduce(+, log_likelihoods)

    x = log_potential.ctdna[1:5]
    println("params:$params")
    println("GLOBAL_CTDNA:$x")
    println("log_potentia:$log_likelihood")
    return log_likelihood
end


function Pigeons.initialization(log_potential::CtDNALogPotential, rng::AbstractRNG, ::Int)
    alpha = 1.0  
    rho = rand(rng, Dirichlet(log_potential.num_clones, alpha))
    @assert abs(sum(rho) - 1) < 1e-5 "density not 1!"

    return rho
end


function Pigeons.sample_iid!(log_potential::CtDNALogPotential, replica, shared)
    rng = replica.rng
    new_state = rand(rng, Dirichlet(log_potential.num_clones, 1.0))

    @assert abs(sum(new_state) - 1) < 1e-5 "density not 1!"

    replica.state = new_state
end


function load_data(ctdna_path, clones_path)
    ctdna_data = CSV.read(ctdna_path, DataFrame, delim='\t', header=false,types=[Float64])
    clones_data = CSV.read(clones_path, DataFrame, delim='\t')
    return ctdna_data, clones_data
end

function default_reference(log_potential::CtDNALogPotential)
    neutral_ctdna = randn(log_potential.n)
    neutral_cn_profiles = abs.(randn(size(log_potential.clone_cn_profiles)))
    return CtDNALogPotential(CuArray(neutral_ctdna), CuArray(neutral_cn_profiles), log_potential.num_clones, log_potential.n, log_potential.scale)
end

# function main()
#     ctdna_path = "data/ctdna-500.tsv"
#     clones_path = "data/2-clones-500.tsv"
#     ctdna_data, clones_data = load_data(ctdna_path, clones_path)

#     n = size(clones_data, 1)
#     num_clones = size(clones_data, 2) - 1
#     clone_cn_profiles = CuMatrix(Matrix(clones_data[:, 2:end]))
#     ctdna = CuArray(Vector{Float64}(ctdna_data[:, 1]))
#     scale = 1.0

#     log_potential = CtDNALogPotential(ctdna, clone_cn_profiles, num_clones, n, scale)
#     reference_potential = default_reference(log_potential)

#     pt = pigeons(
#         target = log_potential,
#         reference = reference_potential,
#         record = [traces; record_default()]
#     )
#     # report(pt)

#     println("Model run complete.")
# end

# @time main()

function main(ctdna_paths, clones_paths)
    times = Float64[]

    for (ctdna_path, clones_path) in zip(ctdna_paths, clones_paths)
        println("processing: $ctdna_path and $clones_path")
        ctdna_data, clones_data = load_data(ctdna_path, clones_path)

        n = size(clones_data, 1)
        num_clones = size(clones_data, 2) - 1
        clone_cn_profiles = CuMatrix(Matrix(clones_data[:, 2:end]))
        ctdna = CuArray(Vector{Float64}(ctdna_data[:, 1]))
        scale = 1.0

        log_potential = CtDNALogPotential(ctdna, clone_cn_profiles, num_clones, n, scale)
        reference_potential = default_reference(log_potential)

        time_taken = @elapsed begin
            pt = pigeons(
                target = log_potential,
                reference = reference_potential,
                record = [traces; record_default()],
                n_rounds = 1
            )
            # report(pt)
        end

        push!(times, time_taken)
        println("run complete for $ctdna_path. time taken: $time_taken seconds.")
    end

    return times
end

ctdna_paths = ["data/ctdna-5000.tsv"]
clones_paths = ["data/2-clones-5000.tsv"]
times = main(ctdna_paths, clones_paths)