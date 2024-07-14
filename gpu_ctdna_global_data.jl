using CUDA
using Pigeons
using Random
using Statistics
using Distributions
using CSV
using DataFrames
using InferenceReport
using SpecialFunctions
using Dates

CUDA.allowscalar(false)  

#global gpu data
global GLOBAL_CTDNA = Ref{CuArray{Float32, 1}}()
global GLOBAL_CLONE_CN_PROFILES = Ref{CuArray{Float32, 2}}()
# global GLOBAL_RHO = CuArray{Float32, 1}(undef, 5)# assume 2 clones for now
global GLOBAL_RHO = nothing

struct CtDNALogPotential
    ctdna::CuArray{Float32, 1}
    clone_cn_profiles::CuArray{Float32, 2}
    num_clones::Int
    n::Int
    scale::Float32
end

function load_data_to_gpu(ctdna_path, clones_path)
    ctdna_data = CSV.read(ctdna_path, DataFrame, delim='\t', header=false, types=[Float32])
    clones_data = CSV.read(clones_path, DataFrame, delim='\t')

    GLOBAL_CTDNA[] = CuArray(Float32.(Vector(ctdna_data[:, 1])))
    GLOBAL_CLONE_CN_PROFILES[] = CuArray(Float32.(Matrix(clones_data[:, 2:end])))
end


function log_t_pdf(x, v)
    result = - ((v + 1) / 2) .* log.(1 .+ (x .^ 2) ./ v)
    return result
end 


function (log_potential::CtDNALogPotential)(params)
    # start_time = time_ns()
    if any(x -> x < 0 || x > 1, params) || abs(sum(params)  - 1) > 1e-5
            return -Inf  #ensure rho is valid
    end
    # elapsed_time = (time_ns()-start_time)/1e9
    # println("params:$params ,elapsed_time:$elapsed_time")

    copyto!(GLOBAL_RHO, Float32.(params)) 
    total_sum = log_potential.clone_cn_profiles * GLOBAL_RHO
    mean_total_sum = mean(CUDA.reduce(+, total_sum) / length(total_sum))
    mu = log.(total_sum) .- log(mean_total_sum)
    
    degrees_of_freedom = 2
    scaled_mu = mu * log_potential.scale
    log_likelihoods = log_t_pdf((log_potential.ctdna .- scaled_mu) / log_potential.scale, degrees_of_freedom)
    log_likelihood = CUDA.reduce(+, log_likelihoods)
    return log_likelihood
end


function Pigeons.initialization(log_potential::CtDNALogPotential, rng::AbstractRNG, ::Int)
    #Random.seed!(1234)
    alpha = 1.0  
    rho = rand(rng, Dirichlet(log_potential.num_clones, alpha))  
    return rho  # cannot convert to cuarray
end

function Pigeons.sample_iid!(log_potential::CtDNALogPotential, replica, shared)
    rng = replica.rng
    new_state = rand(rng, Dirichlet(log_potential.num_clones, 1.0))

    @assert abs(sum(new_state) - 1) < 1e-5 "density not 1!"

    replica.state = new_state
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
        load_data_to_gpu(ctdna_path, clones_path)
        n = length(GLOBAL_CTDNA[])
        num_clones = size(GLOBAL_CLONE_CN_PROFILES[], 2)
        scale = 1.0
        global GLOBAL_RHO = CuArray{Float32,1}(undef,num_clones)

        log_potential = CtDNALogPotential(GLOBAL_CTDNA[],GLOBAL_CLONE_CN_PROFILES[],num_clones, n, scale)
        reference_potential = default_reference(log_potential)  

        time_taken = @elapsed begin
            pt = pigeons(
                target = log_potential,
                reference = reference_potential,
                record = [traces; record_default()],
                n_rounds = 10
            )
            #report(pt)
        end
        push!(times, time_taken)
        println(times)
    end
    return times
end

# ctdna_paths = ["data/ctdna-10000.tsv","data/ctdna-10000.tsv","data/ctdna-10000.tsv","data/ctdna-10000.tsv"]
# clones_paths = ["data/3-clones-10000.tsv","data/4-clones-10000.tsv","data/5-clones-10000.tsv","data/6-clones-10000.tsv"]
ctdna_paths = ["data/ctdna-1000.tsv"]
clones_paths = ["data/3-clones-1000-similar.tsv"]


times = main(ctdna_paths, clones_paths)