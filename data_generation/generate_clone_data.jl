using DataFrames
using CSV

function generate_clones(num_copies::Int)
    values1 = (10, 1, 9, 2, 8, 3, 7, 4, 6, 5)
    values2 = (1, 10, 2, 9, 3, 8, 4, 7, 5, 6)

    df = DataFrame(bin_id=String[], Clone_0=Int[], Clone_1=Int[], Clone_2=Int[], 
                    Clone_3=Int[], Clone_4=Int[], Clone_5=Int[], Clone_6=Int[], 
                    Clone_7=Int[], Clone_8=Int[], Clone_9=Int[])

    for i in 1:num_copies
        bin_label = "bin_$(i-1)"
        push!(df, (bin_label, values1...))
    end

    for i in (num_copies + 1):(2 * num_copies)
        bin_label = "bin_$(i-1)"
        push!(df, (bin_label, values2...))
    end

    return df
end

data = generate_clones(500000)

CSV.write("data/10-clones-1000000.tsv", data, delim='\t')

println(first(data, 10))  
println(last(data, 10))   
