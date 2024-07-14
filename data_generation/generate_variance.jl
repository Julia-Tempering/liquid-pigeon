using DataFrames, CSV, Distributions

function generate_random_data()
    
    n1 = 500000  
    n2 = 500000
    mean1 = 0.4
    mean2 = -0.6
    std_dev = 0.02  

    data1 = Float64.(rand(Normal(mean1, std_dev), n1))
    data2 = Float64.(rand(Normal(mean2, std_dev), n2))

    df = DataFrame(Value = [data1; data2])

    return df
end

data = generate_random_data()

CSV.write("data/ctdna-1000000.tsv", data, delim='\t', decimal='.')

