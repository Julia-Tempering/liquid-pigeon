using DataFrames
using CSV 
# df = CSV.read("syns_data/raw_clone_cn_profiles.tsv", DataFrame, delim="\t")
# df.normal = fill(2,nrow(df))
# CSV.write("syns_data/clone_cn_profiles.tsv", df, delim="\t")

# ctdna_df = CSV.read("syns_data/raw_ctdna.tsv", DataFrame, delim="\t")
# ctdna_df.end .= ctdna_df.end .- 1
# CSV.write("syns_data/ctdna.tsv", ctdna_df, delim="\t")

clones_data = CSV.read("syns_data/clone_cn_profiles.tsv",DataFrame, delim="\t")
A_clone = Vector{Int32}(clones_data.A)
normal_clone = Vector{Int32}(clones_data.normal)
diff = A_clone .- normal_clone
count_zero = count(x->x==0, diff)
not_2 = filter(x->x!=2, A_clone)
println(not_2)
println(size(diff))
println(count_zero)