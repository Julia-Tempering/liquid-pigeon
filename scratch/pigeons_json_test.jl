using BridgeStan
using Pigeons
using Random
using CSV
using DataFrames
using JSON

array3d = reshape(1:27, 3, 3, 3)
array = [array3d[:, :, i] for i in 1:size(array3d, 3)]
vector_of_vectors = [[[array3d[i, j, k] for k in 1:size(array3d, 3)] for j in 1:size(array3d, 2)] for i in 1:size(array3d, 1)]
x = Pigeons.json(test=vector_of_vectors)

println(x)
println(typeof(x))
open("pigeonsjson_test.json", "w") do file
    write(file, x)
end
