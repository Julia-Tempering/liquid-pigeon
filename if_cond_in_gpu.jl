using CUDA

params = Float32[0.1376443098077659, 0.015517111934940044, 0.5367500810031308, 0.3100799338684187]

start_time = time_ns()
if any(x -> x < 0 || x > 1, params) #|| abs(sum(params) - 1) > 1e-5
    println("-Inf")  
end
time_cpu = (time_ns()-start_time)/1e9

cu_params = CuArray(params)

start_time = time_ns()
if any(x -> x < 0 || x > 1, cu_params) #|| abs(sum(cu_params) - 1) > 1e-5
    println("-Inf")  
end
time_gpu = (time_ns()-start_time)/1e9

println("gpu time: $time_gpu ,cpu time: $time_cpu")

function test_cuda()
    if any(x -> x < 0 || x > 1, cu_params)
        println("-Inf")
    end
        # CUDA.zeros(1)
end

CUDA.@sync profile = @time begin
    test_cuda()
end

println(profile)
