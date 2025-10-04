include("types.jl")
include("io.jl")
include("residuals.jl")

# Load linear solvers
include("solvers/cudss_sequential.jl")
include("solvers/cudss_batched.jl")
include("solvers/cudss_ubatch.jl")
include("solvers/klu_sequential.jl")

using ArgParse
using BenchmarkTools
using LinearAlgebra
using Printf
using SparseArrays
using Statistics

using CSV
using DataFrames
using JSON

using CUDA
using NVTX

const SOLVER_MAP = Dict{String,Any}(
    "CUDSS_Sequential" => CUDSSSequentialSolver,
    "CUDSSSequential" => CUDSSSequentialSolver,
    "CUDSSBatchSolver" => CUDSSBatchSolver,
    "CUDSS_Batch" => CUDSSBatchSolver,
    "CUDSSUniformBatchSolver" => CUDSSUniformBatchSolver,
    "CUDSS_UniformBatch" => CUDSSUniformBatchSolver,
    "KLUSequentialSolver" => KLUSequentialSolver,
    "KLUSequential" => KLUSequentialSolver,
    "KLU_Sequential" => KLUSequentialSolver,
)

"""
    prep_benchmark_data(h5path; nbatch=-1)

Load batch linear system data from HDF5 file in CPU/GPU format.

!!! warning
    Only supports uniform batch data
"""
function prep_benchmark_data(h5path; nbatch=-1)
    meta = h5read(h5path, "meta")

    B_cpu = load_from_h5(h5path; nbatch=nbatch)
    B_gpu = BatchLinearSystemGPU(B_cpu)

    if !B_cpu.is_uniform
        return B_cpu, B_gpu, nothing
    end

    B_uni = UniformBatchLinearSystemGPU(B_cpu)

    return B_cpu, B_gpu, B_uni
end

function benchmark(B, solver; nsolve=1)
    if !isa(B, batch_type(solver))
        error("Incompatible batch type $(typeof(B)) for solver $(typeof(solver))")
    end

    # Solve batch system once to get residuals
    CUDA.@sync solve!(B, solver)
    R = compute_residuals(B; to_cpu=true)

    # Benchmark analyze/factorize/solve
    b = @benchmark CUDA.@sync begin solve!($B, $solver; nsolve=$nsolve) end

    res = Dict(
        # Benchmark info
        "solver_type" => name(solver),
        "batch_size" => batch_size(B),
        "matrix_type" => string(B.mtype),
        "matrix_view" => string(B.mview),
        "nsolve" => nsolve,
        "num_threads_cpu" => Base.Threads.nthreads(),
        # Accuracy results (residuals)
        "residuals_l2_max" => maximum(norm.(R, 2)),
        "residuals_l2_avg" => mean(norm.(R, 2)),
        "residuals_linf_max" => maximum(norm.(R, Inf)),
        "residuals_linf_avg" => mean(norm.(R, Inf)),
        # Timings for analysis + factorization + nsolve solves
        "time_mean" => mean(b.times) / 1e9,
        "time_median" => median(b.times) / 1e9,
        "time_min" => minimum(b.times) / 1e9,
        "time_max" => maximum(b.times) / 1e9,
        "time_std" => std(b.times) / 1e9,
    )

    return res
end

function profile(B, solver)
    # Solve batch system only once, to generate profile traces
    NVTX.@range "$(name(solver))" begin
        CUDA.@sync solve!(B, solver; nsolve=1)
    end

    return nothing
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--dataset"
            help = "Path to h5-format dataset, or same of supported dataset"
            arg_type = String
            required = true
        "--solver", "-s"
            help = "Name of the linear solver"
            arg_type = String
            action = :append_arg
            required = true
        "--batch-size", "-b"
            help = "Batch size (number of linear systems to solve); multiple values can be provided, e.g., -b 16 -b 32"
            action = :append_arg
            arg_type = Int
            required = true
            range_tester = (x -> x >= 1)
        "--num-solve"
            help = "Number of RHS to solve per factorization (default is 1)"
            arg_type = Int
            default = 1
            range_tester = (x -> x >= 1)
        "--benchmark"
            action = :store_true
        "--blas-threads"
            help = "Number of BLAS threads to use (default is 1)"
            arg_type = Int
            default = 1
        "--profile"
            help = "Run in profiling mode (with Nsight Systems)"
            action = :store_true
        "--output"
            help = "Path to output CSV file"
            arg_type=String
            default=""
    end

    return parse_args(s)
end

function main_benchmark(args)

    # Overall setup
    BLAS.set_num_threads(args["blas-threads"])

    # Load data
    println("Loading data and prepping batch forms...")
    meta = h5read(args["dataset"], "meta")

    all_results = []

    for batch_size in args["batch-size"]
        println("  Batch size: $batch_size")
        if batch_size > meta["num_systems"]
            println("    WARNING: batch size ($batch_size) is larger than number of systems in dataset ($(meta["num_systems"]))")
            continue
        end

        B_cpu, B_gpu, B_uni = prep_benchmark_data(args["dataset"]; nbatch=batch_size)

        for solver_name in args["solver"]
            # Map solver names to solver instances
            println("Benchmarking linear solver $(solver_name)")
            solver_type = SOLVER_MAP[solver_name]
            solver = solver_type()

            # Figure out which data to use
            _batch_data = if typeof(solver) in [CUDSSUniformBatchSolver]
                B_uni
            elseif typeof(solver) in [CUDSSSequentialSolver, CUDSSBatchSolver]
                B_gpu
            elseif typeof(solver) in [KLUSequentialSolver]
                B_cpu
            else
                error("Unknown solver type: $(typeof(solver))")
            end

            # Execute benchmark for this solver & batch size
            res_benchmark = benchmark(_batch_data, solver; nsolve=args["num-solve"])
            # Put results in a dictionary
            res = Dict(
                "num_threads" => Base.Threads.nthreads(),
                "args" => deepcopy(args),
                "meta" => deepcopy(meta),
                "benchmark" => res_benchmark,
            )
            push!(all_results, res)
        end
    end

    # Export benchmark results to JSON for later use
    if args["output"] != ""
        open(args["output"], "w") do io
            JSON.print(io, all_results, 2)
        end
    end

    return res
end

function main_profile(args) end

function main_cl()
    parsed_args = parse_commandline()

    # Overall setup
    BLAS.set_num_threads(parsed_args["blas-threads"])

    # Options sanity checks
    if parsed_args["benchmark"] && parsed_args["profile"]
        println("ERROR: --benchmark and --profile are mutually exclusive")
        exit(1)
    end

    if parsed_args["benchmark"]
        main_benchmark(parsed_args)
        exit(0)
    end
    if parsed_args["profile"]
        main_profile(parsed_args)
        exit(0)
    end
    

    # Done
    exit(0)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cl()
    # TODO: options sanity checks
    # TODO: load data
    # TODO: execute benchmark or profile
    exit(0)
end
