"""
    CUDSSUniformBatchSolver

Solves batch of linear systems in batch using CUDSS' uniform batch solver interface.
"""
@kwdef struct CUDSSUniformBatchSolver
    fac::String="G"
    view::Char='F'
end

function solve!(B::UniformBatchLinearSystemGPU{T}, s::CUDSSUniformBatchSolver; nsolve=1) where{T}
    m = B.nrows
    n = B.ncols
    k = B.batch_size

    cudss_rhs = CudssMatrix(T, m; nbatch=k)
    cudss_set(cudss_rhs, B.b_dat)
    cudss_sol = CudssMatrix(T, n; nbatch=k)
    cudss_set(cudss_sol, B.x_dat)

    CUDA.@sync begin
        NVTX.@range "CUDSS ubatch solver setup" begin
            batchsolver = CudssSolver(B.rowPtr, B.colVal, B.nzVal, s.fac, s.view)
            cudss_set(batchsolver, "ubatch_size", k)
            cudss_set(batchsolver, "ubatch_index", -1)
        end

        NVTX.@range "CUDSS ubatch analysis" begin
            cudss("analysis", batchsolver, cudss_sol, cudss_rhs)
        end
        NVTX.@range "CUDSS ubatch factorization" begin
            cudss("factorization", batchsolver, cudss_sol, cudss_rhs)
        end
        NVTX.@range "CUDSS ubatch solve" begin
            for _ in 1:nsolve
                cudss("solve", batchsolver, cudss_sol, cudss_rhs)
            end
        end
    end

    return nothing
end
