"""
    CUDSSBatchSolver

Solves batch of linear systems in batch using CUDSS' batch solver interface.
"""
@kwdef struct CUDSSBatchSolver end

batch_type(::CUDSSBatchSolver) = BatchLinearSystemGPU
name(::CUDSSBatchSolver) = "CUDSS_Batched"

function solve!(B::BatchLinearSystemGPU, s::CUDSSBatchSolver; nsolve=1)

    fac = cudss_matrix_type(matrix_type(B))
    view = cudss_matrix_view(matrix_view(B))
    
    CUDA.@sync begin
        solver = CudssBatchedSolver(B.As, fac, view)
        NVTX.@range "analysis" begin
            cudss("analysis", solver, B.xs, B.bs)
        end
        NVTX.@range "factorization" begin
            cudss("factorization", solver, B.xs, B.bs)
        end
        NVTX.@range "solve" begin
            for _ in 1:nsolve
                cudss("solve", solver, B.xs, B.bs)
            end
        end
    end

    return nothing
end
