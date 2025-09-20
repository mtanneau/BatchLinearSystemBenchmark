"""
    CUDSSBatchSolver

Solves batch of linear systems in batch using CUDSS' batch solver interface.
"""
@kwdef struct CUDSSBatchSolver
    fac::String="G"
    view::Char='F'
end

function solve!(B::BatchLinearSystemGPU, s::CUDSSBatchSolver; nsolve=1)
    
    CUDA.@sync begin
        solver = CudssBatchedSolver(B.As, s.fac, s.view)
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
